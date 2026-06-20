import copy
import json
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - exercised in minimal envs
    yaml = None


def parse_scalar(raw: str) -> Any:
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    if raw.startswith("[") and raw.endswith("]"):
        body = raw[1:-1].strip()
        if not body:
            return []
        return [parse_scalar(item.strip()) for item in body.split(",")]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def safe_load_yaml(text: str) -> dict[str, Any]:
    if yaml is not None:
        return yaml.safe_load(text)
    return parse_simple_yaml(text)


def parse_simple_yaml(text: str) -> dict[str, Any]:
    """Parse the simple nested mapping subset used by this repo's configs."""

    lines: list[tuple[int, str, str]] = []
    anchors: dict[str, Any] = {}
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        lines.append((indent, raw_line.strip(), raw_line))

    def parse_node(index: int, indent: int) -> tuple[Any, int]:
        if index >= len(lines):
            return {}, index
        current_indent, stripped, _ = lines[index]
        if current_indent < indent:
            return {}, index
        if stripped.startswith("- "):
            return parse_list(index, current_indent)
        return parse_mapping(index, current_indent)

    def parse_mapping(index: int, indent: int) -> tuple[dict[str, Any], int]:
        result: dict[str, Any] = {}
        while index < len(lines):
            current_indent, stripped, raw_line = lines[index]
            if current_indent < indent:
                break
            if current_indent > indent:
                raise ValueError(f"Unexpected nested YAML line: {raw_line}")
            if stripped.startswith("- "):
                break
            if ":" not in stripped:
                raise ValueError(f"Unsupported YAML line without ':': {raw_line}")
            key, value = stripped.split(":", 1)
            key = key.strip()
            value = value.strip()
            index += 1
            if value == "":
                if (
                    index < len(lines)
                    and lines[index][0] == current_indent
                    and lines[index][1].startswith("- ")
                ):
                    result[key], index = parse_node(index, lines[index][0])
                elif index >= len(lines) or lines[index][0] <= current_indent:
                    result[key] = {}
                else:
                    result[key], index = parse_node(index, lines[index][0])
            elif value.startswith("&"):
                anchor_name, anchor_value = _split_anchor(value)
                if anchor_value:
                    result[key] = parse_scalar(anchor_value)
                elif (
                    index < len(lines)
                    and lines[index][0] == current_indent
                    and lines[index][1].startswith("- ")
                ):
                    result[key], index = parse_node(index, lines[index][0])
                elif index >= len(lines) or lines[index][0] <= current_indent:
                    result[key] = {}
                else:
                    result[key], index = parse_node(index, lines[index][0])
                anchors[anchor_name] = copy.deepcopy(result[key])
            elif value.startswith("*"):
                anchor_name = value[1:].strip()
                if anchor_name not in anchors:
                    raise ValueError(f"Unknown YAML anchor reference: {value}")
                result[key] = copy.deepcopy(anchors[anchor_name])
            else:
                result[key] = parse_scalar(value)
        return result, index

    def parse_list(index: int, indent: int) -> tuple[list[Any], int]:
        result: list[Any] = []
        while index < len(lines):
            current_indent, stripped, raw_line = lines[index]
            if current_indent < indent:
                break
            if current_indent > indent:
                raise ValueError(f"Unexpected nested YAML list line: {raw_line}")
            if not stripped.startswith("- "):
                break
            value = stripped[2:].strip()
            index += 1
            if value == "":
                if index >= len(lines) or lines[index][0] <= current_indent:
                    result.append(None)
                else:
                    child, index = parse_node(index, lines[index][0])
                    result.append(child)
            elif ":" in value and not value.startswith(("http://", "https://")):
                key, item_value = value.split(":", 1)
                item: dict[str, Any] = {}
                item[key.strip()] = parse_scalar(item_value.strip()) if item_value.strip() else {}
                result.append(item)
            else:
                result.append(parse_scalar(value))
        return result, index

    parsed, final_index = parse_node(0, lines[0][0] if lines else 0)
    if final_index != len(lines):
        _, _, raw_line = lines[final_index]
        raise ValueError(f"Unsupported YAML structure near: {raw_line}")
    if not isinstance(parsed, dict):
        raise ValueError("Top-level YAML document must be a mapping.")
    return parsed


def _split_anchor(value: str) -> tuple[str, str]:
    parts = value.split(None, 1)
    anchor_name = parts[0][1:].strip()
    if not anchor_name:
        raise ValueError(f"Invalid YAML anchor: {value}")
    anchor_value = parts[1].strip() if len(parts) > 1 else ""
    return anchor_name, anchor_value
