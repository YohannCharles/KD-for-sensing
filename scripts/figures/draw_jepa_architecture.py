from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 7


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "figures"

COLORS = {
    "ink": "#242424",
    "muted": "#6A6A6A",
    "line": "#505050",
    "panel": "#F7F8FA",
    "context": "#DDECF7",
    "context_edge": "#2F6F9F",
    "target": "#F7E7C3",
    "target_edge": "#A87016",
    "gps": "#E9DDF3",
    "gps_edge": "#7D5AA2",
    "loss": "#F4DAD7",
    "loss_edge": "#A84945",
    "downstream": "#DDEFE6",
    "downstream_edge": "#357C5C",
    "artifact": "#ECECEC",
    "white": "#FFFFFF",
}


def add_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    *,
    fc: str,
    ec: str,
    lw: float = 1.15,
    fontsize: float = 6.8,
    weight: str = "normal",
    color: str | None = None,
    zorder: int = 3,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.008,rounding_size=0.010",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        zorder=zorder,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=color or COLORS["ink"],
        fontweight=weight,
        linespacing=1.15,
        zorder=zorder + 1,
    )
    return patch


def add_label(ax, x: float, y: float, text: str, *, size: float = 7, weight: str = "normal", color: str | None = None):
    ax.text(x, y, text, ha="left", va="center", fontsize=size, fontweight=weight, color=color or COLORS["ink"])


def arrow(
    ax,
    xy0: tuple[float, float],
    xy1: tuple[float, float],
    *,
    color: str = "#505050",
    lw: float = 1.25,
    style: str = "solid",
    rad: float = 0.0,
    zorder: int = 2,
    mutation_scale: float = 9,
    alpha: float = 1.0,
):
    patch = FancyArrowPatch(
        xy0,
        xy1,
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=lw,
        linestyle=style,
        color=color,
        shrinkA=3,
        shrinkB=3,
        connectionstyle=f"arc3,rad={rad}",
        zorder=zorder,
        alpha=alpha,
    )
    ax.add_patch(patch)
    return patch


def add_image_stack(ax, x: float, y: float, w: float, h: float):
    for i, off in enumerate((0.018, 0.009, 0.0)):
        ax.add_patch(
            Rectangle(
                (x + off, y + off),
                w,
                h,
                facecolor=["#D8DEE9", "#E6EDF5", "#F9FBFD"][i],
                edgecolor="#637083",
                linewidth=0.8,
                zorder=4 + i,
            )
        )
    ax.plot([x + 0.018, x + w * 0.34, x + w * 0.59, x + w * 0.82], [y + h * 0.22, y + h * 0.48, y + h * 0.34, y + h * 0.70], color="#6E8DA8", lw=0.9, zorder=8)
    ax.scatter([x + w * 0.72], [y + h * 0.73], s=10, color="#D49B4A", zorder=9)


def add_gps_icon(ax, x: float, y: float, w: float, h: float):
    cx, cy = x + w * 0.5, y + h * 0.5
    ax.add_patch(Rectangle((x, y), w, h, facecolor="#FAF7FC", edgecolor=COLORS["gps_edge"], linewidth=0.8, zorder=4))
    ax.plot([x + w * 0.18, x + w * 0.82], [cy, cy], color="#B8A0CE", lw=0.7, zorder=5)
    ax.plot([cx, cx], [y + h * 0.18, y + h * 0.82], color="#B8A0CE", lw=0.7, zorder=5)
    ax.arrow(cx, cy, w * 0.25, h * 0.20, width=0.0015, head_width=0.012, head_length=0.012, color=COLORS["gps_edge"], length_includes_head=True, zorder=7)
    ax.scatter([cx], [cy], s=12, color=COLORS["gps_edge"], zorder=8)


def add_token_grid(ax, x: float, y: float, w: float, h: float):
    rows, cols = 5, 8
    pad = 0.004
    cw = (w - pad * (cols + 1)) / cols
    ch = (h - pad * (rows + 1)) / rows
    target_cells = {(1, 2), (1, 5), (2, 6), (3, 3), (4, 1), (4, 6)}
    context_cells = {(0, 0), (0, 1), (0, 4), (0, 7), (1, 0), (1, 4), (2, 1), (2, 2), (2, 4), (3, 0), (3, 5), (3, 7), (4, 0), (4, 4)}
    for r in range(rows):
        for c in range(cols):
            fc = "#F1F1F1"
            ec = "#D0D0D0"
            if (r, c) in context_cells:
                fc, ec = "#8CBCE0", COLORS["context_edge"]
            if (r, c) in target_cells:
                fc, ec = "#E3B65B", COLORS["target_edge"]
            ax.add_patch(Rectangle((x + pad + c * (cw + pad), y + pad + (rows - 1 - r) * (ch + pad)), cw, ch, facecolor=fc, edgecolor=ec, linewidth=0.35, zorder=5))


def add_mask_sampler_panel(ax, x: float, y: float, w: float, h: float):
    add_box(ax, x, y, w, h, "", fc=COLORS["white"], ec="#7D7D7D", fontsize=5.9)
    ax.text(x + w / 2, y + h - 0.020, "patch mask sampler", ha="center", va="center", fontsize=6.2, fontweight="bold", color=COLORS["ink"], zorder=8)
    add_token_grid(ax, x + 0.018, y + 0.030, w - 0.036, 0.046)
    ax.text(
        x + w / 2,
        y + 0.018,
        "random or GPS angle-biased\ncontext 60% | target 20% | seeded",
        ha="center",
        va="center",
        fontsize=4.8,
        color=COLORS["muted"],
        zorder=8,
    )


def draw_legend(ax):
    items = [
        ("trainable context path", COLORS["context"], COLORS["context_edge"]),
        ("EMA target path, no grad", COLORS["target"], COLORS["target_edge"]),
        ("GPS conditioning", COLORS["gps"], COLORS["gps_edge"]),
        ("loss / metric", COLORS["loss"], COLORS["loss_edge"]),
        ("downstream reuse", COLORS["downstream"], COLORS["downstream_edge"]),
    ]
    x0, y0 = 0.764, 0.407
    add_label(ax, x0, y0 + 0.070, "Encoding paths", size=6.2, weight="bold")
    for i, (label, fc, ec) in enumerate(items):
        y = y0 + 0.050 - i * 0.020
        ax.add_patch(Rectangle((x0, y), 0.016, 0.010, facecolor=fc, edgecolor=ec, linewidth=0.75))
        add_label(ax, x0 + 0.022, y + 0.005, label, size=5.4, color=COLORS["muted"])


def build_figure():
    fig, ax = plt.subplots(figsize=(11.8, 7.0), constrained_layout=False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(0.018, 0.965, "GPS-conditioned JEPA pretraining and downstream reuse", fontsize=12, fontweight="bold", ha="left", va="center", color=COLORS["ink"])
    ax.text(0.018, 0.932, "Self-supervised image token prediction guided by GPS relative-polar geometry; only the context image encoder is reused for supervised beam prediction.", fontsize=7.2, ha="left", va="center", color=COLORS["muted"])

    add_box(ax, 0.018, 0.392, 0.964, 0.500, "", fc=COLORS["panel"], ec="#E1E4E8", lw=0.8, zorder=0)
    ax.text(0.028, 0.872, "a", fontsize=13, fontweight="bold", ha="left", va="center")
    ax.text(0.052, 0.872, "JEPA pretraining path", fontsize=8.5, fontweight="bold", ha="left", va="center")

    add_image_stack(ax, 0.050, 0.752, 0.070, 0.065)
    add_box(ax, 0.035, 0.704, 0.117, 0.046, "image sequence\n[B,T,3,224,224]", fc=COLORS["white"], ec="#7A8696", fontsize=6.2)
    add_gps_icon(ax, 0.054, 0.565, 0.070, 0.064)
    add_box(ax, 0.035, 0.512, 0.117, 0.047, "GPS Rel-Polar\n[B,T,3]", fc=COLORS["gps"], ec=COLORS["gps_edge"], fontsize=6.2)

    add_box(ax, 0.206, 0.733, 0.150, 0.088, "context visual encoder\nConv patch embed\nTransformer(1) + LN", fc=COLORS["context"], ec=COLORS["context_edge"], fontsize=6.4, weight="bold")
    add_box(ax, 0.206, 0.480, 0.150, 0.088, "EMA target encoder\nsame architecture\nfrozen, no grad", fc=COLORS["target"], ec=COLORS["target_edge"], fontsize=6.4, weight="bold")

    add_box(ax, 0.392, 0.748, 0.116, 0.060, "visual tokens\n[B,T,196,64]", fc=COLORS["white"], ec=COLORS["context_edge"], fontsize=6.2)
    add_box(ax, 0.392, 0.495, 0.116, 0.060, "target tokens\n[B,T,196,64]", fc=COLORS["white"], ec=COLORS["target_edge"], fontsize=6.2)
    add_mask_sampler_panel(ax, 0.374, 0.586, 0.166, 0.118)

    add_box(ax, 0.572, 0.746, 0.130, 0.067, "gather context\n[B,T,Nctx,64]", fc=COLORS["context"], ec=COLORS["context_edge"], fontsize=6.3)
    add_box(ax, 0.572, 0.498, 0.130, 0.067, "gather target\n[B,T,Ntgt,64]\ndetach", fc=COLORS["target"], ec=COLORS["target_edge"], fontsize=6.3)
    add_box(ax, 0.572, 0.621, 0.130, 0.067, "GPS conditioner\nFiLM MLP\ngamma, beta", fc=COLORS["gps"], ec=COLORS["gps_edge"], fontsize=6.3)
    add_box(ax, 0.740, 0.646, 0.142, 0.088, "target predictor\nmean context\n+ target position emb\nMLP", fc="#EAF2F8", ec=COLORS["context_edge"], fontsize=6.1, weight="bold")
    add_box(ax, 0.905, 0.650, 0.070, 0.080, "predicted\ntarget\nlatent", fc=COLORS["context"], ec=COLORS["context_edge"], fontsize=6.0)
    add_box(ax, 0.802, 0.498, 0.160, 0.083, "masked latent loss\nMSE / SmoothL1\nmetric: val_jepa_loss", fc=COLORS["loss"], ec=COLORS["loss_edge"], fontsize=6.2, weight="bold")

    arrow(ax, (0.153, 0.774), (0.206, 0.777), color=COLORS["context_edge"])
    arrow(ax, (0.153, 0.735), (0.206, 0.523), color=COLORS["target_edge"], style="dashed", alpha=0.9)
    arrow(ax, (0.356, 0.777), (0.392, 0.778), color=COLORS["context_edge"])
    arrow(ax, (0.356, 0.523), (0.392, 0.525), color=COLORS["target_edge"], style="dashed")
    arrow(ax, (0.450, 0.748), (0.450, 0.704), color=COLORS["line"], lw=1.0)
    arrow(ax, (0.450, 0.586), (0.450, 0.555), color=COLORS["line"], lw=1.0)
    arrow(ax, (0.528, 0.662), (0.572, 0.779), color=COLORS["context_edge"], lw=1.0, rad=-0.08)
    arrow(ax, (0.528, 0.625), (0.572, 0.533), color=COLORS["target_edge"], lw=1.0, rad=0.08)
    arrow(ax, (0.152, 0.536), (0.572, 0.654), color=COLORS["gps_edge"], lw=1.0, rad=0.08)
    arrow(ax, (0.152, 0.536), (0.374, 0.636), color=COLORS["gps_edge"], lw=1.0, rad=-0.05)
    arrow(ax, (0.702, 0.655), (0.740, 0.688), color=COLORS["gps_edge"], lw=1.1)
    arrow(ax, (0.702, 0.779), (0.740, 0.704), color=COLORS["context_edge"])
    arrow(ax, (0.882, 0.690), (0.905, 0.690), color=COLORS["context_edge"])
    arrow(ax, (0.920, 0.650), (0.882, 0.581), color=COLORS["loss_edge"], lw=1.1)
    arrow(ax, (0.702, 0.533), (0.802, 0.538), color=COLORS["target_edge"], style="dashed")
    arrow(ax, (0.285, 0.733), (0.285, 0.568), color=COLORS["target_edge"], lw=1.0, style="dashed", rad=0.0)
    ax.text(
        0.282,
        0.647,
        "EMA after optimizer step\n target <- 0.99 target + 0.01 context",
        fontsize=5.4,
        ha="center",
        va="center",
        color=COLORS["target_edge"],
        bbox={"boxstyle": "round,pad=0.15", "facecolor": "#FFFDF8", "edgecolor": "none", "alpha": 0.88},
        zorder=9,
    )
    ax.text(0.736, 0.778, "target positions select prediction slots", fontsize=5.6, ha="left", va="center", color=COLORS["muted"])
    ax.text(0.054, 0.410, "Pretraining excludes beam labels, legacy KD distillers, and frozen external teacher checkpoints.", fontsize=6.2, ha="left", va="center", color=COLORS["muted"])

    add_box(ax, 0.018, 0.078, 0.964, 0.270, "", fc="#FBFBFC", ec="#E1E4E8", lw=0.8, zorder=0)
    ax.text(0.028, 0.328, "b", fontsize=13, fontweight="bold", ha="left", va="center")
    ax.text(0.052, 0.328, "Downstream reuse for supervised image+GPS beam prediction", fontsize=8.5, fontweight="bold", ha="left", va="center")

    add_box(ax, 0.060, 0.205, 0.148, 0.078, "JEPA checkpoint\nbest.pth / last.pth\ncontext_encoder.*", fc=COLORS["artifact"], ec="#8A8A8A", fontsize=6.2, weight="bold")
    add_box(ax, 0.278, 0.207, 0.162, 0.083, "jepa_context_image\nload context encoder\nmean pool patches\n-> [B,T,64]", fc=COLORS["downstream"], ec=COLORS["downstream_edge"], fontsize=6.1, weight="bold")
    add_box(ax, 0.278, 0.112, 0.162, 0.063, "GPS Rel-Polar\n[B,T,3]\ntrain-only scaler", fc=COLORS["gps"], ec=COLORS["gps_edge"], fontsize=6.1)
    add_box(ax, 0.520, 0.163, 0.156, 0.095, "modular sequence\nimage+GPS fusion\nearly concat GRU\nhidden 64, 2 layers", fc="#EAF4EF", ec=COLORS["downstream_edge"], fontsize=6.1, weight="bold")
    add_box(ax, 0.744, 0.174, 0.118, 0.073, "beam head\n64 classes\nnum_pred = 1", fc=COLORS["white"], ec=COLORS["line"], fontsize=6.2)
    add_box(ax, 0.885, 0.174, 0.082, 0.073, "Top-K\nlinear DBA\nfinal test", fc=COLORS["loss"], ec=COLORS["loss_edge"], fontsize=6.0)
    arrow(ax, (0.208, 0.244), (0.278, 0.248), color=COLORS["downstream_edge"])
    arrow(ax, (0.440, 0.248), (0.520, 0.218), color=COLORS["downstream_edge"])
    arrow(ax, (0.440, 0.144), (0.520, 0.194), color=COLORS["gps_edge"])
    arrow(ax, (0.676, 0.211), (0.744, 0.211), color=COLORS["line"])
    arrow(ax, (0.862, 0.211), (0.885, 0.211), color=COLORS["loss_edge"])
    ax.text(0.060, 0.092, "Current 2604/fair configs reuse S32-S34 JEPA checkpoints; scene31-only checkpoints are excluded from fair defaults.", fontsize=5.8, ha="left", va="center", color=COLORS["muted"])

    return fig


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig = build_figure()
    base = OUT / "jepa_architecture"
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(base.with_suffix(".svg"))
    print(base.with_suffix(".pdf"))
    print(base.with_suffix(".png"))


if __name__ == "__main__":
    main()
