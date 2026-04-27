from __future__ import annotations

import torch


def compute_flops(model, inputs, name: str = "model", batch_size: int = 1):
    try:
        from thop import profile as thop_profile
    except Exception:
        return None, None
    if not isinstance(inputs, (list, tuple)):
        inputs = (inputs,)
    model.eval()
    with torch.no_grad():
        flops, params = thop_profile(model, inputs=inputs, verbose=False)
    print(f"[FLOPs] {name}: {flops / batch_size / 1e6:.3f} M FLOPs, {params / 1e6:.3f} M params")
    return flops, params

