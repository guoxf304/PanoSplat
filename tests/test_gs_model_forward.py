"""Forward-only smoke test (no ODGS rasterizer required)."""

import torch

from panovggt.models.panovggt_gs_model import PanoVGGTGSModel


def test_gs_model_forward_cpu():
    device = torch.device("cpu")
    h, w = 112, 224
    model = PanoVGGTGSModel(
        img_size=h,
        patch_size=14,
        embed_dim=384,
        enable_gaussian=True,
        enable_global_points=True,
        enable_camera=True,
        enable_point=True,
        aggregator={
            "depth": 4,
            "num_heads": 4,
            "patch_embed": "dinov2_vits14_reg",
        },
    ).to(device)
    model.eval()
    x = torch.rand(1, 2, 3, h, w, device=device)
    with torch.no_grad():
        pred = model(x)
    assert pred["gs_raw"].shape == (1, 10, h, w)
    assert len(pred["gaussians"]) == 1
    assert pred["gaussians"][0].get_xyz.ndim == 2


if __name__ == "__main__":
    test_gs_model_forward_cpu()
    print("OK")
