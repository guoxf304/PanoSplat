"""Smoke test for PanoVGGT GS forward and backward (requires CUDA + ODGS)."""

import torch

from panovggt.render.odgs_bridge import check_odgs_available

try:
    import pytest
except ImportError:
    pytest = None


def test_gs_forward_backward():
    from panovggt.models.panovggt_gs_model import PanoVGGTGSModel
    from panovggt.models.loss_gs import GSLoss

    device = torch.device("cuda")
    b, s, h, w = 1, 2, 112, 224
    if h % 14 != 0 or w % 14 != 0:
        raise ValueError("H,W must be divisible by patch_size 14")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for render backward test")
    if not check_odgs_available():
        raise RuntimeError("odgs_gaussian_rasterization not installed")

    model = PanoVGGTGSModel(
        img_size=h,
        patch_size=14,
        embed_dim=1024,
        enable_gaussian=True,
        enable_global_points=True,
        enable_camera=True,
        enable_point=True,
        aggregator={"depth": 4, "num_heads": 4, "patch_embed": "dinov2_vits14_reg"},
    ).to(device)
    model.train()

    images = torch.rand(b, s, 3, h, w, device=device)
    extrinsics = torch.eye(4, device=device).view(1, 1, 4, 4).repeat(b, s, 1, 1)
    extrinsics = extrinsics[..., :3, :4]
    world_points = torch.randn(b, s, h, w, 3, device=device)
    cam_points = torch.randn(b, s, h, w, 3, device=device)
    point_masks = torch.ones(b, s, h, w, dtype=torch.bool, device=device)
    depths = torch.ones(b, s, h, w, device=device)

    batch = {
        "images": images,
        "extrinsics": extrinsics,
        "world_points": world_points,
        "cam_points": cam_points,
        "point_masks": point_masks,
        "depths": depths,
    }

    loss_fn = GSLoss(lambda_rgb=1.0, lambda_geo=0.0).to(device)

    pred = model(images=batch["images"])
    assert "gaussians" in pred
    assert len(pred["gaussians"]) == b

    loss_dict = loss_fn(pred, batch)
    loss = loss_dict["loss_objective"]
    loss.backward()

    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for n, p in model.named_parameters()
        if "gaussian_param_head" in n and p.requires_grad
    )
    assert has_grad, "Gaussian head should receive gradients from render loss"


if pytest is not None:
    test_gs_forward_backward = pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA required"
    )(
        pytest.mark.skipif(
            not check_odgs_available(), reason="ODGS rasterizer not installed"
        )(test_gs_forward_backward)
    )

if __name__ == "__main__":
    test_gs_forward_backward()
    print("OK: GS forward + ODGS render backward")
