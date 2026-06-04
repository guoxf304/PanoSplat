"""Combined geometry + ODGS photometric loss for PanoVGGTGSModel."""

from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

from panovggt.models.loss import Loss
from panovggt.render.camera import build_erp_camera, build_erp_camera_from_w2c
from panovggt.render.coord_frame import (
    erp_ray_convention_note,
    prepare_anchor_world_means,
    prepare_render_gaussian_xyz,
    resolve_axis_align_mode,
    resolve_scene_scale,
    subsample_hw,
)
from panovggt.render.odgs_bridge import (
    ODGSRenderPipe,
    TensorGaussianCloud,
    check_odgs_available,
    render_erp,
    sanitize_gaussian_cloud,
    scale_gaussian_cloud_for_render,
)
from panovggt.utils.erp_loss import dssim_erp, l1_erp, ssim_erp
from panovggt.utils.gs_debug import (
    gs_grad_core_should_run,
    gs_grad_print,
    gs_grad_verbose_should_run,
    is_gs_grad_debug_enabled,
    safe_retain_grad,
    register_grad_hook,
    report_cloud,
)


@dataclass
class AnchorViewAuditRecord:
    """One render pass: Gaussian anchor frame vs camera view index."""

    batch_idx: int
    tensor_bi: int
    anchor_idx: int
    cur_view_idx: int
    extrinsics_slice: str
    extrinsics_ok: bool
    anchor_eq_view: bool
    means_source: str
    global_step: int = -1
    note: str = ""


class GSLoss(nn.Module):
    def __init__(
        self,
        train_conf: bool = False,
        lambda_rgb: float = 1.0,
        lambda_geo: float = 0.1,
        lambda_dssim: float = 0.2,
        lambda_depth_consis: float = 0.0,
        render_views_per_step: int = 2,
        use_gt_pose: bool = False,
        gs_cross_view_axis_align: str = "none",
        background_color: Optional[List[float]] = None,
    ):
        super().__init__()
        self.geo_loss = Loss(train_conf=train_conf)
        self.lambda_rgb = lambda_rgb
        self.lambda_geo = lambda_geo
        self.lambda_dssim = lambda_dssim
        self.lambda_depth_consis = lambda_depth_consis
        self.render_views_per_step = render_views_per_step
        self.use_gt_pose = use_gt_pose
        self.gs_cross_view_axis_align = resolve_axis_align_mode(gs_cross_view_axis_align)
        bg = background_color if background_color is not None else [0.0, 0.0, 0.0]
        self.register_buffer("_bg", torch.tensor(bg, dtype=torch.float32), persistent=False)
        self.render_pipe = ODGSRenderPipe()

    @staticmethod
    def _compute_norm_factor(
        pred: Dict,
        gt: Dict,
    ) -> torch.Tensor:
        """Same scale as ``Loss.normalize_pred`` (from masked local points)."""
        local_points = pred["local_points"]
        masks = gt["valid_masks"]
        b = local_points.shape[0]
        all_pts = local_points.clone()
        all_pts[~masks] = 0
        all_pts = all_pts.reshape(b, -1, 3)
        all_dis = all_pts.norm(dim=-1)
        denom = masks.float().sum(dim=[-1, -2, -3]).clamp(min=1e-8)
        return (all_dis.sum(dim=[-1, -2]) / denom).clamp(min=1e-3, max=1e3)

    def _prepare_render_cloud(
        self,
        cloud,
        norm_factor: torch.Tensor,
        batch_idx: int,
        aligned_xyz: Optional[torch.Tensor] = None,
        *,
        xyz_already_normalized: bool = False,
        aligned_rotation: Optional[torch.Tensor] = None,
    ):
        rot = aligned_rotation if aligned_rotation is not None else cloud._rotation
        if aligned_xyz is not None and aligned_xyz.numel() > 0:
            cloud = TensorGaussianCloud(
                xyz=aligned_xyz,
                scaling=cloud._scaling,
                rotation=rot,
                opacity=cloud._opacity,
                features_dc=cloud._features_dc,
                features_rest=cloud._features_rest,
                sh_degree=cloud.max_sh_degree,
                active_sh_degree=cloud.active_sh_degree,
            )
        scaled = scale_gaussian_cloud_for_render(
            cloud,
            norm_factor[batch_idx],
            scale_xyz=not xyz_already_normalized,
        )
        return sanitize_gaussian_cloud(scaled)

    def _build_aligned_cloud_xyz(
        self,
        pred: Dict,
        gt: Dict,
        adapter_out,
        batch_idx: int,
    ) -> Optional[torch.Tensor]:
        """Replace Gaussian centers with GT world / local_points @ c2w, subsampled + masked."""
        if pred.get("gs_means_mode") == "merged":
            # Adapter already voxel-fused all views; do not substitute raw merged pixels.
            means = adapter_out[batch_idx].means
            return means.float() if means.numel() > 0 else None

        if pred.get("gs_anchor_idx") is None:
            return None
        if (
            not self.use_gt_pose
            and pred.get("local_points") is None
            and pred.get("global_points") is None
        ):
            return None
        means_hw = prepare_anchor_world_means(
            pred, gt, batch_idx, use_gt_pose=self.use_gt_pose
        )
        stride = int(pred.get("gs_stride", 1))
        means_hw = subsample_hw(means_hw, stride)
        mask = adapter_out[batch_idx].mask
        if mask.shape != means_hw.shape[:2]:
            return None
        xyz = means_hw[mask]
        if xyz.numel() == 0:
            return None
        return xyz.float()

    def _scale_camera_poses(
        self,
        poses: torch.Tensor,
        norm_factor: torch.Tensor,
    ) -> torch.Tensor:
        out = poses.clone()
        out[..., :3, 3] = out[..., :3, 3] / norm_factor.view(-1, 1, 1).clamp(min=1e-6)
        return out

    def _denorm_images(self, images: torch.Tensor) -> torch.Tensor:
        """Return linear RGB in [0, 1] for photometric loss / export.

        Dataloader images are already linear RGB; ImageNet norm runs only inside
        the aggregator forward pass.
        """
        return images.float().clamp(0.0, 1.0)

    @staticmethod
    def _align_depth_map(depth: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
        """Resize depth to (H, W) for pixel-wise comparison."""
        while depth.dim() > 2:
            depth = depth.squeeze(0)
        if depth.shape[-2:] == target_hw:
            return depth
        return F.interpolate(
            depth.unsqueeze(0).unsqueeze(0),
            size=target_hw,
            mode="bilinear",
            align_corners=True,
        ).squeeze(0).squeeze(0)

    def _geometry_base_depth(
        self,
        pred: Dict,
        batch_idx: int,
        view_idx: int,
        norm_factor: torch.Tensor,
    ) -> torch.Tensor:
        """Geometry-head depth in the same normalized scale as the render path."""
        scale = norm_factor[batch_idx].clamp(min=1e-6)
        if pred.get("depth") is not None:
            base = pred["depth"][batch_idx, view_idx].squeeze(-1).float()
        else:
            base = pred["local_points"][batch_idx, view_idx].float().norm(dim=-1)
        return base / scale

    def build_anchor_view_record(
        self,
        pred: Dict,
        gt_raw: Dict,
        *,
        batch_idx: int,
        tensor_bi: int,
        cur_view_idx: int,
        global_step: int = -1,
    ) -> AnchorViewAuditRecord:
        """Validate anchor vs render-view indexing before ``render_erp``."""
        anchor_tensor = pred.get("gs_anchor_idx")
        if anchor_tensor is None:
            anchor_idx = -1
        else:
            anchor_idx = int(anchor_tensor[tensor_bi].item())

        extrinsics_slice = f"[{tensor_bi}, {cur_view_idx}]"
        extrinsics_ok = True
        note = "ok"

        if self.use_gt_pose and gt_raw.get("extrinsics") is not None:
            ext = gt_raw["extrinsics"]
            if cur_view_idx >= ext.shape[1]:
                extrinsics_ok = False
                note = f"cur_view_idx={cur_view_idx} >= num_views={ext.shape[1]}"
            elif tensor_bi >= ext.shape[0]:
                extrinsics_ok = False
                note = f"tensor_bi={tensor_bi} >= batch_size={ext.shape[0]}"
            else:
                # Guard against accidental [bi,0] / [0,vi] / [0,0] misuse.
                wrong_slices = {
                    f"[{tensor_bi}, 0]": int(cur_view_idx) != 0,
                    f"[0, {cur_view_idx}]": int(tensor_bi) != 0,
                    "[0, 0]": int(tensor_bi) != 0 or int(cur_view_idx) != 0,
                }
                for wrong, would_mismatch in wrong_slices.items():
                    if would_mismatch and wrong == extrinsics_slice:
                        extrinsics_ok = False
                        note = f"suspicious extrinsics slice {wrong}"
                        break
        elif self.use_gt_pose:
            extrinsics_ok = False
            note = "use_gt_pose but gt_raw['extrinsics'] missing"

        if self.use_gt_pose:
            means_source = f"world_points[{tensor_bi}, {anchor_idx}]"
        else:
            means_source = f"local_points[{tensor_bi}, {anchor_idx}] @ c2w[{anchor_idx}]"

        return AnchorViewAuditRecord(
            batch_idx=batch_idx,
            tensor_bi=tensor_bi,
            anchor_idx=anchor_idx,
            cur_view_idx=int(cur_view_idx),
            extrinsics_slice=extrinsics_slice,
            extrinsics_ok=extrinsics_ok,
            anchor_eq_view=anchor_idx == int(cur_view_idx),
            means_source=means_source,
            global_step=int(global_step),
            note=note,
        )

    @staticmethod
    def format_anchor_audit_report(
        records: List[AnchorViewAuditRecord],
        *,
        epoch: Optional[int] = None,
        global_step: Optional[int] = None,
        use_gt_pose: bool = False,
        view_indices: Optional[List[int]] = None,
    ) -> str:
        lines = []
        if epoch is not None:
            lines.append(f"epoch: {epoch}")
        if global_step is not None:
            lines.append(f"global_step: {global_step}")
        lines.append(f"use_gt_pose: {use_gt_pose}")
        if view_indices is not None:
            lines.append(f"render_view_indices: {view_indices}")
        lines.extend(
            [
                "",
                "[alignment_policy]",
                "  gaussian_cloud: built from gs_means_mode (anchor frame or merged views)",
                "  render_camera: gt_raw['extrinsics'][tensor_bi, cur_view_idx] when use_gt_pose",
                "  forbidden_slices: [bi,0] unless cur_view_idx==0; [0,vi]; [0,0]",
                "",
                "[records]",
            ]
        )
        for rec in records:
            flag = "OK" if rec.extrinsics_ok else "MISMATCH"
            lines.append(
                f"  dataloader_batch={rec.batch_idx} tensor_bi={rec.tensor_bi} "
                f"anchor_idx={rec.anchor_idx} cur_view_idx={rec.cur_view_idx} "
                f"extrinsics_slice={rec.extrinsics_slice} anchor_eq_view={rec.anchor_eq_view} "
                f"means_source={rec.means_source} [{flag}] {rec.note}"
            )
        if not records:
            lines.append("  (no render passes recorded)")
        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def write_anchor_audit_file(
        export_dir: str,
        records: List[AnchorViewAuditRecord],
        *,
        epoch: Optional[int] = None,
        global_step: Optional[int] = None,
        use_gt_pose: bool = False,
        view_indices: Optional[List[int]] = None,
        append: bool = False,
    ) -> str:
        os.makedirs(export_dir, exist_ok=True)
        path = os.path.join(export_dir, "anchor.txt")
        report = GSLoss.format_anchor_audit_report(
            records,
            epoch=epoch,
            global_step=global_step,
            use_gt_pose=use_gt_pose,
            view_indices=view_indices,
        )
        mode = "a" if append else "w"
        with open(path, mode, encoding="utf-8") as f:
            if append:
                f.write("\n---\n")
            f.write(report)
        return path

    def _render_loss_for_views(
        self,
        pred: Dict,
        gt: Dict,
        gt_raw: Dict,
        gt_imgs: torch.Tensor,
        gt_masks: Optional[torch.Tensor],
        poses: torch.Tensor,
        view_indices: List[int],
        norm_factor: torch.Tensor,
        *,
        dataloader_batch_idx: int = 0,
        anchor_audit_dir: Optional[str] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        adapter_out = pred["gaussian_adapter_out"]
        b = gt_imgs.shape[0]
        device = gt_imgs.device
        bg = self._bg.to(device=device, dtype=torch.float32)
        poses = poses.float()

        total: Optional[torch.Tensor] = None
        depth_total: Optional[torch.Tensor] = None
        count = 0
        depth_count = 0
        trace: Dict[str, torch.Tensor] = {}
        debug = is_gs_grad_debug_enabled() and torch.is_grad_enabled()
        gstep = int(pred.get("gs_global_step", -1))
        anchor_records: List[AnchorViewAuditRecord] = []

        for bi in range(b):
            cloud = adapter_out[bi].cloud
            if cloud.get_xyz.numel() == 0:
                continue
            cloud_raw = cloud
            anchor_idx = int(pred["gs_anchor_idx"][bi].item()) if pred.get("gs_anchor_idx") is not None else 0
            base_aligned_xyz = self._build_aligned_cloud_xyz(pred, gt, adapter_out, bi)
            for vi in view_indices:
                if vi >= poses.shape[1]:
                    continue
                aligned_xyz = base_aligned_xyz
                aligned_rot = cloud_raw._rotation
                if aligned_xyz is None or aligned_xyz.numel() == 0:
                    aligned_xyz = cloud_raw.get_xyz.float()
                elif aligned_xyz.shape[0] != cloud_raw.get_xyz.shape[0]:
                    logger.warning(
                        "aligned_xyz count %s != cloud count %s; using cloud xyz",
                        aligned_xyz.shape[0],
                        cloud_raw.get_xyz.shape[0],
                    )
                    aligned_xyz = cloud_raw.get_xyz.float()
                aligned_xyz, aligned_rot = prepare_render_gaussian_xyz(
                    aligned_xyz,
                    cloud_raw._rotation,
                    anchor_idx=anchor_idx,
                    view_idx=int(vi),
                    axis_align_mode=self.gs_cross_view_axis_align,
                    cross_view_only=True,
                    align_rotation=True,
                )
                cloud = self._prepare_render_cloud(
                    cloud_raw,
                    norm_factor,
                    bi,
                    aligned_xyz=aligned_xyz,
                    xyz_already_normalized=(
                        base_aligned_xyz is not None and base_aligned_xyz.numel() > 0
                    ),
                    aligned_rotation=aligned_rot,
                )
                if debug and bi == 0 and gs_grad_verbose_should_run(gstep):
                    if vi == view_indices[0]:
                        gs_grad_print(
                            "render_setup",
                            step=gstep,
                            verbose=True,
                            norm_factor=norm_factor,
                            view_indices=view_indices,
                            axis_align=self.gs_cross_view_axis_align,
                            erp_convention=erp_ray_convention_note(),
                        )
                        report_cloud("cloud_before_scale", cloud_raw, step=gstep)
                        report_cloud("cloud_after_scale", cloud, step=gstep)
                if cloud.get_xyz.numel() == 0:
                    continue
                audit = self.build_anchor_view_record(
                    pred,
                    gt_raw,
                    batch_idx=dataloader_batch_idx,
                    tensor_bi=bi,
                    cur_view_idx=vi,
                    global_step=gstep,
                )
                anchor_records.append(audit)
                if not audit.extrinsics_ok:
                    logger.warning(
                        "anchor/view mismatch: batch=%s bi=%s anchor=%s view=%s slice=%s (%s)",
                        dataloader_batch_idx,
                        bi,
                        audit.anchor_idx,
                        vi,
                        audit.extrinsics_slice,
                        audit.note,
                    )
                elif audit.anchor_idx != vi and self.gs_cross_view_axis_align != "none":
                    logger.info(
                        "cross-view axis_align=%s: batch=%s anchor=%s render_view=%s",
                        self.gs_cross_view_axis_align,
                        dataloader_batch_idx,
                        audit.anchor_idx,
                        vi,
                    )
                elif audit.anchor_idx != vi:
                    logger.info(
                        "anchor render cross-view: batch=%s bi=%s anchor=%s render_view=%s",
                        dataloader_batch_idx,
                        bi,
                        audit.anchor_idx,
                        vi,
                    )
                if self.use_gt_pose and gt_raw.get("extrinsics") is not None:
                    w2c = gt_raw["extrinsics"][bi, vi].float()
                    cam = build_erp_camera_from_w2c(
                        w2c, gt_imgs.shape[-2], gt_imgs.shape[-1], device
                    )
                else:
                    cam = build_erp_camera(poses[bi, vi], gt_imgs.shape[-2], gt_imgs.shape[-1], device)
                try:
                    pkg = render_erp(cloud, cam, bg, pipe=self.render_pipe)
                except RuntimeError as exc:
                    logger.warning(
                        "ODGS render failed (batch=%s view=%s): %s",
                        bi,
                        vi,
                        exc,
                    )
                    continue
                rendered = pkg["render"].float().clamp(0, 1)
                target = gt_imgs[bi, vi].float().clamp(0, 1)
                l1 = l1_erp(rendered.unsqueeze(0), target.unsqueeze(0))
                if self.lambda_dssim > 0:
                    ssim_val = ssim_erp(rendered.unsqueeze(0), target.unsqueeze(0))
                    dssim = dssim_erp(rendered.unsqueeze(0), target.unsqueeze(0))
                    loss_v = (1.0 - self.lambda_dssim) * l1 + self.lambda_dssim * dssim
                    if debug and bi == 0 and count == 0 and gs_grad_verbose_should_run(gstep):
                        gs_grad_print(
                            "loss_terms",
                            step=gstep,
                            verbose=True,
                            l1=l1,
                            ssim=ssim_val,
                            dssim=dssim,
                            loss_view=loss_v,
                        )
                else:
                    loss_v = l1
                    if debug and bi == 0 and count == 0 and gs_grad_verbose_should_run(gstep):
                        gs_grad_print(
                            "loss_terms", step=gstep, verbose=True, l1=l1, loss_view=loss_v
                        )
                if self.lambda_depth_consis > 0 and "local_points" in pred:
                    rendered_depth = self._align_depth_map(
                        pkg["depth"].float(), target.shape[-2:]
                    )
                    base_depth = self._geometry_base_depth(pred, bi, vi, norm_factor)
                    base_depth = self._align_depth_map(base_depth, target.shape[-2:])
                    if gt_masks is not None:
                        valid_mask = gt_masks[bi, vi].bool()
                    else:
                        valid_mask = torch.ones_like(base_depth, dtype=torch.bool)
                    valid_mask = (
                        valid_mask
                        & torch.isfinite(rendered_depth)
                        & torch.isfinite(base_depth)
                        & (rendered_depth > 0)
                        & (base_depth > 0)
                    )
                    if valid_mask.any():
                        depth_loss_v = F.mse_loss(
                            rendered_depth, base_depth, reduction="none"
                        )[valid_mask].mean()
                        depth_total = (
                            depth_loss_v
                            if depth_total is None
                            else depth_total + depth_loss_v
                        )
                        depth_count += 1
                if debug and bi == 0 and count == 0 and gs_grad_verbose_should_run(gstep):
                    safe_retain_grad(rendered)
                    register_grad_hook(
                        rendered, "rendered", step=gstep, verbose=True
                    )
                    sp = pkg.get("viewspace_points")
                    if torch.is_tensor(sp):
                        register_grad_hook(
                            sp, "screenspace_points", step=gstep, verbose=True
                        )
                    trace["rendered"] = rendered
                    trace["screenspace_points"] = sp
                    trace["loss_view"] = loss_v
                total = loss_v if total is None else total + loss_v
                count += 1

        if anchor_audit_dir and anchor_records:
            self.write_anchor_audit_file(
                anchor_audit_dir,
                anchor_records,
                global_step=gstep,
                use_gt_pose=self.use_gt_pose,
                view_indices=view_indices,
                append=True,
            )

        if count == 0:
            if debug and gs_grad_core_should_run(gstep):
                gs_grad_print("render_path", step=gstep, status="fallback_gs_raw")
            zero = self._fallback_gs_loss(pred)
            return zero, zero.new_zeros(())
        if debug and gs_grad_core_should_run(gstep):
            if trace:
                pred["_gs_grad_trace"] = trace
            gs_grad_print(
                "render_path",
                step=gstep,
                status="ok",
                num_views=count,
                loss_rgb=total / count,
            )
        loss_rgb = total / count
        if depth_count > 0 and depth_total is not None:
            loss_depth = depth_total / depth_count
        else:
            loss_depth = loss_rgb.new_zeros(())
        return loss_rgb, loss_depth

    def _fallback_gs_loss(self, pred: Dict) -> torch.Tensor:
        """Differentiable fallback when no Gaussians survive masking / render."""
        gs_raw = pred.get("gs_raw")
        if not isinstance(gs_raw, torch.Tensor) or gs_raw.numel() == 0:
            raise RuntimeError(
                "Photometric loss has no rendered views and no gs_raw tensor for fallback."
            )
        logger.warning(
            "No Gaussians to render (empty cloud); using finite gs_raw regularizer for backward."
        )
        x = torch.nan_to_num(gs_raw.float(), nan=0.0, posinf=1e4, neginf=-1e4)
        return x.pow(2).mean() * 1e-4

    def forward(self, pred: Dict, gt_raw: Dict) -> Dict:
        loss_dict = {}

        device = next(
            (v.device for v in pred.values() if isinstance(v, torch.Tensor)),
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )

        loss_geo = torch.zeros((), device=device, dtype=torch.float32)
        if self.lambda_geo > 0:
            geo = self.geo_loss(pred, gt_raw)
            loss_dict.update(geo)
            loss_geo = geo["loss_objective"].float()

        loss_rgb = torch.zeros((), device=device, dtype=torch.float32)
        loss_depth_consis = torch.zeros((), device=device, dtype=torch.float32)
        gstep = int(pred.get("gs_global_step", -1))
        if is_gs_grad_debug_enabled() and gs_grad_verbose_should_run(gstep):
            gs_raw = pred.get("gs_raw")
            gs_grad_print(
                "loss_forward",
                step=gstep,
                verbose=True,
                gs_raw=gs_raw if torch.is_tensor(gs_raw) else None,
                has_adapter="gaussian_adapter_out" in pred,
            )
            if torch.is_tensor(gs_raw):
                ch_names = [
                    "density",
                    "scale_x",
                    "scale_y",
                    "scale_z",
                    "quat_w",
                    "quat_x",
                    "quat_y",
                    "quat_z",
                    "sh0_r",
                    "sh0_g",
                    "sh0_b",
                    "conf",
                ]
                for ci, cname in enumerate(ch_names[: gs_raw.shape[1]]):
                    gs_grad_print(
                        f"gs_raw.{cname}",
                        step=gstep,
                        verbose=True,
                        **{cname: gs_raw[:, ci]},
                    )

        if self.lambda_rgb > 0 and "gaussians" in pred:
            if not check_odgs_available():
                raise ImportError("ODGS rasterizer required for photometric loss")

            gt = self.geo_loss.prepare_gt(gt_raw)
            norm_factor = resolve_scene_scale(pred, gt, gt_raw)

            pred_render = pred
            if not self.use_gt_pose and self.lambda_geo == 0:
                pred_render = {
                    k: (v.clone() if torch.is_tensor(v) else v)
                    for k, v in pred.items()
                }
                self.geo_loss.normalize_pred(pred_render, gt)

            images = gt_raw["images"]
            if images.dim() == 4:
                images = images.unsqueeze(0)
            imgs_denorm = self._denorm_images(images)

            if self.use_gt_pose:
                poses = gt["camera_poses"]
            else:
                poses = pred.get("camera_poses")
                if poses is None:
                    raise RuntimeError("camera_poses missing for photometric rendering")

            b, s = poses.shape[:2]
            k = min(self.render_views_per_step, s)
            view_ids = list(range(s))
            random.shuffle(view_ids)
            view_ids = view_ids[:k]
            anchor = pred.get("gs_anchor_idx")
            if anchor is not None:
                for bi in range(b):
                    a = int(anchor[bi].item())
                    if a not in view_ids:
                        view_ids[0] = a

            loss_rgb, loss_depth_consis = self._render_loss_for_views(
                pred_render,
                gt,
                gt_raw,
                imgs_denorm,
                gt.get("valid_masks"),
                poses,
                view_ids,
                norm_factor,
                dataloader_batch_idx=int(gt_raw.get("_dataloader_batch_idx", 0)),
                anchor_audit_dir=gt_raw.get("gs_anchor_audit_dir"),
            )

        loss_objective = (
            self.lambda_rgb * loss_rgb
            + self.lambda_geo * loss_geo
            + self.lambda_depth_consis * loss_depth_consis
        )
        if not torch.isfinite(loss_objective.detach().float().cpu()).all():
            logger.warning("Non-finite loss_objective; falling back to gs_raw regularizer.")
            loss_objective = self._fallback_gs_loss(pred)
            loss_rgb = loss_objective
        loss_dict["loss_rgb"] = loss_rgb.detach() if isinstance(loss_rgb, torch.Tensor) else loss_rgb
        loss_dict["loss_depth_consis"] = (
            loss_depth_consis.detach()
            if isinstance(loss_depth_consis, torch.Tensor)
            else loss_depth_consis
        )
        loss_dict["loss_objective"] = loss_objective
        if "loss_camera" not in loss_dict:
            loss_dict["loss_camera"] = loss_geo.new_tensor(0.0)
        if "loss_global_point" not in loss_dict:
            loss_dict["loss_global_point"] = loss_geo.new_tensor(0.0)
        return loss_dict
