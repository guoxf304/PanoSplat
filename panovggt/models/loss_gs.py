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
    rebuild_anchor_render_cloud,
    rebuild_merged_render_cloud,
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
from panovggt.utils.lpips_metric import lpips_loss
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
        lambda_anti_leak: float = 0.0,
        lambda_lpips: float = 0.0,
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
        self.lambda_anti_leak = lambda_anti_leak
        self.lambda_lpips = lambda_lpips
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

    def _try_rebuild_render_cloud(
        self,
        pred: Dict,
        gt: Dict,
        batch_idx: int,
        *,
        use_gt_mask: bool = True,
    ):
        """Rebuild render-aligned cloud from GS head features + world-frame means."""
        adapter = pred.get("_gs_adapter_ref")
        if adapter is None:
            return None
        if pred.get("gs_means_mode") == "merged":
            return rebuild_merged_render_cloud(
                adapter,
                pred,
                gt,
                batch_idx,
                use_gt_pose=self.use_gt_pose,
                use_gt_mask=use_gt_mask,
            )
        return rebuild_anchor_render_cloud(
            adapter,
            pred,
            gt,
            batch_idx,
            use_gt_pose=self.use_gt_pose,
            use_gt_mask=use_gt_mask,
        )

    def prepare_render_bundle(
        self, pred: Dict, gt_raw: Dict
    ) -> tuple[Dict, Dict, torch.Tensor, torch.Tensor]:
        """Shared GT/pred normalization for photometric loss, export, and inference."""
        gt = self.geo_loss.prepare_gt(gt_raw)
        merged_mode = pred.get("gs_means_mode") == "merged"

        need_normalize = merged_mode or (
            not self.use_gt_pose and self.lambda_geo == 0
        )
        pred_render = pred
        if need_normalize:
            pred_render = {
                k: (v.clone() if torch.is_tensor(v) else v)
                for k, v in pred.items()
            }
            self.geo_loss.normalize_pred(pred_render, gt)

        # Merged cloud + render share pred-normalized frame (inference_gs.py).
        if merged_mode:
            norm_factor = resolve_scene_scale(pred_render, gt, None)
        else:
            norm_factor = resolve_scene_scale(pred_render, gt, gt_raw)

        images = gt_raw["images"]
        if images.dim() == 4:
            images = images.unsqueeze(0)
        imgs_denorm = self._denorm_images(images)
        gt["imgs"] = imgs_denorm
        return pred_render, gt, norm_factor, imgs_denorm

    def cloud_for_view_render(
        self,
        cloud: TensorGaussianCloud,
        pred: Dict,
        batch_idx: int,
        view_idx: int,
    ) -> TensorGaussianCloud:
        """Apply cross-view axis alignment immediately before ``render_erp``."""
        anchor_idx = (
            int(pred["gs_anchor_idx"][batch_idx].item())
            if pred.get("gs_anchor_idx") is not None
            else int(view_idx)
        )
        xyz, rot = prepare_render_gaussian_xyz(
            cloud.get_xyz,
            cloud._rotation,
            anchor_idx=anchor_idx,
            view_idx=view_idx,
            axis_align_mode=self.gs_cross_view_axis_align,
        )
        return TensorGaussianCloud(
            xyz=xyz,
            scaling=cloud._scaling,
            rotation=rot,
            opacity=cloud._opacity,
            features_dc=cloud._features_dc,
            features_rest=cloud._features_rest,
            sh_degree=cloud.max_sh_degree,
            active_sh_degree=cloud.active_sh_degree,
        )

    def build_render_camera(
        self,
        gt_raw: Dict,
        gt: Dict,
        pred_render: Dict,
        batch_idx: int,
        view_idx: int,
        height: int,
        width: int,
        device: torch.device,
    ):
        merged_mode = pred_render.get("gs_means_mode") == "merged"
        if (
            self.use_gt_pose
            and not merged_mode
            and gt_raw.get("extrinsics") is not None
        ):
            w2c = gt_raw["extrinsics"][batch_idx, view_idx].float()
            return build_erp_camera_from_w2c(w2c, height, width, device)
        if merged_mode or not self.use_gt_pose:
            poses = pred_render["camera_poses"]
        else:
            poses = gt["camera_poses"]
        return build_erp_camera(poses[batch_idx, view_idx], height, width, device)

    def _build_aligned_cloud_xyz(
        self,
        pred: Dict,
        gt: Dict,
        adapter_out,
        batch_idx: int,
    ) -> Optional[torch.Tensor]:
        """Replace Gaussian centers with GT world / local_points @ c2w, subsampled + masked."""
        if pred.get("gs_means_mode") == "merged":
            # Merged clouds must be rebuilt with per-view world means; adapter xyz
            # lives in the model geometry frame and misaligns with GT cameras.
            return None

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

    def prepare_render_cloud_for_batch_item(
        self,
        pred: Dict,
        gt: Dict,
        adapter_out,
        batch_idx: int,
        norm_factor: torch.Tensor,
    ):
        """
        Build a render-ready Gaussian cloud whose centers share the same world
        frame as the render cameras (GT or normalized prediction).
        """
        cloud = self._try_rebuild_render_cloud(pred, gt, batch_idx, use_gt_mask=True)
        if cloud is None or cloud.get_xyz.numel() == 0:
            cloud = self._try_rebuild_render_cloud(
                pred, gt, batch_idx, use_gt_mask=False
            )

        if cloud is None or cloud.get_xyz.numel() == 0:
            if pred.get("gs_means_mode") == "merged":
                logger.warning(
                    "GT-aligned merged cloud rebuild failed for batch %s; "
                    "skipping render/export for this item.",
                    batch_idx,
                )
                return None
            aligned_xyz = self._build_aligned_cloud_xyz(
                pred, gt, adapter_out, batch_idx
            )
            if aligned_xyz is not None and aligned_xyz.numel() > 0:
                cloud = self._prepare_render_cloud(
                    adapter_out[batch_idx].cloud,
                    norm_factor,
                    batch_idx,
                    aligned_xyz=aligned_xyz,
                    xyz_already_normalized=True,
                )
            else:
                cloud = adapter_out[batch_idx].cloud
                cloud = self._prepare_render_cloud(
                    cloud,
                    norm_factor,
                    batch_idx,
                    xyz_already_normalized=not self.use_gt_pose,
                )
        else:
            cloud = self._prepare_render_cloud(
                cloud,
                norm_factor,
                batch_idx,
                xyz_already_normalized=True,
            )
        return cloud

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

    @staticmethod
    def _render_alpha_from_pkg(pkg: Dict) -> torch.Tensor:
        """Per-pixel accumulated coverage A from ODGS rasterizer (1 - transmittance)."""
        if "render_alpha" in pkg:
            return pkg["render_alpha"]
        if "accuracy" in pkg:
            return pkg["accuracy"]
        raise KeyError("render pkg missing render_alpha / accuracy")

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

        if pred.get("gs_means_mode") == "merged":
            means_source = (
                f"normalize_pred: local_points[{tensor_bi}, {cur_view_idx}] "
                f"@ camera_poses[{tensor_bi}, {cur_view_idx}]"
            )
        elif self.use_gt_pose:
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        adapter_out = pred["gaussian_adapter_out"]
        b = gt_imgs.shape[0]
        device = gt_imgs.device
        bg = self._bg.to(device=device, dtype=torch.float32)
        poses = poses.float()

        total: Optional[torch.Tensor] = None
        depth_total: Optional[torch.Tensor] = None
        anti_leak_total: Optional[torch.Tensor] = None
        count = 0
        depth_count = 0
        anti_leak_count = 0
        lpips_total: Optional[torch.Tensor] = None
        lpips_count = 0
        trace: Dict[str, torch.Tensor] = {}
        debug = is_gs_grad_debug_enabled() and torch.is_grad_enabled()
        gstep = int(pred.get("gs_global_step", -1))
        anchor_records: List[AnchorViewAuditRecord] = []

        for bi in range(b):
            cloud = self.prepare_render_cloud_for_batch_item(
                pred, gt, adapter_out, bi, norm_factor
            )
            if cloud is None or cloud.get_xyz.numel() == 0:
                continue
            cloud_raw = cloud
            anchor_idx = int(pred["gs_anchor_idx"][bi].item()) if pred.get("gs_anchor_idx") is not None else 0
            for vi in view_indices:
                if vi >= poses.shape[1]:
                    continue
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
                view_cloud = self.cloud_for_view_render(cloud, pred, bi, vi)
                cam = self.build_render_camera(
                    gt_raw,
                    gt,
                    pred,
                    bi,
                    vi,
                    gt_imgs.shape[-2],
                    gt_imgs.shape[-1],
                    device,
                )
                try:
                    pkg = render_erp(view_cloud, cam, bg, pipe=self.render_pipe)
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
                if self.lambda_anti_leak > 0:
                    rendered_alpha = self._align_depth_map(
                        self._render_alpha_from_pkg(pkg).float(),
                        target.shape[-2:],
                    ).clamp(0.0, 1.0)
                    anti_leak_v = F.mse_loss(
                        rendered_alpha,
                        torch.ones_like(rendered_alpha),
                    )
                    anti_leak_total = (
                        anti_leak_v
                        if anti_leak_total is None
                        else anti_leak_total + anti_leak_v
                    )
                    anti_leak_count += 1
                if self.lambda_lpips > 0:
                    lpips_v = lpips_loss(rendered, target)
                    if lpips_v.numel() > 0 and torch.isfinite(lpips_v):
                        lpips_total = (
                            lpips_v if lpips_total is None else lpips_total + lpips_v
                        )
                        lpips_count += 1
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
            return zero, zero.new_zeros(()), zero.new_zeros(()), zero.new_zeros(())
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
        if anti_leak_count > 0 and anti_leak_total is not None:
            loss_anti_leak = anti_leak_total / anti_leak_count
        else:
            loss_anti_leak = loss_rgb.new_zeros(())
        if lpips_count > 0 and lpips_total is not None:
            loss_lpips = lpips_total / lpips_count
        else:
            loss_lpips = loss_rgb.new_zeros(())
        return loss_rgb, loss_depth, loss_anti_leak, loss_lpips

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

    @torch.no_grad()
    def build_tb_visual_batch(
        self, pred: Dict, gt_raw: Dict
    ) -> Dict[str, torch.Tensor]:
        """Render GT / prediction / error maps for TensorBoard (all views, detached)."""
        if not check_odgs_available() or "gaussians" not in pred:
            return {}

        pred_render, gt, norm_factor, imgs = self.prepare_render_bundle(pred, gt_raw)
        adapter_out = pred_render.get("gaussian_adapter_out")
        if not adapter_out:
            return {}

        if self.use_gt_pose:
            poses = gt["camera_poses"]
        else:
            poses = pred_render.get("camera_poses")
        if poses is None:
            return {}

        b, s = poses.shape[:2]
        device = imgs.device
        bg = self._bg.to(device=device, dtype=torch.float32)
        gt_rows: List[torch.Tensor] = []
        render_rows: List[torch.Tensor] = []
        error_rows: List[torch.Tensor] = []

        for bi in range(b):
            cloud = self.prepare_render_cloud_for_batch_item(
                pred_render, gt, adapter_out, bi, norm_factor
            )
            if cloud is None or cloud.get_xyz.numel() == 0:
                continue

            bi_gt: List[torch.Tensor] = []
            bi_render: List[torch.Tensor] = []
            bi_error: List[torch.Tensor] = []
            for vi in range(s):
                view_cloud = self.cloud_for_view_render(cloud, pred_render, bi, vi)
                cam = self.build_render_camera(
                    gt_raw,
                    gt,
                    pred_render,
                    bi,
                    vi,
                    imgs.shape[-2],
                    imgs.shape[-1],
                    device,
                )
                try:
                    pkg = render_erp(view_cloud, cam, bg, pipe=self.render_pipe)
                except RuntimeError as exc:
                    logger.warning(
                        "TB visual render failed (batch=%s view=%s): %s",
                        bi,
                        vi,
                        exc,
                    )
                    continue

                rendered = pkg["render"].float().clamp(0.0, 1.0)
                target = imgs[bi, vi].float().clamp(0.0, 1.0)
                error_map = torch.abs(rendered - target)
                bi_gt.append(target)
                bi_render.append(rendered)
                bi_error.append(error_map)

            if not bi_gt:
                continue
            gt_rows.append(torch.stack(bi_gt, dim=0))
            render_rows.append(torch.stack(bi_render, dim=0))
            error_rows.append(torch.stack(bi_error, dim=0))

        if not gt_rows:
            return {}

        return {
            "gt_rgb": torch.stack(gt_rows, dim=0),
            "rendered_rgb": torch.stack(render_rows, dim=0),
            "error_map": torch.stack(error_rows, dim=0),
        }

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
        loss_anti_leak = torch.zeros((), device=device, dtype=torch.float32)
        loss_lpips = torch.zeros((), device=device, dtype=torch.float32)
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

            pred_render, gt, norm_factor, imgs_denorm = self.prepare_render_bundle(
                pred, gt_raw
            )

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

            loss_rgb, loss_depth_consis, loss_anti_leak, loss_lpips = self._render_loss_for_views(
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
            + self.lambda_anti_leak * loss_anti_leak
            + self.lambda_lpips * loss_lpips
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
        loss_dict["loss_anti_leak"] = (
            loss_anti_leak.detach()
            if isinstance(loss_anti_leak, torch.Tensor)
            else loss_anti_leak
        )
        loss_dict["loss_lpips"] = (
            loss_lpips.detach() if isinstance(loss_lpips, torch.Tensor) else loss_lpips
        )
        loss_dict["loss_objective"] = loss_objective
        if "loss_camera" not in loss_dict:
            loss_dict["loss_camera"] = loss_geo.new_tensor(0.0)
        if "loss_global_point" not in loss_dict:
            loss_dict["loss_global_point"] = loss_geo.new_tensor(0.0)
        return loss_dict
