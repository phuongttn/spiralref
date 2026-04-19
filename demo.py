from pathlib import Path
import torch
import argparse
import os
import cv2
import numpy as np

from hamer.configs import CACHE_DIR_HAMER
from hamer.models import HAMER, download_models, load_hamer, DEFAULT_CHECKPOINT
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hamer.utils.renderer import Renderer, cam_crop_to_full

LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)

from vitpose_model import ViTPoseModel

from torch.serialization import add_safe_globals
from omegaconf import DictConfig, ListConfig
from omegaconf.base import ContainerMetadata
import typing as _typing

# ---- same "safe globals" trick as eval.py ----
add_safe_globals([DictConfig, ListConfig, ContainerMetadata, _typing.Any])

# ---- same "weights_only=False" compatibility as eval.py ----
_orig_load = torch.load
def _load_compat(*args, **kwargs):
    kwargs["weights_only"] = False
    if not torch.cuda.is_available():
        kwargs.setdefault("map_location", "cpu")
    return _orig_load(*args, **kwargs)
torch.load = _load_compat


# ----------------------------
# Hand skeleton (21 joints) edges
# ----------------------------
HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index
    (0, 9), (9, 10), (10, 11), (11, 12),   # middle
    (0, 13), (13, 14), (14, 15), (15, 16), # ring
    (0, 17), (17, 18), (18, 19), (19, 20)  # pinky
]

def hamer_kp_crop_to_full(kp_crop, batch, n, model_cfg):
    """
    kp_crop: (K,2) in crop pixel coords
    return: (K,2) in full-image pixel coords
    """
    kp_crop = kp_crop.copy().astype(np.float32)

    # meta from ViTDetDataset
    box_center = batch["box_center"][n].detach().cpu().numpy()  # (2,)
    box_size   = batch["box_size"][n].detach().cpu().numpy()    # scalar or (2,)
    if np.ndim(box_size) == 0:
        bw = bh = float(box_size)
    else:
        bw, bh = float(box_size[0]), float(box_size[1])

    # crop (square) is resized to IMAGE_SIZE x IMAGE_SIZE
    S = float(model_cfg.MODEL.IMAGE_SIZE)

    x1 = box_center[0] - bw * 0.5
    y1 = box_center[1] - bh * 0.5

    kp_full = kp_crop.copy()
    kp_full[:, 0] = x1 + kp_full[:, 0] * (bw / S)
    kp_full[:, 1] = y1 + kp_full[:, 1] * (bh / S)
    return kp_full

def draw_hand_skeleton_bgr(img_bgr, kpts_xy, edges=HAND_EDGES,
                           pt_color=(0, 255, 0), ln_color=(255, 0, 0),
                           radius=3, thickness=2):
    """
    img_bgr: HxWx3 uint8
    kpts_xy: (K,2) float or int in pixel coords
    """
    if img_bgr is None:
        return None
    out = img_bgr.copy()
    if kpts_xy is None:
        return out
    if not isinstance(kpts_xy, np.ndarray):
        kpts_xy = np.asarray(kpts_xy)

    if kpts_xy.ndim != 2 or kpts_xy.shape[1] < 2:
        return out

    K = kpts_xy.shape[0]

    # lines
    for a, b in edges:
        if a < K and b < K:
            xa, ya = kpts_xy[a, 0], kpts_xy[a, 1]
            xb, yb = kpts_xy[b, 0], kpts_xy[b, 1]
            cv2.line(out, (int(xa), int(ya)), (int(xb), int(yb)), ln_color, thickness, cv2.LINE_AA)

    # points
    for i in range(K):
        x, y = kpts_xy[i, 0], kpts_xy[i, 1]
        cv2.circle(out, (int(x), int(y)), radius, pt_color, -1, cv2.LINE_AA)

    return out


def get_hamer_kp2d_crop(out, n=0):
    """
    Return HAMER predicted 2D keypoints in CROP pixel coords.
    Supports (B,K,2) or (B,2K).
    """
    if "pred_keypoints_2d" not in out:
        return None

    kp = out["pred_keypoints_2d"]
    if torch.is_tensor(kp):
        kp = kp.detach().cpu().numpy()
    else:
        kp = np.asarray(kp)

    if kp.ndim == 3 and kp.shape[-1] == 2:
        return kp[n]  # (K,2)
    if kp.ndim == 2 and kp.shape[1] % 2 == 0:
        K = kp.shape[1] // 2
        return kp[n].reshape(K, 2)

    return None


def clamp_kp(kp_xy, W, H):
    kp = kp_xy.copy()
    kp[:, 0] = np.clip(kp[:, 0], 0, W - 1)
    kp[:, 1] = np.clip(kp[:, 1], 0, H - 1)
    return kp

def kp_to_crop_pixels(kp2d, crop_size):
    """
    kp2d: (K,2) from out["pred_keypoints_2d"]
    crop_size: int, e.g. model_cfg.MODEL.IMAGE_SIZE (usually 256)

    Return kp in crop pixel coords [0..crop_size)
    """
    kp = kp2d.astype(np.float32).copy()

    # Heuristic detect format:
    mn, mx = float(kp.min()), float(kp.max())

    # Case A: normalized in [-1, 1]
    if mn >= -1.5 and mx <= 1.5:
        # map [-1,1] -> [0, crop_size]
        kp = (kp + 1.0) * 0.5 * float(crop_size)
        return kp

    # Case B: normalized in [0, 1]
    if mn >= -0.1 and mx <= 1.1:
        kp = kp * float(crop_size)
        return kp

    # Case C: already pixel
    return kp

def project_joints_to_crop(j3d, cam_t, focal, img_res):
    """
    j3d: (K,3)  3D joints from MANO / HAMER
    cam_t: (3,)
    focal: float
    img_res: int
    """
    X = j3d + cam_t[None, :]
    x = X[:, 0] / X[:, 2]
    y = X[:, 1] / X[:, 2]

    cx = cy = img_res / 2.0
    u = focal * x + cx
    v = focal * y + cy

    return np.stack([u, v], axis=1)

def main():
    parser = argparse.ArgumentParser(description='HaMeR demo code (with keypoints drawing)')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT, help='Path to pretrained model checkpoint')
    parser.add_argument('--img_folder', type=str, default='images', help='Folder with input images')
    parser.add_argument('--out_folder', type=str, default='out_demo', help='Output folder to save rendered results')
    parser.add_argument('--side_view', dest='side_view', action='store_true', default=False, help='If set, render side view also')
    parser.add_argument('--full_frame', dest='full_frame', action='store_true', default=True, help='If set, render all people together also')
    parser.add_argument('--save_mesh', dest='save_mesh', action='store_true', default=False, help='If set, save meshes to disk also')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference/fitting')
    parser.add_argument('--rescale_factor', type=float, default=2.0, help='Factor for padding the bbox')
    parser.add_argument('--body_detector', type=str, default='vitdet', choices=['vitdet', 'regnety'],
                        help='Using regnety improves runtime and reduces memory')
    parser.add_argument('--file_type', nargs='+', default=['*.jpg', '*.png'], help='List of file extensions to consider')

    # new flags
    parser.add_argument('--draw_hamer_kp', action='store_true', default=True,
                        help='Draw HAMER pred_keypoints_2d on crop images')
    parser.add_argument('--draw_vitpose_kp_full', action='store_true', default=False,
                        help='Also save a debug full image with ViTPose hand keypoints')
    parser.add_argument('--det_score', type=float, default=0.5,
                        help='Person detection score threshold (used in valid_idx)')

    args = parser.parse_args()

    # Download and load checkpoints
    download_models(CACHE_DIR_HAMER)
    model, model_cfg = load_hamer(args.checkpoint)

    # Setup HaMeR model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()

    # Load detector (Detectron2)
    from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
    if args.body_detector == 'vitdet':
        from detectron2.config import LazyConfig
        import hamer
        cfg_path = Path(hamer.__file__).parent / 'configs' / 'cascade_mask_rcnn_vitdet_h_75ep.py'
        detectron2_cfg = LazyConfig.load(str(cfg_path))
        detectron2_cfg.train.init_checkpoint = (
            "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/"
            "cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
        )
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
        detector = DefaultPredictor_Lazy(detectron2_cfg)
    elif args.body_detector == 'regnety':
        from detectron2 import model_zoo
        detectron2_cfg = model_zoo.get_config(
            'new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py', trained=True
        )
        detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
        detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh = 0.4
        detector = DefaultPredictor_Lazy(detectron2_cfg)

    # keypoint detector
    cpm = ViTPoseModel(device)

    # Setup the renderer
    renderer = Renderer(model_cfg, faces=model.mano.faces)

    # Make output directory if it does not exist
    os.makedirs(args.out_folder, exist_ok=True)

    # Get all demo images ends with patterns
    img_paths = [img for end in args.file_type for img in Path(args.img_folder).glob(end)]
    img_paths = sorted(img_paths)

    for img_path in img_paths:
        img_cv2 = cv2.imread(str(img_path))
        if img_cv2 is None:
            continue

        # Detect humans in image
        det_out = detector(img_cv2)
        img_rgb = img_cv2.copy()[:, :, ::-1]  # RGB for ViTPoseModel

        det_instances = det_out['instances']
        valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > float(args.det_score))
        pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        pred_scores = det_instances.scores[valid_idx].cpu().numpy()

        if pred_bboxes.shape[0] == 0:
            continue

        # Detect human keypoints for each person
        vitposes_out = cpm.predict_pose(
            img_rgb,
            [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
        )

        # (optional) save debug: full image with ViTPose hand keypoints
        if args.draw_vitpose_kp_full:
            dbg = img_cv2.copy()
            for vitposes in vitposes_out:
                left_hand_keyp = vitposes['keypoints'][-42:-21]   # (21,3)
                right_hand_keyp = vitposes['keypoints'][-21:]     # (21,3)

                # draw only confident points
                for keyp in [left_hand_keyp, right_hand_keyp]:
                    valid = keyp[:, 2] > 0.5
                    if valid.sum() > 3:
                        kp_xy = keyp[:, :2].copy()
                        # even if some are low conf, draw full skeleton for visibility
                        dbg = draw_hand_skeleton_bgr(dbg, kp_xy, pt_color=(0, 255, 255), ln_color=(0, 128, 255))
            img_fn, _ = os.path.splitext(os.path.basename(str(img_path)))
            cv2.imwrite(os.path.join(args.out_folder, f'{img_fn}_vitpose_full.jpg'), dbg)

        bboxes = []
        is_right = []

        # Use hands based on hand keypoint detections
        for vitposes in vitposes_out:
            left_hand_keyp = vitposes['keypoints'][-42:-21]
            right_hand_keyp = vitposes['keypoints'][-21:]

            # Left hand bbox
            keyp = left_hand_keyp
            valid = keyp[:, 2] > 0.5
            if int(valid.sum()) > 3:
                bbox = [keyp[valid, 0].min(), keyp[valid, 1].min(), keyp[valid, 0].max(), keyp[valid, 1].max()]
                bboxes.append(bbox)
                is_right.append(0)

            # Right hand bbox
            keyp = right_hand_keyp
            valid = keyp[:, 2] > 0.5
            if int(valid.sum()) > 3:
                bbox = [keyp[valid, 0].min(), keyp[valid, 1].min(), keyp[valid, 0].max(), keyp[valid, 1].max()]
                bboxes.append(bbox)
                is_right.append(1)

        if len(bboxes) == 0:
            continue

        boxes = np.stack(bboxes).astype(np.float32)  # (N,4)
        right = np.stack(is_right).astype(np.int64)  # (N,)

        # Run reconstruction on all detected hands
        dataset = ViTDetDataset(model_cfg, img_cv2, boxes, right, rescale_factor=args.rescale_factor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

        all_verts = []
        all_cam_t = []
        all_right = []

        for batch in dataloader:
            batch = recursive_to(batch, device)
            with torch.no_grad():
                out = model(batch)

            multiplier = (2 * batch['right'] - 1)
            pred_cam = out['pred_cam'].clone()
            pred_cam[:, 1] = multiplier * pred_cam[:, 1]

            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()

            scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            pred_cam_t_full = cam_crop_to_full(
                pred_cam, box_center, box_size, img_size, scaled_focal_length
            ).detach().cpu().numpy()

            # Render the result
            batch_size = batch['img'].shape[0]
            for n in range(batch_size):
                img_fn, _ = os.path.splitext(os.path.basename(str(img_path)))
                person_id = int(batch['personid'][n])

                # Denorm input patch (RGB float 0..1)
                input_patch = batch['img'][n].detach().cpu() * (DEFAULT_STD[:, None, None] / 255.0) + (
                    DEFAULT_MEAN[:, None, None] / 255.0
                )
                input_patch = input_patch.permute(1, 2, 0).numpy()  # HWC RGB float [0,1]

                # Render mesh on crop
                regression_img = renderer(
                    out['pred_vertices'][n].detach().cpu().numpy(),
                    out['pred_cam_t'][n].detach().cpu().numpy(),
                    batch['img'][n],
                    mesh_base_color=LIGHT_BLUE,
                    scene_bg_color=(1, 1, 1),
                )  # HWC RGB float [0,1]

                # --- Draw HAMER keypoints on crop images (input_patch + regression_img) ---
                if args.draw_hamer_kp and "pred_keypoints_3d" in out:
                    j3d = out["pred_keypoints_3d"][n].detach().cpu().numpy()[:21]
                    cam_t = out["pred_cam_t"][n].detach().cpu().numpy()
                
                    img_res = int(model_cfg.MODEL.IMAGE_SIZE)
                    focal = float(model_cfg.EXTRA.FOCAL_LENGTH)
                
                    # 1️⃣ project joints → 2D crop (luôn khớp mesh)
                    kp_crop21 = project_joints_to_crop(j3d, cam_t, focal, img_res)
                    kp_crop21 = clamp_kp(kp_crop21, img_res, img_res)
                
                    # 2️⃣ vẽ trên input_patch (crop)
                    inp_bgr = (255 * input_patch[:, :, ::-1]).astype(np.uint8)
                    inp_bgr = draw_hand_skeleton_bgr(inp_bgr, kp_crop21,
                                                     pt_color=(0,255,0), ln_color=(255,0,0))
                    input_patch = inp_bgr[:, :, ::-1] / 255.0
                
                    # 3️⃣ vẽ trên regression_img (crop)
                    #reg_bgr = (255 * regression_img[:, :, ::-1]).astype(np.uint8)
                    #reg_bgr = draw_hand_skeleton_bgr(reg_bgr, kp_crop21,pt_color=(0,255,0), ln_color=(255,0,0))
                    #regression_img = reg_bgr[:, :, ::-1] / 255.0
                
                    # 4️⃣ map ra full-frame & vẽ lên ảnh gốc
                    kp_full = hamer_kp_crop_to_full(kp_crop21.copy(), batch, n, model_cfg)
                    full_dbg = draw_hand_skeleton_bgr(img_cv2.copy(), kp_full,
                                                      pt_color=(0,255,255), ln_color=(0,165,255))
                    cv2.imwrite(os.path.join(args.out_folder,
                                f"{img_fn}_{person_id}_hamer_fullkp.jpg"), full_dbg)


                # Side view (optional)
                if args.side_view:
                    white_img = (torch.ones_like(batch['img'][n]).cpu() - DEFAULT_MEAN[:, None, None] / 255.0) / (
                        DEFAULT_STD[:, None, None] / 255.0
                    )
                    side_img = renderer(
                        out['pred_vertices'][n].detach().cpu().numpy(),
                        out['pred_cam_t'][n].detach().cpu().numpy(),
                        white_img,
                        mesh_base_color=LIGHT_BLUE,
                        scene_bg_color=(1, 1, 1),
                        side_view=True
                    )
                    final_img = np.concatenate([input_patch, regression_img, side_img], axis=1)
                else:
                    final_img = np.concatenate([input_patch, regression_img], axis=1)

                cv2.imwrite(
                    os.path.join(args.out_folder, f'{img_fn}_{person_id}.png'),
                    (255.0 * final_img[:, :, ::-1]).astype(np.uint8)
                )

                # Collect for full-frame overlay
                verts = out['pred_vertices'][n].detach().cpu().numpy()
                is_r = batch['right'][n].detach().cpu().numpy()
                verts[:, 0] = (2 * is_r - 1) * verts[:, 0]
                cam_t = pred_cam_t_full[n]
                all_verts.append(verts)
                all_cam_t.append(cam_t)
                all_right.append(is_r)

                # Save mesh (optional)
                if args.save_mesh:
                    camera_translation = cam_t.copy()
                    tmesh = renderer.vertices_to_trimesh(verts, camera_translation, LIGHT_BLUE, is_right=is_r)
                    tmesh.export(os.path.join(args.out_folder, f'{img_fn}_{person_id}.obj'))

        # Render full-frame overlay (mesh only)
        if args.full_frame and len(all_verts) > 0:
            misc_args = dict(
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
                focal_length=scaled_focal_length,
            )
            # use last img_size[n] like original demo (works)
            cam_view = renderer.render_rgba_multiple(
                all_verts, cam_t=all_cam_t, render_res=img_size[n], is_right=all_right, **misc_args
            )

            input_img = img_cv2.astype(np.float32)[:, :, ::-1] / 255.0
            input_img = np.concatenate([input_img, np.ones_like(input_img[:, :, :1])], axis=2)
            input_img_overlay = input_img[:, :, :3] * (1 - cam_view[:, :, 3:]) + cam_view[:, :, :3] * cam_view[:, :, 3:]

            cv2.imwrite(
                os.path.join(args.out_folder, f'{img_fn}_all.jpg'),
                (255.0 * input_img_overlay[:, :, ::-1]).astype(np.uint8)
            )


if __name__ == '__main__':
    main()
