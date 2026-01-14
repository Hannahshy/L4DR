# Modified from OpenPCDet (https://github.com/open-mmlab/OpenPCDet)
# Based on PVRCNNPlusPlus & Detector3DTemplate
import os
import torch
import torch.nn as nn
import numpy as np
from easydict import EasyDict
import json, time
from ops.iou3d_nms import iou3d_nms_utils
from utils.spconv_utils import find_all_spconv_keys
from models import backbone_2d, backbone_3d, head, roi_head
from models.backbone_2d import map_to_bev
from models.backbone_3d import pfe, vfe
from models.model_utils import model_nms_utils
from .utils import common_utils
from tools.dynamic_denoise import DynamicDenoisePredictor
from models.pre_processor.motion_compensator import TemporalMotionCompensator
tv = None
try:
    import cumm.tensorview as tv
except Exception:
    tv = None

# Try importing a dedicated doppler feature derivation if present; otherwise we'll use a simple fallback
try:
    from models.pre_processor.doppler_features import derive_doppler_features
except Exception:
    derive_doppler_features = None


# Robust VoxelGeneratorWrapper to handle different spconv versions / kwarg names
class VoxelGeneratorWrapper():
    def __init__(self, vsize_xyz, coors_range_xyz, num_point_features,
                 max_num_points_per_voxel=None, max_num_voxels=None, max_voxels=None):
        """
        Robust wrapper for spconv VoxelGenerator. Accepts either keyword
        'max_num_voxels' or 'max_voxels' for compatibility with different call-sites.
        """
        # resolve max_num_voxels from either name
        if max_num_voxels is None and max_voxels is not None:
            max_num_voxels = max_voxels

        try:
            from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
            self.spconv_ver = 1
        except Exception:
            try:
                from spconv.utils import VoxelGenerator
                self.spconv_ver = 1
            except Exception:
                from spconv.utils import Point2VoxelCPU3d as VoxelGenerator
                self.spconv_ver = 2

        if self.spconv_ver == 1:
            # Try common signature paths with defensive fallbacks
            try:
                self._voxel_generator = VoxelGenerator(
                    voxel_size=vsize_xyz,
                    point_cloud_range=coors_range_xyz,
                    max_num_points=max_num_points_per_voxel,
                    max_voxels=max_num_voxels
                )
            except TypeError:
                try:
                    self._voxel_generator = VoxelGenerator(
                        voxel_size=vsize_xyz,
                        point_cloud_range=coors_range_xyz,
                        max_num_points=max_num_points_per_voxel,
                        max_num_voxels=max_num_voxels
                    )
                except TypeError:
                    # positional fallback
                    self._voxel_generator = VoxelGenerator(
                        vsize_xyz,
                        coors_range_xyz,
                        max_num_points_per_voxel,
                        max_num_voxels
                    )
        else:
            try:
                self._voxel_generator = VoxelGenerator(
                    vsize_xyz=vsize_xyz,
                    coors_range_xyz=coors_range_xyz,
                    num_point_features=num_point_features,
                    max_num_points_per_voxel=max_num_points_per_voxel,
                    max_num_voxels=max_num_voxels
                )
            except TypeError:
                self._voxel_generator = VoxelGenerator(
                    vsize_xyz, coors_range_xyz, num_point_features,
                    max_num_points_per_voxel, max_num_voxels
                )

    def generate(self, points):
        if self.spconv_ver == 1:
            voxel_output = self._voxel_generator.generate(points)
            if isinstance(voxel_output, dict):
                voxels, coordinates, num_points = \
                    voxel_output['voxels'], voxel_output['coordinates'], voxel_output['num_points_per_voxel']
            else:
                voxels, coordinates, num_points = voxel_output
        else:
            assert tv is not None, f"Unexpected error, library: 'cumm' wasn't imported properly."
            voxel_output = self._voxel_generator.point_to_voxel(tv.from_numpy(points))
            tv_voxels, tv_coordinates, tv_num_points = voxel_output
            voxels = tv_voxels.numpy()
            coordinates = tv_coordinates.numpy()
            num_points = tv_num_points.numpy()
        return voxels, coordinates, num_points


class PointPillar_RLF(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model_cfg = cfg.MODEL
        self.dataset_cfg = cfg.DATASET

        # class setup
        self.num_class = 0
        self.class_names = []
        dict_label = self.cfg.DATASET.label.copy()
        list_for_pop = ['calib', 'onlyR', 'Label', 'consider_cls', 'consider_roi', 'remove_0_obj']
        for temp_key in list_for_pop:
            dict_label.pop(temp_key, None)
        self.dict_cls_name_to_id = dict()
        for k, v in dict_label.items():
            _, logit_idx, _, _ = v
            self.dict_cls_name_to_id[k] = logit_idx
            self.dict_cls_name_to_id['Background'] = 0
            if logit_idx > 0:
                self.num_class += 1
                self.class_names.append(k)
        # keep dense head mapping consistent
        try:
            self.model_cfg.DENSE_HEAD.CLASS_NAMES_EACH_HEAD.append(self.class_names)
        except Exception:
            pass

        # Common params - read raw per-point dims from dataset cfg
        ldr_n_used = int(self.dataset_cfg.ldr64.n_used)
        rdr_n_used = int(self.dataset_cfg.rdr_sparse.n_used)
        extra_lidar_channels = max(0, ldr_n_used - 3)
        num_rawpoint_features = [ldr_n_used, rdr_n_used + extra_lidar_channels]
        self.num_point_features = num_rawpoint_features

        point_cloud_range = np.array(self.dataset_cfg.roi.xyz)
        voxel_size = self.dataset_cfg.roi.voxel_size
        grid_size = (point_cloud_range[3:6] - point_cloud_range[0:3]) / np.array(voxel_size)
        grid_size = np.round(grid_size).astype(np.int64)
        model_info_dict = dict(
            module_list=[],
            num_rawpoint_features=num_rawpoint_features,
            num_point_features=num_rawpoint_features,
            grid_size=grid_size,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
        )

        # voxel generators
        self.ldr_voxel_generator_train = VoxelGeneratorWrapper(
            vsize_xyz=voxel_size,
            coors_range_xyz=point_cloud_range,
            num_point_features=ldr_n_used,
            max_num_points_per_voxel=self.model_cfg.PRE_PROCESSING.MAX_POINTS_PER_VOXEL,
            max_num_voxels=self.model_cfg.PRE_PROCESSING.MAX_NUMBER_OF_VOXELS['train'],
        )
        self.ldr_voxel_generator_test = VoxelGeneratorWrapper(
            vsize_xyz=voxel_size,
            coors_range_xyz=point_cloud_range,
            num_point_features=ldr_n_used,
            max_num_points_per_voxel=self.model_cfg.PRE_PROCESSING.MAX_POINTS_PER_VOXEL,
            max_num_voxels=self.model_cfg.PRE_PROCESSING.MAX_NUMBER_OF_VOXELS['test'],
        )
        self.rdr_voxel_generator_train = VoxelGeneratorWrapper(
            vsize_xyz=voxel_size,
            coors_range_xyz=point_cloud_range,
            num_point_features=rdr_n_used,
            max_num_points_per_voxel=self.model_cfg.PRE_PROCESSING.MAX_POINTS_PER_VOXEL,
            max_num_voxels=self.model_cfg.PRE_PROCESSING.MAX_NUMBER_OF_VOXELS['train'],
        )
        self.rdr_voxel_generator_test = VoxelGeneratorWrapper(
            vsize_xyz=voxel_size,
            coors_range_xyz=point_cloud_range,
            num_point_features=rdr_n_used,
            max_num_points_per_voxel=self.model_cfg.PRE_PROCESSING.MAX_POINTS_PER_VOXEL,
            max_num_voxels=self.model_cfg.PRE_PROCESSING.MAX_NUMBER_OF_VOXELS['test'],
        )

        self.point_head = None

        # Doppler fusion config
        self._use_doppler_in_point_features = False
        dop_cfg = None
        if hasattr(self.model_cfg, 'POINT_HEAD'):
            dop_cfg = self.model_cfg.POINT_HEAD.get('DOPPLER_FUSION', None)

        self._use_doppler_in_point_features = False
        self._use_score_doppler_fusion = False
        self._dop_feat_dim = 0
        self._score_fusion_mode = 'learnable_alpha'
        if dop_cfg is not None and dop_cfg.get('ENABLED', False):
            if getattr(self.dataset_cfg, 'rdr_sparse', None) is not None and int(self.dataset_cfg.rdr_sparse.get('n_used', 0)) >= 5:
                self._dop_feat_dim = int(dop_cfg.get('DOP_FEAT_DIM', 6))
                self._use_doppler_in_point_features = bool(dop_cfg.get('ADD_TO_POINT_FEATURES', False))
                self._use_score_doppler_fusion = bool(dop_cfg.get('SCORE_FUSION', False))
                self._score_fusion_mode = dop_cfg.get('FUSION_MODE', 'learnable_alpha')
                print(f"* Info: DOPPLER_FUSION enabled. ADD_TO_POINT_FEATURES={self._use_doppler_in_point_features}, SCORE_FUSION={self._use_score_doppler_fusion}, MODE={self._score_fusion_mode}, DOP_FEAT_DIM={self._dop_feat_dim}")

        if self._use_score_doppler_fusion:
            self.doppler_score_net = nn.Linear(self._dop_feat_dim, 1)
            if self._score_fusion_mode == 'learnable_alpha':
                self.score_alpha = nn.Parameter(torch.tensor(0.5))
            elif self._score_fusion_mode == 'mlp':
                try:
                    mlp_hidden = int(self.model_cfg.POINT_HEAD.DOPPLER_FUSION.get('MLP_HIDDEN', 16))
                    mlp_dropout = float(self.model_cfg.POINT_HEAD.DOPPLER_FUSION.get('MLP_DROPOUT', 0.1))
                except Exception:
                    mlp_hidden = 16
                    mlp_dropout = 0.1
                self.score_fuser = nn.Sequential(
                    nn.Linear(2, mlp_hidden),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=mlp_dropout),
                    nn.Linear(mlp_hidden, 1)
                )
                print(f"* Info: Using MLP fusion (hidden={mlp_hidden}, dropout={mlp_dropout}).")
            elif self._score_fusion_mode == 'mlp' and not hasattr(self, 'score_fuser'):
                self.score_fuser = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1))
            else:
                print(f"* Info: Using fusion mode {self._score_fusion_mode} (no learnable fusion module created here).")

        # dynamic denoise config
        try:
            dyn_ckpt = cfg.MODEL.PRE_PROCESSING.get('DYNAMIC_DENOISE_CKPT', None)
            dyn_alpha = cfg.MODEL.PRE_PROCESSING.get('DYNAMIC_THR_ALPHA', 0.9)
            dyn_default = cfg.MODEL.PRE_PROCESSING.get('DYNAMIC_THR_DEFAULT', 0.3)
        except Exception:
            dyn_ckpt = None; dyn_alpha = 0.9; dyn_default = 0.3

        dyn_enabled = False
        try:
            dyn_enabled = bool(getattr(cfg.MODEL.PRE_PROCESSING, 'DYNAMIC_DENOISE', False))
        except Exception:
            dyn_enabled = False
        print(f"* Info: DYNAMIC_DENOISE flag = {dyn_enabled}")

        self._dynamic_denoise_predictor = None
        if dyn_enabled:
            try:
                self._dynamic_denoise_predictor = DynamicDenoisePredictor(
                    cfg=cfg,
                    ckpt_path=dyn_ckpt,
                    device=device_str,
                    ema_alpha=dyn_alpha,
                    default_thr=dyn_default
                )
                print(f"* DynamicDenoisePredictor: loaded ckpt {dyn_ckpt} device={device_str}")
            except Exception as e:
                print(f"* Warning: failed to instantiate DynamicDenoisePredictor: {e}")
                self._dynamic_denoise_predictor = None
        else:
            self._dynamic_denoise_predictor = None
            print("* Info: Dynamic denoise disabled by cfg; predictor not instantiated.")

        # precomputed frame-level features (optional)
        self._precomp_feats = None
        self._precomp_frame_ids = None
        precomp_dir = 'tmp/frame_features'
        feat_file = os.path.join(precomp_dir, 'features.npy')
        ids_file = os.path.join(precomp_dir, 'frame_ids.txt')
        weather_file = os.path.join(precomp_dir, 'weathers.txt')
        if os.path.exists(feat_file) and os.path.exists(ids_file):
            try:
                self._precomp_feats = np.load(feat_file)
                with open(ids_file, 'r') as f:
                    self._precomp_frame_ids = [x.strip() for x in f.readlines()]
                if os.path.exists(weather_file):
                    with open(weather_file, 'r') as f:
                        self._precomp_weathers = [x.strip() for x in f.readlines()]
                else:
                    self._precomp_weathers = None
                print(f"* Info: loaded precomputed features from {feat_file}, shape={self._precomp_feats.shape}")
            except Exception as e:
                print(f"* Warning: failed to load precomputed frame features: {e}")
                self._precomp_feats = None
                self._precomp_frame_ids = None
                self._precomp_weathers = None
        else:
            self._precomp_feats = None
            self._precomp_frame_ids = None
            self._precomp_weathers = None

        # motion config
        motion_cfg = None
        try:
            motion_cfg = cfg.MODEL.PRE_PROCESSING.MOTION_COMPENSATION
        except Exception:
            try:
                motion_cfg = getattr(cfg.MODEL.PRE_PROCESSING, 'MOTION_COMPENSATION', None)
            except Exception:
                motion_cfg = None

        if motion_cfg is not None and isinstance(motion_cfg, dict):
            self._add_temporal_to_point_features = bool(motion_cfg.get('ADD_TO_POINT_FEATURES', False))
            self._temporal_feat_dim = int(motion_cfg.get('FEAT_DIM', 1))
        else:
            self._temporal_feat_dim = 1

        # Build VFE
        self.vfe = vfe.__all__[self.model_cfg.VFE.NAME](
            model_cfg=self.model_cfg.VFE,
            num_point_features=model_info_dict['num_rawpoint_features'],
            point_cloud_range=model_info_dict['point_cloud_range'],
            voxel_size=model_info_dict['voxel_size'],
            grid_size=model_info_dict['grid_size'],
        )

        model_info_dict['num_point_features'] = self.vfe.get_output_feature_dim()

        # doppler runtime concat accounted later (after backbone)
        if self._use_doppler_in_point_features:
            model_info_dict['num_point_features'] = int(model_info_dict['num_point_features']) + self._dop_feat_dim
            print(f"* Info: PointHead input_channels increased by {self._dop_feat_dim} to account for doppler-derived features (ADD_TO_POINT_FEATURES=True).")

        # save interim
        try:
            self.num_point_feature = int(model_info_dict['num_point_features'])
        except Exception:
            self.num_point_feature = model_info_dict['num_point_features']

        # Build backbone_3d
        self.backbone_3d = backbone_3d.__all__[self.model_cfg.BACKBONE_3D.NAME](
            model_cfg=self.model_cfg.BACKBONE_3D,
            input_channels=model_info_dict['num_rawpoint_features'][0],
            grid_size=model_info_dict['grid_size'],
            voxel_size=model_info_dict['voxel_size'],
            point_cloud_range=model_info_dict['point_cloud_range']
        )
        model_info_dict['num_point_features'] = self.backbone_3d.num_point_features
        try:
            self.num_point_feature = int(model_info_dict['num_point_features'])
        except Exception:
            self.num_point_feature = model_info_dict['num_point_features']

        model_info_dict['backbone_channels'] = self.backbone_3d.backbone_channels if hasattr(self.backbone_3d, 'backbone_channels') else None

        # Ensure temporal feature dim is counted after backbone output dim is known
        if getattr(self, '_add_temporal_to_point_features', False):
            try:
                model_info_dict['num_point_features'] = int(model_info_dict['num_point_features']) + int(self._temporal_feat_dim)
            except Exception:
                model_info_dict['num_point_features'] = int(self.backbone_3d.num_point_features) + int(self._temporal_feat_dim)
            print(f"* Info: adding temporal feature dim {self._temporal_feat_dim} to point features (post-BACKBONE)")

        # Build point head
        self.point_head = head.__all__[self.model_cfg.POINT_HEAD.NAME](
            model_cfg=self.model_cfg.POINT_HEAD,
            input_channels=model_info_dict['num_point_features'],
            num_class=self.num_class if not self.model_cfg.POINT_HEAD.CLASS_AGNOSTIC else 1,
            predict_boxes_when_training=self.model_cfg.get('ROI_HEAD', False)
        )

        # map_to_bev, backbone_2d, dense_head
        self.map_to_bev_module = map_to_bev.__all__[self.model_cfg.MAP_TO_BEV.NAME](
            model_cfg=self.model_cfg.MAP_TO_BEV,
            grid_size=model_info_dict['grid_size']
        )
        model_info_dict['num_bev_features'] = np.array(self.map_to_bev_module.num_bev_features).sum()
        self.backbone_2d = backbone_2d.__all__[self.model_cfg.BACKBONE_2D.NAME](
            model_cfg=self.model_cfg.BACKBONE_2D,
            input_channels=model_info_dict.get('num_bev_features', None)
        )
        model_info_dict['num_bev_features'] = self.backbone_2d.num_bev_features
        self.dense_head = head.__all__[self.model_cfg.DENSE_HEAD.NAME](
            model_cfg=self.model_cfg.DENSE_HEAD,
            input_channels=model_info_dict['num_bev_features'] if 'num_bev_features' in model_info_dict else self.model_cfg.DENSE_HEAD.INPUT_FEATURES,
            num_class=self.num_class if not self.model_cfg.DENSE_HEAD.CLASS_AGNOSTIC else 1,
            class_names=self.class_names,
            grid_size=model_info_dict['grid_size'],
            point_cloud_range=model_info_dict['point_cloud_range'],
            predict_boxes_when_training=self.model_cfg.get('ROI_HEAD', False),
            voxel_size=model_info_dict.get('voxel_size', False)
        )

        self.model_info_dict = model_info_dict

        # instantiate MME sparse processor (with safe fallback)
        try:
            from models.pre_processor.lrf_mme_sparse_processor import MMESparseProcessor
            self.mme = MMESparseProcessor(cfg)
            print("* MMESparseProcessor instantiated")
        except Exception as e:
            print(f"* Warning: failed to instantiate MMESparseProcessor: {e}")
            self.mme = None

        # Instantiate temporal motion compensator
        try:
            try:
                motion_cfg = cfg.MODEL.PRE_PROCESSING.MOTION_COMPENSATION
            except Exception:
                motion_cfg = getattr(cfg.MODEL.PRE_PROCESSING, 'MOTION_COMPENSATION', None)
            self.temporal_comp = TemporalMotionCompensator(motion_cfg)
            print("* TemporalMotionCompensator instantiated (enabled=%s)" % self.temporal_comp.enabled)
        except Exception as e:
            print(f"* Warning: failed to instantiate TemporalMotionCompensator: {e}")
            self.temporal_comp = None

        # Pre-processor flags
        self.is_pre_processing = self.model_cfg.PRE_PROCESSING.get('VER', None)
        self.shuffle_points = self.model_cfg.PRE_PROCESSING.get('SHUFFLE_POINTS', False)
        self.transform_points_to_voxels = self.model_cfg.PRE_PROCESSING.get('TRANSFORM_POINTS_TO_VOXELS', False)
        self.TP = common_utils.AverageMeter()
        self.P = common_utils.AverageMeter()
        self.TP_FN = common_utils.AverageMeter()
        self.TP_FP_FN = common_utils.AverageMeter()
        self.is_logging = cfg.GENERAL.LOGGING.IS_LOGGING

    @property
    def mode(self):
        return 'TRAIN' if self.training else 'TEST'

    def pre_processor(self, batch_dict):
        if self.is_pre_processing is None:
            return batch_dict
        elif self.is_pre_processing == 'v1_0':
            # gather batched radar points (use original rdr_n_used slicing)
            batched_rdr = batch_dict['rdr_sparse'].detach()
            batched_indices_rdr = batch_dict['batch_indices_rdr_sparse'].detach()
            list_points = []
            list_voxels = []
            list_voxel_coords = []
            list_voxel_num_points = []
            rdr_n_used = int(self.dataset_cfg.rdr_sparse.n_used)
            for batch_idx in range(batch_dict['batch_size']):
                idxs = torch.where(batched_indices_rdr == batch_idx)[0]
                temp_points = batched_rdr[idxs, :rdr_n_used]
                if (self.shuffle_points) and (self.training):
                    shuffle_idx = np.random.permutation(temp_points.shape[0])
                    temp_points = temp_points[shuffle_idx, :]
                list_points.append(temp_points)

                if self.transform_points_to_voxels:
                    if self.training:
                        voxels, coordinates, num_points = self.rdr_voxel_generator_train.generate(temp_points.cpu().numpy())
                    else:
                        voxels, coordinates, num_points = self.rdr_voxel_generator_test.generate(temp_points.cpu().numpy())
                    voxel_batch_idx = np.full((coordinates.shape[0], 1), batch_idx, dtype=np.int64)
                    coordinates = np.concatenate((voxel_batch_idx, coordinates), axis=-1)
                    list_voxels.append(voxels)
                    list_voxel_coords.append(coordinates)
                    list_voxel_num_points.append(num_points)

            batched_points = torch.cat(list_points, dim=0) if len(list_points) > 0 else torch.zeros((0, rdr_n_used), device=batched_rdr.device)
            batch_dict['radar_points'] = torch.cat((batched_indices_rdr.reshape(-1,1), batched_points), dim=1).cuda()
            batch_dict['radar_voxels'] = torch.from_numpy(np.concatenate(list_voxels, axis=0)).cuda() if len(list_voxels) > 0 else torch.zeros((0,32,rdr_n_used), device='cuda')
            batch_dict['radar_voxel_coords'] = torch.from_numpy(np.concatenate(list_voxel_coords, axis=0)).cuda() if len(list_voxel_coords) > 0 else torch.zeros((0,4), device='cuda')
            batch_dict['radar_voxel_num_points'] = torch.from_numpy(np.concatenate(list_voxel_num_points, axis=0)).cuda() if len(list_voxel_num_points) > 0 else torch.zeros((0,), device='cuda')
            batch_dict['gt_boxes'] = batch_dict['gt_boxes'].cuda()
            batch_dict['points'] = batch_dict['radar_points']

            # lidar similarly
            batched_ldr64 = batch_dict['ldr64']
            batched_indices_ldr64 = batch_dict['batch_indices_ldr64']
            list_points = []
            list_voxels = []
            list_voxel_coords = []
            list_voxel_num_points = []
            ldr_n_used = int(self.dataset_cfg.ldr64.n_used)
            for batch_idx in range(batch_dict['batch_size']):
                idxs = torch.where(batched_indices_ldr64 == batch_idx)[0]
                temp_points = batched_ldr64[idxs, :ldr_n_used]
                if (self.shuffle_points) and (self.training):
                    shuffle_idx = np.random.permutation(temp_points.shape[0])
                    temp_points = temp_points[shuffle_idx, :]
                list_points.append(temp_points)

                if self.transform_points_to_voxels:
                    if self.training:
                        voxels, coordinates, num_points = self.ldr_voxel_generator_train.generate(temp_points.cpu().numpy())
                    else:
                        voxels, coordinates, num_points = self.ldr_voxel_generator_test.generate(temp_points.cpu().numpy())
                    voxel_batch_idx = np.full((coordinates.shape[0], 1), batch_idx, dtype=np.int64)
                    coordinates = np.concatenate((voxel_batch_idx, coordinates), axis=-1)
                    list_voxels.append(voxels)
                    list_voxel_coords.append(coordinates)
                    list_voxel_num_points.append(num_points)

            batched_points = torch.cat(list_points, dim=0) if len(list_points)>0 else torch.zeros((0, ldr_n_used), device=batched_ldr64.device)
            batch_dict['lidar_points'] = torch.cat((batched_indices_ldr64.reshape(-1,1), batched_points), dim=1).cuda()
            batch_dict['lidar_voxels'] = torch.from_numpy(np.concatenate(list_voxels, axis=0)).cuda() if len(list_voxels)>0 else torch.zeros((0,32,ldr_n_used), device='cuda')
            batch_dict['lidar_voxel_coords'] = torch.from_numpy(np.concatenate(list_voxel_coords, axis=0)).cuda() if len(list_voxel_coords)>0 else torch.zeros((0,4), device='cuda')
            batch_dict['lidar_voxel_num_points'] = torch.from_numpy(np.concatenate(list_voxel_num_points, axis=0)).cuda() if len(list_voxel_num_points)>0 else torch.zeros((0,), device='cuda')

            return batch_dict

    # 在 PointPillar_RLF 类中添加
    def _safe_point_head_forward_no_grad(self, batch_dict):
        """
        Safe no_grad forward of point_head: if current batch point_features channel count
        doesn't match point_head.first_linear.in_features, temporarily pad or truncate
        point_features for this temporary forward. Does NOT modify the original batch_dict.
        Returns the temporary forward output (dict).
        """
        # find first linear module in point_head (cls_layers or fallback)
        first_linear = None
        try:
            for m in self.point_head.cls_layers:
                if isinstance(m, torch.nn.Linear):
                    first_linear = m
                    break
        except Exception:
            for m in self.point_head.modules():
                if isinstance(m, torch.nn.Linear):
                    first_linear = m
                    break
    
        expected_in = first_linear.in_features if first_linear is not None else None
    
        # shallow copy dict so we do not mutate original entries
        tmp_batch = batch_dict.copy()
    
        pf = batch_dict.get('point_features', None)
        if pf is None or expected_in is None:
            # fall back to normal forward (let it raise if incompatible)
            with torch.no_grad():
                return self.point_head(tmp_batch)
    
        cur_ch = pf.shape[1]
        if cur_ch == expected_in:
            with torch.no_grad():
                return self.point_head(tmp_batch)
        elif cur_ch < expected_in:
            pad_ch = expected_in - cur_ch
            zeros = pf.new_zeros((pf.shape[0], pad_ch))
            tmp_batch['point_features'] = torch.cat([pf, zeros], dim=1)
            with torch.no_grad():
                return self.point_head(tmp_batch)
        else:
            # cur_ch > expected_in: temporarily truncate
            tmp_batch['point_features'] = pf[:, :expected_in].contiguous()
            with torch.no_grad():
                return self.point_head(tmp_batch)
    
    def forward(self, batch_dict):
        # pre_processor
        batch_dict = self.pre_processor(batch_dict)

        # run backbone 3d (includes VFE)
        batch_dict = self.backbone_3d(batch_dict)

        # If configured, compute doppler-derived features and append to point_features BEFORE point_head
        if self._use_doppler_in_point_features:
            pf = batch_dict.get('point_features', None)
            rdr_pts = batch_dict.get('radar_points', None)
            if rdr_pts is None:
                rdr_pts = batch_dict.get('rdr_sparse', None)

            if pf is not None and rdr_pts is not None:
                device = pf.device
                power_idx = int(self.dataset_cfg.rdr_sparse.get('power_idx', 3))
                dop_idx = int(self.dataset_cfg.rdr_sparse.get('doppler_idx', 4))
                try:
                    raw = rdr_pts[:, 1:1 + int(self.dataset_cfg.rdr_sparse.n_used)].to(device)
                except Exception:
                    raw = rdr_pts[:, :int(self.dataset_cfg.rdr_sparse.n_used)].to(device)

                if derive_doppler_features is not None:
                    try:
                        dop_feats = derive_doppler_features(raw, cfg=self.model_cfg.POINT_HEAD.DOPPLER_FUSION)
                        if isinstance(dop_feats, np.ndarray):
                            dop_feats = torch.from_numpy(dop_feats).to(device)
                        else:
                            dop_feats = dop_feats.to(device)
                    except Exception as e:
                        print(f"* Warning: derive_doppler_features failed: {e}. Falling back to simple doppler features.")
                        dop_feats = None
                else:
                    dop_feats = None

                if dop_feats is None:
                    dop = torch.zeros((raw.shape[0],), device=device)
                    power = torch.zeros((raw.shape[0],), device=device)
                    if raw.shape[1] > dop_idx:
                        dop = raw[:, dop_idx]
                    if raw.shape[1] > power_idx:
                        power = raw[:, power_idx]

                    abs_dop = dop.abs().unsqueeze(1)
                    power_norm = torch.zeros_like(abs_dop)
                    if rdr_pts is not None and rdr_pts.shape[1] > 0:
                        if 'radar_points' in batch_dict and batch_dict['radar_points'].shape[1] > (1 + int(self.dataset_cfg.rdr_sparse.n_used)):
                            batch_idxs = batch_dict['radar_points'][:, 0].long().to(device)
                        else:
                            batch_idxs = torch.zeros((raw.shape[0],), dtype=torch.long, device=device)
                        unique_bs = torch.unique(batch_idxs)
                        neigh_mean = torch.zeros_like(abs_dop)
                        neigh_std = torch.zeros_like(abs_dop)
                        support = torch.zeros_like(abs_dop)
                        dop_diff = torch.zeros_like(abs_dop)
                        for b in unique_bs:
                            bmask = (batch_idxs == b)
                            pvals = power[bmask]
                            if pvals.numel() > 0:
                                pmax = pvals.max()
                                if pmax.abs() < 1e-6:
                                    pnorm = torch.zeros_like(pvals)
                                else:
                                    pnorm = pvals / (pmax + 1e-6)
                                power_norm[bmask, 0] = pnorm
                                dvals = dop[bmask]
                                if dvals.numel() > 0:
                                    dmean = dvals.mean()
                                    dstd = dvals.std(unbiased=False)
                                    neigh_mean[bmask, 0] = dmean
                                    neigh_std[bmask, 0] = dstd
                                    dop_diff[bmask, 0] = (dvals - dmean)
                                    thr_low = torch.quantile(pvals, 0.10) if pvals.numel() > 1 else 0.0
                                    support[bmask, 0] = (pvals > thr_low).float()
                        dop_feats = torch.cat([abs_dop, power_norm, neigh_mean, neigh_std, support, dop_diff], dim=1)
                    else:
                        dop_feats = torch.zeros((raw.shape[0], self._dop_feat_dim), device=device)

                if dop_feats.shape[1] != self._dop_feat_dim:
                    if dop_feats.shape[1] > self._dop_feat_dim:
                        dop_feats = dop_feats[:, :self._dop_feat_dim].contiguous()
                    else:
                        pad = dop_feats.new_zeros((dop_feats.shape[0], self._dop_feat_dim - dop_feats.shape[1]))
                        dop_feats = torch.cat([dop_feats, pad], dim=1)

                try:
                    pf = batch_dict['point_features']
                    if pf.device != dop_feats.device:
                        dop_feats = dop_feats.to(pf.device)
                    batch_dict['point_features'] = torch.cat([pf, dop_feats], dim=1)
                except Exception as e:
                    print(f"* Warning: failed to concat doppler features into point_features: {e}")

# 改为如下逻辑：

        # -------------------------
        # Candidate-based temporal matching flow:
        # 1) Do a lightweight preliminary point scoring (no temporal features).
        # 2) Select candidates (score>prefilter_thr or top_k per batch).
        # 3) Compute temporal scores ONLY for candidates using TemporalMotionCompensator.
        # 4) Fill temporal scores into a full-length tensor and concat to point_features.
        # 5) Re-run point_head for final predictions.
        # -------------------------

        # ---------- candidate-based temporal matching & prefilter (替换原有 temporal concat 段) ----------
        # 将原先“把 temporal score 直接全部计算并 concat 再跑 point_head”的流程
        # 替换为“先做一次 no_grad 预打分选 candidate -> 仅对候选点计算 temporal scores ->
        # 把 temporal scores 填回全量并 concat -> 再次做 point_head（有梯度）”。
        if getattr(self, '_add_temporal_to_point_features', False) and getattr(self, 'temporal_comp', None) is not None and self.temporal_comp.enabled:
            # --- 从 cfg 读取 prefilter 配置（兼容 dict/EasyDict） ---
            try:
                prefilter_cfg = self.model_cfg.PRE_PROCESSING.get('PREFILTER', None)
            except Exception:
                try:
                    prefilter_cfg = getattr(self.model_cfg.PRE_PROCESSING, 'PREFILTER', None)
                except Exception:
                    prefilter_cfg = None

            if prefilter_cfg is None:
                prefilter_point_score_th = 0.3
                prefilter_topk_per_batch = 2000
            else:
                try:
                    prefilter_point_score_th = float(prefilter_cfg.get('POINT_SCORE_TH', 0.3))
                except Exception:
                    try:
                        prefilter_point_score_th = float(getattr(prefilter_cfg, 'POINT_SCORE_TH', 0.3))
                    except Exception:
                        prefilter_point_score_th = 0.3
                try:
                    prefilter_topk_per_batch = int(prefilter_cfg.get('TOPK_PER_BATCH', 2000))
                except Exception:
                    try:
                        prefilter_topk_per_batch = int(getattr(prefilter_cfg, 'TOPK_PER_BATCH', 2000))
                    except Exception:
                        prefilter_topk_per_batch = 2000

            # --- 1) 轻量预筛（no_grad 前向一次 point_head 获取初始点得分，并可包含 doppler 融合） ---
            orig_point_features = batch_dict.get('point_features', None)
            s_final_init = None
            with torch.no_grad():
                try:
                    batch_tmp = self._safe_point_head_forward_no_grad(batch_dict)
                    s_point_raw_init = batch_tmp.get('point_cls_scores', None)
                except Exception as e:
                    print(f"* Warning: preliminary point_head forward failed: {e}. Falling back to no score.")
                    s_point_raw_init = None

                # 得到 s_point_prob_init（分类概率）
                if s_point_raw_init is None:
                    s_point_prob_init = None
                else:
                    if s_point_raw_init.dim() == 2 and s_point_raw_init.shape[1] > 1:
                        s_point_prob_init = torch.sigmoid(s_point_raw_init).max(dim=-1)[0]
                    else:
                        s_point_prob_init = torch.sigmoid(s_point_raw_init).squeeze(-1)

                # 默认将 s_final_init 设为分类分数
                s_final_init = s_point_prob_init

                # 若配置了 score-level doppler 融合，则尽量在no_grad阶段计算与最终一致的融合得分用于预筛
                try:
                    if getattr(self, '_use_score_doppler_fusion', False) and s_point_prob_init is not None:
                        rdr_pts = batch_dict.get('radar_points', None)
                        if rdr_pts is None:
                            rdr_pts = batch_dict.get('rdr_sparse', None)

                        if rdr_pts is None:
                            s_final_init = s_point_prob_init
                        else:
                            try:
                                raw = rdr_pts[:, 1:1 + int(self.dataset_cfg.rdr_sparse.n_used)].to(s_point_prob_init.device)
                            except Exception:
                                raw = rdr_pts[:, :int(self.dataset_cfg.rdr_sparse.n_used)].to(s_point_prob_init.device)

                            dop_feats = None
                            try:
                                if 'derive_doppler_features' in globals() and derive_doppler_features is not None:
                                    dop_feats = derive_doppler_features(raw, cfg=self.model_cfg.POINT_HEAD.DOPPLER_FUSION)
                                    if isinstance(dop_feats, np.ndarray):
                                        dop_feats = torch.from_numpy(dop_feats).to(s_point_prob_init.device)
                            except Exception:
                                dop_feats = None

                            if dop_feats is None:
                                dop_idx = int(self.dataset_cfg.rdr_sparse.get('doppler_idx', 4))
                                power_idx = int(self.dataset_cfg.rdr_sparse.get('power_idx', 3))
                                dop = torch.zeros((raw.shape[0],), device=s_point_prob_init.device)
                                power = torch.zeros((raw.shape[0],), device=s_point_prob_init.device)
                                if raw.shape[1] > dop_idx:
                                    dop = raw[:, dop_idx]
                                if raw.shape[1] > power_idx:
                                    power = raw[:, power_idx]

                                abs_dop = dop.abs().unsqueeze(1)
                                power_norm = torch.zeros_like(abs_dop)
                                neigh_mean = torch.zeros_like(abs_dop)
                                neigh_std = torch.zeros_like(abs_dop)
                                support = torch.zeros_like(abs_dop)
                                dop_diff = torch.zeros_like(abs_dop)

                                try:
                                    if 'radar_points' in batch_dict and batch_dict['radar_points'].shape[1] > (1 + int(self.dataset_cfg.rdr_sparse.n_used)):
                                        batch_idxs_for_dop = batch_dict['radar_points'][:, 0].long().to(s_point_prob_init.device)
                                    else:
                                        batch_idxs_for_dop = torch.zeros((raw.shape[0],), dtype=torch.long, device=s_point_prob_init.device)
                                    unique_bs = torch.unique(batch_idxs_for_dop)
                                    for b in unique_bs:
                                        bmask = (batch_idxs_for_dop == b)
                                        pvals = power[bmask]
                                        if pvals.numel() > 0:
                                            pmax = pvals.max()
                                            if pmax.abs() < 1e-6:
                                                pnorm = torch.zeros_like(pvals)
                                            else:
                                                pnorm = pvals / (pmax + 1e-6)
                                            power_norm[bmask, 0] = pnorm
                                            dvals = dop[bmask]
                                            if dvals.numel() > 0:
                                                dmean = dvals.mean()
                                                dstd = dvals.std(unbiased=False)
                                                neigh_mean[bmask, 0] = dmean
                                                neigh_std[bmask, 0] = dstd
                                                dop_diff[bmask, 0] = (dvals - dmean)
                                                thr_low = torch.quantile(pvals, 0.10) if pvals.numel() > 1 else 0.0
                                                support[bmask, 0] = (pvals > thr_low).float()
                                    dop_feats = torch.cat([abs_dop, power_norm, neigh_mean, neigh_std, support, dop_diff], dim=1)
                                except Exception:
                                    dop_feats = torch.cat([abs_dop, torch.zeros_like(abs_dop), torch.zeros_like(abs_dop),
                                                           torch.zeros_like(abs_dop), torch.zeros_like(abs_dop), torch.zeros_like(abs_dop)], dim=1)

                            dop_feat_dim = int(self.model_cfg.POINT_HEAD.DOPPLER_FUSION.get('DOP_FEAT_DIM', 6))
                            if dop_feats.shape[1] != dop_feat_dim:
                                if dop_feats.shape[1] > dop_feat_dim:
                                    dop_feats = dop_feats[:, :dop_feat_dim].contiguous()
                                else:
                                    pad = dop_feats.new_zeros((dop_feats.shape[0], dop_feat_dim - dop_feats.shape[1]))
                                    dop_feats = torch.cat([dop_feats, pad], dim=1)

                            try:
                                s_dop_logits = self.doppler_score_net(dop_feats)
                                s_dop = torch.sigmoid(s_dop_logits).squeeze(1)
                            except Exception:
                                s_dop = dop_feats[:, 0].abs()
                                s_dop = (s_dop - s_dop.min()) / (s_dop.max() - s_dop.min() + 1e-6)

                            fusion_mode = getattr(self, '_score_fusion_mode', 'learnable_alpha')
                            if fusion_mode == 'learnable_alpha' and hasattr(self, 'score_alpha'):
                                alpha = torch.sigmoid(self.score_alpha)
                                s_final_init = alpha * s_point_prob_init + (1.0 - alpha) * s_dop
                            elif fusion_mode == 'mlp' and hasattr(self, 'score_fuser'):
                                inp = torch.stack([s_point_prob_init, s_dop], dim=1)
                                s_final_init = torch.sigmoid(self.score_fuser(inp)).squeeze(1)
                            else:
                                fixed_alpha = float(self.model_cfg.POINT_HEAD.DOPPLER_FUSION.get('FIXED_ALPHA', 0.5))
                                s_final_init = fixed_alpha * s_point_prob_init + (1.0 - fixed_alpha) * s_dop
                except Exception as e:
                    print(f"* Warning: failed to compute doppler fusion in prefilter: {e}")
                    s_final_init = s_point_prob_init

            # 恢复原始 point_features（确保下一步最终前向使用正确输入）
            if orig_point_features is not None:
                batch_dict['point_features'] = orig_point_features
            # 清理临时可能被写入的字段，避免对后续前向造成副作用
            if 'point_cls_scores' in batch_dict:
                try:
                    del batch_dict['point_cls_scores']
                except Exception:
                    pass
            if 'point_cls_preds' in batch_dict:
                try:
                    del batch_dict['point_cls_preds']
                except Exception:
                    pass

            # --- 2) 构造 candidate_mask（按阈值并按 batch 保证 top-K） ---
            N = batch_dict['rdr_sparse'].shape[0]
            device = batch_dict['rdr_sparse'].device
            candidate_mask = torch.zeros((N,), dtype=torch.bool, device=device)

            if s_final_init is None:
                # 如果没有预前向得分，退化为全选或按 top-K（这里选择按 power/topk 回退）
                if prefilter_topk_per_batch <= 0:
                    candidate_mask[:] = True
                else:
                    batch_idxs = batch_dict.get('batch_indices_rdr_sparse', None)
                    if batch_idxs is None:
                        candidate_mask[:] = True
                    else:
                        batch_idxs = batch_idxs.long().to(device)
                        B = int(batch_dict.get('batch_size', 1))
                        power_idx = None
                        try:
                            power_idx = int(self.dataset_cfg.rdr_sparse.get('power_idx', 3))
                        except Exception:
                            power_idx = 3
                        try:
                            rdr_raw = batch_dict['rdr_sparse']
                            if rdr_raw.shape[1] > power_idx:
                                fallback_scores = rdr_raw[:, power_idx].float().to(device)
                            else:
                                fallback_scores = torch.randn((N,), device=device)
                        except Exception:
                            fallback_scores = torch.randn((N,), device=device)
                        for b in range(B):
                            idxs_b = torch.nonzero((batch_idxs == b), as_tuple=False).view(-1)
                            if idxs_b.numel() == 0:
                                continue
                            k = min(prefilter_topk_per_batch, idxs_b.numel())
                            topk_vals, topk_idx = torch.topk(fallback_scores[idxs_b], k)
                            candidate_mask[idxs_b[topk_idx]] = True
            else:
                # 使用 s_final_init 作为预筛分数（已包含分类与可选 doppler 融合）
                if s_final_init.dim() == 2 and s_final_init.shape[1] > 1:
                    s_score_init = torch.sigmoid(s_final_init).max(dim=-1)[0]
                else:
                    s_score_init = s_final_init.squeeze(-1)
                thr_mask = (s_score_init > prefilter_point_score_th)
                candidate_mask = thr_mask.clone()
                batch_idxs = batch_dict['batch_indices_rdr_sparse'].long().to(device)
                B = int(batch_dict.get('batch_size', 1))
                if prefilter_topk_per_batch > 0:
                    for b in range(B):
                        idxs_b = torch.nonzero((batch_idxs == b), as_tuple=False).view(-1)
                        if idxs_b.numel() == 0:
                            continue
                        selected_b = candidate_mask[idxs_b]
                        if selected_b.sum() > prefilter_topk_per_batch:
                            sel_idxs = torch.nonzero(selected_b, as_tuple=False).view(-1)
                            sel_global = idxs_b[sel_idxs]
                            sel_scores = s_score_init[sel_global]
                            topk_vals, topk_idx = torch.topk(sel_scores, prefilter_topk_per_batch)
                            keep = torch.zeros_like(sel_scores, dtype=torch.bool)
                            keep[topk_idx] = True
                            candidate_mask[sel_global] = False
                            candidate_mask[sel_global[keep]] = True
                        elif selected_b.sum() == 0:
                            scores_b = s_score_init[idxs_b]
                            k = min(prefilter_topk_per_batch, idxs_b.numel())
                            topk_vals, topk_idx = torch.topk(scores_b, k)
                            candidate_mask[idxs_b[topk_idx]] = True

            # --- 3) 仅对 candidate 计算 temporal scores ---
            try:
                t_scores_full = self.temporal_comp.get_temporal_scores(batch_dict, candidate_mask=candidate_mask)
                t_scores_full = t_scores_full.to(device=device).float()
            except Exception as e:
                print(f"* Warning: temporal_comp.get_temporal_scores(candidate_mask=...) failed: {e}. Falling back to full compute.")
                t_scores_full = self.temporal_comp.get_temporal_scores(batch_dict)
                t_scores_full = t_scores_full.to(device=device).float()

            # --- 4) 处理 temporal feature dim 并 concat 回 point_features ---
            t_feat_dim = int(getattr(self, '_temporal_feat_dim', 1))
            if t_feat_dim == 1:
                t_feat = t_scores_full.unsqueeze(1)  # (N,1)
            else:
                t_feat = t_scores_full.unsqueeze(1).repeat(1, t_feat_dim)

            pf = batch_dict.get('point_features', None)
            if pf is None:
                print("* Warning: no point_features found; skipping temporal concat.")
            else:
                if t_feat.device != pf.device:
                    t_feat = t_feat.to(pf.device)
                batch_dict['point_features'] = torch.cat([pf, t_feat], dim=1)

            # --- 5) 最终一次前向 point_head（参与梯度） ---
            batch_dict = self.point_head(batch_dict)

        else:
            # 若未启用 temporal_comp 或未配置 temporal concat，保持原有单次前向
            batch_dict = self.point_head(batch_dict)
        # ---------- end candidate-based temporal matching ----------
            
        # run point head
        batch_dict = self.point_head(batch_dict)

        # compute s_point
        s_point_raw = batch_dict['point_cls_scores']
        if s_point_raw.dim() == 2 and s_point_raw.shape[1] > 1:
            s_point_prob = torch.sigmoid(s_point_raw).max(dim=-1)[0]
        else:
            s_point_prob = torch.sigmoid(s_point_raw).squeeze(-1)

        s_final = s_point_prob

        # doppler score fusion
        if self._use_score_doppler_fusion:
            rdr_pts = batch_dict.get('radar_points', None)
            if rdr_pts is None:
                rdr_pts = batch_dict.get('rdr_sparse', None)

            dop_idx = int(self.dataset_cfg.rdr_sparse.get('doppler_idx', 4)) if getattr(self.dataset_cfg, 'rdr_sparse', None) is not None else 4
            power_idx = int(self.dataset_cfg.rdr_sparse.get('power_idx', 3)) if getattr(self.dataset_cfg, 'rdr_sparse', None) is not None else 3

            dop_feats = None
            if rdr_pts is not None:
                try:
                    raw = rdr_pts[:, 1:1 + int(self.dataset_cfg.rdr_sparse.n_used)].to(s_point_prob.device)
                except Exception:
                    raw = rdr_pts[:, :int(self.dataset_cfg.rdr_sparse.n_used)].to(s_point_prob.device)

                try:
                    from models.pre_processor.doppler_features import derive_doppler_features
                except Exception:
                    derive_doppler_features = None

                dop_feats = None
                if derive_doppler_features is not None:
                    try:
                        dop_feats = derive_doppler_features(raw, cfg=self.model_cfg.POINT_HEAD.DOPPLER_FUSION)
                        if isinstance(dop_feats, np.ndarray):
                            dop_feats = torch.from_numpy(dop_feats).to(s_point_prob.device)
                        else:
                            dop_feats = dop_feats.to(s_point_prob.device)
                    except Exception as e:
                        print(f"* Warning: derive_doppler_features failed during score fusion: {e}. Using fallback.")
                        dop_feats = None

                if dop_feats is None:
                    dop = raw[:, dop_idx] if raw.shape[1] > dop_idx else torch.zeros((raw.shape[0],), device=s_point_prob.device)
                    power = raw[:, power_idx] if raw.shape[1] > power_idx else torch.zeros((raw.shape[0],), device=s_point_prob.device)
                    abs_dop = dop.abs().unsqueeze(1)
                    power_norm = torch.zeros_like(abs_dop)
                    neigh_mean = torch.zeros_like(abs_dop)
                    neigh_std = torch.zeros_like(abs_dop)
                    support = torch.zeros_like(abs_dop)
                    dop_diff = torch.zeros_like(abs_dop)

                    if ('radar_points' in batch_dict) and batch_dict['radar_points'].shape[1] >= 1:
                        if batch_dict['radar_points'].shape[1] >= (1 + int(self.dataset_cfg.rdr_sparse.n_used)):
                            batch_idxs = batch_dict['radar_points'][:, 0].long().to(s_point_prob.device)
                        else:
                            batch_idxs = torch.zeros((raw.shape[0],), dtype=torch.long, device=s_point_prob.device)
                    else:
                        batch_idxs = torch.zeros((raw.shape[0],), dtype=torch.long, device=s_point_prob.device)

                    unique_bs = torch.unique(batch_idxs)
                    for b in unique_bs:
                        bmask = (batch_idxs == b)
                        pvals = power[bmask]
                        if pvals.numel() > 0:
                            pmax = pvals.max()
                            if pmax.abs() < 1e-6:
                                pnorm = torch.zeros_like(pvals)
                            else:
                                pnorm = pvals / (pmax + 1e-6)
                            power_norm[bmask, 0] = pnorm
                            dvals = dop[bmask]
                            if dvals.numel() > 0:
                                dmean = dvals.mean()
                                dstd = dvals.std(unbiased=False)
                                neigh_mean[bmask, 0] = dmean
                                neigh_std[bmask, 0] = dstd
                                dop_diff[bmask, 0] = (dvals - dmean)
                                thr_low = torch.quantile(pvals, 0.10) if pvals.numel() > 1 else 0.0
                                support[bmask, 0] = (pvals > thr_low).float()

                    dop_feats = torch.cat([abs_dop, power_norm, neigh_mean, neigh_std, support, dop_diff], dim=1)
                    if dop_feats.shape[1] != self._dop_feat_dim:
                        if dop_feats.shape[1] > self._dop_feat_dim:
                            dop_feats = dop_feats[:, :self._dop_feat_dim].contiguous()
                        else:
                            pad = dop_feats.new_zeros((dop_feats.shape[0], self._dop_feat_dim - dop_feats.shape[1]))
                            dop_feats = torch.cat([dop_feats, pad], dim=1)
            else:
                dop_feats = s_point_prob.new_zeros((s_point_prob.shape[0], self._dop_feat_dim))

            try:
                s_dop_logits = self.doppler_score_net(dop_feats)
                s_dop = torch.sigmoid(s_dop_logits).squeeze(1)
            except Exception as e:
                s_dop = dop_feats[:, 0].abs()
                s_dop = (s_dop - s_dop.min()) / (s_dop.max() - s_dop.min() + 1e-6)

            if self._score_fusion_mode == 'learnable_alpha' and hasattr(self, 'score_alpha'):
                alpha = torch.sigmoid(self.score_alpha)
                s_final = alpha * s_point_prob + (1.0 - alpha) * s_dop
            elif self._score_fusion_mode == 'mlp' and hasattr(self, 'score_fuser'):
                inp = torch.stack([s_point_prob, s_dop], dim=1)
                s_final = torch.sigmoid(self.score_fuser(inp)).squeeze(1)
            else:
                alpha_fixed = float(dop_cfg.get('FIXED_ALPHA', 0.5)) if dop_cfg is not None else 0.5
                s_final = alpha_fixed * s_point_prob + (1.0 - alpha_fixed) * s_dop

        # produce temporal mask for diagnostics/optional postprocess
        temporal_mask = None
        try:
            if getattr(self, 'temporal_comp', None) is not None:
                temporal_mask = self.temporal_comp.get_temporal_mask(batch_dict)
                if temporal_mask is not None:
                    temporal_mask = temporal_mask.to(s_final.device)
        except Exception as e:
            print(f"* Info: temporal_comp.get_temporal_mask failed: {e}")
            temporal_mask = None

        # dynamic/static denoise selection (unchanged)
        if getattr(self, '_dynamic_denoise_predictor', None) is not None and self._dynamic_denoise_predictor.enabled:
            try:
                feats_np = None
                weather_list = None
                frame_ids = None
                if getattr(self, '_precomp_feats', None) is not None and getattr(self, '_precomp_frame_ids', None) is not None:
                    frame_ids_batch = None
                    if 'meta_list' in batch_dict and isinstance(batch_dict['meta_list'], (list, tuple)):
                        frame_ids_batch = [(m.get('seq', '') + ',' + (m.get('label_v2_0', '') or m.get('label', ''))) for m in batch_dict['meta_list']]
                    elif 'meta' in batch_dict and isinstance(batch_dict['meta'], (list, tuple)):
                        frame_ids_batch = [(m.get('seq', '') + ',' + (m.get('label_v2_0', '') or m.get('label', ''))) for m in batch_dict['meta']]
                    else:
                        bidx = batch_dict.get('batch_indices_rdr_sparse', None)
                        if bidx is not None:
                            try:
                                bidx_np = bidx.detach().cpu().numpy()
                                B = int(bidx_np.max()) + 1
                                frame_ids_batch = [f"pos_{i}" for i in range(B)]
                            except Exception:
                                frame_ids_batch = None

                    if frame_ids_batch is not None:
                        fid2idx = {fid: i for i, fid in enumerate(self._precomp_frame_ids)}
                        idxs = []
                        for fid in frame_ids_batch:
                            if fid in fid2idx:
                                idxs.append(fid2idx[fid])
                            else:
                                found = None
                                tail = fid.split(',')[-1]
                                for j, ff in enumerate(self._precomp_frame_ids):
                                    if tail and tail in ff:
                                        found = j; break
                                if found is not None:
                                    idxs.append(found)
                                else:
                                    raise ValueError(f"Frame id {fid} not found in precomputed frame ids.")
                        feats_np = self._precomp_feats[idxs]
                        frame_ids = frame_ids_batch
                        if getattr(self, '_precomp_weathers', None) is not None:
                            weather_list = [self._precomp_weathers[i] for i in idxs]
                        else:
                            if 'meta_list' in batch_dict:
                                weather_list = [m.get('weather', '') if isinstance(m, dict) else '' for m in batch_dict['meta_list']]
                            elif 'meta' in batch_dict and isinstance(batch_dict['meta'], (list, tuple)):
                                weather_list = [m.get('weather', '') if isinstance(m, dict) else '' for m in batch_dict['meta']]
                            else:
                                weather_list = [''] * len(idxs)

                if feats_np is None:
                    for k in ['frame_feats', 'frame_features', 'feats', 'features']:
                        if k in batch_dict:
                            val = batch_dict[k]
                            try:
                                feats_np = val.detach().cpu().numpy() if hasattr(val, 'detach') else np.array(val)
                            except Exception:
                                feats_np = np.array(val)
                            if 'meta_list' in batch_dict and isinstance(batch_dict['meta_list'], (list, tuple)):
                                frame_ids = [(m.get('seq', '') + ',' + (m.get('label_v2_0', '') or m.get('label', ''))) for m in batch_dict['meta_list']]
                                weather_list = [m.get('weather', '') if isinstance(m, dict) else '' for m in batch_dict['meta_list']]
                            elif 'meta' in batch_dict and isinstance(batch_dict['meta'], (list, tuple)):
                                frame_ids = [(m.get('seq', '') + ',' + (m.get('label_v2_0', '') or m.get('label', ''))) for m in batch_dict['meta']]
                                weather_list = [m.get('weather', '') if isinstance(m, dict) else '' for m in batch_dict['meta']]
                            else:
                                B = feats_np.shape[0]
                                frame_ids = [f"pos_{i}" for i in range(B)]
                                weather_list = [''] * B
                            break

                if feats_np is None:
                    raise ValueError("Could not construct feats_np for dynamic predictor (no precomputed features and no per-batch frame features).")

                thr_preds = self._dynamic_denoise_predictor.predict_frame_thresholds(feats_np, weather_list=weather_list, frame_ids=frame_ids, device=str(self._dynamic_denoise_predictor.device))
                pre_mask = self._dynamic_denoise_predictor.apply_thresholds_to_scores(s_final, thr_preds, batch_dict['batch_indices_rdr_sparse'])

            except Exception as e:
                print(f"* Warning: dynamic denoise predictor failed: {e}. Falling back to static DENOISE_T")
                static_t = getattr(self.cfg.MODEL.PRE_PROCESSING, 'DENOISE_T', 0.3) if hasattr(self, 'cfg') else 0.3
                pre_mask = s_final > static_t
        else:
            static_t = getattr(self.cfg.MODEL.PRE_PROCESSING, 'DENOISE_T', 0.3) if hasattr(self, 'cfg') else 0.3
            pre_mask = s_final > static_t

        # ensure some points kept
        if pre_mask.sum() < 10:
            pre_mask[:10] = 1
        extra_choice = torch.ones(batch_dict['point_cls_scores'][pre_mask].shape, dtype=bool)
        batch_dict['raw_rdr_sparse'] = batch_dict['rdr_sparse'][pre_mask]
        batch_dict['batch_indices_rdr_sparse'] = batch_dict['batch_indices_rdr_sparse'][pre_mask][extra_choice]
        try:
            batch_dict['rdr_sparse'] = torch.cat([batch_dict['rdr_sparse'][pre_mask][extra_choice], batch_dict['point_cls_scores'][pre_mask][extra_choice].reshape(-1,1)], dim=1)
        except Exception:
            batch_dict['rdr_sparse'] = torch.cat([batch_dict['rdr_sparse'][pre_mask][extra_choice], batch_dict['point_cls_scores'][pre_mask][extra_choice].reshape(-1,1).detach().cpu()], dim=1)

        # continue pipeline
        batch_dict = self.pre_processor(batch_dict)
        batch_dict = self.vfe(batch_dict)
        batch_dict = self.map_to_bev_module(batch_dict)
        batch_dict = self.backbone_2d(batch_dict)
        batch_dict = self.dense_head(batch_dict)

        if self.training:
            return batch_dict
        else:
            batch_dict = self.post_processing(batch_dict)
            return batch_dict

    def loss(self, dict_item):
        loss_rpn, tb_dict = self.dense_head.get_loss()
        if self.point_head is not None:
            loss_point, tb_dict = self.point_head.get_loss(tb_dict)
        else:
            loss_point = 0
        loss = loss_rpn + loss_point

        if self.is_logging:
            dict_item['logging'] = dict()
            dict_item['logging'].update(tb_dict)

        return loss

    def post_processing(self, batch_dict):
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        recall_dict = {}
        pred_dicts = []
        for index in range(batch_size):
            if batch_dict.get('batch_index', None) is not None:
                assert batch_dict['batch_box_preds'].shape.__len__() == 2
                batch_mask = (batch_dict['batch_index'] == index)
            else:
                assert batch_dict['batch_box_preds'].shape.__len__() == 3
                batch_mask = index

            box_preds = batch_dict['batch_box_preds'][batch_mask]
            src_box_preds = box_preds

            if not isinstance(batch_dict['batch_cls_preds'], list):
                cls_preds = batch_dict['batch_cls_preds'][batch_mask]

                src_cls_preds = cls_preds
                assert cls_preds.shape[1] in [1, self.num_class]

                if not batch_dict['cls_preds_normalized']:
                    cls_preds = torch.sigmoid(cls_preds)
            else:
                cls_preds = [x[batch_mask] for x in batch_dict['batch_cls_preds']]
                src_cls_preds = cls_preds
                if not batch_dict['cls_preds_normalized']:
                    cls_preds = [torch.sigmoid(x) for x in cls_preds]

            if post_process_cfg.NMS_CONFIG.MULTI_CLASSES_NMS:
                if not isinstance(cls_preds, list):
                    cls_preds = [cls_preds]
                    multihead_label_mapping = [torch.arange(1, self.num_class, device=cls_preds[0].device)]
                else:
                    multihead_label_mapping = batch_dict['multihead_label_mapping']

                cur_start_idx = 0
                pred_scores, pred_labels, pred_boxes = [], [], []
                for cur_cls_preds, cur_label_mapping in zip(cls_preds, multihead_label_mapping):
                    assert cur_cls_preds.shape[1] == len(cur_label_mapping)
                    cur_box_preds = box_preds[cur_start_idx: cur_start_idx + cur_cls_preds.shape[0]]
                    cur_pred_scores, cur_pred_labels, cur_pred_boxes = model_nms_utils.multi_classes_nms(
                        cls_scores=cur_cls_preds, box_preds=cur_box_preds,
                        nms_config=post_process_cfg.NMS_CONFIG,
                        score_thresh=post_process_cfg.SCORE_THRESH
                    )
                    cur_pred_labels = cur_label_mapping[cur_pred_labels]
                    pred_scores.append(cur_pred_scores)
                    pred_labels.append(cur_pred_labels)
                    pred_boxes.append(cur_pred_boxes)
                    cur_start_idx += cur_cls_preds.shape[0]

                final_scores = torch.cat(pred_scores, dim=0)
                final_labels = torch.cat(pred_labels, dim=0)
                final_boxes = torch.cat(pred_boxes, dim=0)
            else:
                cls_preds, label_preds = torch.max(cls_preds, dim=-1)
                if batch_dict.get('has_class_labels', False):
                    label_key = 'roi_labels' if 'roi_labels' in batch_dict else 'batch_pred_labels'
                    label_preds = batch_dict[label_key][index]
                else:
                    label_preds = label_preds + 1
                selected, selected_scores = model_nms_utils.class_agnostic_nms(
                    box_scores=cls_preds, box_preds=box_preds,
                    nms_config=post_process_cfg.NMS_CONFIG,
                    score_thresh=post_process_cfg.SCORE_THRESH
                )

                if post_process_cfg.OUTPUT_RAW_SCORE:
                    max_cls_preds, _ = torch.max(src_cls_preds, dim=-1)
                    selected_scores = max_cls_preds[selected]

                final_scores = selected_scores
                final_labels = label_preds[selected]
                final_boxes = box_preds[selected]

            recall_dict = self.generate_recall_record(
                box_preds=final_boxes if 'rois' not in batch_dict else src_box_preds,
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST
            )

            record_dict = {
                'pred_boxes': final_boxes,
                'pred_scores': final_scores,
                'pred_labels': final_labels
            }
            pred_dicts.append(record_dict)

            batch_dict['pred_dicts'] = pred_dicts
            batch_dict['recall_dict'] = recall_dict

        return batch_dict
        
    def generate_recall_record(self, box_preds, recall_dict, batch_index, data_dict=None, thresh_list=None):
        if 'gt_boxes' not in data_dict:
            return recall_dict
    
        rois = data_dict['rois'][batch_index] if 'rois' in data_dict else None
        gt_boxes = data_dict['gt_boxes'][batch_index]
    
        if recall_dict.__len__() == 0:
            recall_dict = {'gt': 0}
            for cur_thresh in thresh_list:
                recall_dict['roi_%s' % (str(cur_thresh))] = 0
                recall_dict['rcnn_%s' % (str(cur_thresh))] = 0
    
        cur_gt = gt_boxes
        k = cur_gt.__len__() - 1
        while k >= 0 and cur_gt[k].sum() == 0:
            k -= 1
        cur_gt = cur_gt[:k + 1]
    
        if cur_gt.shape[0] > 0:
            if box_preds.shape[0] > 0:
                iou3d_rcnn = iou3d_nms_utils.boxes_iou3d_gpu(box_preds[:, 0:7], cur_gt[:, 0:7])
            else:
                iou3d_rcnn = torch.zeros((0, cur_gt.shape[0]))
    
            if rois is not None:
                iou3d_roi = iou3d_nms_utils.boxes_iou3d_gpu(rois[:, 0:7], cur_gt[:, 0:7])
    
            for cur_thresh in thresh_list:
                if iou3d_rcnn.shape[0] == 0:
                    recall_dict['rcnn_%s' % str(cur_thresh)] += 0
                else:
                    rcnn_recalled = (iou3d_rcnn.max(dim=0)[0] > cur_thresh).sum().item()
                    recall_dict['rcnn_%s' % str(cur_thresh)] += rcnn_recalled
                if rois is not None:
                    roi_recalled = (iou3d_roi.max(dim=0)[0] > cur_thresh).sum().item()
                    recall_dict['roi_%s' % str(cur_thresh)] += roi_recalled
    
            recall_dict['gt'] += cur_gt.shape[0]
        else:
            gt_iou = box_preds.new_zeros(box_preds.shape[0])
        return recall_dict
