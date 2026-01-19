# models/pre_processor/motion_compensator.py
# Simple temporal motion compensation + persistence filter for radar sparse points
# See assistant notes for usage and expected batch_dict keys.
import torch
import numpy as np

class TemporalMotionCompensator:
    def __init__(self, cfg=None):
        # cfg may be EasyDict or dict or None
        mcfg = {}
        if cfg is not None:
            try:
                mcfg = cfg.MOTION_COMPENSATION if hasattr(cfg, 'MOTION_COMPENSATION') else cfg.get('MOTION_COMPENSATION', {})
            except Exception:
                try:
                    mcfg = cfg.get('MOTION_COMPENSATION', {})
                except Exception:
                    mcfg = {}
        self.enabled = bool(mcfg.get('ENABLED', True))
        self.num_history = int(mcfg.get('NUM_HISTORY', 3))
        self.dist_tol = float(mcfg.get('DIST_TOL', 0.5))         # meters for NN match
        self.min_persist = int(mcfg.get('MIN_PERSIST', 1))       # require matches in >= this many histories
        self.use_ego = bool(mcfg.get('USE_EGO_POSE', True))
        self.use_doppler = bool(mcfg.get('USE_DOPPLER', True))
        self.doppler_scale = float(mcfg.get('DOPPLER_SCALE', 1.0))
        # default slices (assume first col possibly batch idx): [start, end) python indexing
        self.xyz_slice = tuple(mcfg.get('XYZ_SLICE', (1,4)))
        self.dop_idx = int(mcfg.get('DOPPLER_IDX', 4))
        self.power_idx = int(mcfg.get('POWER_IDX', 3))
        # inside __init__ (after existing self.* assignments)
        # optional maximum prev points to use per history (to limit KDTree work)
        self.max_prev_points = int(mcfg.get('MAX_PREV_POINTS', 5000))
        # prefer scipy cKDTree if available (fallback to torch.cdist if not)
        self._use_cKDTree_prefer = bool(mcfg.get('USE_CKDTREE', True))
        # debug flag
        self.debug = bool(mcfg.get('DEBUG', False))
        # ---- dataset / io related (NEW) ----
        self.rdr_sparse_dir = mcfg.get('RDR_SPARSE_DIR', None)   # e.g. cfg.rdr_sparse.dir
        self.rdr_file_prefix = mcfg.get('RDR_FILE_PREFIX', 'rdr_sparse_doppler_')
        self.rdr_file_ext = mcfg.get('RDR_FILE_EXT', '.npy')

        # feature dimension (fallback only)
        self.n_used = int(mcfg.get('N_USED', 5))
        # history indexing
        self.require_meta_idx = True   # assert meta contains rdr_idx_int



    @staticmethod
    def _ensure_tensor(x, device=None, dtype=torch.float32):
        if isinstance(x, torch.Tensor):
            return x.to(device=device, dtype=dtype)
        else:
            return torch.tensor(x, device=device, dtype=dtype)

    @staticmethod
    def _transform_points(points, src_pose, dst_pose):
        # points: (N,3), src_pose/dst_pose: (4,4) torch (sensor->world poses)
        if src_pose is None or dst_pose is None:
            return points
        N = points.shape[0]
        homo = torch.cat([points, points.new_ones((N,1))], dim=1)  # (N,4)
        try:
            rel = torch.matmul(torch.inverse(dst_pose), src_pose)   # (4,4)
            p_trans = torch.matmul(homo, rel.t())[:, :3]
            return p_trans
        except Exception:
            return points

    def _apply_doppler_shift(self, pts, dop, dt):
        if pts.shape[0] == 0:
            return pts
        norms = torch.norm(pts, dim=1, keepdim=True)
        norms = torch.where(norms < 1e-6, torch.ones_like(norms), norms)
        los = pts / norms
        displ = (dop.view(-1,1) * dt * float(self.doppler_scale)) * los
        return pts + displ

    def _safe_get_ts(self, ts, b):
        """Helper: parse timestamp which may be None, scalar, tensor, list-of-lists, etc.
           Returns float timestamp or None.
        """
        if ts is None:
            return None
        try:
            # ts could be list-of-lists (per-history k: list of length B)
            if isinstance(ts, (list, tuple)):
                # if it's a list-of-lists: ts[b] might be the item
                val = ts[b] if b < len(ts) else None
                if val is None:
                    return None
                # val may itself be list-like (we expect a scalar), try to reduce
                if isinstance(val, (list, tuple, np.ndarray)):
                    try:
                        return float(val[0])
                    except Exception:
                        return None
                try:
                    return float(val)
                except Exception:
                    return None
            elif torch.is_tensor(ts):
                try:
                    # if it's a tensor of shape (B,) or scalar
                    if ts.dim() == 0:
                        return float(ts.item())
                    else:
                        return float(ts[b].item()) if b < ts.shape[0] else None
                except Exception:
                    return None
            elif isinstance(ts, np.ndarray):
                try:
                    return float(ts[b]) if ts.size > b else None
                except Exception:
                    return None
            else:
                # scalar
                return float(ts)
        except Exception:
            return None

    def _load_prev_radar(self, seq, curr_idx_int, k_hist):
        """
        Load k-th previous radar frame for a single sample.
        Returns np.ndarray (N, C) or None.
        """
        if self.rdr_sparse_dir is None:
            return None
        if curr_idx_int is None:
            return None
    
        prev_idx = curr_idx_int - k_hist
        if prev_idx < 0:
            return None
    
        # zero-pad width: assume same width as current idx
        curr_str = str(curr_idx_int)
        width = len(curr_str)
        prev_str = str(prev_idx).zfill(width)
    
        path = os.path.join(
            self.rdr_sparse_dir,
            str(seq),
            f'{self.rdr_file_prefix}{prev_str}{self.rdr_file_ext}'
        )
    
        if not os.path.exists(path):
            return None
    
        try:
            arr = np.load(path, mmap_mode='r')  # ★关键：mmap 防 RAM 爆炸
            if arr.ndim != 2 or arr.shape[0] == 0:
                return None
            return arr
        except Exception:
            return None


    def _compute_match_counts(self, batch_dict, candidate_mask=None):
        """
        RAM-safe temporal match counting.
        - Only converts prev radar to numpy ONCE per history
        - No global cache of large tensors
        - Candidate-aware
        """
        if (not self.enabled) or ('rdr_sparse' not in batch_dict):
            N = batch_dict['rdr_sparse'].shape[0]
            return torch.zeros((N,), dtype=torch.int32, device=batch_dict['rdr_sparse'].device), 1.0
    
        device = batch_dict['rdr_sparse'].device
        curr_pts = batch_dict['rdr_sparse']
        batch_idxs = batch_dict['batch_indices_rdr_sparse'].long()
    
        metas = batch_dict['meta']
    
        curr_timestamps = batch_dict.get('timestamp', None) or batch_dict.get('timestamps', None)
        curr_poses = batch_dict.get('poses', None)
    
        N = curr_pts.shape[0]
        match_counts_all = torch.zeros((N,), dtype=torch.int32, device=device)
    
    
        start, end = self.xyz_slice
        B = int(batch_dict.get('batch_size', 1))
    
        # ---- 1. candidate mask（CPU bool，一次性）
        if candidate_mask is not None:
            cand_np = candidate_mask.detach().cpu().numpy().astype(np.bool_)
        else:
            cand_np = None
    
    
        # ---- 3. KDTree 可用性
        try:
            from scipy.spatial import cKDTree
            use_kdtree = True
        except Exception:
            use_kdtree = False
            
        if not use_kdtree:
            return match_counts_all, 1.0
    
        # ---- 4. 主循环（batch）
        batch_idxs_np = batch_idxs.detach().cpu().numpy()
        for b in range(B):
            mask_b = (batch_idxs_np == b)
            if cand_np is not None:
                mask_b &= cand_np
    
            idxs = np.nonzero(mask_b)[0]
            if idxs.size == 0:
                continue
    
            curr_xyz = curr_pts[idxs, start:end].to(device)
            local_counts = torch.zeros((curr_xyz.shape[0],), dtype=torch.int32, device=device)
    
            meta_b = metas[b]
            seq = meta_b.get('seq', None)
            curr_idx_int = meta_b.get('rdr_idx_int', None)
    
            # ---- timestamps (optional)
            try:
                cur_ts = float(meta_b.get('timestamp', None))
            except Exception:
                cur_ts = None
    
            # ---- poses (optional)
            curr_pose = None
            if self.use_ego:
                curr_pose = meta_b.get('pose', None)
    
        
            # ================= history loop =================
            for k in range(1, self.num_history + 1):
                # ---- load prev radar (mmap)
                prev_np = self._load_prev_radar(seq, curr_idx_int, k)
                if prev_np is None or prev_np.size == 0:
                    continue
    
                # ---- xyz
                if prev_np.shape[1] < end:
                    continue
                prev_xyz = prev_np[:, start:end].astype(np.float32)
    
                # ---- ego-motion compensation
                if self.use_ego and curr_pose is not None:
                    try:
                        prev_pose = meta_b.get('prev_pose', None)
                        if prev_pose is not None:
                            homo = np.concatenate(
                                [prev_xyz, np.ones((prev_xyz.shape[0], 1))], axis=1
                            )
                            rel = np.linalg.inv(curr_pose).dot(prev_pose)
                            prev_xyz = homo.dot(rel.T)[:, :3]
                    except Exception:
                        pass
    
                # ---- doppler compensation
                if self.use_doppler and prev_np.shape[1] > self.dop_idx:
                    try:
                        prev_ts = None
                        if cur_ts is not None:
                            # simple constant-dt fallback (safe & stable)
                            prev_ts = cur_ts - 0.1 * k
                        if prev_ts is not None:
                            dt = cur_ts - prev_ts
                            if abs(dt) > 1e-6:
                                dop = prev_np[:, self.dop_idx].astype(np.float32)
                                norms = np.linalg.norm(prev_xyz, axis=1, keepdims=True)
                                norms = np.clip(norms, 1e-6, None)
                                prev_xyz += (
                                    dop[:, None] * dt * self.doppler_scale
                                ) * (prev_xyz / norms)
                    except Exception:
                        pass
    
                # ---- downsample (critical for RAM / speed)
                if self.max_prev_points > 0 and prev_xyz.shape[0] > self.max_prev_points:
                    sel = np.random.choice(prev_xyz.shape[0],
                                           self.max_prev_points,
                                           replace=False)
                    prev_xyz = prev_xyz[sel]
    
                # ---- KDTree match
                try:
                    tree = cKDTree(prev_xyz)
                    curr_xyz_np = curr_xyz.detach().cpu().numpy()
                    hits = tree.query_ball_point(curr_xyz_np, r=self.dist_tol)
    
                    hit_mask = np.fromiter(
                        (1 if len(h) > 0 else 0 for h in hits),
                        count=len(hits),
                        dtype=np.int32
                    )
                    local_counts += torch.from_numpy(hit_mask).to(device)
                except Exception:
                    pass
                finally:
                    # explicit cleanup
                    del prev_xyz
                    if 'tree' in locals():
                        del tree
    
            match_counts_all[idxs] = local_counts

    
        # ---- denom
        denom = float(max(1, self.num_history))
    
        return match_counts_all, denom


    def get_temporal_scores(self, batch_dict, candidate_mask=None):
        """
        Returns length-N float tensor of temporal scores in [0,1].
        If candidate_mask provided (torch.BoolTensor length N), only compute for those entries.
        """
        match_counts, denom = self._compute_match_counts(batch_dict, candidate_mask=candidate_mask)
        if denom <= 0:
            denom = 1.0
        score = match_counts.float() / float(denom)
        return score.clamp(0.0, 1.0)
        
    def get_temporal_mask(self, batch_dict):
        """
        Returns boolean mask with length N (for batch_dict['rdr_sparse']) showing temporal persistence.
        """
        match_counts, _ = self._compute_match_counts(batch_dict)
        pers = (match_counts >= self.min_persist)
        return pers
