# models/pre_processor/motion_compensator.py
# Simple temporal motion compensation + persistence filter for radar sparse points
# RAM-safe: uses np.load(..., mmap_mode='r') and scipy.spatial.cKDTree for matching.
import os
import torch
import numpy as np

class TemporalMotionCompensator:
    def __init__(self, cfg=None):
        # cfg may be EasyDict or dict or None
        mcfg = {}
        full_cfg = None
        if cfg is not None:
            # Accept either full cfg (with MODEL.PRE_PROCESSING.MOTION_COMPENSATION)
            # or the motion sub-dict itself.
            try:
                if hasattr(cfg, 'MODEL') and hasattr(cfg.MODEL, 'PRE_PROCESSING'):
                    full_cfg = cfg
                    try:
                        mcfg = cfg.MODEL.PRE_PROCESSING.MOTION_COMPENSATION
                    except Exception:
                        mcfg = cfg.MODEL.PRE_PROCESSING.get('MOTION_COMPENSATION', {})
                elif isinstance(cfg, dict) and 'MODEL' in cfg and 'PRE_PROCESSING' in cfg['MODEL']:
                    full_cfg = cfg
                    mcfg = cfg['MODEL']['PRE_PROCESSING'].get('MOTION_COMPENSATION', {})
                else:
                    if isinstance(cfg, dict) or hasattr(cfg, 'get'):
                        mcfg = cfg
                    else:
                        mcfg = {}
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
        # optional maximum prev points to use per history (to limit KDTree work)
        self.max_prev_points = int(mcfg.get('MAX_PREV_POINTS', 5000))
        # prefer scipy cKDTree if available (fallback to torch.cdist if not)
        self._use_cKDTree_prefer = bool(mcfg.get('USE_CKDTREE', True))
        # debug flag
        self.debug = bool(mcfg.get('DEBUG', False))

        # ---- dataset / io related (align with cfg.rdr_sparse) ----
        self.rdr_sparse_dir = None
        self.rdr_file_prefix = mcfg.get('RDR_FILE_PREFIX', 'rdr_sparse_doppler_')
        self.rdr_file_ext = mcfg.get('RDR_FILE_EXT', '.npy')
        self.n_used = int(mcfg.get('N_USED', 5))
        # if full cfg was passed, try to read DATASET.rdr_sparse
        if full_cfg is not None:
            try:
                try:
                    rdr_cfg = full_cfg.DATASET.rdr_sparse
                except Exception:
                    rdr_cfg = full_cfg.get('DATASET', {}).get('rdr_sparse', {})
                self.rdr_sparse_dir = rdr_cfg.get('dir', self.rdr_sparse_dir)
                self.n_used = int(rdr_cfg.get('n_used', self.n_used))
                self.dop_idx = int(rdr_cfg.get('doppler_idx', self.dop_idx))
                self.power_idx = int(rdr_cfg.get('power_idx', self.power_idx))
            except Exception:
                pass

        # history indexing
        self.require_meta_idx = True   # assert meta contains rdr_idx_int
        # temporal dt handling
        self.default_dt = float(mcfg.get('DEFAULT_DT', 0.1))   # fallback per-frame dt (seconds)
        self.seq_median_dt = {}   # {seq_id: median_dt}  # dataset or trainer should populate this mapping
        self.use_median_dt = bool(mcfg.get('USE_MEDIAN_DT', True))
        # soft matching (doppler-aware)
        self.use_soft = bool(mcfg.get('USE_SOFT_MATCH', True))
        self.soft_doppler_sigma = float(mcfg.get('SOFT_DOPPLER_SIGMA', 1.0))  # m/s
        self.soft_weight = float(mcfg.get('SOFT_WEIGHT', 0.5))               # [0,1]


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
                val = ts[b] if b < len(ts) else None
                if val is None:
                    return None
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
                return float(ts)
        except Exception:
            return None

    def _load_prev_radar(self, seq, curr_idx_int, k_hist):
        """
        Load k-th previous radar frame using mmap (RAM-safe).
        Returns np.ndarray view or None.
        """
        if self.rdr_sparse_dir is None:
            return None
        if curr_idx_int is None:
            return None

        prev_idx = curr_idx_int - k_hist
        if prev_idx < 0:
            return None

        # keep same zero-padding width as current index
        curr_str = str(curr_idx_int)
        width = len(curr_str)
        prev_str = str(prev_idx).zfill(width)

        path = os.path.join(
            self.rdr_sparse_dir,
            str(seq),
            f'{self.rdr_file_prefix}{prev_str}{self.rdr_file_ext}'
        )

        if not os.path.isfile(path):
            return None

        try:
            arr = np.load(path, mmap_mode='r')   # ★ mmap not loading whole file into RAM
            if arr.ndim != 2 or arr.shape[1] < self.n_used:
                return None
            return arr
        except Exception:
            return None


    def _compute_match_counts(self, batch_dict, candidate_mask=None):
        """
        Hard + Soft (Doppler-aware) temporal match counting.
        RAM-safe, mmap-based, KDTree local search.
        Uses timestamps if available in batch_dict['meta'] per-sample entries:
          - meta_b.get('timestamp') for current frame (float)
          - meta_b.get('prev_timestamps') for list of previous timestamps (list-like)
        Fallback order for dt (for k-th previous frame):
          1) if prev_timestamps[k-1] exists -> dt = cur_ts - prev_ts_k
          2) elif seq_median_dt[seq] present and use_median_dt=True -> dt = seq_median_dt[seq] * k
          3) else -> dt = default_dt * k
        """
        if (not self.enabled) or ('rdr_sparse' not in batch_dict):
            N = batch_dict['rdr_sparse'].shape[0]
            return torch.zeros((N,), dtype=torch.int32,
                               device=batch_dict['rdr_sparse'].device), 1.0

        device = batch_dict['rdr_sparse'].device
        curr_pts = batch_dict['rdr_sparse']
        batch_idxs = batch_dict['batch_indices_rdr_sparse'].long()

        metas = batch_dict['meta']
        N = curr_pts.shape[0]
        match_counts_all = torch.zeros((N,), dtype=torch.float32, device=device)

        start, end = self.xyz_slice
        B = int(batch_dict.get('batch_size', 1))

        # ---- candidate mask (CPU once)
        if candidate_mask is not None:
            cand_np = candidate_mask.detach().cpu().numpy().astype(np.bool_)
        else:
            cand_np = None

        # ---- KDTree availability
        try:
            from scipy.spatial import cKDTree
            use_kdtree = True
        except Exception:
            use_kdtree = False

        if not use_kdtree:
            return match_counts_all, 1.0

        batch_idxs_np = batch_idxs.detach().cpu().numpy()

        # ================= batch loop =================
        for b in range(B):
            mask_b = (batch_idxs_np == b)
            if cand_np is not None:
                mask_b &= cand_np

            idxs = np.nonzero(mask_b)[0]
            if idxs.size == 0:
                continue

            curr_xyz = curr_pts[idxs, start:end].to(device)
            curr_xyz_np = curr_xyz.detach().cpu().numpy()

            local_score = torch.zeros((curr_xyz.shape[0],),
                                      dtype=torch.float32, device=device)

            meta_b = metas[b]
            seq = meta_b.get('seq', None)
            curr_idx_int = meta_b.get('rdr_idx_int', None)

            # ---- timestamps
            try:
                cur_ts = float(meta_b.get('timestamp', None))
            except Exception:
                cur_ts = None
            prev_timestamps_list = meta_b.get('prev_timestamps', None)

            # ---- ego pose
            curr_pose = meta_b.get('pose', None) if self.use_ego else None

            # ================= history loop =================
            for k in range(1, self.num_history + 1):
                prev_np = self._load_prev_radar(seq, curr_idx_int, k)
                if prev_np is None or prev_np.size == 0:
                    continue
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

                # ---- doppler compensation (robust fallback using timestamps or seq_median_dt)
                if self.use_doppler and prev_np.shape[1] > self.dop_idx:
                    try:
                        # Try per-prev timestamps if available
                        dt = None
                        if (cur_ts is not None) and (prev_timestamps_list is not None):
                            try:
                                # prev_timestamps_list is expected to be list-like with order [t_prev1, t_prev2, ...]
                                if isinstance(prev_timestamps_list, (list, tuple, np.ndarray)) and len(prev_timestamps_list) >= k:
                                    prev_ts_k = float(prev_timestamps_list[k-1])
                                    dt = float(cur_ts) - float(prev_ts_k)
                            except Exception:
                                dt = None

                        # If no per-prev timestamps, try seq median dt
                        if dt is None:
                            if self.use_median_dt and (seq in self.seq_median_dt) and (self.seq_median_dt[seq] is not None):
                                dt = float(self.seq_median_dt[seq]) * float(k)
                            else:
                                dt = float(self.default_dt) * float(k)

                        if abs(dt) > 1e-9:
                            dop = prev_np[:, self.dop_idx].astype(np.float32)
                            norms = np.linalg.norm(prev_xyz, axis=1, keepdims=True)
                            norms = np.clip(norms, 1e-6, None)
                            prev_xyz += (dop[:, None] * dt * self.doppler_scale) * (prev_xyz / norms)
                    except Exception:
                        pass

                # ---- downsample
                if self.max_prev_points > 0 and prev_xyz.shape[0] > self.max_prev_points:
                    sel = np.random.choice(prev_xyz.shape[0],
                                           self.max_prev_points,
                                           replace=False)
                    prev_xyz = prev_xyz[sel]
                    prev_np = prev_np[sel]

                # ---- KDTree local search
                try:
                    tree = cKDTree(prev_xyz)
                    hits = tree.query_ball_point(curr_xyz_np, r=self.dist_tol)

                    # -------- HARD + SOFT MATCHING --------
                    hard_hit = np.fromiter(
                        (1 if len(h) > 0 else 0 for h in hits),
                        count=len(hits),
                        dtype=np.float32
                    )

                    if self.use_soft and self.use_doppler and prev_np.shape[1] > self.dop_idx:
                        curr_dop = curr_pts[idxs, self.dop_idx].detach().cpu().numpy()
                        prev_dop = prev_np[:, self.dop_idx]

                        soft_scores = np.zeros_like(hard_hit, dtype=np.float32)

                        for i, h in enumerate(hits):
                            if len(h) == 0:
                                continue
                            dv = prev_dop[h] - curr_dop[i]
                            w = np.exp(-(dv * dv) /
                                       (self.soft_doppler_sigma ** 2))
                            soft_scores[i] = float(w.max())

                        combined = hard_hit + self.soft_weight * soft_scores
                    else:
                        combined = hard_hit

                    local_score += torch.from_numpy(combined).to(device)

                except Exception:
                    pass
                finally:
                    if 'tree' in locals():
                        del tree
                    del prev_xyz, prev_np

            match_counts_all[idxs] = local_score

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