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

# (替换或插入到 TemporalMotionCompensator 类中，替换原有 _compute_match_counts 和 get_temporal_scores)
    def _compute_match_counts(self, batch_dict, candidate_mask=None):
        """
        Compute match_counts for current points. If candidate_mask is provided (boolean Tensor
        of length N on same device as batch_dict['rdr_sparse']), only compute matches for
        the candidate subset to save memory/time. Returns (match_counts_all, denom).
        """
        # If not enabled or missing data, return zeros as before
        cache = batch_dict.setdefault('_tmc_cache', {})
        key = id(self)

        # If no rdr data, quick exit
        if (not self.enabled) or ('rdr_sparse' not in batch_dict) or ('batch_indices_rdr_sparse' not in batch_dict):
            N = batch_dict['rdr_sparse'].shape[0] if 'rdr_sparse' in batch_dict else 0
            match_counts_all = torch.zeros((N,), dtype=torch.int32, device=batch_dict['rdr_sparse'].device if 'rdr_sparse' in batch_dict else 'cpu')
            denom = 1.0
            # When candidates were provided, avoid caching to keep semantics simple
            if candidate_mask is None:
                cache[key] = {'match_counts': match_counts_all, 'denom': denom}
            return match_counts_all, denom

        # If candidate_mask not provided and cache exists, return cache
        if candidate_mask is None and key in cache:
            return cache[key]['match_counts'], cache[key]['denom']

        # Otherwise, compute (candidate-aware)
        device = batch_dict['rdr_sparse'].device
        curr_pts_all = batch_dict['rdr_sparse']
        batch_idxs = batch_dict['batch_indices_rdr_sparse'].long().to(device)

        prev_rdrs = batch_dict.get('prev_rdrs', None)
        prev_batch_inds = batch_dict.get('prev_batch_indices', None)
        prev_timestamps = batch_dict.get('prev_timestamps', None)
        prev_poses = batch_dict.get('prev_poses', None)
        curr_timestamps = batch_dict.get('timestamp', None) or batch_dict.get('timestamps', None)
        curr_poses = batch_dict.get('poses', None)

        N = curr_pts_all.shape[0]
        match_counts_all = torch.zeros((N,), dtype=torch.int32, device=device)

        if (not prev_rdrs) or len(prev_rdrs) == 0:
            denom = 1.0
            # don't cache when candidate_mask provided
            if candidate_mask is None:
                cache[key] = {'match_counts': match_counts_all, 'denom': denom}
            return match_counts_all, denom

        start, end = self.xyz_slice
        B = int(batch_dict.get('batch_size', 1))

        # prefer cKDTree if available
        use_ckd = False
        if self._use_cKDTree_prefer:
            try:
                from scipy.spatial import cKDTree
                use_ckd = True
            except Exception:
                use_ckd = False

        # helper to downsample prev points
        def _maybe_downsample(prev_arr_np):
            M = prev_arr_np.shape[0]
            if M > self.max_prev_points and self.max_prev_points > 0:
                idx = np.random.choice(M, self.max_prev_points, replace=False)
                return prev_arr_np[idx]
            else:
                return prev_arr_np

        CHUNK = 8192

        # candidate_mask: convert to CPU bool numpy if provided
        if candidate_mask is not None:
            # ensure boolean tensor on same device
            if not torch.is_tensor(candidate_mask):
                raise ValueError("candidate_mask must be a torch.BoolTensor or None")
            # move to CPU numpy bool array for indexing
            candidate_mask_cpu = candidate_mask.detach().cpu().numpy().astype(bool)
        else:
            candidate_mask_cpu = None

        for b in range(B):
            # global indices in curr_pts_all for this batch
            mask_idxs = (batch_idxs == b).cpu().numpy()
            if candidate_mask_cpu is not None:
                # only keep those that are both in this batch and marked candidate
                mask_idxs = mask_idxs & candidate_mask_cpu
            # get positions (global indices) of candidates in this batch
            idx_positions = np.nonzero(mask_idxs)[0]
            if idx_positions.size == 0:
                continue

            # Build curr_xyz for candidates only
            curr_xyz = curr_pts_all[idx_positions, start:end].to(device)
            # match_counts local
            match_counts = torch.zeros((curr_xyz.shape[0],), dtype=torch.int32, device=device)
            cur_ts = self._safe_get_ts(curr_timestamps, b)

            for k, prev in enumerate(prev_rdrs[: self.num_history]):
                if prev is None:
                    continue

                # convert prev to numpy safely
                try:
                    if torch.is_tensor(prev):
                        prev_np_all = prev.cpu().numpy()
                    else:
                        prev_np_all = np.array(prev)
                except Exception:
                    try:
                        prev_np_all = np.asarray(prev)
                    except Exception:
                        continue

                if prev_np_all.size == 0:
                    continue

                # filter prev by batch index if provided
                if prev_batch_inds and k < len(prev_batch_inds) and prev_batch_inds[k] is not None:
                    pbi = prev_batch_inds[k]
                    if torch.is_tensor(pbi):
                        pbi_np = pbi.cpu().numpy()
                    else:
                        pbi_np = np.array(pbi)
                    mask = (pbi_np == b)
                    if mask.sum() == 0:
                        continue
                    prev_pts_np = prev_np_all[mask]
                else:
                    prev_pts_np = prev_np_all
                    try:
                        if prev_pts_np.shape[1] > end:
                            col0 = prev_pts_np[:, 0].astype(np.int64)
                            pm = (col0 == b)
                            if pm.sum() > 0:
                                prev_pts_np = prev_pts_np[pm]
                    except Exception:
                        pass

                if prev_pts_np.shape[0] == 0:
                    continue

                # prev xyz slice
                try:
                    prev_xyz_np = prev_pts_np[:, start:end].astype(np.float32)
                except Exception:
                    prev_xyz_np = prev_pts_np[:, :3].astype(np.float32)

                # ego pose transform if available
                if self.use_ego and (prev_poses is not None) and (curr_poses is not None) and (k < len(prev_poses)):
                    try:
                        ppose = prev_poses[k]
                        if torch.is_tensor(ppose):
                            ppose_np = ppose.cpu().numpy()
                        else:
                            ppose_np = np.array(ppose)
                        if ppose_np.ndim == 3:
                            ppose_b = ppose_np[b]
                        else:
                            ppose_b = ppose_np
                        cpose = curr_poses
                        if torch.is_tensor(cpose):
                            cpose_np = cpose.cpu().numpy()
                        else:
                            cpose_np = np.array(cpose)
                        if cpose_np.ndim == 3:
                            cpose_b = cpose_np[b]
                        else:
                            cpose_b = cpose_np
                        Np = prev_xyz_np.shape[0]
                        homo = np.concatenate([prev_xyz_np, np.ones((Np, 1), dtype=np.float32)], axis=1)
                        rel = np.linalg.inv(cpose_b).dot(ppose_b)
                        transformed = homo.dot(rel.T)[:, :3]
                        prev_xyz_np = transformed
                    except Exception:
                        pass

                # optional downsample prev
                prev_xyz_np = _maybe_downsample(prev_xyz_np)

                # doppler correction if enabled
                if self.use_doppler:
                    try:
                        dop_vals = prev_pts_np[:, self.dop_idx].astype(np.float32)
                        prev_ts = self._safe_get_ts(prev_timestamps[k] if prev_timestamps is not None and k < len(prev_timestamps) else None, b)
                        if (cur_ts is not None) and (prev_ts is not None):
                            dt = float(cur_ts) - float(prev_ts)
                        else:
                            dt = 0.0
                        if abs(dt) > 1e-9:
                            norms = np.linalg.norm(prev_xyz_np, axis=1, keepdims=True)
                            norms = np.where(norms < 1e-6, 1.0, norms)
                            los = prev_xyz_np / norms
                            displ = (dop_vals.reshape(-1, 1) * dt * float(self.doppler_scale)) * los
                            prev_xyz_np = prev_xyz_np + displ
                    except Exception:
                        pass

                # Now do radius search ONLY for curr_xyz candidates
                any_close_np = None
                if use_ckd:
                    try:
                        from scipy.spatial import cKDTree
                        tree = cKDTree(prev_xyz_np)
                        curr_xyz_np = curr_xyz.cpu().numpy()
                        L = curr_xyz_np.shape[0]
                        any_close_list = []
                        for s in range(0, L, CHUNK):
                            e = min(L, s + CHUNK)
                            chunk = curr_xyz_np[s:e]
                            res = tree.query_ball_point(chunk, r=self.dist_tol)
                            any_close_chunk = np.array([1 if len(rr) > 0 else 0 for rr in res], dtype=np.int32)
                            any_close_list.append(any_close_chunk)
                        any_close_np = np.concatenate(any_close_list, axis=0)
                    except Exception:
                        any_close_np = None

                if any_close_np is None:
                    try:
                        prev_t = torch.from_numpy(prev_xyz_np).to(curr_xyz.device)
                        L = curr_xyz.shape[0]
                        any_close_list = []
                        for s in range(0, L, CHUNK):
                            e = min(L, s + CHUNK)
                            chunk = curr_xyz[s:e]
                            dists = torch.cdist(prev_t, chunk)  # (M_prev, chunk_len)
                            any_close_chunk = (dists <= self.dist_tol).any(dim=0).to(torch.int32).cpu().numpy()
                            any_close_list.append(any_close_chunk)
                        any_close_np = np.concatenate(any_close_list, axis=0)
                    except Exception:
                        any_close = []
                        prev_t_cpu = torch.from_numpy(prev_xyz_np)
                        for i in range(curr_xyz.shape[0]):
                            pi = curr_xyz[i:i + 1].cpu()
                            dd = torch.norm(prev_t_cpu - pi, dim=1)
                            any_close.append(1 if (dd <= self.dist_tol).any().item() else 0)
                        any_close_np = np.array(any_close, dtype=np.int32)

                # map back to torch and add
                any_close_t = torch.from_numpy(any_close_np.astype(np.int32)).to(device=device)
                match_counts += any_close_t.int()

            # write local match_counts to global positions (idx_positions)
            idxs = torch.from_numpy(idx_positions).long().to(device=device)
            if idxs.shape[0] != match_counts.shape[0]:
                L = min(idxs.shape[0], match_counts.shape[0])
                match_counts_all[idxs[:L]] = match_counts[:L]
            else:
                match_counts_all[idxs] = match_counts

        # denom
        try:
            valid_hist = sum([1 for p in prev_rdrs[: self.num_history] if (p is not None and getattr(p, 'shape', None) is not None and ((isinstance(p, np.ndarray) and p.shape[0] > 0) or (torch.is_tensor(p) and p.numel() > 0)))])
            denom = float(min(self.num_history, max(1, int(valid_hist))))
        except Exception:
            denom = float(min(self.num_history, max(1, len(prev_rdrs))))

        # cache only when candidate_mask is None
        if candidate_mask is None:
            cache[key] = {'match_counts': match_counts_all, 'denom': denom}
            batch_dict['_tmc_cache'] = cache
        else:
            # do not populate cache to avoid mismatch across different candidate subsets
            batch_dict['_tmc_cache'] = cache

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
