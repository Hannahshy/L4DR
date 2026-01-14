# New: doppler_features.py
# Utility functions to derive per-point Doppler features (numpy) and
# small torch helpers to compute P_dop and to fuse scores (torch).
#
# Expected input point format (per-point): [x, y, z, power, doppler]
# Derived features (order in output):
#   [abs_dop_n, power_norm_n, neigh_mean_n, neigh_std_n, neighbor_support, dop_diff_n]
# All returned features are float32 and normalized roughly to 0..1 (robust percentiles).
#
# NOTE: For neighborhood stats we use sklearn.NearestNeighbors (CPU). If you need GPU KNN,
#       replace with faiss/torch-kNN.

import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch
from scipy import stats

EPS = 1e-9

def _robust_normalize(x, lop=5, hip=95):
    # Percentile-based robust 0-1 normalization
    lo = np.percentile(x, lop)
    hi = np.percentile(x, hip)
    if hi - lo < 1e-9:
        return np.zeros_like(x, dtype=np.float32)
    y = (x - lo) / (hi - lo + EPS)
    return np.clip(y, 0.0, 1.0).astype(np.float32)

def derive_doppler_features(points_np, k=8, radius=None, power_scale=None, cfg=None):
    """
    points_np: ndarray (N, >=5) columns: x,y,z,power,doppler (others ignored)
               or a torch.Tensor (will be converted to numpy on CPU)
    cfg: optional dict/EasyDict of parameters (keys: k, radius, power_scale, lop, hip)
    returns: feat_np (N,6) in order:
        [abs_dop_n, power_norm_n, neigh_mean_n, neigh_std_n, neighbor_support, dop_diff_n]
    Notes:
      - This function is backward compatible: callers that pass either (points_np)
        or (points_np, cfg=...) will work.
      - Always returns a numpy ndarray of dtype float32.
    """
    # Allow cfg to override parameters
    if cfg is not None:
        try:
            # cfg might be an EasyDict / dict / object
            if isinstance(cfg, dict) or hasattr(cfg, 'get'):
                k = int(cfg.get('k', k)) if cfg.get('k', None) is not None else k
                radius = cfg.get('radius', radius)
                power_scale = cfg.get('power_scale', power_scale)
                lop = cfg.get('lop', 5)
                hip = cfg.get('hip', 95)
            else:
                # attempt attribute access
                k = int(getattr(cfg, 'k', k))
                radius = getattr(cfg, 'radius', radius)
                power_scale = getattr(cfg, 'power_scale', power_scale)
                lop = getattr(cfg, 'lop', 5)
                hip = getattr(cfg, 'hip', 95)
        except Exception:
            lop, hip = 5, 95
    else:
        lop, hip = 5, 95

    # If input is torch tensor, convert to numpy on CPU
    was_torch = False
    device = None
    if isinstance(points_np, torch.Tensor):
        was_torch = True
        device = points_np.device
        points_np = points_np.detach().cpu().numpy()

    assert isinstance(points_np, np.ndarray), "points_np must be numpy.ndarray or torch.Tensor converted to numpy"
    assert points_np.shape[1] >= 5, "points_np must have at least 5 columns: x,y,z,power,doppler"
    coords = points_np[:, :3].astype(np.float32)
    power = points_np[:, 3].astype(np.float32)
    dop = points_np[:, 4].astype(np.float32)
    N = points_np.shape[0]

    # power normalization (robust)
    if power_scale is None:
        med = np.median(power) + EPS
        power_norm = power / med
    else:
        power_norm = power / (power_scale + EPS)

    abs_dop = np.abs(dop)

    # neighbors
    if radius is None:
        k_use = min(max(1, int(k)), N)
        nbrs = NearestNeighbors(n_neighbors=k_use, algorithm='auto').fit(coords)
        dists, inds = nbrs.kneighbors(coords, return_distance=True)
    else:
        nbrs = NearestNeighbors(radius=radius, algorithm='auto').fit(coords)
        inds = nbrs.radius_neighbors(coords, radius=radius, return_distance=False)

    neigh_mean = np.zeros(N, dtype=np.float32)
    neigh_std = np.zeros(N, dtype=np.float32)
    neigh_count = np.zeros(N, dtype=np.int32)
    neighbor_support = np.zeros(N, dtype=np.float32)

    med_power = np.median(power) + EPS
    pthr = 0.5 * med_power

    if radius is None:
        for i in range(N):
            idx = inds[i]
            vals_dop = dop[idx]
            vals_power = power[idx]
            neigh_mean[i] = float(vals_dop.mean())
            neigh_std[i]  = float(vals_dop.std() if vals_dop.size > 0 else 0.0)
            neigh_count[i] = len(idx)
            neighbor_support[i] = float((vals_power > pthr).sum()) / float(len(idx)) if len(idx) > 0 else 0.0
    else:
        for i in range(N):
            idx = inds[i]
            if len(idx) == 0:
                neigh_mean[i] = 0.0
                neigh_std[i]  = 0.0
                neigh_count[i] = 0
                neighbor_support[i] = 0.0
            else:
                vals_dop = dop[idx]
                vals_power = power[idx]
                neigh_mean[i] = float(vals_dop.mean())
                neigh_std[i]  = float(vals_dop.std() if vals_dop.size > 0 else 0.0)
                neigh_count[i] = len(idx)
                neighbor_support[i] = float((vals_power > pthr).sum()) / float(len(idx))

    dop_diff = np.abs(dop - neigh_mean)

    # Normalize features to 0..1 robustly using lop/hip if provided via cfg
    abs_dop_n = _robust_normalize(abs_dop, lop=lop, hip=hip)
    power_norm_n = _robust_normalize(power_norm, lop=lop, hip=hip)
    neigh_mean_n = _robust_normalize(np.abs(neigh_mean), lop=lop, hip=hip)
    neigh_std_n = _robust_normalize(neigh_std, lop=lop, hip=hip)
    neighbor_support_n = np.clip(neighbor_support, 0.0, 1.0)
    dop_diff_n = _robust_normalize(dop_diff, lop=lop, hip=hip)

    feat = np.stack([
        abs_dop_n,
        power_norm_n,
        neigh_mean_n,
        neigh_std_n,
        neighbor_support_n,
        dop_diff_n
    ], axis=1).astype(np.float32)

    # If the caller passed a torch tensor originally, convert back to torch on same device if requested.
    # But to keep backward compatibility with existing callers (which expected numpy), we return numpy.
    return feat


# ---------------- Torch helpers -----------------
def compute_pdop_from_feat_torch(dop_feats, params=None, device=None):
    """
    dop_feats: torch.Tensor (N,6) order same as derive_doppler_features output
    params: dict options (weights & thresholds)
    returns: P_dop torch.Tensor (N,) in (0,1)
    Default formula (interpretable): combine motion evidence + static-power support + neighbor consistency.
    """
    if params is None:
        params = {
            'w_power': 0.8,
            'w_motion': 1.0,
            'w_consistency': 0.6,
            'power_thresh_keep_static': 0.7,  # (since power_norm_n in 0..1)
            'beta': 6.0,
            'bias': -0.2
        }
    if device is None:
        device = dop_feats.device

    # dop_feats columns: [abs_dop_n, power_norm_n, neigh_mean_n, neigh_std_n, neighbor_support, dop_diff_n]
    abs_dop_n = dop_feats[:, 0]
    power_norm_n = dop_feats[:, 1]
    neigh_mean_n = dop_feats[:, 2]
    neigh_std_n = dop_feats[:, 3]
    neighbor_support = dop_feats[:, 4]
    dop_diff_n = dop_feats[:, 5]

    # motion evidence: abs_dop and dop_diff (both normalized)
    motion_raw = 0.6 * abs_dop_n + 0.4 * dop_diff_n
    # consistency reduces motion confidence if neighbor std is high
    motion_score = torch.sigmoid(params['beta'] * (params['w_motion'] * motion_raw - 0.6 * neigh_std_n))

    # static support: if power_norm_n is high -> static candidate
    static_support = torch.sigmoid(params['beta'] * (power_norm_n - params['power_thresh_keep_static']))

    # combination
    raw = params['w_motion'] * motion_score + params['w_power'] * static_support + params['w_consistency'] * neighbor_support + params.get('bias', 0.0)
    P_dop = torch.sigmoid(raw)
    return P_dop


def fuse_scores_torch(p_net, p_dop, method='logit_sum', alpha=0.6, beta=1.0):
    """
    p_net: torch.Tensor (N,) in (0,1)
    p_dop: torch.Tensor (N,) in (0,1)
    returns fused p_final (N,)
    methods: 'weighted', 'mul', 'logit_sum'
    """
    eps = 1e-6
    p = torch.clamp(p_net, eps, 1.0 - eps)
    d = torch.clamp(p_dop, eps, 1.0 - eps)
    if method == 'weighted':
        return torch.clamp(alpha * p + (1.0 - alpha) * d, 0.0, 1.0)
    elif method == 'mul':
        pf = p * d
        # avoid 0 by renormalization to max 1
        maxv = pf.max()
        if maxv > 0:
            pf = pf / (maxv + EPS)
        return pf
    elif method == 'logit_sum':
        logit = lambda x: torch.log(x / (1.0 - x))
        invlogit = lambda x: torch.sigmoid(x)
        lf = logit(p) + beta * logit(d)
        return invlogit(lf)
    else:
        raise ValueError("Unknown fusion method: {}".format(method))