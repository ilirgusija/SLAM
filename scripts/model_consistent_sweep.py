#!/usr/bin/env python3
# Auto-generated from: notebooks/mapping/cost_comparisons/model_consistent_sweep.ipynb
# Do not edit manually; edit the notebook and re-export.

# ---- code cell 2 ----
import time
from scipy.spatial import KDTree as ScipyKDTree
from tqdm.auto import tqdm
from scipy.stats import spearmanr
import os
import sys
import subprocess
import ctypes
from pathlib import Path
from datetime import datetime
import numpy as onp
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Keep tqdm.auto imports everywhere, but force text backend globally before project imports.

PROJECT_ROOT = Path('/global/home/hpc5656/SLAM')
os.chdir(PROJECT_ROOT)
sys.path.append(str(PROJECT_ROOT))
print('CWD:', Path.cwd())

# CRITICAL: Preload libcuda from system path before importing array backend/CuPy.
try:
    ctypes.CDLL('/usr/lib64/libcuda.so.1', mode=ctypes.RTLD_GLOBAL)
    print('✓ Preloaded libcuda.so.1 from /usr/lib64')
except Exception as e:
    print(f'⚠ Could not preload libcuda.so.1: {e}')

# Ensure CUDA env is populated in notebook shells.
if 'CUDA_PATH' not in os.environ:
    try:
        result = subprocess.run(
            'module load cuda/12.2 && env',
            shell=True,
            executable='/bin/bash',
            capture_output=True,
            text=True,
            timeout=20,
        )
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value
            print(f"✓ Modules loaded. CUDA_PATH: {os.environ.get('CUDA_PATH', 'Not set')}")
        else:
            raise RuntimeError('CUDA_PATH not set. Run: module load cuda/12.2')
    except Exception as e:
        raise RuntimeError(f'Failed to load CUDA: {e}') from e

# Make sure runtime libraries are discoverable by CuPy.
cuda_path = os.environ.get('CUDA_PATH')
if cuda_path:
    cuda_lib_paths = [
        os.path.join(cuda_path, 'lib64'),
        os.path.join(cuda_path, 'targets', 'x86_64-linux', 'lib'),
    ]
    current_ld = os.environ.get('LD_LIBRARY_PATH', '')
    ld_paths = current_ld.split(':') if current_ld else []
    for p in cuda_lib_paths:
        if os.path.exists(p) and p not in ld_paths:
            ld_paths.insert(0, p)
    if ld_paths != (current_ld.split(':') if current_ld else []):
        os.environ['LD_LIBRARY_PATH'] = ':'.join(ld_paths)
        print('✓ Updated LD_LIBRARY_PATH')

    for p in cuda_lib_paths:
        if not os.path.exists(p):
            continue
        for name in ['libcudart.so', 'libcudart.so.12', 'libnvrtc.so.12']:
            path = os.path.join(p, name)
            if os.path.exists(path):
                try:
                    ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                    print(f'✓ Preloaded {name}')
                except Exception as e:
                    print(f'⚠ Could not preload {name}: {e}')
                break

print('\n=== Environment before backend import ===')
print(f"CUDA_PATH: {os.environ.get('CUDA_PATH', 'Not set')}")
print('=========================================\n')

from src.utils.array_backend import np, is_cupy
from src.belief_quantized.belief_mdp_n_M import BeliefMDP_n_M_Mapping
from src.classes.model import SingleIntegratorModel, RangeBearingSensor
from src.classes.mapping import LandmarkMap
from src.belief_quantized.value_iteration import ValueIteration


if is_cupy:
    try:
        import cupy as cp
        print('Using backend: CuPy (GPU)')
        print(f'✓ CuPy {cp.__version__}, CUDA devices: {cp.cuda.runtime.getDeviceCount()}')
    except Exception as e:
        print(f'✗ CuPy check failed: {e}')
else:
    print('Using backend: NumPy (CPU)')


def to_numpy(x):
    return x.get() if hasattr(x, 'get') else onp.asarray(x)

def sample_true_map_continuous(rng, n_landmarks=2):
    xs = rng.uniform(0.0, 10.0, size=n_landmarks)
    ys = rng.uniform(0.0, 10.0, size=n_landmarks)
    return onp.stack([xs, ys], axis=1)

# ---- code cell 3 ----
# Base problem config
pose_n = 4
map_n = 4
obs_n = 5
M = 6
beta = 0.95
action_n = 8
sigma_w = 0.1
horizon = 20
n_trials = 1000
seed = 9090
n_gpus = 8
j_batch_size = None
# i_batch_size = 300
i_batch_size = 256+512
epsilon = 1e-6
SENSOR_EPSILON = 0.1
SENSOR_R_MAX = 4.0
SENSOR_R0 = SENSOR_EPSILON + 0.1 * (SENSOR_R_MAX - SENSOR_EPSILON)
SENSOR_R1 = SENSOR_EPSILON + 0.9 * (SENSOR_R_MAX - SENSOR_EPSILON)

# Shared-setup diagnostics/speed controls
# - default is "none" to keep sweeps fast; diagnostics are opt-in
# - set SWEEP_KERNEL_REPORT_MODE=sampled or full to enable checks
KERNEL_REPORT_MODE = os.environ.get("SWEEP_KERNEL_REPORT_MODE", "none").strip().lower()
if KERNEL_REPORT_MODE not in {"none", "sampled", "full"}:
    print(f"⚠ Invalid SWEEP_KERNEL_REPORT_MODE='{KERNEL_REPORT_MODE}', defaulting to 'none'")
    KERNEL_REPORT_MODE = "none"
KERNEL_REPORT_SAMPLE_ACTIONS = int(os.environ.get("SWEEP_KERNEL_REPORT_SAMPLE_ACTIONS", "2"))
KERNEL_REPORT_SAMPLE_ACTIONS = max(1, KERNEL_REPORT_SAMPLE_ACTIONS)

# Sweep config
SWEEP_SIGMA_R = [0.5, 0.75, 1, 1.25]
SWEEP_SIGMA_PHI = [0.3, 0.5, 0.7]
# SWEEP_SIGMA_R = [0.75]
# SWEEP_SIGMA_PHI = [0.5]
# SWEEP_LAMBDA = [50, 100]
# SWEEP_LAMBDA = [200]
SWEEP_LAMBDA = [1, 2, 10, 20, 50, 100, 200,  500, 1000, 2000]
MAX_SETTINGS = None  # set int for quick tests

# Robustness config (paired, balanced trials across seeds)
MASTER_SEEDS = [9090, 9091, 9092]
BALANCE_TRIALS_OVER_MAPS = True

landmark_map = LandmarkMap(
    x_min=0.0, x_max=10.0,
    y_min=0.0, y_max=10.0,
    n=map_n,
    num_landmarks=1,
)

motion_model = SingleIntegratorModel(i_x=5.0, i_y=5.0, dt=1.0, max_v=4.0)
obstacles = []

def build_mdp(beta, obs_n, action_n, sigma_r, sigma_phi):
    sensor = RangeBearingSensor(
        r_max=SENSOR_R_MAX,
        epsilon=SENSOR_EPSILON,
        sigma_r=sigma_r,
        sigma_phi=sigma_phi,
        r0=SENSOR_R0,
        r1=SENSOR_R1,
    )
    cov_y = onp.diag(onp.tile([sigma_r**2, sigma_phi**2], landmark_map.num_candidate_landmarks))
    mdp = BeliefMDP_n_M_Mapping(
        M=M,
        β=beta,
        n=pose_n,
        motion_model=motion_model,
        measurement_model=sensor,
        obstacles=obstacles,
        _map=landmark_map,
        sigma_w=sigma_w,
        cov_y=cov_y,
        obs_n=obs_n,
        action_n=action_n,
        j_batch_size=j_batch_size,
        i_batch_size=i_batch_size,
        n_gpus=n_gpus,
    )

    # Sentinel must always be present in Y_n.
    Y_n_np = to_numpy(mdp.Y_n)
    sentinel_rows = onp.isnan(Y_n_np).all(axis=1).sum()
    assert sentinel_rows >= 1, "Sentinel observation missing from Y_n"

    return mdp

def build_cost_tensors(mdp, lam=10.0):
    B = np.asarray(mdp.BQ.Π_n_M)
    U = np.asarray(mdp.AQ.U)
    effort = np.asarray(mdp.c_effort(U))            # (n_u,)
    sh = np.asarray(mdp.r_information_gain(B))       # (card,)
    ra = np.asarray(mdp.rao_entropy(B))              # (card,)

    # normalize to [0, 1] so lambda means the same thing for both
    sh_max = float(np.max(sh))
    ra_max = float(np.max(ra))
    sh_n = sh / sh_max if sh_max > 0 else sh
    ra_n = ra / ra_max if ra_max > 0 else ra

    eff_max = float(np.max(effort))
    eff_n = effort / eff_max if eff_max > 0 else effort

    c_sh_2d = lam * sh_n[:, None] + eff_n[None, :]
    c_ra_2d = lam * ra_n[:, None] + eff_n[None, :]

    c_sh = np.broadcast_to(c_sh_2d[:, None, :],
                           (mdp.BQ.cardinality, mdp.SQ.m_n, mdp.AQ.n_u)).copy()
    c_ra = np.broadcast_to(c_ra_2d[:, None, :],
                           (mdp.BQ.cardinality, mdp.SQ.m_n, mdp.AQ.n_u)).copy()
    return c_sh, c_ra, sh, ra

def run_vi(mdp, c_tensor):
    old = mdp.c_n_M
    try:
        mdp.c_n_M = c_tensor
        vi = ValueIteration(mdp, epsilon=epsilon)
        vi.run(verbose=False)
        return to_numpy(vi.policy).astype(onp.int32), vi.iteration_count
    finally:
        mdp.c_n_M = old

# ---- code cell 4 ----

def make_rollout_backend(mdp, sigma_r, sigma_phi):
    all_maps = to_numpy(mdp.all_maps_3d)
    Q_n = to_numpy(mdp.Q_n)
    Y_n = to_numpy(mdp.Y_n)
    X_n = to_numpy(mdp.SQ.X_n)
    U_n = to_numpy(mdp.AQ.U)
    codebook = to_numpy(mdp.BQ.Π_n_M)
    D_map = to_numpy(mdp._get_map_distance_matrix())
    cov_x = to_numpy(mdp.cov_x)

    state_dim = mdp.state_dim
    L = all_maps.shape[1]
    len_M = all_maps.shape[0]
    dt = float(mdp.motion_model.dt)
    lower = to_numpy(mdp.state_bounds[:, 0])
    upper = to_numpy(mdp.state_bounds[:, 1])
    eps = float(mdp.sensor.epsilon)
    r_max = float(mdp.sensor.r_max)
    r0 = float(mdp.sensor.r0)
    r1 = float(mdp.sensor.r1)

    tree = ScipyKDTree(X_n)
    Y_n_nan = onp.isnan(Y_n)
    # Pre-index observation cells by NaN visibility pattern, then query nearest
    # within that subset using KDTree. This avoids O(|Y_n|) scans per step.
    obs_pattern_groups = {}
    for idx in range(Y_n.shape[0]):
        key = Y_n_nan[idx].tobytes()
        if key not in obs_pattern_groups:
            obs_pattern_groups[key] = []
        obs_pattern_groups[key].append(idx)

    obs_index = {}
    for key, idx_list in obs_pattern_groups.items():
        idx_arr = onp.asarray(idx_list, dtype=onp.int64)
        valid_mask = ~Y_n_nan[idx_arr[0]]
        points = Y_n[idx_arr][:, valid_mask]
        tree_local = ScipyKDTree(points) if points.shape[0] > 1 and points.shape[1] > 0 else None
        obs_index[key] = {
            "indices": idx_arr,
            "valid_mask": valid_mask,
            "points": points,
            "tree": tree_local,
        }

    Mq = int(mdp.BQ.M)
    cb_int = onp.rint(codebook * Mq).astype(onp.int64)
    cb_hash = {tuple(r): i for i, r in enumerate(cb_int)}

    def reznik_quantize(b):
        k = onp.floor(Mq * b + 0.5).astype(onp.int64)
        delta = int(k.sum()) - Mq
        if delta != 0:
            d = k - Mq * b
            order = onp.argsort(d)
            if delta > 0:
                k[order[-delta:]] -= 1
            else:
                k[order[:-delta]] += 1
        return k

    def belief_to_idx(b):
        k = reznik_quantize(b)
        idx = cb_hash.get(tuple(k))
        if idx is not None:
            return idx
        q = k.astype(onp.float64) / Mq
        return int(onp.argmin(onp.linalg.norm(codebook - q, axis=1)))

    def state_idx(x):
        return int(tree.query(x)[1])

    def obs_idx(y):
        nan_y = onp.isnan(y)
        key = nan_y.tobytes()
        group = obs_index.get(key)
        if group is not None:
            idx_arr = group["indices"]
            valid_mask = group["valid_mask"]
            points = group["points"]
            tree_local = group["tree"]

            # all-NaN observation pattern (sentinel): pick first matching cell
            if not onp.any(valid_mask):
                return int(idx_arr[0])

            yv = y[valid_mask]
            if tree_local is not None:
                local_idx = int(tree_local.query(yv)[1])
                return int(idx_arr[local_idx])

            # tiny group fallback
            d = onp.linalg.norm(points - yv, axis=1)
            return int(idx_arr[int(onp.argmin(d))])

        # Defensive fallback for unseen patterns
        d = onp.linalg.norm(onp.nan_to_num(Y_n) - onp.nan_to_num(y), axis=1)
        d = onp.where(Y_n_nan.any(axis=1), onp.inf, d)
        return int(onp.argmin(d))

    def fast_filter(b, x_idx, y):
        oi = obs_idx(y)
        log_q = onp.log(onp.maximum(Q_n[oi, x_idx, :], 1e-300))
        log_b = onp.log(onp.maximum(b, 1e-300))
        z = log_q + log_b
        z -= z.max()
        b_new = onp.exp(z)
        s = b_new.sum()
        return b_new / s if s > 0 else onp.full(len_M, 1.0 / len_M)

    def fast_step(x, u, w):
        return onp.clip(x + u * dt + w, lower, upper)

    def smoothstep01(z):
        z = onp.clip(z, 0.0, 1.0)
        return 3.0 * z**2 - 2.0 * z**3

    def detection_probability(r):
        eta_in = onp.where(
            r <= eps,
            0.0,
            onp.where(r >= r0, 1.0, smoothstep01((r - eps) / (r0 - eps))),
        )
        eta_out = onp.where(
            r <= r1,
            0.0,
            onp.where(r >= r_max, 1.0, smoothstep01((r - r1) / (r_max - r1))),
        )
        return onp.clip(eta_in * (1.0 - eta_out), 0.0, 1.0)

    def sample_truncated_range(rng, mu):
        samples = mu + rng.normal(0.0, sigma_r, size=mu.shape)
        invalid = (samples < eps) | (samples > r_max)
        while invalid.any():
            samples[invalid] = mu[invalid] + rng.normal(0.0, sigma_r, size=int(invalid.sum()))
            invalid = (samples < eps) | (samples > r_max)
        return samples

    def fast_observe(x, true_map, rng):
        delta = true_map - x[:2]
        ranges = onp.sqrt((delta**2).sum(axis=1))
        bearings = onp.arctan2(delta[:, 1], delta[:, 0])

        p_det = detection_probability(ranges)
        detected = rng.random(L) < p_det
        z = onp.full((L, 2), onp.nan)
        if detected.any():
            z[detected, 0] = sample_truncated_range(rng, ranges[detected])
            phi = bearings[detected] + rng.normal(0.0, sigma_phi, size=int(detected.sum()))
            z[detected, 1] = onp.arctan2(onp.sin(phi), onp.cos(phi))
        return z.ravel()

    def entropy(b):
        bp = b[b > 0]
        return float(-onp.dot(bp, onp.log(bp)))

    def rao(b):
        return float(b @ D_map @ b)

    def map_mean(b):
        return onp.einsum('k,klj->lj', b, all_maps)

    def rollout(policy, trial_seed, true_map, horizon):
        rng = onp.random.default_rng(trial_seed)
        x = onp.array([5.0, 5.0], dtype=float)
        b = onp.full(len_M, 1.0 / len_M)

        sh, ra, mse = onp.empty(horizon), onp.empty(horizon), onp.empty(horizon)
        effort = onp.empty(horizon)         
        for t in range(horizon):
            sh[t] = entropy(b)
            ra[t] = rao(b)
            mse[t] = float(onp.mean((map_mean(b) - true_map) ** 2))

            ai = int(policy[belief_to_idx(b), state_idx(x)])
            u = U_n[ai]
            effort[t] = float(onp.sum(u**2)) 
            w = rng.multivariate_normal(onp.zeros(state_dim), cov_x)
            x_next = fast_step(x, u, w)

            y = fast_observe(x_next, true_map, rng)

            b = fast_filter(b, state_idx(x_next), y)
            x = x_next

        return {'shannon': sh, 'rao': ra, 'mse': mse, 'effort': effort}

    def rollout_random(trial_seed, true_map, horizon):
        rng = onp.random.default_rng(trial_seed)
        x = onp.array([5.0, 5.0], dtype=float)
        b = onp.full(len_M, 1.0 / len_M)

        sh, ra, mse = onp.empty(horizon), onp.empty(horizon), onp.empty(horizon)
        effort = onp.empty(horizon)
        for t in range(horizon):
            sh[t] = entropy(b)
            ra[t] = rao(b)
            mse[t] = float(onp.mean((map_mean(b) - true_map) ** 2))

            ai = int(rng.integers(0, U_n.shape[0]))
            u = U_n[ai]
            effort[t] = float(onp.sum(u**2))
            w = rng.multivariate_normal(onp.zeros(state_dim), cov_x)
            x_next = fast_step(x, u, w)

            y = fast_observe(x_next, true_map, rng)

            b = fast_filter(b, state_idx(x_next), y)
            x = x_next

        return {'shannon': sh, 'rao': ra, 'mse': mse, 'effort': effort}

    return rollout, rollout_random

def make_trials(n_trials, seed, n_landmarks=2):
    rng = onp.random.default_rng(seed)
    out = []
    for _ in range(n_trials):
        out.append((int(rng.integers(0, 2**31 - 1)), sample_true_map_continuous(rng, n_landmarks=n_landmarks)))
    return out

def make_trials_model_consistent(n_trials, seed, all_maps, balanced=True):
    rng = onp.random.default_rng(seed)
    len_M = len(all_maps)

    if not balanced:
        out = []
        for _ in range(n_trials):
            tseed = int(rng.integers(0, 2**31 - 1))
            tidx = int(rng.integers(0, len_M))
            out.append((tseed, all_maps[tidx]))
        return out

    # Balanced trial list: each map appears nearly equally often.
    base = n_trials // len_M
    rem = n_trials % len_M
    map_ids = onp.repeat(onp.arange(len_M, dtype=onp.int64), base)
    if rem > 0:
        map_ids = onp.concatenate([map_ids, rng.permutation(len_M)[:rem]])
    rng.shuffle(map_ids)

    trial_seeds = rng.integers(0, 2**31 - 1, size=n_trials, dtype=onp.int64)
    return [(int(ts), all_maps[int(mi)]) for ts, mi in zip(trial_seeds, map_ids)]



# ---- code cell 5 ----

def stack_field(traces, key):
    return onp.stack([t[key] for t in traces], axis=0)

def shaded_mean(ax, t, data, color, label, alpha_fill=0.15, band='ci95', linestyle='-', zorder=None):
    mu = data.mean(axis=0)
    sigma = data.std(axis=0)
    n = max(1, data.shape[0])

    if band == 'std':
        lo = mu - sigma
        hi = mu + sigma
    else:
        half = 1.96 * sigma / onp.sqrt(n)
        lo = mu - half
        hi = mu + half

    kwargs = {'color': color, 'linewidth': 1.8, 'label': label, 'linestyle': linestyle}
    if zorder is not None:
        kwargs['zorder'] = zorder
    ax.plot(t, mu, **kwargs)
    ax.fill_between(t, lo, hi, color=color, alpha=alpha_fill, zorder=(zorder - 0.1) if zorder is not None else None)

def fmt_s(seconds):
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    if m > 0:
        return f"{m}m {s:02d}s"
    return f"{s}s"


POLICY_STYLE = {
    "random": {"label": "Random", "color": "#95a5a6", "linestyle": "-"},
    "shannon": {"label": r"$\gamma_{\mathbb{H}}$ (Shannon)", "color": "#2980b9", "linestyle": "--"},
    "rao": {"label": r"$\gamma_{\tilde{W}}$ (Rao)", "color": "#e67e22", "linestyle": "-"},
}

VIOLIN_STAT_STYLE = {
    "q95": {"label": r"$q_{95}$ (solid)", "linestyle": "-", "color": "#1f1f1f"},
    "cvar90": {"label": r"CVaR$_{90}$ (dotted)", "linestyle": ":", "color": "#1f1f1f"},
    "median": {"label": "Median (black marker)", "marker": "o", "color": "#000000"},
}

noise_settings = [(sr, sp) for sr in SWEEP_SIGMA_R for sp in SWEEP_SIGMA_PHI]
max_runs = int(MAX_SETTINGS) if MAX_SETTINGS is not None else None
planned_total = len(noise_settings) * len(SWEEP_LAMBDA)
if max_runs is not None:
    planned_total = min(planned_total, max_runs)

stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
out_dir = PROJECT_ROOT / 'output' / 'experiment2' / f'sweep_model_consistent_streamlined_{stamp}'
out_dir.mkdir(parents=True, exist_ok=True)
print('Output dir:', out_dir)
print(f'Planned settings: {planned_total} ({len(noise_settings)} noise groups x {len(SWEEP_LAMBDA)} lambdas)')
print(f'Master seeds ({len(MASTER_SEEDS)}): {MASTER_SEEDS}')
print(f'Balanced map trials: {BALANCE_TRIALS_OVER_MAPS}')
print(
    "Sensor detection taper: "
    f"epsilon={SENSOR_EPSILON}, r0={SENSOR_R0:.3f}, r1={SENSOR_R1:.3f}, r_max={SENSOR_R_MAX}"
)

setting_rows = []
seed_rows = []
all_start = time.time()
setting_durations = []
run_idx = 0
for g_idx, (sr, sp) in enumerate(noise_settings, start=1):
    if run_idx >= planned_total:
        break

    print(f"\n=== Noise group {g_idx}/{len(noise_settings)}: sigma_r={sr}, sigma_phi={sp} ===")
    t0_group = time.time()
    t0 = time.time()
    mdp_shared = build_mdp(beta, obs_n, action_n, sr, sp)
    t_build_mdp = time.time() - t0
    print(f"  Shared setup stage build_mdp (includes cache I/O): {fmt_s(t_build_mdp)}")

    t0 = time.time()
    if KERNEL_REPORT_MODE == "full":
        report_shared = mdp_shared.transition_kernel_report(verbose=False)
        report_mode_used = "full"
        report_actions_used = int(mdp_shared.AQ.n_u)
    elif KERNEL_REPORT_MODE == "none":
        report_shared = {
            "aggregate": {
                "row_coverage_mean": float("nan"),
                "bad_rows_total": -1,
                "n_actions_analyzed": 0,
            }
        }
        report_mode_used = "none"
        report_actions_used = 0
    else:
        report_actions_used = min(int(mdp_shared.AQ.n_u), KERNEL_REPORT_SAMPLE_ACTIONS)
        sampled_actions = list(range(report_actions_used))
        report_shared = mdp_shared.transition_kernel_report(
            action_indices=sampled_actions,
            verbose=False,
        )
        report_mode_used = "sampled"
    t_kernel_report = time.time() - t0
    print(
        f"  Shared setup stage transition_kernel_report [{report_mode_used}, "
        f"actions={report_actions_used}]: {fmt_s(t_kernel_report)}"
    )

    t0 = time.time()
    rollout_shared, rollout_rand_shared = make_rollout_backend(mdp_shared, sr, sp)
    all_maps_shared = to_numpy(mdp_shared.all_maps_3d)
    t_rollout_setup = time.time() - t0
    print(f"  Shared setup stage rollout backend: {fmt_s(t_rollout_setup)}")

    t_group = time.time() - t0_group
    print(
        "Shared setup complete: "
        f"{fmt_s(t_group)} | "
        f"kernel_report_mode={report_mode_used}, "
        f"row_coverage_mean={report_shared['aggregate']['row_coverage_mean']:.4f}, "
        f"bad_rows={report_shared['aggregate']['bad_rows_total']}"
    )

    remaining_budget = planned_total - run_idx
    group_lambdas = SWEEP_LAMBDA[:remaining_budget]
    for lambd in group_lambdas:
        setting_start = time.time()
        run_idx += 1

        print(f"\n[{run_idx}/{planned_total}] beta={beta}, action_n={action_n}, obs_n={obs_n}, sigma_r={sr}, sigma_phi={sp}, lambda={lambd:g}")

        if len(setting_durations) > 0:
            avg_prev = onp.mean(setting_durations)
            remaining = planned_total - run_idx + 1
            print(f"  ETA (based on avg of previous {len(setting_durations)} settings): ~{fmt_s(avg_prev * remaining)}")

        # Stage 1: Reuse shared model and kernel caches for this noise group
        mdp = mdp_shared
        report = report_shared
        print(
            "  Stage build+kernel: 0s | reused shared MDP/kernel "
            f"(row_coverage_mean={report['aggregate']['row_coverage_mean']:.4f}, bad_rows={report['aggregate']['bad_rows_total']})"
        )

        # Stage 2: Build cost tensors
        t0 = time.time()
        c_sh, c_ra, sh_cost, ra_cost = build_cost_tensors(mdp, lambd)
        t_costs = time.time() - t0
        print(f"  Stage costs: {fmt_s(t_costs)}")

        # Stage 3: Value iteration (Shannon)
        t0 = time.time()
        pol_sh, it_sh = run_vi(mdp, c_sh)
        t_vi_sh = time.time() - t0
        print(f"  Stage VI Shannon: {fmt_s(t_vi_sh)} | iterations={it_sh}")

        # Stage 4: Value iteration (Rao)
        t0 = time.time()
        pol_ra, it_ra = run_vi(mdp, c_ra)
        t_vi_ra = time.time() - t0
        print(f"  Stage VI Rao: {fmt_s(t_vi_ra)} | iterations={it_ra}")

        policy_diff = float(onp.mean(pol_sh != pol_ra))
        rank_corr = float(spearmanr(to_numpy(sh_cost), to_numpy(ra_cost)).statistic)
        print(f"  Separation metrics: policy_diff={100*policy_diff:.2f}%, rank_corr={rank_corr:.4f}")

        # Stage 5: Reuse shared rollout backend
        rollout, rollout_rand = rollout_shared, rollout_rand_shared
        all_maps = all_maps_shared
        print(f"  Stage rollout setup: 0s | reused shared backend | trials/seed={n_trials}, horizon={horizon}")

        # Stage 6: Monte Carlo over master seeds (paired common-random-number trials)
        t0 = time.time()
        tr_sh_all, tr_ra_all, tr_rand_all = [], [], []
        mse_sh_all, mse_ra_all = [], []
        seed_gap_means = []
        eff_sh_all, eff_ra_all, eff_rand_all = [], [], []

        for midx, master_seed in enumerate(MASTER_SEEDS, start=1):
            trials = make_trials_model_consistent(
                n_trials,
                master_seed,
                all_maps,
                balanced=BALANCE_TRIALS_OVER_MAPS,
            )

            tr_sh, tr_ra, tr_rand = [], [], []
            for tseed, tmap in tqdm(trials, desc=f'MC {run_idx}/{planned_total} seed {midx}/{len(MASTER_SEEDS)}', leave=False):
                tr_sh.append(rollout(pol_sh, tseed, tmap, horizon))
                tr_ra.append(rollout(pol_ra, tseed, tmap, horizon))
                tr_rand.append(rollout_rand(tseed, tmap, horizon))

            sh_self_seed = stack_field(tr_sh, 'shannon')
            ra_self_seed = stack_field(tr_ra, 'rao')
            mse_sh_seed = stack_field(tr_sh, 'mse')
            mse_ra_seed = stack_field(tr_ra, 'mse')
            mse_rand_seed = stack_field(tr_rand, 'mse')
            eff_sh_seed = stack_field(tr_sh, 'effort')
            eff_ra_seed = stack_field(tr_ra, 'effort')
            eff_rand_seed = stack_field(tr_rand, 'effort')

            delta_seed = mse_sh_seed[:, -1] - mse_ra_seed[:, -1]
            gap_mean_seed = float(delta_seed.mean())
            gap_std_seed = float(delta_seed.std(ddof=1)) if delta_seed.size > 1 else 0.0
            gap_ci95_seed = float(1.96 * gap_std_seed / max(1, onp.sqrt(delta_seed.size)))
            rao_win_seed = float((delta_seed > 0).mean())

            q95_sh_seed = float(onp.quantile(mse_sh_seed[:, -1], 0.95))
            q95_ra_seed = float(onp.quantile(mse_ra_seed[:, -1], 0.95))
            cvar90_sh_seed = float(mse_sh_seed[:, -1][mse_sh_seed[:, -1] >= onp.quantile(mse_sh_seed[:, -1], 0.9)].mean())
            cvar90_ra_seed = float(mse_ra_seed[:, -1][mse_ra_seed[:, -1] >= onp.quantile(mse_ra_seed[:, -1], 0.9)].mean())

            seed_rows.append({
                'master_seed': int(master_seed),
                'lambda': float(lambd),
                'beta': beta,
                'action_n': action_n,
                'obs_n': obs_n,
                'sigma_r': sr,
                'sigma_phi': sp,
                'kernel_report_mode': report_mode_used,
                'kernel_actions_analyzed': int(report_shared['aggregate'].get('n_actions_analyzed', report_actions_used)),
                'iter_shannon': int(it_sh),
                'iter_rao': int(it_ra),
                'policy_diff_rate': policy_diff,
                'cost_rank_spearman': rank_corr,
                'kernel_row_coverage_mean': float(report['aggregate']['row_coverage_mean']),
                'kernel_bad_rows_total': int(report['aggregate']['bad_rows_total']),
                'final_gap_mean_sh_minus_ra': gap_mean_seed,
                'final_gap_ci95_trials': gap_ci95_seed,
                'rao_win_rate': rao_win_seed,
                'q95_terminal_mse_shannon': q95_sh_seed,
                'q95_terminal_mse_rao': q95_ra_seed,
                'cvar90_terminal_mse_shannon': cvar90_sh_seed,
                'cvar90_terminal_mse_rao': cvar90_ra_seed,
                'mean_cumul_effort_shannon': float(eff_sh_seed.sum(axis=1).mean()),
                'mean_cumul_effort_rao': float(eff_ra_seed.sum(axis=1).mean()),
                'mean_cumul_effort_random': float(eff_rand_seed.sum(axis=1).mean()),
            })

            seed_gap_means.append(gap_mean_seed)
            tr_sh_all.append(sh_self_seed)
            tr_ra_all.append(ra_self_seed)
            tr_rand_all.append(mse_rand_seed)
            mse_sh_all.append(mse_sh_seed)
            mse_ra_all.append(mse_ra_seed)
            eff_sh_all.append(eff_sh_seed)
            eff_ra_all.append(eff_ra_seed)
            eff_rand_all.append(eff_rand_seed)

            print(
                f"    seed {master_seed}: gap={gap_mean_seed:.4f} ± {gap_ci95_seed:.4f}, "
                f"rao_win={100*rao_win_seed:.1f}%"
            )

        t_mc = time.time() - t0
        print(f"  Stage Monte Carlo: {fmt_s(t_mc)}")

        # Stage 7: Pooled postprocess and plot
        t0 = time.time()
        sh_self = onp.concatenate(tr_sh_all, axis=0)
        ra_self = onp.concatenate(tr_ra_all, axis=0)
        mse_rand = onp.concatenate(tr_rand_all, axis=0)

        mse_sh = onp.concatenate(mse_sh_all, axis=0)
        mse_ra = onp.concatenate(mse_ra_all, axis=0)
        eff_sh = onp.concatenate(eff_sh_all, axis=0)
        eff_ra = onp.concatenate(eff_ra_all, axis=0)
        eff_rand = onp.concatenate(eff_rand_all, axis=0)

        delta = mse_sh[:, -1] - mse_ra[:, -1]
        gap_mean_pooled = float(delta.mean())
        gap_std_pooled = float(delta.std(ddof=1)) if delta.size > 1 else 0.0
        gap_ci95_pooled = float(1.96 * gap_std_pooled / max(1, onp.sqrt(delta.size)))
        rao_win_pooled = float((delta > 0).mean())

        seed_gap_arr = onp.asarray(seed_gap_means, dtype=float)
        gap_mean_across_seeds = float(seed_gap_arr.mean())
        gap_std_across_seeds = float(seed_gap_arr.std(ddof=1)) if seed_gap_arr.size > 1 else 0.0
        gap_ci95_across_seeds = float(1.96 * gap_std_across_seeds / max(1, onp.sqrt(seed_gap_arr.size)))

        q95_sh = float(onp.quantile(mse_sh[:, -1], 0.95))
        q95_ra = float(onp.quantile(mse_ra[:, -1], 0.95))
        cvar90_sh = float(mse_sh[:, -1][mse_sh[:, -1] >= onp.quantile(mse_sh[:, -1], 0.9)].mean())
        cvar90_ra = float(mse_ra[:, -1][mse_ra[:, -1] >= onp.quantile(mse_ra[:, -1], 0.9)].mean())

        # ---- 1x3 plot: MSE trajectory | cumulative effort | violin ----
        t = onp.arange(horizon)
        fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6))

        # (a) MSE trajectory
        shaded_mean(
            axes[0], t, mse_rand,
            POLICY_STYLE["random"]["color"], POLICY_STYLE["random"]["label"],
            linestyle=POLICY_STYLE["random"]["linestyle"]
        )
        shaded_mean(
            axes[0], t, mse_ra,
            POLICY_STYLE["rao"]["color"], POLICY_STYLE["rao"]["label"],
            linestyle=POLICY_STYLE["rao"]["linestyle"]
        )
        shaded_mean(
            axes[0], t, mse_sh,
            POLICY_STYLE["shannon"]["color"], POLICY_STYLE["shannon"]["label"],
            linestyle=POLICY_STYLE["shannon"]["linestyle"], zorder=10
        )
        axes[0].set_title('(a) Map estimation error')
        axes[0].set_xlabel('Time step $t$')
        axes[0].set_ylabel('MSEE')

        # (b) Cumulative effort
        cum_eff_sh = onp.cumsum(eff_sh, axis=1)
        cum_eff_ra = onp.cumsum(eff_ra, axis=1)
        cum_eff_rand = onp.cumsum(onp.concatenate(eff_rand_all, axis=0), axis=1)
        shaded_mean(
            axes[1], t, cum_eff_sh,
            POLICY_STYLE["shannon"]["color"], POLICY_STYLE["shannon"]["label"],
            linestyle=POLICY_STYLE["shannon"]["linestyle"], zorder=10
        )
        shaded_mean(
            axes[1], t, cum_eff_ra,
            POLICY_STYLE["rao"]["color"], POLICY_STYLE["rao"]["label"],
            linestyle=POLICY_STYLE["rao"]["linestyle"]
        )
        shaded_mean(
            axes[1], t, cum_eff_rand,
            POLICY_STYLE["random"]["color"], POLICY_STYLE["random"]["label"],
            linestyle=POLICY_STYLE["random"]["linestyle"]
        )
        axes[1].set_title(r'(b) Cumulative effort $\sum_{k=0}^{t}\|u_k\|^2$')
        axes[1].set_xlabel('Time step $t$')
        axes[1].set_ylabel('Cumulative effort')

        # (c) Violin: terminal MSE distribution
        terminal = [mse_rand[:, -1], mse_sh[:, -1], mse_ra[:, -1]]
        parts = axes[2].violinplot(terminal, positions=[1, 2, 3], showmeans=False, showmedians=False, showextrema=False)
        colors_v = [
            POLICY_STYLE["random"]["color"],
            POLICY_STYLE["shannon"]["color"],
            POLICY_STYLE["rao"]["color"],
        ]
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors_v[i])
            pc.set_edgecolor(colors_v[i])
            pc.set_alpha(0.5)

        # annotate q95 and cvar90 per policy
        labels_v = ['Random', r'$\gamma_{\mathbb{H}}$', r'$\gamma_{\tilde{W}}$']
        for i, dat in enumerate(terminal):
            med = float(onp.median(dat))
            q95_v = float(onp.quantile(dat, 0.95))
            tail_mask = dat >= onp.quantile(dat, 0.90)
            cvar_v = float(dat[tail_mask].mean()) if tail_mask.any() else q95_v
            pos = i + 1
            axes[2].scatter(
                pos, med, color=VIOLIN_STAT_STYLE["median"]["color"], s=28, zorder=8,
                marker=VIOLIN_STAT_STYLE["median"]["marker"], edgecolors='white', linewidths=0.5
            )
            axes[2].hlines(
                q95_v, pos - 0.16, pos + 0.16,
                colors=VIOLIN_STAT_STYLE["q95"]["color"],
                linewidths=1.6,
                linestyles=VIOLIN_STAT_STYLE["q95"]["linestyle"],
                zorder=7,
            )
            axes[2].hlines(
                cvar_v, pos - 0.16, pos + 0.16,
                colors=VIOLIN_STAT_STYLE["cvar90"]["color"],
                linewidths=1.6,
                linestyles=VIOLIN_STAT_STYLE["cvar90"]["linestyle"],
                zorder=7,
            )

        axes[2].set_xticks([1, 2, 3])
        axes[2].set_xticklabels(labels_v)
        axes[2].set_title('(c) Terminal MSEE distribution')
        axes[2].set_ylabel('Terminal MSEE')

        # Figure-level legends to avoid repeated clutter and overlapping text.
        policy_handles = [
            Line2D([0], [0], color=POLICY_STYLE["random"]["color"], lw=2, linestyle=POLICY_STYLE["random"]["linestyle"], label=POLICY_STYLE["random"]["label"]),
            Line2D([0], [0], color=POLICY_STYLE["shannon"]["color"], lw=2, linestyle=POLICY_STYLE["shannon"]["linestyle"], label=POLICY_STYLE["shannon"]["label"]),
            Line2D([0], [0], color=POLICY_STYLE["rao"]["color"], lw=2, linestyle=POLICY_STYLE["rao"]["linestyle"], label=POLICY_STYLE["rao"]["label"]),
        ]
        violin_stat_handles = [
            Line2D([0], [0], color=VIOLIN_STAT_STYLE["q95"]["color"], lw=1.8, linestyle=VIOLIN_STAT_STYLE["q95"]["linestyle"], label=VIOLIN_STAT_STYLE["q95"]["label"]),
            Line2D([0], [0], color=VIOLIN_STAT_STYLE["cvar90"]["color"], lw=1.8, linestyle=VIOLIN_STAT_STYLE["cvar90"]["linestyle"], label=VIOLIN_STAT_STYLE["cvar90"]["label"]),
            Line2D([0], [0], marker=VIOLIN_STAT_STYLE["median"]["marker"], color=VIOLIN_STAT_STYLE["median"]["color"], lw=0, markersize=5, label=VIOLIN_STAT_STYLE["median"]["label"]),
        ]

        # Compact single-row bottom legend to reduce excess whitespace.
        fig.tight_layout(rect=[0, 0.12, 1, 1])
        fig.legend(
            handles=policy_handles + violin_stat_handles,
            loc='lower center',
            ncol=6,
            frameon=False,
            bbox_to_anchor=(0.5, 0.02),
            fontsize=8.2,
            handlelength=2.0,
            columnspacing=1.0,
            handletextpad=0.5,
            borderaxespad=0.3,
        )

        seed_tag = f"s{len(MASTER_SEEDS)}_{min(MASTER_SEEDS)}-{max(MASTER_SEEDS)}"
        img_name = (
            f"sameplot_M{M}_n{pose_n}_nmap{map_n}_obs{obs_n}_act{action_n}_beta{beta}"
            f"_lam{lambd:g}_H{horizon}_sr{sr}_sp{sp}_trials{n_trials}_{seed_tag}.png"
        )
        fig.savefig(out_dir / img_name, bbox_inches='tight')
        plt.close(fig)
        t_plot = time.time() - t0
        print(f"  Stage postprocess+plot: {fmt_s(t_plot)}")

        setting_rows.append({
            'lambda': float(lambd),
            'beta': beta,
            'action_n': action_n,
            'obs_n': obs_n,
            'sigma_r': sr,
            'sigma_phi': sp,
            'kernel_report_mode': report_mode_used,
            'kernel_actions_analyzed': int(report_shared['aggregate'].get('n_actions_analyzed', report_actions_used)),
            'n_master_seeds': int(len(MASTER_SEEDS)),
            'iter_shannon': int(it_sh),
            'iter_rao': int(it_ra),
            'policy_diff_rate': policy_diff,
            'cost_rank_spearman': rank_corr,
            'kernel_row_coverage_mean': float(report['aggregate']['row_coverage_mean']),
            'kernel_bad_rows_total': int(report['aggregate']['bad_rows_total']),
            'final_gap_mean_sh_minus_ra_pooled': gap_mean_pooled,
            'final_gap_ci95_pooled_trials': gap_ci95_pooled,
            'final_gap_mean_sh_minus_ra_across_seeds': gap_mean_across_seeds,
            'final_gap_ci95_across_seeds': gap_ci95_across_seeds,
            'rao_win_rate_pooled': rao_win_pooled,
            'q95_terminal_mse_shannon': q95_sh,
            'q95_terminal_mse_rao': q95_ra,
            'cvar90_terminal_mse_shannon': cvar90_sh,
            'cvar90_terminal_mse_rao': cvar90_ra,
            'mean_cumul_effort_shannon': float(eff_sh.sum(axis=1).mean()),
            'mean_cumul_effort_rao': float(eff_ra.sum(axis=1).mean()),
        })

        setting_elapsed = time.time() - setting_start
        setting_durations.append(setting_elapsed)
        avg_so_far = onp.mean(setting_durations)
        remaining_after = planned_total - run_idx
        eta_after = avg_so_far * remaining_after

        print(
            f"  done {run_idx}/{planned_total} | pooled_gap={gap_mean_pooled:.4f} ± {gap_ci95_pooled:.4f} | "
            f"across_seed_gap={gap_mean_across_seeds:.4f} ± {gap_ci95_across_seeds:.4f} | "
            f"setting_time={fmt_s(setting_elapsed)} | avg={fmt_s(avg_so_far)} | ETA_remaining~{fmt_s(eta_after)}"
        )

summary = pd.DataFrame(setting_rows).sort_values(
    by=['sigma_r', 'sigma_phi', 'lambda', 'policy_diff_rate', 'final_gap_mean_sh_minus_ra_across_seeds'],
    ascending=[True, True, True, False, False],
)
seed_summary = pd.DataFrame(seed_rows).sort_values(
    by=['sigma_r', 'sigma_phi', 'lambda', 'beta', 'action_n', 'obs_n', 'master_seed']
)

seed_tag = f"s{len(MASTER_SEEDS)}_{min(MASTER_SEEDS)}-{max(MASTER_SEEDS)}"
summary_name = (
    f"summary_M{M}_n{pose_n}_nmap{map_n}"
    f"_lambda{min(SWEEP_LAMBDA)}-{max(SWEEP_LAMBDA)}"
    f"_sr{min(SWEEP_SIGMA_R)}-{max(SWEEP_SIGMA_R)}"
    f"_sp{min(SWEEP_SIGMA_PHI)}-{max(SWEEP_SIGMA_PHI)}"
    f"_trials{n_trials}_{seed_tag}.csv"
)
seed_summary_name = (
    f"seedwise_M{M}_n{pose_n}_nmap{map_n}"
    f"_lambda{min(SWEEP_LAMBDA)}-{max(SWEEP_LAMBDA)}"
    f"_sr{min(SWEEP_SIGMA_R)}-{max(SWEEP_SIGMA_R)}"
    f"_sp{min(SWEEP_SIGMA_PHI)}-{max(SWEEP_SIGMA_PHI)}"
    f"_trials{n_trials}_{seed_tag}.csv"
)
summary.to_csv(out_dir / summary_name, index=False)
seed_summary.to_csv(out_dir / seed_summary_name, index=False)

all_elapsed = time.time() - all_start
print('\nFinished all settings.')
print(f'Total elapsed: {fmt_s(all_elapsed)}')
print('Saved:', out_dir)
print(summary.head(20).to_string(index=False))

# ---- code cell 6 ----
pass
