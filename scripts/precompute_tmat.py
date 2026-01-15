from src.classes.belief_mdp_n import BeliefMDP_n
from src.classes.model import DoubleIntegratorModel, LIDAR
from src.classes.mapping import LidarGridMapVec
from src.utils.map import load_obstacles_config
import argparse
import time
from pathlib import Path
import sys
import os
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
os.chdir(str(PROJECT_ROOT))



def main():
    p = argparse.ArgumentParser(description="Precompute and cache T_mat for BeliefMDP_n (DoubleIntegratorModel)")
    p.add_argument("--n", type=int, default=4, help="quantization level for state/action")
    p.add_argument("--dt", type=float, default=1.0, help="timestep for model")
    p.add_argument("--max_a", type=float, default=5.0, help="max acceleration")
    p.add_argument("--env", type=str, default="toy2", help="environment config name")
    p.add_argument("--map_q", type=int, default=None, help="map quantization level (defaults to n)")
    args = p.parse_args()

    all_obstacles, area = load_obstacles_config(environment=args.env)
    model = DoubleIntegratorModel(p_x=5.0, p_y=5.0, v_x=0.0, v_y=0.0, dt=args.dt, max_a=args.max_a)
    sensor = LIDAR(fov=360, r_max=10.0, B=8)
    map_q = args.map_q if args.map_q is not None else args.n
    grid_map = LidarGridMapVec(x_min=area[0], x_max=area[1], y_min=area[2], y_max=area[3], quantization_level=map_q)

    print(f"Starting precompute: n={args.n}, dt={args.dt}, max_a={args.max_a}, env={args.env}, map_q={map_q}")
    print(f"Action space: polar quantization with n={args.n} (radial and angular levels)")
    print(f"State space: hybrid quantization (square lattice for position, polar for velocity)")
    t0 = time.time()
    mdp = BeliefMDP_n(n=args.n, motion_model=model, measurement_model=sensor, obstacles=all_obstacles, _map=grid_map)
    elapsed = time.time() - t0
    cache_path = mdp._get_cache_path()
    print(f"Done. Elapsed: {elapsed:.2f}s")
    print(f"Cache saved to: {cache_path}")
    print(f"T_mat shape: {mdp.T_mat.shape}")
    print(f"State space size: {mdp.SQ.m_n}, Action space size: {mdp.AQ.n_u}")
    print(
        f"Action magnitudes: min={np.min(np.linalg.norm(mdp.AQ.U, axis=1)):.3f}, max={np.max(np.linalg.norm(mdp.AQ.U, axis=1)):.3f}")


if __name__ == "__main__":
    main()
