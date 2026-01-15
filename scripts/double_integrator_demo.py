import numpy as np
import matplotlib.pyplot as plt
import argparse

from src.classes.model import DoubleIntegratorModel


def simulate(model: DoubleIntegratorModel, x0: np.ndarray, u_seq: np.ndarray, noise_std: float = 0.0, seed: int | None = 0):
    """
    Simulate DoubleIntegratorModel for a given control sequence.

    Args:
        model: DoubleIntegratorModel instance (its dt, max_a used; internal x is unused here)
        x0: initial state [px, py, vx, vy]
        u_seq: sequence of accelerations shape (T, 2)
        noise_std: standard deviation of additive Gaussian process noise on state (per-dimension)
        seed: RNG seed

    Returns:
        states: (T+1, 4) array of states
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    T = u_seq.shape[0]
    states = np.zeros((T + 1, 4), dtype=float)
    states[0] = x0

    for t in range(T):
        u = u_seq[t]
        w = rng.normal(0.0, noise_std, size=4) if noise_std > 0 else 0.0
        states[t + 1] = model.f(states[t], u, w)

    return states


def build_controls(T: int, dt: float, max_a: float):
    t = np.arange(T) * dt
    controls = {}
    # Constant acceleration in +x
    controls["const_ax"] = np.tile(np.array([0.5 * max_a, 0.0]), (T, 1))
    # Sinusoidal acceleration in x, small y
    controls["sin_ax"] = np.stack([0.7 * max_a * np.sin(0.5 * t), 0.2 * max_a * np.cos(0.4 * t)], axis=1)
    # Step in y then zero
    u = np.zeros((T, 2))
    u[: T // 4, 1] = 0.8 * max_a
    controls["step_ay"] = u
    # Circular acceleration pattern
    controls["circle"] = 0.5 * max_a * np.stack([np.cos(0.8 * t), np.sin(0.8 * t)], axis=1)
    return controls


def plot_trajectories(results: dict, dt: float, title: str):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # XY plot
    for label, states in results.items():
        axs[0].plot(states[:, 0], states[:, 1], label=label)
    axs[0].set_xlabel("x [m]")
    axs[0].set_ylabel("y [m]")
    axs[0].set_title("Position trajectories")
    axs[0].axis("equal")
    axs[0].grid(True, ls=":", alpha=0.6)
    axs[0].legend()

    # Speed vs time
    for label, states in results.items():
        speed = np.linalg.norm(states[:, 2:4], axis=1)
        axs[1].plot(np.arange(states.shape[0]) * dt, speed, label=label)
    axs[1].set_xlabel("time [s]")
    axs[1].set_ylabel("speed [m/s]")
    axs[1].set_title("Speed over time")
    axs[1].grid(True, ls=":", alpha=0.6)
    axs[1].legend()

    fig.suptitle(title)
    fig.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="DoubleIntegratorModel trajectory demo")
    parser.add_argument("--dt", type=float, default=0.1, help="Timestep [s]")
    parser.add_argument("--T", type=int, default=200, help="Number of steps")
    parser.add_argument("--max_a", type=float, default=2.0, help="Max acceleration magnitude")
    parser.add_argument("--noise_std", type=float, default=0.0, help="Process noise stddev per state component")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    # Set up model and controls
    model = DoubleIntegratorModel(p_x=0.0, p_y=0.0, v_x=0.0, v_y=0.0, dt=args.dt, max_a=args.max_a)
    controls = build_controls(args.T, args.dt, args.max_a)

    x0 = np.array([0.0, 0.0, 0.0, 0.0])

    # Deterministic runs
    det_results = {}
    for name, u_seq in controls.items():
        det_results[f"{name}_det"] = simulate(model, x0, u_seq, noise_std=0.0, seed=args.seed)

    plot_trajectories(det_results, args.dt, title="DoubleIntegratorModel (deterministic)")

    # Noisy runs (if specified)
    if args.noise_std > 0:
        noisy_results = {}
        for name, u_seq in controls.items():
            noisy_results[f"{name}_noisy"] = simulate(model, x0, u_seq, noise_std=args.noise_std, seed=args.seed)
        plot_trajectories(noisy_results, args.dt, title=f"DoubleIntegratorModel (noise_std={args.noise_std})")


if __name__ == "__main__":
    main()


