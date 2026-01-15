from src.utils.map import load_obstacles_config
from src.classes.model import SingleIntegratorModel, LIDAR
from src.classes.mapping import LidarGridMapVec
from src.classes.belief_mdp_n_M import BeliefMDP_n_M
from src.algorithms.value_iteration import ValueIteration


def run_learning(env='toy1'):
    all_obstacles, area = load_obstacles_config(environment=env)

    motion_model = SingleIntegratorModel(
        i_x=0,
        i_y=0,
        dt=0.1,
        max_v=3
    )

    sensor = LIDAR(fov=360, r_max=12, B=360)

    M = 3
    β = 0.95
    n = 3
    sigma_w = 0.01
    sigma_v = 0.01

    map = LidarGridMapVec(
        x_min=area[0],
        x_max=area[1],
        y_min=area[2],
        y_max=area[3],
        # since n = (x_max-x_min+1)/resolution
        resolution=(area[1] - area[0] + 1) / (n)
    )

    model = BeliefMDP_n_M(
        M=M,
        β=β,
        n=n,
        motion_model=motion_model,
        measurement_model=sensor,
        obstacles=all_obstacles,
        _map=map,
        sigma_w=sigma_w,
        sigma_v=sigma_v,
    )
    value_iteration = ValueIteration(model)
    value_iteration.run()
    print(value_iteration.V)


if __name__ == "__main__":
    run_learning()
