import yaml
import numpy as np
import numpy.ma as ma
from ..classes.obstacle import Obstacle


def load_obstacles_config(environment):
    """
    :param environment: name of the yaml config file
    :return: all obstacles, area of the environment
    """
    with open('src/config/'+environment+'.yaml') as file:
        yaml_data = yaml.load(file, Loader=yaml.FullLoader)

        # load environment area parameters
        area = yaml_data['area']
        area = (area['x_min'], area['x_max'], area['y_min'], area['y_max'])

        # load static and dynamic obstacles
        obs = yaml_data['obstacles']
        all_obstacles = []
        for i in range(len(obs)):
            obs_i = Obstacle(centroid=[obs[i]['centroid_x'], obs[i]['centroid_y']], dx=obs[i]['dx'], dy=obs[i]['dy'],
                             angle=obs[i]['orientation']*np.pi/180, vel=[obs[i]['velocity_x'], obs[i]['velocity_y']],
                             acc=[obs[i]['acc_x'], obs[i]['acc_y']])
            all_obstacles.append(obs_i)
    return all_obstacles, area


def bresenham_vec(start, ends):
    """
    Implementation of Bresenham's line drawing algorithm
    See en.wikipedia.org/wiki/Bresenham's_line_algorithm
    Bresenham's Line Algorithm
    Produces a np.array from start and end (original from roguebasin.com)
    >>> points1 = bresenham((4, 4), (6, 10))
    >>> print(points1)
    np.array([[4,4], [4,5], [5,6], [5,7], [5,8], [6,9], [6,10]])
    """
    # setup initial conditions

    n_lines = len(ends)
    starts = np.tile(start, (n_lines, 1))
    d = ends[:] - starts[:]

    # determine how steep the line is
    is_steep = abs(d[:, 1]) > abs(d[:, 0])

    steep_mask = is_steep[:, np.newaxis]
    starts = np.where(steep_mask, starts[:, ::-1], starts)
    ends = np.where(steep_mask, ends[:, ::-1], ends)

    # Ensure lines go left to right
    swap_mask = (starts[:, 0] > ends[:, 0])[:, np.newaxis]
    starts_swapped = np.where(swap_mask, ends, starts)
    ends_swapped = np.where(swap_mask, starts, ends)

    starts = starts_swapped
    ends = ends_swapped
    d = ends - starts  # recalculate differentials
    errors = d[:, 0] // 2  # calculate error

    # iterate over bounding box generating points between start and end
    y = starts[:, 1:2].flatten()
    y_step = np.where(starts[:, 1] < ends[:, 1],
                      1, -1)[:, np.newaxis].flatten()
    print(f"y_step:\n{y_step}")

    lengths = ends[:, 0] - starts[:, 0]
    max_length = np.max(lengths) + 1

    # Create a range array for the second dimension
    col_indices = np.arange(max_length)
    # Broadcasting will compare each length against all column indices
    valid_mask = col_indices >= lengths[:, np.newaxis]

    # Instead of the loop, we can do:
    x_coords = starts[:, 0, np.newaxis] + np.arange(max_length)

    # points = np.zeros((n_lines, max_length, 2), dtype=int)
    # for i, x in enumerate(x_coords.T):
    #     coord = np.where(steep_mask, np.stack([y, x], axis=-1), np.stack([x, y], axis=-1))
    #     points[:, i, :] = coord
    points = np.zeros((n_lines, max_length), dtype=[
        ('f0', 'i'), ('f1', 'i')])
    for i, x in enumerate(x_coords.T):
        x_vals = np.where(steep_mask.flatten(), y, x)
        y_vals = np.where(steep_mask.flatten(), x, y)
        points[:, i] = np.rec.fromarrays([x_vals, y_vals], dtype='i,i')
        errors -= abs(d[:, 1])
        # if errors < 0:
        #     y += y_step
        #     errors += d[:, 0]
        y = np.where(errors < 0, y + y_step, y)
        errors = np.where(errors < 0, errors + d[:, 0], errors)
    points = np.where(swap_mask, points[:, ::-1], points)
    print(f"points:\n{points}")
    valid_mask = np.where(swap_mask, valid_mask[:, ::-1], valid_mask)
    print(f"valid_mask:\n{valid_mask}")
    points = ma.array(points, mask=valid_mask)

    def shave_leading_invalid_entries(x):
        return np.split(x.ravel(), (np.stack((np.zeros(x.shape[0], dtype=int), np.argmin(x.mask['f0'], axis=1)), axis=-1) + np.arange(x.shape[0])[:, None] * x.shape[1]).ravel())[2::2]
    return shave_leading_invalid_entries(points)
