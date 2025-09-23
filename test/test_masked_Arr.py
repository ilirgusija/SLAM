import numpy as np
import numpy.ma as ma
from matplotlib import pyplot as plt
import classes.mapping as mapping

def shave_leading_invalid_entries(x):
    return np.split(x.ravel(), (np.stack((np.zeros(x.shape[0], dtype=int), np.argmin(x.mask['f0'], axis=1)), axis=-1) + np.arange(x.shape[0])[:, None] * x.shape[1]).ravel())[2::2]

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
    y_step = np.where(starts[:, 1] < ends[:, 1], 1, -1)[:, np.newaxis].flatten()
    print(f"y_step:\n{y_step}")

    lengths = ends[:, 0] - starts[:, 0]
    max_length = np.max(lengths) + 1

    # Create a range array for the second dimension
    col_indices = np.arange(max_length)
    # Broadcasting will compare each length against all column indices
    valid_mask = col_indices > lengths[:, np.newaxis]

    # Instead of the loop, we can do:
    x_coords = starts[:, 0, np.newaxis] + np.arange(max_length)

    # points = np.zeros((n_lines, max_length, 2), dtype=int)
    # for i, x in enumerate(x_coords.T):
    #     coord = np.where(steep_mask, np.stack([y, x], axis=-1), np.stack([x, y], axis=-1))
    #     points[:, i, :] = coord
    points = np.zeros((n_lines, max_length), dtype=[('f0', 'i'), ('f1', 'i')])
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
    return shave_leading_invalid_entries(points)


def test_shaver():
    x = np.array([[(1, 1), (2, 2), (3, 2)],
                  [(1, 1), (2, 1), (3, 2)],
                  [(1, 1), (2, 2), (3, 3)]], dtype='i,i')
    x_arr = np.array([[[1, 1], [2, 2], [3, 2]],
                  [[1, 1], [2, 1], [3, 2]],
                  [[1, 1], [2, 2], [3, 3]]], dtype='int')
    # print("Original array:")
    # print(x)
    # print("Shape:", x.shape)

    # Create a 2D mask array instead of a mask for each component
    mask = np.array([[1, 1, 0],  # First row: mask first pair
                     [1, 0, 0],  # Second row: mask first two pairs
                     [0, 0, 0]], dtype=bool)  # Third row: mask all pairs
    
    mask_expanded = np.repeat(mask[:, :, np.newaxis], 2, axis=2)

    x = ma.array(x, mask=mask)
    x_arr = ma.array(x_arr, mask=mask_expanded)
    print(shave_leading_invalid_entries(x))
    # print(x[~x.mask])
    print(x_arr[~x_arr.mask])
    sys.exit()
    print(mask)
    print(x.mask['f0'])

    start = np.argmin(mask, axis=1)
    print(start)

    pad = np.zeros(x.shape[0], dtype=int)
    print(pad)

    indices = np.ravel(np.stack((pad, start), axis=-1) +
                       np.arange(x.shape[0])[:, None] * x.shape[1])
    print(indices)
    x = np.split(x.ravel(), indices)[2::2]
    print(x)


def plot_points(starts, ends):
    plt.scatter(ends[:, 0], ends[:, 1], c='red')
    plt.scatter(starts[:, 0], starts[:, 1], c='blue')
    # plot lines connecting start to ends
    for i in range(ends.shape[0]):
        plt.plot([starts[i, 0], ends[i, 0]], [
                 starts[i, 1], ends[i, 1]], c='black')
    plt.show()


def main():
    start = np.array([1, 1])
    ends = np.array([[3, 7], [-5, 8], [-5, -9], [4, -10]])

    # plot_points(start, ends)
    # print(bresenham_vec(start, ends))
    # sys.exit()
    x = mapping.FloatGrid(0.5)
    y = 0.5
    #check size of variable in bytes 
    print(sys.getsizeof(x))
    print(sys.getsizeof(y))

if __name__ == "__main__":
    # main()
    test_shaver()