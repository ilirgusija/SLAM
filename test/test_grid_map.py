import numpy as np
from matplotlib import pyplot as plt

# --- New class: GridMapNP ---

class GridMapNP:
    def __init__(self, x_min, x_max, y_min, y_max, resolution=1.0,
                 init_val=0.5):
        """__init__
        :param x_min, x_max, y_min, y_max: positions of extremities of map [m]
        :param resolution: number of levels of quantization per meter [m]
        :param init_val: initial value for all grid cells
        """
        self.left_lower = np.array([x_min,y_min])
        self.resolution = resolution
        self.center = np.array([(x_max - x_min) / 2,(y_max - y_min) / 2])

        self.width = int((x_max-x_min+1) * 1/self.resolution)
        self.height = int((y_max-y_min+1) * 1/self.resolution)

        self.data = np.full((self.width, self.height),
                            init_val, dtype=type(init_val))
        self.data_type = type(init_val)
    
    # Getters##################################################################
    def get_value_from_xy_index(self, x_ind, y_ind):
        if x_ind is None or y_ind is None:
            return None
        if 0 <= x_ind < self.width and 0 <= y_ind < self.height:
            return self.data[x_ind, y_ind]
        else:
            return None

    def get_xy_index_from_xy_pos(self, pos):
        x_ind, y_ind = int(np.floor((pos - self.left_lower) / self.resolution))
        if 0 <= x_ind < self.width and 0 <= y_ind < self.height:
            return x_ind, y_ind
        else:
            return None

    # Setters##################################################################
    def set_value_from_xy_pos(self, pos, val):
        x_ind, y_ind = self.get_xy_index_from_xy_pos(pos)
        if (not x_ind) or (not y_ind):
            raise ValueError("Not GOOD!")
            # return False
        flag = self.set_value_from_xy_index(x_ind, y_ind, val)
        return flag

    def set_value_from_xy_index(self, x_ind, y_ind, val):
        if (x_ind is None) or (y_ind is None):
            raise ValueError("returning false from set_value_from_xy_index")
        if 0 <= x_ind < self.width and 0 <= y_ind < self.height and isinstance(val, self.data_type):
            self.data[x_ind, y_ind] = val
            return True
        else:
            print(f"x_ind: {x_ind}, y_ind: {y_ind} out of bounds")
            print(f"val: {val} vs self.data_type: {self.data_type}")
            raise ValueError(f"y_ind: {y_ind} or x_ind: {x_ind} out of bounds")

    # Rest of the methods are the same as in the original GridMap class

    def check_occupied_from_xy_index(self, x_ind, y_ind, occupied_val):
        val = self.get_value_from_xy_index(x_ind, y_ind)
        if val is None or val >= occupied_val:
            return True
        else:
            return False

    def expand_grid(self, occupied_val=1.0):
        x_inds, y_inds, values = [], [], []
        for ix in range(self.width):
            for iy in range(self.height):
                if self.check_occupied_from_xy_index(ix, iy, occupied_val):
                    x_inds.append(ix)
                    y_inds.append(iy)
                    values.append(self.get_value_from_xy_index(ix, iy))
        for (ix, iy, value) in zip(x_inds, y_inds, values):
            self.set_value_from_xy_index(ix + 1, iy, val=value)
            self.set_value_from_xy_index(ix, iy + 1, val=value)
            self.set_value_from_xy_index(ix + 1, iy + 1, val=value)
            self.set_value_from_xy_index(ix - 1, iy, val=value)
            self.set_value_from_xy_index(ix, iy - 1, val=value)
            self.set_value_from_xy_index(ix - 1, iy - 1, val=value)

    def print_grid_map_info(self):
        print("width:", self.width)
        print("height:", self.height)
        print("resolution:", self.resolution)
        print("center_x:", self.center_x)
        print("center_y:", self.center_y)
        print("left_lower_x:", self.left_lower_x)
        print("left_lower_y:", self.left_lower_y)

    def plot_grid_map(self, ax=None):
        grid_data = self.data
        if not ax:
            fig, ax = plt.subplots()
        heat_map = ax.pcolor(grid_data, cmap="Blues", vmin=0.0, vmax=1.0)
        plt.axis("equal")
        return heat_map


def main():
    grid_map = GridMapNP(0, 10, 0, 10, 1.0, 0.5)
    grid_map.plot_grid_map()
    plt.show()


if __name__ == "__main__":
    main()
