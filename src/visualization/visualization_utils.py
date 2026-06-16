import numpy as np
import pandas as pd

def create_meshgrid(x1: pd.Series, x2: pd.Series, resolution=100):
    # Cartesian meshgrid utility for decision boundaries
    x_min, x_max = x1.min() - 0.5, x1.max() + 0.5
    y_min, y_max = x2.min() - 0.5, x2.max() + 0.5
    grid_x, grid_y = np.meshgrid(np.linspace(x_min, x_max, resolution),
                                np.linspace(y_min, y_max, resolution))
    grid_cartesian = pd.DataFrame({'x1': grid_x.ravel(), 'x2': grid_y.ravel()})

    return grid_x, grid_y, grid_cartesian
