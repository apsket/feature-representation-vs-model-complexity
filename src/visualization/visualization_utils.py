import numpy as np
import pandas as pd

def create_meshgrid(X, resolution=100):
    # Cartesian meshgrid utility for decision boundaries
    x_min, x_max = X['x'].min() - 0.5, X['x'].max() + 0.5
    y_min, y_max = X['y'].min() - 0.5, X['y'].max() + 0.5
    grid_x, grid_y = np.meshgrid(np.linspace(x_min, x_max, resolution),
                                np.linspace(y_min, y_max, resolution))
    grid_cartesian = pd.DataFrame({'x': grid_x.ravel(), 'y': grid_y.ravel()})

    return grid_x, grid_y, grid_cartesian
