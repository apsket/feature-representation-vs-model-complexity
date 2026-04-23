import numpy as np
import pandas as pd

# Convert 2D Cartesian to polar coordinates on numpy arrays
def cartesian_to_polar(x, y):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return pd.DataFrame({"r": r, "theta": theta})

# Convert 2D Cartesian to elliptic coordinates on numpy arrays
def cartesian_to_elliptic(x, y, c=1):
    u = np.arccosh((1/(2*c)) * (np.sqrt((x**2 + c**2)**2 + y**2) + np.sqrt((x**2 - c**2)**2 + y**2)))
    v = np.arctan2(y, x)
    return pd.DataFrame({"u": u, "v": v})
