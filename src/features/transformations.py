import numpy as np
import pandas as pd

# Convert 2D Cartesian to polar coordinates on numpy arrays
def cartesian_to_polar(x, y):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return pd.DataFrame({"x1": r, "x2": theta})

# Convert 2D Cartesian to elliptic coordinates on numpy arrays
# elliptic is a family of coordinates characterizable by a parameter (c)
def cartesian_to_elliptic(x, y, c=1):
    u = np.arccosh((1/(2*c)) * (np.sqrt((x**2 + c**2)**2 + y**2) + np.sqrt((x**2 - c**2)**2 + y**2)))
    v = np.arctan2(y, x)
    return pd.DataFrame({"x1": u, "x2": v})

# Convert polar to custom curvilinear coordinates (r, cos^2(theta))
def polar_to_curvilinear_cos2theta(r, theta):
    return pd.DataFrame({"x1": r, "x2": np.cos(theta)**2})

