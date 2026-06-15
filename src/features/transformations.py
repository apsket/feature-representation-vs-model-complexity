import numpy as np
import pandas as pd

# Convert 2D Cartesian to polar coordinates on numpy arrays
def cartesian_to_polar(df: pd.DataFrame):
    x = df['x']
    y = df['y']
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return pd.DataFrame({"r": r, "theta": theta})

# Convert 2D Cartesian to elliptic coordinates on numpy arrays
# elliptic is a family of coordinates characterizable by a parameter (c)
def cartesian_to_elliptic(df: pd.DataFrame, c=1):
    x = df['x']
    y = df['y']
    u = np.arccosh((1/(2*c)) * (np.sqrt((x**2 + c**2)**2 + y**2) + np.sqrt((x**2 - c**2)**2 + y**2)))
    v = np.arctan2(y, x)
    return pd.DataFrame({"u": u, "v": v})

def polar_to_cartesian(df: pd.DataFrame):
    r = df['r']
    theta = df['theta']
    return pd.DataFrame({'x': r * np.cos(theta), 'y': r * np.sin(theta)})

# Convert polar to custom curvilinear coordinates (r^2, cos^2(theta))
def polar_to_curvilinear_r2cos2theta(df):
    r = df['r']
    theta = df['theta']
    return pd.DataFrame({"r^2": r**2, "cos^2()": np.cos(theta)**2})


def extend_cartesian_to_quadratic(df):
    return pd.DataFrame(
        {
            'x^2': df['x']**2,
            'y^2': df['y']**2,
            'xy': df['x']*df['y'],
            'x': df['x'],
            'y': df['y']
        }
    )

