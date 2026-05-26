from typing import Tuple
import numpy as np
import pandas as pd


def generate_uniform_ring_points(
        num_points: int,
        r_min: float = 0.0,
        r_max: float = 1.0,
        coord_system: str = 'polar',
        noise_std: float = 0.0,
    ):
    """
    Generates random points within a circle of a given radius with uniform density.
    
    Parameters:
    - cutoff_radius (float): The radius of the circle.
    - num_points (int): Number of points to generate.
    - noise_std (float): noise_std (float): Standard deviation of the Gaussian noise applied to the radius.
    - coord_system (str): The coordinate system to represent the points (defaulted to polar).
    
    Returns:
    - numpy.ndarray: An array of shape (num_points, 2) containing the coordinates.
    """

    coord_transforms = {
        'cartesian': lambda r, theta: (r * np.cos(theta), r * np.sin(theta)), 
        'polar': lambda r, theta: (r, theta)
    }

    if coord_system not in coord_transforms:
        raise ValueError(f"coord_system must be one of {tuple(coord_transforms.keys())}")
    elif not r_max > r_min:
        raise ValueError(f"Inconsistent r_min {r_min} and r_max {r_max}, must have r_max > r_min")
    elif r_max <= 0:
        raise ValueError(f"r_max must be greater than zero")
    elif r_min < 0:
        raise ValueError(f"r_min must not be less than zero")

    theta = np.random.uniform(-np.pi, np.pi, num_points)
    r = np.sqrt(np.random.uniform(r_min**2, r_max**2, num_points))
    r = np.clip(r, r_min, None)

    if noise_std != 0:
        r += np.random.normal(loc=0.0, scale=noise_std, size=num_points)
        r = np.abs(r)

    return np.column_stack(coord_transforms[coord_system](r, theta))


def generate_uniform_ring_synthetic_dataset(
    num_points1: int,
    num_points2: int,
    r_min: float = 0.0,
    r_bound: float = 0.5,
    delta_r: float = 0.0,
    r_max: float = 1.0, 
    coord_system: str = 'polar',
    noise_std: Tuple[float, float] = (0.0, 0.0),
    ):

    if not (r_max > r_min + delta_r):
        raise ValueError("Radial parameters must satisfy r_max > r_min + delta_r")

    X1 = pd.DataFrame(
    generate_uniform_ring_points(
        num_points=num_points1, 
        r_min=r_min, r_max=r_bound, 
        noise_std=noise_std[0],
        coord_system=coord_system
        ), columns=['x1', 'x2']
    )

    X2 = pd.DataFrame(
    generate_uniform_ring_points(
        num_points=num_points2, 
        r_min=r_bound+delta_r, r_max=r_max, 
        noise_std=noise_std[1],
        coord_system=coord_system
        ), columns=['x1', 'x2']
    )

    X = pd.concat([X1, X2])
    y = np.concat([np.ones(len(X1)), np.zeros(len(X2))])

    return X, y


def generate_custom_polar_dataset(num_points, r_bound_func=None):

    theta_inner = np.random.uniform(-np.pi, np.pi, num_points)
    r_bounds = r_bound_func(theta_inner)
    r_inner = np.random.uniform(np.zeros(r_bounds.size), r_bounds)

    return r_inner
