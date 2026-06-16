from typing import Tuple
import numpy as np
import pandas as pd
from src.data.loaders import PointsDataset


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


def generate_custom_polar_points(
        num_points: int,
        theta_lb: float,
        theta_ub: float,
        theta_pdf: callable,
        r_inner_bound_func: callable,
        r_outer_bound_func: callable,
        r_pdf: callable,
        theta_noise: float = 0.0,
        r_noise: float = 0.0,
        random_seed: int = None
    ):

    rng = np.random.default_rng(seed=random_seed)

    theta = theta_pdf(theta_lb, theta_ub, num_points)
    r_inner_bounds = r_inner_bound_func(theta)
    r_outer_bounds = r_outer_bound_func(theta)
    r = r_pdf(r_inner_bounds, r_outer_bounds)

    if theta_noise != 0:
        theta += rng.vonmises(mu=0.0, kappa=theta_noise, size=num_points)

    if r_noise != 0:
        r += rng.normal(loc=0.0, scale=r_noise, size=num_points)
        r = np.abs(r)

    return np.column_stack((r, theta))


def generate_custom_polar_dataset(
        label: int,
        num_points: int,
        theta_lb: float = -np.pi,
        theta_ub: float = np.pi,
        theta_pdf: callable = None,
        r_inner_bound_func: callable = lambda theta: np.array([0]*theta.size),
        r_outer_bound_func: callable = lambda theta: np.array([1]*theta.size),
        r_pdf: callable = None,
        theta_noise: float = 0.0,
        r_noise: float = 0.0,
        random_seed: int = None,
    ):
    if label not in {0, 1}:
        raise ValueError("Invalid dataset label. Supported labels are 0 and 1.")
    
    rng = np.random.default_rng(seed=random_seed)
    if theta_pdf is None:
        theta_pdf = rng.uniform
    if r_pdf is None:
        r_pdf = lambda low, high: np.sqrt(rng.uniform(low**2, high**2))
    
    polar_points = generate_custom_polar_points(
        num_points=num_points, 
        theta_lb=theta_lb, theta_ub=theta_ub, theta_pdf=theta_pdf, 
        r_inner_bound_func=r_inner_bound_func, r_outer_bound_func=r_outer_bound_func, r_pdf=r_pdf,
        theta_noise=theta_noise, r_noise=r_noise,
        random_seed=random_seed
    )
    
    return PointsDataset(
        pd.DataFrame(polar_points, columns=["r", "theta"]),
        y=np.zeros(num_points) if label == 0 else np.ones(num_points)
    )