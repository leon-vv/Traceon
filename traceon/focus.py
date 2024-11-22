"""
Module containing a single function to find the focus of a beam of electron trajecories.
"""

import numpy as np


def focus_position(positions):
    """
    Find the focus of the given trajectories (which are returned from `traceon.tracing.Tracer.__call__`).
    The focus is found using a least square method by considering the final positions and velocities of
    the given trajectories and linearly extending the trajectories backwards.
     
    
    Parameters
    ------------
    positions: iterable of (N,6) np.ndarray float64
        Trajectories of electrons, as returned by `traceon.tracing.Tracer.__call__`
    
    
    Returns
    --------------
    (3,) np.ndarray of float64, representing the position of the focus
    """
    assert all(p.shape == (len(p), 6) for p in positions)
     
    angles_x = np.array([p[-1, 3]/p[-1, 5] for p in positions])
    angles_y = np.array([p[-1, 4]/p[-1, 5] for p in positions])
    x, y, z = [np.array([p[-1, i] for p in positions]) for i in [0, 1, 2]]
     
    N = len(positions)
    first_column = np.concatenate( (-angles_x, -angles_y) )
    second_column = np.concatenate( (np.ones(N), np.zeros(N)) )
    third_column = np.concatenate( (np.zeros(N), np.ones(N)) )
    right_hand_side = np.concatenate( (x - z*angles_x, y - z*angles_y) )
     
    (z, x, y) = np.linalg.lstsq(
        np.array([first_column, second_column, third_column]).T,
        right_hand_side, rcond=None)[0]
    
    return np.array([x, y, z])
    
