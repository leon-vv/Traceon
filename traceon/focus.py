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
    positions: iterable of (N,4) or (N,6) np.ndarray float64
        Trajectories of electrons, as returned by `traceon.tracing.Tracer.__call__`
    
    
    Returns
    --------------
    A tuple of size two or three, depending on whether the input positions are 2D or 3D trajectories. The \
    returned position is the (r, z) or (x, y, z) position of the focus.

    """
    two_d = positions[0].shape[1] == 4
    
    # Also for 2D, extend to 3D
    if two_d:
        positions = [np.column_stack(
            (p[:, 0], np.zeros(len(p)), p[:, 1], p[:, 2], np.zeros(len(p)), p[:, 3])) for p in positions]
      
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

    return (x, y, z) if not two_d else (x, z)
    
