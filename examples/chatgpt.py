import numpy as np
from scipy.special import sph_harm, factorial

def cartesian_to_spherical(coords):
    """
    Convert Cartesian coordinates to spherical coordinates (r, theta, phi).
    - r: radial distance
    - theta: polar angle (0 <= theta <= pi)
    - phi: azimuthal angle (0 <= phi < 2*pi)
    """
    x, y, z = coords
    r = np.linalg.norm(coords)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    return r, theta, phi

def spherical_harmonic_expansion(y, x_alpha, phi_alpha, max_order=5):
    """
    Evaluate f(y) using spherical harmonic expansion for a single source alpha.

    Parameters:
    - y: Target point (3D Cartesian coordinates).
    - x_alpha: Source point (3D Cartesian coordinates).
    - phi_alpha: Source charge.
    - max_order: Maximum order of the spherical harmonic expansion.

    Returns:
    - The potential f(y).
    """
    # Relative position vector and spherical coordinates
    rel_vec = y - x_alpha
    r, theta, phi = cartesian_to_spherical(y)
    
    # Compute the multipole expansion
    f_y = 0
    for l in range(max_order + 1):
        for m in range(-l, l + 1):
            # Coefficient: charge times source spherical harmonics
            r_alpha, theta_alpha, phi_alpha = cartesian_to_spherical(x_alpha)
            norm_factor = ((2 * l + 1) / (4 * np.pi))**-1 #* (factorial(l - abs(m)) / factorial(l + abs(m)))
            
            C_lm = norm_factor * phi_alpha * np.conj(sph_harm(m, l, phi_alpha, theta_alpha)) * (r_alpha ** l)
            
            # Contribution to the potential
            f_y += C_lm * sph_harm(m, l, phi, theta) / (r ** (l + 1))
    
    return f_y.real  # Only the real part contributes to the potential

# Example usage
x_alpha = np.array([0.5, 0.5, 0.5])  # Source position
phi_alpha = 2.0  # Source charge
y = np.array([1.0, 1.0, 1.0])*5  # Target point
max_order = 5  # Maximum expansion order

result = spherical_harmonic_expansion(y, x_alpha, phi_alpha, max_order)
print("Potential at y:", result)

# Output conventional

out = phi_alpha / np.linalg.norm(y - x_alpha)
print(out)
