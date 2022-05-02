r"""
Aberration coefficients give the deviation from perfect focussing in the focal plane. The deviations are notated as $\Delta r$ and
are a function of the initial angle with the optical axis $\alpha$ and the deviation from the mean beam energy $\frac{\Delta E}{E}$.

$$
\Delta r = \underbrace{-\sum_{i=1, i \text{ odd}}C_{i0}\alpha^i}_\text{spherical} +
\underbrace{\sum_{i,j=1, i \text{ odd}} C_{ij} \alpha^i \left( \frac{\Delta E}{E} \right)^j}_\text{chromatic}
$$

Aberration coefficients are notated as $C_{ij}$ where i is the spherical power and j the chromatic power. 
The rank is $i+j$. i is in [1, 3, 5, 7] while j is in [0, 1, 2, 3], this gives 12 aberration coefficients.
The spherical coefficients $C_{10}$, $C_{30}$, $C_{50}$ can be fitted independently from the chromatic terms.

The aberrations are stored in memory as an array of shape (8, 4). The array is padded
with zeros to allow straightforward indexing. For example, the spherical aberration $C_{30}$
can be indexed as C[3, 0].
"""

import time, pickle

import matplotlib.pyplot as plt
import numpy as np
import numba as nb

from . import tracing as T

@nb.njit(cache=True)
def compute_intersections_fast(coeff, dE, angles):
    """Compute the deviation $\Delta r$.

    Args:
        coeff: the aberration coefficients (shape (8,4)).
        dE: energy deviations with shape (N,)
        angles: initial angles with shape (N,)

    Returns:
        The deviations $\Delta r$ with shape (N,)
    """
    assert coeff.shape == (8,4)
    assert dE.size == angles.size
    
    intersections = np.zeros_like(angles) 

    for n in range(intersections.size):
        
        sum_ = 0.0
        
        for i in range(1, 8, 2):
            for j in range(4):
                sign = -1 if j == 0 else 1
                sum_ += sign * coeff[i, j] * angles[n]**i * dE[n]**j

        intersections[n] = sum_

    return intersections

class AberrationCoefficients:
    """Wrapper class of the aberration coefficient matrix. Provides some utility method."""
    
    def __init__(self, C=None):
        if C is None:
            self.C = np.zeros( (8, 4) )
        else:
            self.C = C
            assert C.shape == (8,4)
        self.intersection = np.vectorize(self._intersection, excluded='self')
    
    def __setitem__(self, *args, **kwargs):
        self.C.__setitem__(*args, **kwargs)
    
    def __getitem__(self, *args, **kwargs):
        return self.C.__getitem__(*args, **kwargs)
    
    def factors(dE, angle):
        F = np.zeros( (8, 4) )
        energy_powers, angle_powers = np.meshgrid([1, dE, dE**2, dE**3], [angle, angle**3, angle**5, angle**7])
        F[1::2, :] = energy_powers * angle_powers
        
        # We want to follow the following convention:
        # For chromatic aberrations, a faster electron hitting positive r (so focusing too far) should count as a positive aberration
        # For spherical aberrations, an electron at larger angle hitting a negative r (so focusing too early) should count as a positive aberration.
        # We therefore have to introduce a minus sign in front of the coefficients of spherical aberration
        F[:, 0] *= -1
        
        return F
        
    def terms(self, dE, angle):
        # Important: by convention a positive aberration must lead to too strong
        # focusing. Therefore we add a minus sign such that positive aberrations
        # lead to negative r values.
        return self.C * AberrationCoefficients.factors(dE, angle)
    
    def spherical_contribution(self, angle):
        terms = self.terms(0.0, angle)
        return np.sum(terms[:, 0])

    def chromatic_contribution(self, dE, angle):
        terms = np.terms(dE, angle)
        return np.sum(terms[:, 1:])
     
    def _intersection(self, dE, angle):
        return np.sum(self.terms(dE, angle))
    
    # Which of the terms contribute how much
    def contribution(self, dE, angle, intersection=None):
        terms = self.terms(dE, angle)
        int_ = np.sum(terms)
        
        assert int_ == self.intersection(dE, angle)
        
        if intersection is not None:
            assert np.isclose(int_, intersection)
        
        return np.abs(terms)/np.abs(int_)
    
    # Which of the coefficients can be considered to have
    # a negligible contribution to the intersection coordinate.
    def _negligible(self, dE, angle, intersection=None):
        contribution = self.contribution(dE, angle, intersection=intersection)
        
        # Terms are negligible if they contribute less than 1% to final answer
        neg_x, neg_y = np.where(contribution < 1e-2)
         
        # Term with even spherical aberration do not contribute anyway, so keep only odd
        mask = neg_x % 2 == 1
        neg_x, neg_y = neg_x[mask], neg_y[mask]
        
        return neg_x, neg_y

    def to_string(self, dE=None, angle=None):
        
        if dE != None and angle != None:
            neg_x, neg_y = self._negligible(dE, angle)
            neg = list(zip(neg_x, neg_y))
            contribution = self.contribution(dE, angle)
        else:
            neg = []
            contribution = None
         
        C = self.C
        txt = ''
        
        for i in range(1, C.shape[0], 2):
            for j in range(C.shape[1]):
                
                if (i, j) in neg:
                    continue
                 
                if C[i, j] == 0.0:
                    continue
                
                txt += f'C{i}{j} = {C[i,j]:+9.2e}'
                if contribution is not None:
                    txt += f' ({contribution[i, j]*100:+6.1f}%)'

                txt += '\t'
            
            txt += '\n'
        
        return txt
    
    def __str__(self):
        return self.to_string()
        
class AberrationCurve:
    
    def __init__(self, geometry,
                aberrations,
                scan_electrode, focus_electrode,
                scan_voltages, focus_voltages,
                z_coords,
                aux_electrodes=None,
                aux_voltages=None):
        
        self.geometry = geometry
        self.scan_voltages = np.array(scan_voltages)
        self.focus_voltages = np.array(focus_voltages)
        self.aberrations = np.array(aberrations)
        self.scan_electrode = scan_electrode
        self.focus_electrode = focus_electrode
        assert isinstance(self.scan_electrode, str) and isinstance(self.focus_electrode, str)
        self.aux_electrodes = list(aux_electrodes)
        self.aux_voltages = list(aux_voltages)
        self.z_coords = z_coords
        
        assert aux_electrodes is None or (len(aux_electrodes) == len(aux_voltages))
    
    def write(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
    
    def read(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)

def spherical(tracer, energy, dr=0.006, N=7):
    """Compute spherical aberration coefficients.

    Args:
        tracer: a tracer object (for example tracing.PlaneTracer)
        energy: mean beam energy
        dr: The largest angle used will result in a beam spread at z=0 of $2\Delta r$ (in millimeters).
        N: number of electrons traced
    
    Returns:
       A tuple containg the aberration coefficients, the initial angles used for tracing and
       the resulting intersections.
    """
    angle = dr / tracer.get_z0()
     
    angles = np.linspace(0, angle, N)
    energies = np.full_like(angles, energy)
     
    intersections, mask = tracer.trace(angles, energies)

    assert np.sum(mask) > N-3, 'Lost too many electrons while tracing' # Do not lose more than three electrons
     
    A = np.array([AberrationCoefficients.factors(energy, a)[1::2, 0] for a in angles[mask]])
    
    C = AberrationCoefficients()
    
    C[1::2, 0] = np.linalg.lstsq(A, intersections[mask, 0], rcond=None)[0]

    # If any is exactly zero there is likely a mistake in the fitting procedure
    if 0.0 in [C[1, 0], C[3, 0], C[5, 0]]:
        raise RuntimeError('Failed obtaining correct spherical aberration coefficients')
    
    assert np.all(np.isfinite(C.C))
     
    return C, angles[mask], intersections[mask]

def compute_coefficients(tracer, energy, dE=0.25e-2, dr=0.006, N=6):
    r"""Compute spherical and chromatic aberration coefficients.

    Args:
        tracer: a tracer object (for example tracing.PlaneTracer)
        energy: mean beam energy in electronvolt
        dE: the maximum energy deviation $\frac{\Delta E}{E}$
        dr: the largest angle used will result in a beam spread at z=0 of $2\Delta r$ (in millimeters).
        N: sampling of energy deviations and angles. Total number of electrons traced scales as $N^2$.
    
    Returns:
       A tuple containg the aberration coefficients, the initial angles used for tracing and
       the resulting intersections.
    """
     
    angle = dr / tracer.get_z0()
     
    dE = np.repeat(np.linspace(-dE, dE, N), N) # Relative energy deviation
    energies = (1+dE)*energy
     
    angles = np.linspace(0, angle, N)
    angles = np.repeat(angles[np.newaxis, :], N, axis=0).flatten() # Repeat such that all combinations are considered

    assert dE.size == N*N and angles.size == N*N
    
    spherical_coeff, _, _ = spherical(tracer, energy, dr=dr, N=2*N)
    assert np.all(np.isfinite(spherical_coeff.C))
    assert not spherical_coeff.C[3, 0] == 0.0 or spherical_coeff[5, 0] == 0.0
    
    intersections, mask = tracer.trace(angles, energies)
    intersections_sph = compute_intersections_fast(spherical_coeff.C, np.zeros(np.sum(mask)), angles[mask]) #spherical_coeff.intersection(0.0, angles)
    
    A = np.array([AberrationCoefficients.factors(de, a)[1::2, 1:].flatten() for (de, a) in zip(dE[mask], angles[mask])])

    assert len(A) == len(intersections_sph)
     
    coeff = np.linalg.lstsq(A, intersections[mask, 0] - intersections_sph, rcond=None)[0]
    assert np.all(np.isfinite(coeff))
     
    # Note that C[0, j] is always zero, as electrons along the optical axis
    # always go through the focus again
    C = AberrationCoefficients()
    C[1::2, 1:] = np.resize(coeff, (4, 3))
    C[1::2, 0]  = spherical_coeff[1::2, 0]
    
    assert np.all(np.isfinite(C.C))
    
    return C, dE, angles, intersections


