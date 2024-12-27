import scipy
import numpy as np
from scipy.constants import mu_0
import scipy.special as sp

def spherical_to_cartesian(r, theta, phi):
    x = r*np.sin(theta)*np.cos(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z = r*np.cos(theta)
    return x, y, z

def cartesian_to_spherical(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.sign(y)*np.arccos(x/(x**2 + y**2))
    return r, theta, phi

def Y(l,m,theta, phi):
    if np.abs(m)> l:
        return 0
    return sp.sph_harm(m,l,phi, theta)

def Y_star(l,m,theta,phi):
    return np.conj(Y(l,m,theta, phi))

def dYdtheta(l,m,theta,phi):
    a =  m*np.cos(theta)/np.sin(theta)*Y(l,m,theta, phi)+np.sqrt((l-m)*(l+m+1)+0*1j)*np.exp(-1j*phi)*Y(l,m+1,theta, phi)
    return a

def dYdphi(l,m,theta,phi):
    return 1j*m*Y(l,m,theta, phi)

def vp(L, r, theta, phi, I):
    rp, phip, thetap, Ix, Iy, Iz = I
    sum = 0
    factor = 4*np.pi/(rp)
    for l in range(0, L):
        for m in range(-l, l+1):
            prefactor = (r/rp)**l * 1/(2*l+1)* Y_star(l, m,thetap, phip)
            Ylm = Y(l, m, theta, phi)
            print(f'term {l,m} {prefactor*Ylm}')
            sum += prefactor*Ylm
            
    return factor*sum* np.array([Ix, Iy, Iz])

def B_r(L, r, theta, phi, I):
    rp, phip, thetap, Ir, Iphi, Itheta = I
    sum_r = 0
    sum_theta = 0
    sum_phi = 0
    factor = mu_0/(4*np.pi)/(rp)  * (4*np.pi)
    for l in range(0, L):
        for m in range(-l, l+1):
            # prefactor = mu_0 /(4*np.pi* rp) * (r/rp)**l * (4*np.pi)/(2*l+1)* Y_star(l, m,thetap, phip)
            # print(l,m)
            prefactor = (r/rp)**l/(2*l+1)* Y_star(l,m,thetap, phip)
            Ylm = Y(l, m, theta, phi)
            dtheta = dYdtheta(l,m,theta,phi)
            dphi = dYdphi(l,m,theta,phi)
            dYdr = (l)/r * Ylm

            # print(f'pref {prefactor}')
            # print(f'theta {dYdtheta}')
            # print(f'phi {dYdphi}')
            
            sum_r += prefactor*((np.sin(theta)*dtheta+np.cos(theta)*Ylm)*Iphi - dphi*Itheta)
            sum_theta += prefactor*(1/np.sin(theta)*dphi*Ir - r*dYdr*Iphi - Ylm*Iphi)
            sum_phi += prefactor*(r*dYdr*Itheta + Ylm*Itheta - dtheta*Ir)
    return factor*sum_r/(np.sin(theta)*r), sum_theta/r, sum_phi/r

def BiotSavart(r, theta, phi, I):
    rp, thetap, phip, Ir, Itheta, Iphi = I
    xp, yp, zp = spherical_to_cartesian(rp, thetap, phip)
    x, y, z = spherical_to_cartesian(r, theta, phi)
    Ix, Iy, Iz = spherical_to_cartesian(Ir, Itheta, Iphi)

    difference = np.array([x,y,z]) - np.array([xp, yp, zp])
    norm = np.linalg.norm(difference)
    cross = np.cross(np.array([Ix,Iy,Iz]), difference)
    xyz = cross/norm**3
    return cartesian_to_spherical(xyz[0],xyz[1],xyz[2])

def VectorPotential(r, theta, phi, I):
    rp, thetap, phip, Ir, Itheta, Iphi = I
    xp, yp, zp = spherical_to_cartesian(rp, thetap, phip)
    x, y, z = spherical_to_cartesian(r, theta, phi)
    I = [Ir, Itheta, Iphi]

    difference = np.array([x,y,z]) - np.array([xp, yp, zp])
    norm = np.linalg.norm(difference)
    current = np.array(I)

    return current/norm


# print(B_r(1, 0.1, np.pi/4, np.pi/4, (2, np.pi/2, np.pi/4, 0, 1, 0)))
# print(mu_0)


r = np.sqrt(2)/200, np.pi/8, 0
I = (3, np.pi/2,0.3, 6, np.pi/2, np.pi/2)

#check if derivatives are correct
delta = 1e-7
l, m = 1, 0
theta, phi = np.pi/3, np.pi/3


print('Value', Y(l, m, theta, phi))
print(-np.exp(1j*phi)*np.sin(theta)*np.sqrt(1.5/np.pi)/2)

f = lambda theta: Y(l, m , theta, phi)
derivative_numerically = (f(theta+delta)-f(theta))/delta
derivative_analytically = dYdtheta(l, m, theta, phi)

print('numerically', derivative_numerically)
print('analytically', derivative_analytically)
# assert np.isclose(derivative_analytically, derivative_numerically)
derivative_numerically = (Y(l, m, theta, phi+delta)-Y(l, m , theta, phi))/delta
derivative_analytically = dYdphi(l, m, theta, phi)

print('numerically', derivative_numerically)
print('analytically', derivative_analytically)

# assert np.isclose(derivative_numerically, derivative_analytically)
vp1= VectorPotential(*r, I)
vp2 = vp(2, *r, I)
print('vp', vp1)
print('vp2', vp2)
print(vp1/vp2)