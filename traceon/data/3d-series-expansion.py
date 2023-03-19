from math import sqrt, cos, sin, atan2
import os.path as path

import mpmath as mp
from mpmath import mpf

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.interpolate import CubicSpline
import numba as nb


INTEGRATION_FACTORS = np.array([[3.183098861837907e-1, 3.183098861837907e-1, 1.591549430918953e-1, 5.305164769729845e-2, 1.326291192432461e-2, 2.652582384864922e-3, 4.420970641441537e-4, 6.315672344916482e-5, 7.894590431145602e-6, 8.771767145717336e-7, 8.771767145717336e-8, 7.974333768833941e-9, 6.645278140694951e-10],
 [ -6.366197723675813e-1, -4.244131815783876e-1, -1.591549430918953e-1, -4.244131815783876e-2, -8.841941282883074e-3, -1.515761362779956e-3, -2.210485320720769e-4, -2.806965486629547e-5, -3.157836172458241e-6, -3.189733507533577e-7, -2.923922381905779e-8, -2.453641159641213e-9, -1.898650897341415e-10],
 [8.488263631567751e-1, 5.092958178940651e-1, 1.69765272631355e-1, 4.042030300746548e-2, 7.578806813899778e-3, 1.17892550438441e-3, 1.571900672512547e-4, 1.83728650033934e-5, 1.913840104520146e-6, 1.799336850403556e-7, 1.542288728917334e-8, 1.215136574298505e-9, 8.860370854259935e-11],
 [ -1.01859163578813e0, -5.820523633075029e-1, -1.818913635335947e-1, -4.042030300746548e-2, -7.073553026306459e-3, -1.02888044019003e-3, -1.286100550237538e-4, -1.413297307953339e-5, -1.3880598560256e-6, -1.233830983133867e-7, -1.002487673796267e-8, -7.50525531184371e-10, -5.211982855447021e-11],
 [ 1.164104726615006e0, 6.467248481194477e-1, 1.940174544358343e-1, 4.115521760760122e-2, 6.859202934600203e-3, 9.497357909446435e-4, 1.130637846362671e-4, 1.184477743808512e-5, 1.11044788482048e-6, 9.435178106317806e-8, 7.338471860469405e-9, 5.266845832872779e-10, 3.511230555248519e-11],
 [ -1.293449696238895e0, -7.055180161303066e-1, -2.057760880380061e-1, -4.221047959753971e-2, -6.783827078176025e-3, -9.045102770901367e-4, -1.036418025832448e-4, -1.045127421007511e-5, -9.435178106317806e-7, -7.724707221546742e-8, -5.793530416160056e-9, -4.012834920284022e-10, -2.584022486546529e-11],
 [ 1.411036032260613e0, 7.597886327557148e-1, 2.170824665016328e-1, 4.341649330032656e-2, 6.783827078176025e-3, 8.779070336463091e-4, 9.754522596070101e-5, 9.534495770594836e-6, 8.342683799270481e-7, 6.621177618468636e-8, 4.815401904340826e-9, 3.235645548371306e-10, 2.022278467732066e-11],
 [ -1.51957726551143e0, -8.104412082727624e-1, -2.279365898267144e-1, -4.469344898563028e-2, -6.828165817249071e-3, -8.625051558630405e-4, -9.343805855182939e-5, -8.898862719221847e-6, -7.584257999336801e-7, -5.862228405284484e-8, -4.15241178707651e-9, -2.717942260631897e-10, -1.65515714589763e-11],
 [ 1.620882416545525e0, 8.581142205241014e-1, 2.383650612566948e-1, 4.600027497936216e-2, 6.900041246904324e-3, 8.542908210452973e-4, 9.060660223207699e-5, 8.441608903609657e-6, 7.034674086341381e-7, 5.315087087457932e-8, 3.679675675932415e-9, 2.354001274165518e-10, 1.401191234622332e-11],
 [ -1.716228441048203e0, -9.032781268674752e-1, -2.484014848885557e-1, -4.731456855020108e-2, -6.989652172188796e-3, -8.509141774838534e-4, -8.86368934879014e-5, -8.103944547465271e-6, -6.623416216678346e-7, -4.906234234576553e-8, -3.329230373462661e-9, -2.087291770196026e-10, -1.217586865947682e-11], 
 [ 1.80655625373495e0, 9.462913710040216e-1, 2.580794648192786e-1, 4.862366728479162e-2, 7.090951479032112e-3, 8.509141774838534e-4, 8.727324897270292e-5, 7.849974775322484e-6, 6.308015444455568e-7, 4.592041894431256e-8, 3.061361262954171e-9, 1.88529579243512e-10, 1.080117381082621e-11],
 [ -1.892582742008043e0, -9.87434474091153e-1, -2.674301700663539e-1, -4.992029841238607e-2, -7.200043040247991e-3, -8.533384343997618e-4, -8.634972252854733e-5, -7.656625642925379e-6, -6.061495300649258e-7, -4.345157921612372e-8, -2.851509886058119e-9, -1.728187809732193e-10, -9.742235201921678e-12],
 [ 1.974868948182306e0, 1.026931853054799e0, 2.764816527455228e-1, 5.120030606398571e-2, 7.314329437712244e-3, 8.575420720076424e-4, 8.575420720076424e-5, 7.508432888546178e-6, 5.865963194176702e-7, 4.147650743357264e-8, 2.683774010407641e-9, 1.60329356465911e-10, 8.907186470328391e-12]])

INTEGRATION_FACTORS[:, 0] *= 1/2 # The integration formula needs to be adjusted for m=0, since the cos(m*phi) term in the integral vanishes.

SERIES_FACTORS = np.array([
[1.0e0, 1.0e0, 1.0e0, 1.0e0, 1.0e0, 1.0e0, 1.0e0, 1.0e0, 1.0e0, 1.0e0, 1.0e0, 1.0e0, 1.0e0],
[ -2.5e-1, -1.25e-1, -8.333333333333333e-2, -6.25e-2, -5.0e-2, -4.166666666666667e-2, -3.571428571428571e-2, -3.125e-2, -2.777777777777778e-2, -2.5e-2, -2.272727272727273e-2, -2.083333333333333e-2, -1.923076923076923e-2],
[ 1.5625e-2, 5.208333333333333e-3, 2.604166666666667e-3, 1.5625e-3, 1.041666666666667e-3, 7.44047619047619e-4, 5.580357142857143e-4, 4.340277777777778e-4, 3.472222222222222e-4, 2.840909090909091e-4, 2.367424242424242e-4, 2.003205128205128e-4, 1.717032967032967e-4],
[ -4.340277777777778e-4, -1.085069444444444e-4, -4.340277777777778e-5, -2.170138888888889e-5, -1.240079365079365e-5, -7.750496031746032e-6, -5.166997354497354e-6, -3.616898148148148e-6, -2.63047138047138e-6, -1.972853535353535e-6, -1.517579642579643e-6, -1.192384004884005e-6, -9.539072039072039e-7],
[ 6.781684027777778e-6, 1.356336805555556e-6, 4.521122685185185e-7, 1.937624007936508e-7, 9.68812003968254e-8, 5.382288910934744e-8, 3.229373346560847e-8, 2.055055765993266e-8, 1.370037177328844e-8, 9.484872766122766e-9, 6.774909118659119e-9, 4.968266687016687e-9, 3.726200015262515e-9],
[ -6.781684027777778e-8, -1.130280671296296e-8, -3.229373346560847e-9, -1.211015004960317e-9, -5.382288910934744e-10, -2.691144455467372e-10, -1.467896975709476e-10, -8.562732358305275e-11, -5.269373758957092e-11, -3.387454559329559e-11, -2.25830303955304e-11, -1.552583339692715e-11, -1.095941180959563e-11],
[ 4.709502797067901e-10, 6.72786113866843e-11, 1.681965284667108e-11, 5.606550948890359e-12, 2.242620379556143e-12, 1.019372899798247e-12, 5.096864498991235e-13, 2.744465499456819e-13, 1.568265999689611e-13, 9.409595998137665e-14, 5.880997498836041e-14, 3.805351322776262e-14, 2.536900881850841e-14],
[ -2.402807549524439e-12, -3.003509436905549e-13, -6.674465415345665e-14, -2.0023396246037e-14, -7.281234998558907e-15, -3.033847916066211e-15, -1.400237499722867e-15, -7.001187498614334e-16, -3.733966665927645e-16, -2.1003562495843e-16, -1.235503676226059e-16, -7.550300243603693e-17, -4.768610680170754e-17],
[ 9.385966990329841e-15, 1.04288522114776e-15, 2.08577044229552e-16, 5.688464842624146e-17, 1.896154947541382e-17, 7.292903644389931e-18, 3.125530133309971e-18, 1.458580728877986e-18, 7.292903644389931e-19, 3.860948988206434e-19, 2.144971660114686e-19, 1.241825697961134e-19, 7.450954187766803e-20],
[ -2.896903392077112e-17, -2.896903392077112e-18, -5.267097076503839e-19, -1.31677426912596e-19, -4.051613135772184e-20, -1.447004691347209e-20, -5.788018765388834e-21, -2.532258209857615e-21, -1.191650922285936e-21, -5.958254611429682e-22, -3.135923479699833e-22, -1.724757913834908e-22, -9.855759507628046e-23],
[ 7.242258480192779e-20, 6.583871345629799e-21, 1.0973118909383e-21, 2.532258209857615e-22, 7.235023456736043e-23, 2.411674485578681e-23, 9.043779320920054e-24, 3.723909132143551e-24, 1.655070725397134e-24, 7.839808699249582e-25, 3.919904349624791e-25, 2.053283230755843e-25, 1.119972671321369e-25],
[ -1.496334396734045e-22, -1.246945330611704e-23, -1.918377431710314e-24, -4.110808782236388e-25, -1.096215675263037e-25, -3.42567398519699e-26, -1.209061406540114e-26, -4.701905469878222e-27, -1.979749671527672e-27, -8.908873521874525e-28, -4.242320724702155e-28, -2.121160362351077e-28, -1.10669236296578e-28], 
[ 2.597802772107717e-25, 1.998309824698244e-26, 2.854728320997492e-27, 5.709456641994983e-28, 1.427364160498746e-28, 4.198129883819841e-29, 1.399376627939947e-29, 5.155598102936646e-30, 2.062239241174659e-30, 8.838168176462822e-31, 4.017349171119465e-31, 1.921340907926701e-31, 9.606704539633503e-32]])

assert INTEGRATION_FACTORS.shape == (13, 13)
assert SERIES_FACTORS.shape == (13,13)

NU_MAX = 4
M_MAX = 8

def directional_derivative(x, y, z, z0, alpha, N):
    x0, y0 = 0.0, 0.0 
    dx, dy, dz = x0-x, y0-y, z0-z 
    n_x, n_y, n_z = (sin(alpha)*dy+cos(alpha)*dx)/sqrt(dz**2+dy**2+dx**2),(sin(alpha)*dx-cos(alpha)*dy)/sqrt(dy**2+dx**2),-(sin(alpha)*dy*dz+cos(alpha)*dx*dz)/sqrt((dy**2+dx**2)*dz**2+dy**4+2*dx**2*dy**2+dx**4)
    r = sqrt(dx**2 + dy**2 + dz**2)
    
    if N == 0:
        return 1/(4*sqrt( (x0-x)**2 + (y0-y)**2 + (z0-z)**2))
    elif N == 1:
        return -n_x/(4*r**2)
    elif N == 2:
        return -(n_z**2+n_y**2-2*n_x**2)/(4*r**3)
    elif N == 3:
        return (3*n_x*(3*n_z**2+3*n_y**2-2*n_x**2))/(4*r**4)
    elif N == 4:
        return (3*(3*n_z**4+6*n_y**2*n_z**2-24*n_x**2*n_z**2+3*n_y**4-24*n_x**2*n_y**2+8*n_x**4))/(4*r**5)
    elif N == 5:
        return -(15*n_x*(15*n_z**4+30*n_y**2*n_z**2-40*n_x**2*n_z**2+15*n_y**4-40*n_x**2*n_y**2+8*n_x**4))/(4*r**6)
    elif N == 6:
        return -(45*(5*n_z**6+15*n_y**2*n_z**4-90*n_x**2*n_z**4+15*n_y**4*n_z**2-180*n_x**2*n_y**2*n_z**2+120*n_x**4*n_z**2+5*n_y**6-90*n_x**2*n_y**4+120*n_x**4*n_y**2-16*n_x**6))/(4*r**7)
    elif N == 7:
        return (315*n_x*(35*n_z**6+105*n_y**2*n_z**4-210*n_x**2*n_z**4+105*n_y**4*n_z**2-420*n_x**2*n_y**2*n_z**2+168*n_x**4*n_z**2+35*n_y**6-210*n_x**2*n_y**4+168*n_x**4*n_y**2-16*n_x**6))/(4*r**8)
    elif N == 8:
        return (315*(35*n_z**8+140*n_y**2*n_z**6-1120*n_x**2*n_z**6+210*n_y**4*n_z**4-3360*n_x**2*n_y**2*n_z**4+3360*n_x**4*n_z**4+140*n_y**6*n_z**2-3360*n_x**2*n_y**4*n_z**2+6720*n_x**4*n_y**2*n_z**2-1792*n_x**6*n_z**2+35*n_y**8-1120*n_x**2*n_y**6+3360*n_x**4*n_y**4-1792*n_x**6*n_y**2+128*n_x**8))/(4*r**9)
    elif N == 9:
        return -(2835*n_x*(315*n_z**8+1260*n_y**2*n_z**6-3360*n_x**2*n_z**6+1890*n_y**4*n_z**4-10080*n_x**2*n_y**2*n_z**4+6048*n_x**4*n_z**4+1260*n_y**6*n_z**2-10080*n_x**2*n_y**4*n_z**2+12096*n_x**4*n_y**2*n_z**2-2304*n_x**6*n_z**2+315*n_y**8-3360*n_x**2*n_y**6+6048*n_x**4*n_y**4-2304*n_x**6*n_y**2+128*n_x**8))/(4*r**10)
    elif N == 10:
        return -(14175*(63*n_z**10+315*n_y**2*n_z**8-3150*n_x**2*n_z**8+630*n_y**4*n_z**6-12600*n_x**2*n_y**2*n_z**6+16800*n_x**4*n_z**6+630*n_y**6*n_z**4-18900*n_x**2*n_y**4*n_z**4+50400*n_x**4*n_y**2*n_z**4-20160*n_x**6*n_z**4+315*n_y**8*n_z**2-12600*n_x**2*n_y**6*n_z**2+50400*n_x**4*n_y**4*n_z**2-40320*n_x**6*n_y**2*n_z**2+5760*n_x**8*n_z**2+63*n_y**10-3150*n_x**2*n_y**8+16800*n_x**4*n_y**6-20160*n_x**6*n_y**4+5760*n_x**8*n_y**2-256*n_x**10))/(4*r**11)
    elif N == 11:
        return (155925*n_x*(693*n_z**10+3465*n_y**2*n_z**8-11550*n_x**2*n_z**8+6930*n_y**4*n_z**6-46200*n_x**2*n_y**2*n_z**6+36960*n_x**4*n_z**6+6930*n_y**6*n_z**4-69300*n_x**2*n_y**4*n_z**4+110880*n_x**4*n_y**2*n_z**4-31680*n_x**6*n_z**4+3465*n_y**8*n_z**2-46200*n_x**2*n_y**6*n_z**2+110880*n_x**4*n_y**4*n_z**2-63360*n_x**6*n_y**2*n_z**2+7040*n_x**8*n_z**2+693*n_y**10-11550*n_x**2*n_y**8+36960*n_x**4*n_y**6-31680*n_x**6*n_y**4+7040*n_x**8*n_y**2-256*n_x**10))/(4*r**12)
    elif N == 12:
        return (467775*(231*n_z**12+1386*n_y**2*n_z**10-16632*n_x**2*n_z**10+3465*n_y**4*n_z**8-83160*n_x**2*n_y**2*n_z**8+138600*n_x**4*n_z**8+4620*n_y**6*n_z**6-166320*n_x**2*n_y**4*n_z**6+554400*n_x**4*n_y**2*n_z**6-295680*n_x**6*n_z**6+3465*n_y**8*n_z**4-166320*n_x**2*n_y**6*n_z**4+831600*n_x**4*n_y**4*n_z**4-887040*n_x**6*n_y**2*n_z**4+190080*n_x**8*n_z**4+1386*n_y**10*n_z**2-83160*n_x**2*n_y**8*n_z**2+554400*n_x**4*n_y**6*n_z**2-887040*n_x**6*n_y**4*n_z**2+380160*n_x**8*n_y**2*n_z**2-33792*n_x**10*n_z**2+231*n_y**12-16632*n_x**2*n_y**10+138600*n_x**4*n_y**8-295680*n_x**6*n_y**6+190080*n_x**8*n_y**4-33792*n_x**10*n_y**2+1024*n_x**12))/(4*r**13)
    elif N == 13:
        return -(6081075*n_x*(3003*n_z**12+18018*n_y**2*n_z**10-72072*n_x**2*n_z**10+45045*n_y**4*n_z**8-360360*n_x**2*n_y**2*n_z**8+360360*n_x**4*n_z**8+60060*n_y**6*n_z**6-720720*n_x**2*n_y**4*n_z**6+1441440*n_x**4*n_y**2*n_z**6-549120*n_x**6*n_z**6+45045*n_y**8*n_z**4-720720*n_x**2*n_y**6*n_z**4+2162160*n_x**4*n_y**4*n_z**4-1647360*n_x**6*n_y**2*n_z**4+274560*n_x**8*n_z**4+18018*n_y**10*n_z**2-360360*n_x**2*n_y**8*n_z**2+1441440*n_x**4*n_y**6*n_z**2-1647360*n_x**6*n_y**4*n_z**2+549120*n_x**8*n_y**2*n_z**2-39936*n_x**10*n_z**2+3003*n_y**12-72072*n_x**2*n_y**10+360360*n_x**4*n_y**8-549120*n_x**6*n_y**6+274560*n_x**8*n_y**4-39936*n_x**10*n_y**2+1024*n_x**12))/(4*r**14)
    elif N == 14:
        return -(42567525*(429*n_z**14+3003*n_y**2*n_z**12-42042*n_x**2*n_z**12+9009*n_y**4*n_z**10-252252*n_x**2*n_y**2*n_z**10+504504*n_x**4*n_z**10+15015*n_y**6*n_z**8-630630*n_x**2*n_y**4*n_z**8+2522520*n_x**4*n_y**2*n_z**8-1681680*n_x**6*n_z**8+15015*n_y**8*n_z**6-840840*n_x**2*n_y**6*n_z**6+5045040*n_x**4*n_y**4*n_z**6-6726720*n_x**6*n_y**2*n_z**6+1921920*n_x**8*n_z**6+9009*n_y**10*n_z**4-630630*n_x**2*n_y**8*n_z**4+5045040*n_x**4*n_y**6*n_z**4-10090080*n_x**6*n_y**4*n_z**4+5765760*n_x**8*n_y**2*n_z**4-768768*n_x**10*n_z**4+3003*n_y**12*n_z**2-252252*n_x**2*n_y**10*n_z**2+2522520*n_x**4*n_y**8*n_z**2-6726720*n_x**6*n_y**6*n_z**2+5765760*n_x**8*n_y**4*n_z**2-1537536*n_x**10*n_y**2*n_z**2+93184*n_x**12*n_z**2+429*n_y**14-42042*n_x**2*n_y**12+504504*n_x**4*n_y**10-1681680*n_x**6*n_y**8+1921920*n_x**8*n_y**6-768768*n_x**10*n_y**4+93184*n_x**12*n_y**2-2048*n_x**14))/(4*r**15)
    elif N == 15:
        return (638512875*n_x*(6435*n_z**14+45045*n_y**2*n_z**12-210210*n_x**2*n_z**12+135135*n_y**4*n_z**10-1261260*n_x**2*n_y**2*n_z**10+1513512*n_x**4*n_z**10+225225*n_y**6*n_z**8-3153150*n_x**2*n_y**4*n_z**8+7567560*n_x**4*n_y**2*n_z**8-3603600*n_x**6*n_z**8+225225*n_y**8*n_z**6-4204200*n_x**2*n_y**6*n_z**6+15135120*n_x**4*n_y**4*n_z**6-14414400*n_x**6*n_y**2*n_z**6+3203200*n_x**8*n_z**6+135135*n_y**10*n_z**4-3153150*n_x**2*n_y**8*n_z**4+15135120*n_x**4*n_y**6*n_z**4-21621600*n_x**6*n_y**4*n_z**4+9609600*n_x**8*n_y**2*n_z**4-1048320*n_x**10*n_z**4+45045*n_y**12*n_z**2-1261260*n_x**2*n_y**10*n_z**2+7567560*n_x**4*n_y**8*n_z**2-14414400*n_x**6*n_y**6*n_z**2+9609600*n_x**8*n_y**4*n_z**2-2096640*n_x**10*n_y**2*n_z**2+107520*n_x**12*n_z**2+6435*n_y**14-210210*n_x**2*n_y**12+1513512*n_x**4*n_y**10-3603600*n_x**6*n_y**8+3203200*n_x**8*n_y**6-1048320*n_x**10*n_y**4+107520*n_x**12*n_y**2-2048*n_x**14))/(4*r**16)
    elif N == 16:
        return (638512875*(6435*n_z**16+51480*n_y**2*n_z**14-823680*n_x**2*n_z**14+180180*n_y**4*n_z**12-5765760*n_x**2*n_y**2*n_z**12+13453440*n_x**4*n_z**12+360360*n_y**6*n_z**10-17297280*n_x**2*n_y**4*n_z**10+80720640*n_x**4*n_y**2*n_z**10-64576512*n_x**6*n_z**10+450450*n_y**8*n_z**8-28828800*n_x**2*n_y**6*n_z**8+201801600*n_x**4*n_y**4*n_z**8-322882560*n_x**6*n_y**2*n_z**8+115315200*n_x**8*n_z**8+360360*n_y**10*n_z**6-28828800*n_x**2*n_y**8*n_z**6+269068800*n_x**4*n_y**6*n_z**6-645765120*n_x**6*n_y**4*n_z**6+461260800*n_x**8*n_y**2*n_z**6-82001920*n_x**10*n_z**6+180180*n_y**12*n_z**4-17297280*n_x**2*n_y**10*n_z**4+201801600*n_x**4*n_y**8*n_z**4-645765120*n_x**6*n_y**6*n_z**4+691891200*n_x**8*n_y**4*n_z**4-246005760*n_x**10*n_y**2*n_z**4+22364160*n_x**12*n_z**4+51480*n_y**14*n_z**2-5765760*n_x**2*n_y**12*n_z**2+80720640*n_x**4*n_y**10*n_z**2-322882560*n_x**6*n_y**8*n_z**2+461260800*n_x**8*n_y**6*n_z**2-246005760*n_x**10*n_y**4*n_z**2+44728320*n_x**12*n_y**2*n_z**2-1966080*n_x**14*n_z**2+6435*n_y**16-823680*n_x**2*n_y**14+13453440*n_x**4*n_y**12-64576512*n_x**6*n_y**10+115315200*n_x**8*n_y**8-82001920*n_x**10*n_y**6+22364160*n_x**12*n_y**4-1966080*n_x**14*n_y**2+32768*n_x**16))/(4*r**17)
    elif N == 17:
        return -(10854718875*n_x*(109395*n_z**16+875160*n_y**2*n_z**14-4667520*n_x**2*n_z**14+3063060*n_y**4*n_z**12-32672640*n_x**2*n_y**2*n_z**12+45741696*n_x**4*n_z**12+6126120*n_y**6*n_z**10-98017920*n_x**2*n_y**4*n_z**10+274450176*n_x**4*n_y**2*n_z**10-156828672*n_x**6*n_z**10+7657650*n_y**8*n_z**8-163363200*n_x**2*n_y**6*n_z**8+686125440*n_x**4*n_y**4*n_z**8-784143360*n_x**6*n_y**2*n_z**8+217817600*n_x**8*n_z**8+6126120*n_y**10*n_z**6-163363200*n_x**2*n_y**8*n_z**6+914833920*n_x**4*n_y**6*n_z**6-1568286720*n_x**6*n_y**4*n_z**6+871270400*n_x**8*n_y**2*n_z**6-126730240*n_x**10*n_z**6+3063060*n_y**12*n_z**4-98017920*n_x**2*n_y**10*n_z**4+686125440*n_x**4*n_y**8*n_z**4-1568286720*n_x**6*n_y**6*n_z**4+1306905600*n_x**8*n_y**4*n_z**4-380190720*n_x**10*n_y**2*n_z**4+29245440*n_x**12*n_z**4+875160*n_y**14*n_z**2-32672640*n_x**2*n_y**12*n_z**2+274450176*n_x**4*n_y**10*n_z**2-784143360*n_x**6*n_y**8*n_z**2+871270400*n_x**8*n_y**6*n_z**2-380190720*n_x**10*n_y**4*n_z**2+58490880*n_x**12*n_y**2*n_z**2-2228224*n_x**14*n_z**2+109395*n_y**16-4667520*n_x**2*n_y**14+45741696*n_x**4*n_y**12-156828672*n_x**6*n_y**10+217817600*n_x**8*n_y**8-126730240*n_x**10*n_y**6+29245440*n_x**12*n_y**4-2228224*n_x**14*n_y**2+32768*n_x**16))/(4*r**18)
    
    raise ValueError()

mp.mp.dps = 50

def Cacc(x, y, z, z0, nu, m):
    x = mpf(x); y = mpf(y); z = mpf(z); z0 = mpf(z0);
    return float(mp.quad(lambda alpha: cos(m*alpha)*directional_derivative(x, y, z, z0, alpha, 2*nu+m), [0, 2*mp.pi]))

def Sacc(x, y, z, z0, nu, m):
    x = mpf(x); y = mpf(y); z = mpf(z); z0 = mpf(z0);
    return float(mp.quad(lambda alpha: sin(m*alpha)*directional_derivative(x, y, z, z0, alpha, 2*nu+m), [0, 2*mp.pi]))

dd_numba = nb.njit(fastmath=True)(directional_derivative)
EPS = np.finfo(np.float64).eps

def C(x, y, z, z0, nu, m):
    return quad(lambda alpha: cos(m*alpha)*dd_numba(x, y, z, z0, alpha, 2*nu+m), 0, 2*np.pi, epsabs=-1, epsrel=51*EPS)[0]

def S(x, y, z, z0, nu, m):
    return quad(lambda alpha: sin(m*alpha)*dd_numba(x, y, z, z0, alpha, 2*nu+m), 0, 2*np.pi, epsabs=-1, epsrel=51*EPS)[0]


def axial_coefficients(x, y, z, z0, accurate=False):
    
    A_coeffs = np.zeros( (NU_MAX, M_MAX) )
    B_coeffs = np.zeros( (NU_MAX, M_MAX) )

    Cint = Cacc if accurate else C
    Sint = Sacc if accurate else S
    
    for nu in range(NU_MAX):
        for m in range(M_MAX):
            A_coeffs[nu, m] = SERIES_FACTORS[nu, m] * INTEGRATION_FACTORS[nu, m] * Cint(x, y, z, z0, nu, m)
            B_coeffs[nu, m] = SERIES_FACTORS[nu, m] * INTEGRATION_FACTORS[nu, m] * Sint(x, y, z, z0, nu, m)

    return A_coeffs, B_coeffs

theta = np.linspace(-np.pi/2, np.pi/2, 1000)
A_coeff = np.zeros( (len(theta), NU_MAX, M_MAX) )

for i, t in enumerate(theta):
    print(i)
    x_, z_ = cos(t), sin(t)
    A_coeff[i], _ = axial_coefficients(x_, 0.0, z_, 0.0)

theta_interpolation = CubicSpline(theta, A_coeff)
thetas_interpolation_coefficients = theta_interpolation.c
thetas_interpolation_coefficients = np.moveaxis(thetas_interpolation_coefficients, 0, -1)
print(thetas_interpolation_coefficients.shape)

# Save...

np.save('radial-series-3D-thetas.npy', theta)
np.save('radial-series-3D-theta-dependent-coefficients.npy',thetas_interpolation_coefficients)

exit(1)

'''
dir_ = path.dirname(__file__)
data = path.join(dir_, 'traceon', 'data')
thetas_file = path.join(data, 'radial-series-3D-thetas.npy')
coefficients_file = path.join(data, 'radial-series-3D-theta-dependent-coefficients.npy')

thetas = np.load(thetas_file)
radial_coefficients = np.load(coefficients_file)

theta_interpolation = CubicSpline(theta, radial_coefficients)
'''


def compute_interpolated_potential(A, B, xp, yp):
    r = sqrt(xp**2 + yp**2)
    phi = atan2(yp, xp)
    
    sum_ = 0.0
    
    for nu in range(NU_MAX):
        for m in range(M_MAX):
            #sum_ += SERIES_FACTORS[nu, m] * (A[nu, m]*cos(m*phi) + B[nu, m]*sin(m*phi)) * r**(m+2*nu)
            sum_ += (A[nu, m]*cos(m*phi) + B[nu, m]*sin(m*phi)) * r**(m+2*nu)
        
    return sum_

MIN_DISTANCE_AXIS = 1e-5

def interpolated_coefficients(zp, zs, coeffs):
    dz = zs[1]-zs[0]
    index = int((zp-zs[0])/dz)
     
    z_ = zp-zs[index]
    A, B = z_**3*coeffs[0,index] + z_**2*coeffs[1,index] + z_*coeffs[2,index] + coeffs[3,index]
    #Adiff, Bdiff = 3*z_**2*coeffs[0, index] + 2*z_*coeffs[1, index] + coeffs[2, index]

    return A, B
     

def compute_interpolated_field(point, zs, coeffs):
    xp, yp, zp = point[0], point[1], point[2]
     
    if not (zs[0] <= zp < zs[-1]):
        return np.array([0.0, 0.0, 0.0])
    
    dz = zs[1]-zs[0]
    index = int((zp-zs[0])/dz)
     
    z_ = zp-zs[index]
    A, B = z_**3*coeffs[0,index] + z_**2*coeffs[1,index] + z_*coeffs[2,index] + coeffs[3,index]
    Adiff, Bdiff = 3*z_**2*coeffs[0, index] + 2*z_*coeffs[1, index] + coeffs[2, index]

    print('Coefficients used: ')
     
    Ex = 0.0
    Ey = 0.0
    Ez = 0.0
    
    r = sqrt(xp**2 + yp**2)
    phi = atan2(yp, xp)
    
    if r < MIN_DISTANCE_AXIS:
        return np.array([-A[0, 1], -B[0, 1], -Adiff[0, 0]])
     
    for nu in range(NU_MAX):
        for m in range(M_MAX):
            exp = 2*nu + m
             
            diff_r = (A[nu, m]*cos(m*phi) + B[nu, m]*sin(m*phi)) * exp*r**(exp-1)
            diff_theta = m *(-A[nu, m]*sin(m*phi) + B[nu, m]*cos(m*phi)) * r**exp
             
            Ex -= diff_r * xp/r + diff_theta * -yp/r**2
            Ey -= diff_r * yp/r + diff_theta *  xp/r**2
            Ez -= (Adiff[nu, m]*cos(m*phi) + Bdiff[nu, m]*sin(m*phi)) * r**exp
     
    return np.array([Ex, Ey, Ez])
 


nu_map, m_map = np.meshgrid( np.arange(NU_MAX), np.arange(M_MAX), indexing='ij')

def axial_coefficients_interpolated(x, y, z, z0):
    r = sqrt(x**2 + y**2 + (z-z0)**2) 
    theta = atan2((z-z0), sqrt(x**2 + y**2))
    mu = atan2(y, x)
     
    C = theta_interpolation(theta)
    
    A_factor = np.cos(m_map*mu) * r**(-2*nu_map - m_map - 1)
    B_factor = np.sin(m_map*mu) * r**(-2*nu_map - m_map - 1)
     
    return np.array([C*A_factor, C*B_factor])

def field(x, y, z, x0, y0, z0):
    return np.array([
        -(x-x0)/(4*((z-z0)**2+(y-y0)**2+(x-x0)**2)**(3/2)),
        -(y-y0)/(4*((z-z0)**2+(y-y0)**2+(x-x0)**2)**(3/2)),
        -(z-z0)/(4*((z-z0)**2+(y-y0)**2+(x-x0)**2)**(3/2))])

def potential(x, y, z, x0, y0, z0):
    return 1/(4*sqrt( (x-x0)**2 + (y-y0)**2 + (z-z0)**2 ))


### Validation for paper
D = 10
points = np.array([
    [D, D, D],
    [-D, D, D],
    [-D, -D, D],
    [D, -D, D],
    [D, D, -D],
    [-D, D, -D],
    [-D, -D, -D],
    [D, -D, -D]])

z0 = 0.0
coeffs_int = np.sum([axial_coefficients_interpolated(*p, 0.0) for p in points], axis=0)
coeffs_exact = np.sum([axial_coefficients(*p, 0.0) for p in points], axis=0)

angle = np.linspace(0, 2*np.pi, 400)

def potential_circle(r, coeffs):
    circle_x = r*np.cos(angle)
    circle_y = r*np.sin(angle)

    pot = np.array([compute_interpolated_potential(coeffs[0], coeffs[1], x_, y_) for x_, y_ in zip(circle_x, circle_y)])
    return pot

plt.figure()

errors_int = []
errors_exact = []

for r in [1.0, 2.0, 3.0]:
    pot_int = potential_circle(r, coeffs_int)
    pot_exact = potential_circle(r, coeffs_exact)

    pot_formula = np.array([sum(potential(*p, x_, y_, z0) for p in points) for x_, y_ in zip(r*np.cos(angle), r*np.sin(angle))])
    
    pzero = 0.1154700538379251
    plt.plot(angle, pot_formula - pzero, label=f'Formula (r={r/D:.1f}D)')

    errors_int.append( np.abs( (pot_int-pot_formula)/pzero ) )
    errors_exact.append( np.abs( (pot_exact-pot_formula)/pzero ) )

   
    #plt.plot(angle, pot_int, linestyle='dashed', color='orange')

#plt.plot([],[], label='Interpolated', linestyle='dashed', color='orange')

plt.xlabel('Angle (rad)')
plt.ylabel('$\phi - \phi_0$ (1/m)')
plt.legend(loc='upper right')
plt.savefig('validation-pot.png')
plt.figure()

for r, ei, ee in zip([1.0, 2.0, 3.0], errors_int, errors_exact):
    #(p,) = plt.plot(angle, ee, label='Integrated, (r={r/D:.1f}D)')
    plt.plot(angle, ei, label=f'Interpolated, (r={r/D:.1f}D)')#,  linestyle='dashed')
 
plt.legend(loc='upper right')
plt.ylabel('Relative error $(\phi - \phi_{correct})/\phi_0$')
plt.xlabel('Angle (rad)')
plt.yscale('log')
plt.savefig('validation-error.png')

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='black')
for r in [1.0, 2.0, 3.0]:
    ax.plot(r*np.cos(angle), r*np.sin(angle), 0.0)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()
exit(1)
 




#### Generate random points

N_p = 25

z = np.linspace(-10, 10, 100)

x_0 = 0.0
y_0 = 0.0
z_0 = z[6] - (z[6]-z[5])/20

points = np.random.uniform(-5.0, 5.0, size=(N_p, 3))

mask = [ p[0]**2 + p[1]**2 > 4**2*(x_0**2 + y_0**2) for p in points ]
points = points[mask]
print(f'{len(points)} far enough')

coeffs = np.array([np.sum([axial_coefficients_interpolated(*p, z_) for p in points], axis=0) for z_ in z])
coeffs = np.array([np.sum([axial_coefficients(*p, z_) for p in points], axis=0) for z_ in z])
interpolated_z = CubicSpline(z, coeffs).c


#mask = coeffs_exact != 0.0
#print('Error: ', coeffs[mask]/coeffs_exact[mask] - 1)
#print('Max error: ', np.max(coeffs[mask]/coeffs_exact[mask] - 1))

field_exact = np.sum([field(*p, x_0, y_0, z_0) for p in points], axis=0)
field_interpolated = compute_interpolated_field(np.array([x_0, y_0, z_0]), z, interpolated_z)

#potential_exact = np.sum([potential(*p, x_0, y_0, z_0) for p in points], axis=0)
#potential_interpolated = compute_interpolated_potential(np.array([x_0, y_0, z_0]), z, interpolated_z)

#print(potential_interpolated/potential_exact - 1)
print(field_interpolated/field_exact - 1)
