cimport cython
import numpy as np
cimport numpy as np
from cython.parallel import prange
from libc.math cimport sqrt, fabs, log, abs, copysign
BT_Z_LAYERS = np.array([     0.,   1293.,   2586.,   3879.,   5172.,   6465.,   7758.,
                          9051.,  10344.,  11637.,  12930.,  14223.,  15516.,  16809.,
                         18102.,  19395.,  20688.,  21981.,  23274.,  24567.,  25860.,
                         27153.,  28446.,  29739.,  31032.,  32325.,  33618.,  34911.,
                         36204.,  37497.,  38790.,  40083.,  41376.,  42669.,  43962.,
                         45255.,  46548.,  47841.,  49134.,  50427.,  51720.,  53013.,
                         54306.,  55599.,  56892.,  58185.,  59478.,  60771.,  62064.,
                         63357.,  64650.,  65943.,  67236.,  68529.,  69822.,  71115.,
                         72408.,  73701.])
cdef double DISTANCE = 1293.
cdef double EPS = 1e-6

@cython.linetrace(True)
@cython.binding(True)
@cython.wraparound(True)
@cython.boundscheck(True)
@cython.cdivision(True)
cpdef opera_distance_metric(double[:] basetrack_left, double[:] basetrack_right):
    cdef double dx, dy, dz, dtx, dty
    dz = basetrack_right[3] - basetrack_left[3]
    dx = basetrack_left[1] - (basetrack_right[1] - basetrack_right[4] * dz)
    dy = basetrack_left[2] - (basetrack_right[2] - basetrack_right[5] * dz)
    
    #dtx = basetrack_left[4] # * copysign(1.0, dz)
    dtx = (basetrack_left[4] - basetrack_right[4]) # * copysign(1.0, dz)
    
    #dty = basetrack_left[5] # * copysign(1.0, dz)
    dty = (basetrack_left[5] - basetrack_right[5]) # * copysign(1.0, dz)
    
    dz = DISTANCE
    
    cdef double a = (dtx * dz) ** 2 + (dty * dz) ** 2
    cdef double b = 2 * (dty * dz * dx +  dty * dz * dy)
    cdef double c = dx ** 2 + dy ** 2
    
    if a == 0.:
        return fabs(sqrt(c))
    
    cdef double discriminant = (b ** 2 - 4 * a * c)
    cdef double log_denominator = 2 * sqrt(a) * sqrt(fabs(a + b + c)) + 2 * a + b + EPS
    cdef double log_numerator = 2 * sqrt(a) * sqrt(c) + b
    cdef double first_part = ( (2 * a + b) * sqrt(fabs(a + b + c)) - b * sqrt(c) ) / (4 * a)
    
    if fabs(discriminant) < EPS:
        return fabs(first_part)
    else:
        result = fabs((discriminant * log(fabs(log_numerator / log_denominator)) / (8 * sqrt(a * a * a)) + first_part))
        return result