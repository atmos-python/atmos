# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 17:31:42 2015

@author: mcgibbon
"""
import numpy as np
cimport numpy as np
cimport cython

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

@cython.boundscheck(False)
def AH_from_q_rho(np.ndarray[DTYPE_t, ndim=1] q,
                  np.ndarray[DTYPE_t, ndim=1] rho):
    '''
    Calculates absolute humidity (kg/m^3) from specific humidity (kg/kg) and
    air density (kg/m^3).

    AH = q*rho
    '''
    #cdef tuple q.shape
    #cdef tuple rho.shape
    #if q.shape != rho.shape:
    #    raise ValueError('q and rho must be same shape')
    #if len(q.shape) > 0:
    #    raise ValueError('q and rho must be 1D arrays')
    assert q.dtype == DTYPE and rho.dtype == DTYPE
    cdef np.ndarray[DTYPE_t, ndim=1] return_array = np.empty(q.shape[0],
                                                             dtype=DTYPE)
    cdef int i
    for i in range(q.shape[0]):
        return_array[i] = q[i]*rho[i]
    return return_array
