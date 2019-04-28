#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

def Vector(*args):
    """Convenience function to create numpy vector to use with gtsam
    cython wrapper.

    Example:
        Vector(1)
        Vector(1,2,3)
        Vector(3,2,4)
    """
    ret = np.squeeze(np.asarray(args, dtype='float'))
    if ret.ndim == 0:
        ret = np.expand_dims(ret, axis=0)
    return ret

def Matrix(*args):
    """Convenience function to create numpy matrix to use with gtsam
    cython wrapper

    Example:
        Matrix([1])
        Matrix([1,2,3])
        Matrix((3,2,4))
        Matrix([1,2,3],[2,3,4])
        Matrix((1,2,3),(2,3,4))
    """
    ret = np.asarray(args, dtype='float', order='F')
    if ret.ndim == 1:
        ret = np.expand_dims(ret, axis=0)
    return ret
