# !/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import (leastsq, curve_fit)
from scipy import optimize as opt
import os

'''
Este script encuentreala línea recta que mejor modela la relación entre el
flujo en la banda i y la banda z
'''

def datos():
    '''
    Descarga los datos de un archivo y los retorna en columnas multiplicados
    por 3.631 que es la equivalencia para tener los datos en Jy
    '''
    file_path = os.path.join('data', 'SNIa.dat')
    band_i = 3.631 * np.loadtxt(fname=file_path, usecols=(80,))
    err_i = 3.631 * np.loadtxt(fname=file_path, usecols=(81,))
    band_z = 3.631 * np.loadtxt(fname=file_path, usecols=(82,))
    err_z = 3.631 * np.loadtxt(fname=file_path, usecols=(83,))
    return [band_i, err_i, band_z, err_z]
