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
    file_path = os.path.join('data', 'DR9Q.dat')
    band_i = 3.631 * np.loadtxt(fname=file_path, usecols=(80,))
    err_i = 3.631 * np.loadtxt(fname=file_path, usecols=(81,))
    band_z = 3.631 * np.loadtxt(fname=file_path, usecols=(82,))
    err_z = 3.631 * np.loadtxt(fname=file_path, usecols=(83,))
    return [band_i, err_i, band_z, err_z]


def mostrar_datos(banda_i, error_i, banda_z, error_z, c):
    '''grafica los datos originales con sus errores asociados,
    grafica el ajuste lineal polyfit'''
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.errorbar(banda_i, banda_z, xerr=error_i, yerr=error_z, fmt="o",
                 color='c', label="Muestra con error")
    x = np.linspace(-100, 500, 600)
    ax1.plot(x, c[1] + x*c[0], color="r", label="Ajuste lineal")
    ax1.set_title("Muestra de Banda y Ajuste lineal")
    ax1.set_xlabel("Flujo banda i [$10^{-6}Jy$]")
    ax1.set_ylabel("Flujo banda z [$10^{-6}Jy$]")
    plt.legend(loc=2)
    plt.savefig("bandasiz.jpg")


def MonteCarlo(banda_i, error_i, banda_z, error_z, c_0):
    '''realiza una simulación de montecarlo para obtener el intervalo de
    confianza del 95%'''
    Nmc = 10000
    C = np.zeros(Nmc)
    m = np.zeros(Nmc)
    for j in range(Nmc):
        r = np.random.normal(0, 1, size=len(banda_i))
        muestra_i = banda_i + error_i * r
        muestra_z = banda_z + error_z * r
        m[j], C[j] = np.polyfit(muestra_i, muestra_z, 1)
    m = np.sort(m)
    C = np.sort(C)
    limite_bajo_1 = m[int(Nmc * 0.025)]
    limite_alto_1 = m[int(Nmc * 0.975)]
    limite_bajo_2 = C[int(Nmc * 0.025)]
    limite_alto_2 = C[int(Nmc * 0.975)]
    print """El intervalo de confianza al
             95% para la pendiente es: [{}:{}]""".format(limite_bajo_1,
                                                         limite_alto_1)
    print """El intervalo de confianza al
             95% para el coef de posicion es: [{}:{}]""".format(limite_bajo_2,
                                                                limite_alto_2)
bi, erri, bz, errz = datos()
c = np.polyfit(bi, bz, 1)
print("recta : {}x + {}".format(c[0], c[1]))
intervalo_confianza = MonteCarlo(bi, erri, bz, errz, c)
mostrar_datos(bi, erri, bz, errz, c)
plt.show()
