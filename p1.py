# !/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import (leastsq, curve_fit)
from scipy import optimize as opt
import os

'''
Este script calcula la constante de Hubble con los primeros datos
experimentales.
'''

def datos():
    '''
    Descarga los datos de un archivo y los retorna en columnas
    '''
    file_path = os.path.join('data', 'hubble_original.dat')
    D = np.loadtxt(fname=file_path, usecols=(-2,))
    v = np.loadtxt(fname=file_path, usecols=(-1,))
    return D, v


def f_modelo1(H, x):
    '''
    Modelo 1 en funcion de la distancia
    '''
    return H * x

def f_modelo2(H, v):
    '''
    Modelo 2 en funcion de la velocidad
    '''
    return v / H

def func_a_minimizar_1cf(x, H):
    '''
    Funcion a minimizar con curve_fit, modelo 1
    '''
    params = H
    return f_modelo1(params, x)


def func_a_minimizar_2cf(v, H):
    '''
    Funcion a minimizar con curve_fit, modelo 2
    '''
    params = H
    return f_modelo2(params, v)

def bootstrap(d, v):
    '''
    simulacion de bootstrap para encontrar el
    intervalo de  confianza (95%), retorna el valor promedio de H tanto
    para la funcion modelo 1 y 2 como para su promedio además de plotear
    histogramas de para los Hs
    '''
    N = len(d)
    N_boot = 5000
    H1 = np.zeros(N_boot)
    H2 = np.zeros(N_boot)
    H_prom = np.zeros(N_boot)
    mean_values_d = np.zeros(N_boot)
    mean_values_v = np.zeros(N_boot)
    for i in range(N_boot):
        s = np.random.randint(low=0, high=N, size=N)
        mean_values_d[i] = np.mean(d[s])
        mean_values_v[i] = np.mean(v[s])
        H_optimo_1, a_covarianza_1 = curve_fit(func_a_minimizar_1cf,
                                               d[s], v[s], 2)
        H_optimo_2, a_covarianza_2 = curve_fit(func_a_minimizar_2cf,
                                               v[s], d[s], 2)
        H_prom[i] = (H_optimo_2 + H_optimo_1) / 2
        H1[i] = H_optimo_1
        H2[i] = H_optimo_2
    H_prom_0 = np.mean(H_prom)
    H1_0 = np.mean(H1)
    H2_0 = np.mean(H2)
    Hprom = np.sort(H_prom)
    limite_bajo = Hprom[int(N_boot * 0.025)]
    limite_alto = Hprom[int(N_boot * 0.975)]
    fig2, ax2 = plt.subplots()
    plt.hist(H_prom, bins=30, facecolor='b')
    plt.axvline(H_prom_0, color='r')
    plt.axvline(limite_bajo, color='m')
    plt.axvline(limite_alto, color='m')
    ax2.set_title("Pregunta 1: Simulacion con Bootstrap")
    ax2.set_xlabel("H [Km/s /Mpc]")
    ax2.set_ylabel("frecuencia")
    plt.savefig("Histograma_p1.jpg")

    print "El intervalo de confianza al 95% es: [{}:{}]".format(limite_bajo,
                                                                limite_alto)
    return H_prom_0


def aprox(distancia, vel, H_prom):
    '''
    recibe el arreglo de distancias ,velocidades y Hs, las grafica y compara
    con los datos de muestra
    '''
    ax, fig = plt.subplots()
    plt.scatter(distancia, vel, label="Datos originales")
    fig.plot(distancia, H_prom*distancia, label="Modelo con H promedio")
    fig.set_title("Pregunta 1: Datos originales y ajuste lineal")
    fig.set_xlabel("Distancia [Mpc]")
    fig.set_ylabel("Velocidad [Km/s]polyfit")
    plt.legend(loc=2)
    plt.savefig("hubble_p1.jpg")
    plt.show()

# main


np.random.seed(43760)
d, v = datos()
H = bootstrap(d, v)
aprox(d, v, H)
plt.show()
