# !/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
from scipy import optimize as opt
import os

'''
Este script calcula la constante de Hubble con los primeros datos
experimentales.
'''

def datos(nombre):
'''
Descarga los datos de un archivo y los retorna en columnas
'''
    file_path = os.path.join('data', 'nombre')
    D = np.loadtxt(fname=file_path, usecols=(0,))
    v = np.loadtxt(fname=file_path, usecols=(1,))
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
    return f_modelo_1(params, x)


def func_a_minimizar_2cf(v, H):
    '''
    Funcion a minimizar con curve_fit, modelo 2
    '''
    params = H
    return f_modelo_2(params, v)

def bootstrap(d, v, H_0):
    '''
    simulacion de bootstrap para encontrar el
    intervalo de  confianza (95%), retorna el valor promedio de H tanto
    para la funcion modelo 1 y 2 como para su promedio además de plotear
    histogramas de para los Hs
    '''
    N = len(d)
    N_boot = 10000
    H1 = np.zeros(N_boot)
    H2 = np.zeros(N_boot)
    H_prom = np.zeros(N_boot)
    for i in range(N_boot):
        s = np.random.randint(low=0, high=N, size=N)
        mean_values_d[i] = np.mean(d[s])
        mean_values_v[i] = np.mean(v[s])
        fake_data = data[s][s]
        distancia = fake_data[:, 0]
        vel = fake_data[:, 1]
        H_optimo_1, a_covarianza_1 = curve_fit(func_a_minimizar_1,
                                               distancia, vel, 2)
        H_optimo_2, a_covarianza_2 = curve_fit(func_a_minimizar_2,
                                               vel, distancia, 2)
        H_prom[i] = (H_optimo_2 + H_optimo_1) / 2
        H1[i] = H_optimo_1
        H2[i] = H_optimo_2
    H_prom_0 = np.mean(H_prom)
    H1_0 = np.mean(H1)
    H2_0 = np.mean(H2)
    fig2, ax2 = plt.subplots()
    plt.hist(H1, bins=30)
    plt.axvline(H1_0, color='r')
    plt.hist(H2, bins=30)
    plt.axvline(H2_0, color='r')
    plt.hist(H_prom, bins=30)
    plt.axvline(H_prom_0, color='r')
    ax2.set_title("Simulacion de bootstrap")
    ax2.set_xlabel("H [Km/s /Mpc]")
    ax2.set_ylabel("frecuencia")
    # plt.savefig("bootstrap_p1.jpg")
    H1 = np.sort(H1)
    H2 = np.sort(H2)
    Hprom = np.sort(H_prom)
    limite_bajo = H[int(N_boot * 0.025)]
    limite_alto = H[int(N_boot * 0.975)]
    print "El intervalo de confianza al 95% es: [{}:{}]".format(limite_bajo,
                                                                limite_alto)
    return [H_prom_0, H1_0, H2_0]


def aprox(distancia, vel, H1, H2, H_prom):
    '''
    recibe el arreglo de distancias ,velocidades y Hs, las grafica y compara
    con los datos de muestra
    '''
    ax, fig = plt.subplots()
    plt.scatter(distancia, vel, label="Datos originales")
    fig.plot(distancia, f_modelo_1(H1, distancia), label="modelo con V=D*H")
    fig.plot(f_modelo_2(H2, vel), vel, label="modelo con V/H = D")
    fig.plot(distancia, H_prom*distancia, label="Modelo con H promedio")
    fig.set_title("Datos originales y ajuste lineal")
    fig.set_xlabel("Distancia [Mpc]")
    fig.set_ylabel("Velocidad [Km/s]polyfit")
    plt.legend(loc=2)
    plt.savefig("hubble_1.jpg")
    plt.show()

# main

name = "data/hubble_original.dat"
y_scales = ?
d, v = datos(name)
H = bootstrap(d, v, H_0)
aprox(d, v, H[1], H[2], H[0])
