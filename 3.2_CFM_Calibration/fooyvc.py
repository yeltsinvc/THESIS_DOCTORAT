# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 17:16:55 2019

@author: valero
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 16:10:59 2019

@author: valero
"""

import numpy as np
from xml.etree import ElementTree as ET
import run_stage as rs


def evaulate_sumo(w, parameters,carFollowingModelChoise):
    """It evaluates the mean squared error of trajectory"""
    #read the .rou.xml --> Configuration of car-following
    tree = ET.parse('input_routes.rou.xml')
    root = tree.getroot()

    #Change car-following parameters of e-scooters
    for i in root.iter('vType'):
        if i.get('id') == 'bicycle':
            iter=0
            i.set('carFollowModel', carFollowingModelChoise)
            for parametersCar in parameters:
                i.set(parametersCar, str(w[iter]))
                iter+=1
    # Write modifications
    tree.write('input_routes_mod.rou.xml')

    # Run SUMO with the new car-following parameters x, y --> position
    x,y=rs.runStage()

    #Insert trajectory of the cameras
    prov=len(x)
    x_data = np.ones(prov)
    y_data = np.ones(prov)

    #Mean squared error of trajectory
    error = np.mean(np.power(np.add(np.power(x-x_data,2),np.power(y-y_data,2)),0.5))

    #Return error, and trajectory
    return error, x, y


def estimator(w_list, noise_coef):
    """ It estiamted the mean vector and covariance matrix based on the list of collected w
    :param w_list: list of weights that we will use to form our estimates
    :type w_list: list of tuples
    :param noise_coef: to estimated covariance matrix we add a noise_coef * identity matrix to increase variance
    :type noise_coef: float
    :return: sample estimate of mean vector (6,) and covariance matrix (6,6)
    :rtype: pair of ndarrays
    """
    w_list_ndarray = np.array(w_list)
    mu_hat = np.mean(w_list_ndarray, axis=0)
    covmat_hat = np.cov(np.transpose(w_list_ndarray)) + noise_coef * np.eye(6, 6)  # ADD SOME CONSTANT TO AVOID
    return mu_hat, covmat_hat


def simulator(n, mu, covmat):
    """ Sampling n samples from multivariate normal with mean vector mu and covariance matrix covmat
    :param n: number of samples to generate
    :type n: int
    :param mu: mean vector of n elements
    :type mu: ndarray
    :param covmat: (n,n) ndarray - covariance matrix
    :type covmat: ndarray
    :return: samples of multivariate normal
    :rtype: list of tuples
    """

    a=np.ndarray(shape=(1,len(mu)))
    k=0

    while k<n:
        a_temp = np.array(np.random.multivariate_normal(mu, covmat, 1))
        if (a_temp[0]>0).all():
            a=np.concatenate((a,a_temp))
            k+=1

    return [tuple(a[i, :]) for i in range(n)]
