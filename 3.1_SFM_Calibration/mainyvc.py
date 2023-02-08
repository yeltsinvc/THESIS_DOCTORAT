# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 17:09:36 2019

@author: valero
"""
import numpy as np
from fooyvc import *

import matplotlib.pyplot as plt

#Car-Following models to evaluate
carFollowingModel=['Krauss', 'IDM', 'Wiedemann']

#Choose the model to evaluates
modelChoose=0 #0--> Krauss, 1-->IDM, 2-->Wiedemann

#Configuration of car-following parameters
if modelChoose==0:
    # PARAMETERS
    parameters = ['acell', 'decel', 'emergencyDecel','tau', 'maxSpeed','minGap']#parameters car-following
    lb = [1.5, 2, 5, 3, 5.4, 0.25] # minGap, accel, decel, emergencydecel, maxSpeed(pedestrian)
    ub = [1.2, 3, 7, 2, 20, 0.5] # minGap, accel, decel, emergencydecel, maxSpeed(bicycle)
elif modelChoose==1:
    # PARAMETERS
    parameters = ['acell', 'decel', 'emergencyDecel', 'tau', 'maxSpeed','minGap']#parameters car-following
    lb = [1.5, 2, 5, 5.4, 0.25] # minGap, accel, decel, emergencydecel, maxSpeed(pedestrian)
    ub = [1.2, 3, 7, 20, 0.5] # minGap, accel, decel, emergencydecel, maxSpeed(bicycle)
else:
    # PARAMETERS
    parameters = ['tau', 'security', 'maxSpeed','minGap']#parameters car-following
    lb = [1.5, 2, 5, 5.4, 0.25] # minGap, accel, decel, emergencydecel, maxSpeed(pedestrian)
    ub = [1.2, 3, 7, 20, 0.5] # minGap, accel, decel, emergencydecel, maxSpeed(bicycle)


# mean and covariance initialization
init_mu = np.add(ub,np.divide(np.subtract(lb,ub),2))
init_covmat = 1 * np.identity(len(init_mu))

init_covmat = np.multiply(np.abs(np.subtract(lb,ub)),init_covmat)


n_samples = 15  # number of sampled weights we generate
n_best_to_keep = 5  # We order policies based on results and then use n_best_to_keep best of them to estimate new param
absolute_winners_threshold = 1 # algorithm terminated when a certain number of winning policies found
initial_noise_coef = 1  # we always add constant_noise_coef * I to the estimated covmat (to increase variance)
noise_decay = 99 / 100


global_max = 245
absolute_winners = []  # Solutions
mu = init_mu
covmat = init_covmat
noise_coef = initial_noise_coef

counter = 0
###############XML
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
samples_results_values=[]

while len(absolute_winners) < absolute_winners_threshold:

    # 1) Sample
    samples = simulator(n_samples, mu, covmat)

    # 2) Evaluation of samples
    samples_result = {w: evaulate_sumo(w, parameters,carFollowingModel[modelChoose]) for w in samples}
    for w, res in samples_result.items():
        if res[0] == 0:
            absolute_winners.append(w)

    # 3) Pick the n_best_to_keep best policies
    local_winners = sorted(samples_result.keys(), key=lambda my_key: samples_result[my_key][0])[0:n_best_to_keep]

    # 4) Estimate new parameters and decay noise - in case we managed to find a better solution
    for temp in list(samples_result.values()):
        samples_results_values.append(temp[0])

    current_max = np.mean(sorted(samples_results_values)[0:n_best_to_keep])

    print('Current max is: ' + str(current_max))
    if global_max > current_max:
        global_max = current_max
        mu, covmat = estimator(local_winners, noise_coef)
        noise_coef *= noise_decay

    # 5) Counter and results
    counter += 1
    print('The algorithm finished its ' + str(counter) + ' iteration')
    print('So far, it has found ' + str(len(absolute_winners)) + ' winners')
    print('The noise_coef: ' + str(noise_coef))
    if counter>100:
        break
