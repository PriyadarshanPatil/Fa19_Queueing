# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 13:38:16 2019

@author: priya
"""

# -*- coding: utf-8 -*-
"""
Queueing_FIFO
"""


import numpy as np
from sklearn.utils import shuffle
s=0
num_iters = 30
mu_range = [0.1, 0.2, 0.5]
lambda_range = [0.1, 0.2, 0.5, 1]
theta_range = [0.0002, 0.0005, 0.001, 0.002]
pct1_range = [0.1, 0.2, 0.3, 0.5]
array_len = len(mu_range) * len(lambda_range) * len(theta_range) * len(pct1_range)
Avg_queue_length_results = np.zeros((num_iters, array_len))
Throughput_results = Avg_queue_length_results = np.zeros((num_iters, array_len))    
w=0

for s in range(num_iters):
    for a in range(len(mu_range)):
        for b in range(len(lambda_range)):
            for c in range(len(theta_range)):
                for d in range(len(pct1_range)):
                    
                    ###Input settings here###
                    t = 10000                   #Simulation time
                    mu = mu_range[a]                    #Service rate
                    lamda = lambda_range[b]                 #Arrival rate
                    theta = theta_range[c]              #Abandonment rate
                    c1_threshold = 3            #Class 1 liver quality threshold
                    c2_threshold = 15           #Class 2 liver quality threshold
                    c_threshold = np.array([c1_threshold, c2_threshold])
                    liver_lb = 1                #Liver quality lower bound
                    liver_ub = 37               #Liver quality upper bound
                    pct_c1 = pct1_range[d]                #Percentage of class 1 patients among random arrivals and in queue already
                    pct_c2 = 1-pct_c1
                    init_queue_length = 1000    #Number of patients already in the queue
    
                    ###Necessary variables and arrays defined here###
                    
                    """These lines below calculate the initial queue"""
                    np.random.seed(s+1)
                    temp1 = init_queue_length
                    temp2 = int(init_queue_length*pct_c1)
                    Init_queue = np.array([1] * temp2 + [2] * (temp1-temp2))
                    np.random.shuffle(Init_queue)
                    
                    #This is the time variable showing remaining lifetime
                    #We reinitialize this every timestep  
                    Init_queue_abandonment_time = np.array([0] * init_queue_length)
                    
                    for i in range(init_queue_length):
                        Init_queue_abandonment_time[i] = np.random.exponential(1/theta)
                    
                    num_new_arrivals_c1 = np.array([0] * t)
                    num_new_arrivals_c2 = np.array([0] * t)
                    num_abandonments = np.array([0] * t)
                    
                    ###Performance metric variables
                    Queue_lengths = np.array([0]*t)
                    System_throughput = 0
                    
                    
                    liver_timestep = np.array([0])
                    liver_quality = np.array([0])
                    
                    
                    ###The liver arrivals are documented here###
                    for i in range(t):
                        num_arrival = np.random.poisson(mu)
                        for j in range(num_arrival):
                            liver_timestep = np.append(liver_timestep,i+1)
                            liver_quality = np.append(liver_quality, np.random.randint(low = liver_lb, high = liver_ub))
                    ###These two arrays store when the livers arrive and 
                    liver_timestep = np.delete(liver_timestep,0)
                    liver_quality = np.delete(liver_quality,0)
                    
                    
                    '''SIRO'''
                    for i in range(t):
                        num_new_arrivals_c1[i] = np.random.poisson(lamda*pct_c1)
                        num_new_arrivals_c2[i] = np.random.poisson(lamda*pct_c2)
                        if (num_new_arrivals_c1[i] > 0):
                            temp1 = np.array([1] * num_new_arrivals_c1[i])
                            Init_queue = np.concatenate([Init_queue, temp1])
                            for k in range(num_new_arrivals_c1[i]):
                                Init_queue_abandonment_time = np.concatenate((Init_queue_abandonment_time, [int(np.random.exponential(1/theta))]))
                        if (num_new_arrivals_c2[i] > 0):
                            temp2 = np.array([2] * num_new_arrivals_c2[i])
                            Init_queue = np.concatenate([Init_queue, temp2])
                            for k in range(num_new_arrivals_c2[i]):
                                Init_queue_abandonment_time = np.concatenate((Init_queue_abandonment_time, [int(np.random.exponential(1/theta))]))
                        j=len(Init_queue)-1
                        while (len(liver_quality)>0  and liver_timestep[0]-1 == i):
                            if liver_quality[0] < min(c1_threshold, c2_threshold):
                                liver_quality = np.delete(liver_quality,0)
                                liver_timestep = np.delete(liver_timestep,0)
                                continue
                            if (j<0):
                                liver_quality = np.delete(liver_quality,0)
                                liver_timestep = np.delete(liver_timestep,0)
                                continue
                    
                    
                    
                            if (c_threshold[Init_queue[j]-1] > liver_quality[0]):
                                #print("Customer number is",j,"and threshold is ",c_threshold[Init_queue[j]-1],". Current liver quality is ",liver_quality[0],"Here")
                                j -= 1
                                pass
                            else:
                                #print("Customer number is",j,"and threshold is ",c_threshold[Init_queue[j]-1],". Current liver quality is ",liver_quality[0])
                                liver_quality = np.delete(liver_quality,0)
                                liver_timestep = np.delete(liver_timestep,0)
                                Init_queue = np.delete(Init_queue, j)
                                Init_queue_abandonment_time = np.delete(Init_queue_abandonment_time, j)
                                Init_queue, Init_queue_abandonment_time  = shuffle(Init_queue, Init_queue_abandonment_time)
                                System_throughput += 1
                                j -= 1
                                
                                
                        length_queue = len(Init_queue)
                        Queue_lengths[i] = length_queue
                        for k in range(length_queue):
                            if (Init_queue_abandonment_time[length_queue - k - 1] == 0):
                                Init_queue = np.delete(Init_queue, length_queue - k - 1)
                                Init_queue_abandonment_time = np.delete(Init_queue_abandonment_time, length_queue - k - 1)
                                num_abandonments[i] += 1
                            else:
                                Init_queue_abandonment_time[length_queue - k - 1] -= 1
                                
                    '''Performance metric generation'''
                    Avg_queue_length = np.average(Queue_lengths)
                    Avg_queue_length_results[s][w] = Avg_queue_length
                    Throughput_results[s][w] = System_throughput
                    print("Iteration ",w)
                    w += 1 


f = open("output_SIRO.txt", "a")
print("Average queue length results below", file=f)
print(Avg_queue_length_results, file=f)
print("Throughput results below",file=f)
print(Throughput_results, file=f)
f.close()

