import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import norm


def gen_normal(mu=5, sigma=0.1, num=10000):
    return np.random.normal(loc=mu, scale=sigma, size=num)

def gen_uniform(low=0, high=1.0, size=10000):
    return np.random.uniform(low=low, high=high, size=size)


mu = 20 
sigma=1
time_steps=60 
deacay=1.0
size=10000
bins=5

high = mu+3
low = mu-3
amp = (high-low)/2
ave = (high+low)/2
x = np.arange(0., 7*np.pi, (1/6))
vals = np.sin(x) * amp + ave

time_steps = int(len(x)) # 132
dists = np.zeros((size, int(time_steps))) # (10000, 132) initial array


for i in range(time_steps):
    #f is_sin == True:
    mu = vals[i]
    # Uniform/Normal split
    decay_per = (i+1)/time_steps
    num_uniform = int(decay_per*size)
    num_normal = size - num_uniform

    # Gernerate data
    norm_ = gen_normal(mu, sigma, num_normal)
    unif_bound = norm.ppf([0.01, 0.999], loc=mu, scale=sigma) # Set uniform distribution to 95th
    unif = gen_uniform(unif_bound[0], unif_bound[1], num_uniform)

    samples = list(norm_) + list(unif)
    dists[:,i] = samples # raw distributions (132) each containing 10,000 points 

#dists.shape

#test = np.array([[1, 2],
##                [4, 5], 
#                [7, 8]])
#test.shape #(2,3)
#test.sum(axis=0).shape #2 Sum ACROSS dimension 1


# Find minimum 1 percentile and maximum 99 percentile, used to create y range
_1, _99 = np.percentile(dists, [1, 99], axis=0)
min = _1.min()
max = _99.max()

num_bins = 100
bin_val = np.linspace(min, max, num_bins)


z_init = np.zeros((num_bins-1, time_steps)) # (99, 132), will be initial histogram (pre-bin aggregation)
#dists[4,:].shape

# For every distribution, create histogram, replate z_init
for i in range(z_init.shape[1]):
    hist = np.histogram(dists[:,i], bins=bin_val) # dists = (10000, 132)
    #hist = hist/num_bins
    z_init[:,i] = hist[0]/(len(samples))


#z_init[50]
z_final = np.zeros((num_bins-1, time_steps)) # (99, 132)

for i in range(z_init.shape[0]):
    if i <=20:
        z_final[i,:] = z_init[0:20, :].sum(axis=0)
    elif i > 20 & i <=40:
        z_final[i,:] = z_init[20:41, :].sum(axis=0)
    elif i > 40 & i <= 60:
        z_final[i,:] = z_init[40:61, :].sum(axis=0)
    elif i > 60 & i <=80:
        z_final[i,:] = z_init[60:81, :].sum(axis=0)
    elif i > 80:
        z_final[i,:] = z_init[80:101, :].sum(axis=0)


y_final = np.array(bin_val[:-1])



#z_final[0,:] = z_init[0:20, :].sum(axis=0)
#z_final[1,:] = z_init[20:41, :].sum(axis=0)
#z_final[2,:] = z_init[40:61, :].sum(axis=0)
#z_final[3,:] = z_init[60:81, :].sum(axis=0)
#z_final[4,:] = z_init[80:101, :].sum(axis=0)
# y_final = np.array([bin_val[20], bin_val[40], bin_val[60], bin_val[80], bin_val[99]])

y_final = np.array([bin_val[20], bin_val[40], bin_val[60], bin_val[80], bin_val[99]])

z_final[:,0].shape