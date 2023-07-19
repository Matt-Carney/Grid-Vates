import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm
from utils import DLR, temp_data, gen_normal, gen_uniform


np.random.seed(123)
gen_normal(num=2)
gen_uniform(size=2)




def contour_data(mu = 20, sigma=1, time_fraction=(1/24), deacay=1.0, size=100000, bins=10, type = 'sine'):
    #x = np.arange(time_steps)
    #x = np.arange(0., 10*np.pi, time_fraction)    
    x = np.linspace(0., 10*3.14, 24*5)
    time_steps = int(len(x))


    dists = np.zeros((size, int(time_steps))) 


    if type == 'sine':
        high = mu+3
        low = mu-3
        amp = (high-low)/2
        ave = (high+low)/2
        vals = np.sin(x) * amp + ave

    #norm_init = gen_normal(mu, sigma, size)
    #unif_bound = norm.ppf([0.1, 0.99], loc=mu, scale=sigma)
    #unif_init = gen_uniform(unif_bound[0], unif_bound[1], size)

    nums = []
    for i in range(time_steps):
        #if type == 'sine':
        mu = vals[i]
        
        decay_per = (i)/time_steps
        num_uniform = int(decay_per*size)
        num_normal = size - num_uniform

        #if type == 'sine':
        np.random.seed(123)
        norm_0 = gen_normal(mu, sigma, num_normal)
        unif_bound = norm.ppf([0.001, 0.9999], loc=mu, scale=sigma)
        unif_0 = gen_uniform(unif_bound[0], unif_bound[1], num_uniform)
        
        norm_ = norm_0[0:num_normal]
        unif_ = unif_0[0:num_uniform]
        samples = list(norm_) + list(unif_)
        dists[:,i] = samples 
        #dists[:,i] = norm_init
        dists = np.array(dists)
        
        nums.append([num_normal, num_uniform])
        
    return dists, list(x), nums




d, t, n = contour_data()

t_new = t*1000
d_new = d.reshape(-1)
n

d[:,-1].mean()

d.shape

fig2 = go.Figure(go.Histogram2d(
        x = t_new,
        y = d_new,
        histnorm='percent',
        colorscale='turbo',
        autobinx=False,
        xbins= dict(size= .2),
        autobiny=False,
        ybins= dict(size = .2)
        ))
fig2.show()

len(t)

t

fig.clo

t_new[1]
d_new[0]
d_new[-1]
d[0:5,0:5]

d_new.shape
x = np.array([[1, 2, 3], [4, 5, 6]])
x.shape
d.shape
x.reshape(-1)
d.shape
t.shape



a = np.array(a)

type(a)

fig.show()