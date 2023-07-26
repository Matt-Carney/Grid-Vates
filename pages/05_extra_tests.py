import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm
from utils import DLR, temp_data, gen_normal, gen_uniform

def contour_data(mu = 20, sigma=1, decay=1.0, size=1000):
    x = np.linspace(1.5*np.pi, 11.5*3.14, 24*5)
    time_steps = int(len(x))


    dists = np.zeros((size, int(time_steps))) 

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
        num_uniform = int(decay_per*size*decay)
        num_normal = size - num_uniform

        #if type == 'sine':
        np.random.seed(123)
        norm_0 = gen_normal(mu, sigma, num_normal)
        norm_0 = np.abs(norm_0)
        unif_bound = norm.ppf([0.001, 0.9999], loc=mu, scale=sigma)
        unif_0 = gen_uniform(max(unif_bound[0], 0), unif_bound[1], num_uniform)
        
        #if rand == 'index':
        norm_ = norm_0[0:num_normal]
        unif_ = unif_0[0:num_uniform]
        #elif rand == 'choice':
        #    norm_ = np.random.choice(norm_0, num_normal, replace=False)
        #    unif_ = np.random.choice(unif_0, num_uniform, replace=False)
        samples = list(norm_) + list(unif_)
        dists[:,i] = samples 
        #dists[:,i] = norm_init
        dists = np.array(dists)
        
        nums.append([num_normal, num_uniform])
        
    #dists[dists<=0] = 0.0
    return dists, list(x)


with st.sidebar:
    st.write("Wind Velocity Variables (m/s)")
vel_decay = st.sidebar.slider('Wind SPeed Temporal Decay', 0.0, 1.0, 1.0)
vel_mean = st.sidebar.slider('Velocity', 1, 20, 9)
vel_sd = st.sidebar.slider('Velocity Init. Standard Deviation', 0.1, 5.0, 2.0)
#sz = st.sidebar.slider('size', 100, 10000, 1000)

d, t = contour_data(mu=vel_mean, sigma=vel_sd, decay=vel_decay)

t_new = t*1000
d_new = d.reshape(-1)


fig2 = go.Figure(go.Histogram2d(
        x = t_new,
        y = d_new,
        histnorm='percent',
        colorscale='turbo',
        autobinx=False,
        xbins= dict(size= .264),
        autobiny=False,
        ybins= dict(size = .264)
        ))

t_text = ['Day 0 00:00',
          'Day 0 12:00',
          'Day 1 00:00',
          'Day 1 12:00',
          'Day 2 00:00',
          'Day 2 12:00',
          'Day 3 00:00',
          'Day 3 12:00',
          'Day 4 00:00',
          'Day 4 12:00',
          'Day 5 00:00']

fig2.update_xaxes(tickangle=-90,
                  tickvals = np.linspace(1.5*np.pi, 11.5*3.14, 11),
                  ticktext = t_text
                  )

st.plotly_chart(fig2)

#d2, t2, n2 = contour_data(mu=vel_mean, sigma=vel_sd, size=sz, rand='choice')

#t_new2 = t2*1000
#d_new2 = d2.reshape(-1)


#fig3 = go.Figure(go.Histogram2d(
#        x = t_new2,
#        y = d_new2,
#        histnorm='percent',
#        colorscale='turbo',
#        autobinx=False,
#        xbins= dict(size= .264),
#        autobiny=False,
#        ybins= dict(size = .264)
#        ))

#fig3.update_xaxes(tickangle=-90,
#                  tickvals = np.linspace(1.5*np.pi, 11.5*3.14, 11),
#                  ticktext = t_text
#                  )


#agree = st.checkbox('Tempeature')

#if agree:
#    st.plotly_chart(fig3)


















#a = gen_normal(num=sz)

#st.write(np.std(a))
#import plotly.express as px
#df = px.data.tips()
#fig = px.histogram(x=a, histnorm='probability density')
#st.plotly_chart(fig)