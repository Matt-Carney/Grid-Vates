import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm
from utils import DLR, temp_data, gen_normal, gen_uniform




# Solar Irradiance
with st.sidebar:
    st.write("Solar Irradiance Variables")
sol_decay = st.sidebar.slider('Solar Temporal Decay', 0.0, 1.0, 1.0)
sol_irr = st.sidebar.slider('Heat Flux (W)', 0, 2000, 1000)
sol_perc = st.sidebar.slider('Percent Cloud Cover', 0.0, 1.0, 0.5)


def solar_irr_data(sol_irr = 1000, sol_perc = 0.5, decay = 1.0):
    x = np.linspace(1.5*np.pi, 11.5*3.14, 24*5)
    size = 1000
    dists = np.zeros((size, int(len(x))))

    irr_perc_day = [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.39, 0.45, 0.59, 0.75, 0.88, 0.97,\
                    1.00, 0.98, 0.90, 0.77, 0.62, 0.47, 0.39, 0.00, 0.00, 0.00, 0.00, 0.00]

    irr_perc = np.array(irr_perc_day*5)

    red = 1.0 * (1.0-(.75* sol_perc))

    for i, val in enumerate(irr_perc):
        if val != 0.00:
            ideal = [val * sol_irr * red]
            #ideals = np.random.normal(ideal, scale=10, size=1000)
            ideals = 1000 * ideal

            decay_per = (i)/int(len(x))
            num_uniform = int(decay_per * size * decay)
            num_ideal = size - num_uniform
            
            ideal_ = np.random.choice(ideals, num_ideal, replace=True)
            
            unif_low = val*sol_irr*.25
            unif_high = val*sol_irr*1
            np.random.seed(123)
            #unif = np.random.uniform(low=unif_low, high=unif_high, size=size)
            unif = np.random.normal(loc=ideal, scale=100, size=size)
            np.random.seed(123)
            unif_ = np.random.choice(unif, num_uniform, replace=True)

            #samples = unif
            #samples = np.concatenate(ideal_, unif_)
            samples = list(ideal_) + list(unif_) #Clean this up
            samples = np.array(samples)
            samples[samples<0] = 0.0
            dists[:,i] = samples 

    t = list(x)*1000


    return dists, t

d_sol, t = solar_irr_data(sol_irr=sol_irr, sol_perc=sol_perc, decay=sol_decay)
dists = d_sol.reshape(-1)

my_colorsc=[[0.0, 'rgb(231, 236, 239)'],
            #[0.1, 'rgb(60, 90, 205)'],
            [0.1, 'rgb(70, 107, 227)'],
            [0.2, 'rgb(62, 155, 254)'],
            [0.3, 'rgb(24, 214, 203)'], 
            [0.4, 'rgb(70, 247, 131)'],
            [0.5, 'rgb(162, 252, 60)'],
            [0.6, 'rgb(225, 220, 55)'], 
            [0.7, 'rgb(253, 165, 49)'],
            [0.8, 'rgb(239, 90, 17)'], 
            [0.9, 'rgb(196, 37, 2)'],
            [1, 'rgb(122, 4, 2)']]



fig = go.Figure(go.Histogram2d(
        x = t,
        y = dists,
        histnorm='percent',
        colorscale='turbo',
        autobinx=False,
        xbins= dict(size= .264),
        autobiny=False,
        ybins= dict(size = 40)
        ))

st.plotly_chart(fig)
