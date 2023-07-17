import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm

st.title('Probabilistic Weather - Sensitivity')
st.markdown('This is some text2')

def gen_normal(mu=5, sigma=0.1, num=1000):
    return np.random.normal(loc=mu, scale=sigma, size=num)

def gen_uniform(low=0, high=1.0, size=1000):
    return np.random.uniform(low=low, high=high, size=size)


mu = 20
sigma = 1
size=1000

x = np.arange(0., 5*np.pi, (1/24))
high = mu+3
low = mu-3
amp = (high-low)/2
ave = (high+low)/2
vals = np.sin(x) * amp + ave

len(vals)
time_steps = int(len(x))


df = pd.DataFrame(columns=['x', 'y'])

for i in range(time_steps):
        decay_per = (i+1)/time_steps
        num_uniform = int(decay_per*size)
        num_normal = size - num_uniform

        # Gernerate data
        mu = vals[i]
        norm_ = gen_normal(mu, sigma, num_normal)
        unif_bound = norm.ppf([0.001, 0.9999], loc=mu, scale=sigma) # Set uniform distribution to 95th
        unif = gen_uniform(unif_bound[0], unif_bound[1], num_uniform)

        samples = list(norm_) + list(unif)
        t_val = x[i]
        t = [t_val for j in range(size)]

        df_temp = pd.DataFrame({'x': t, 'y': samples})

        df = pd.concat([df, df_temp])    


#x_bins = {'size': 20}



fig = go.Figure(go.Histogram2dContour(
        x = df['x'],
        y = df['y'],
        histnorm='percent',
        colorscale = 'turbo',
        #autobinx=False,
        #xbins= dict(size= 0.3),
        #autobiny=False,
        #ybins= dict(size = 0.3)
        ))

#fig.update_layout(xaxis_range=[1,1.2])
st.plotly_chart(fig)