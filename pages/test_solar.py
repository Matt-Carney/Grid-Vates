import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm
from utils import DLR, temp_data, gen_normal, gen_uniform


x = np.linspace(1.5*np.pi, 11.5*3.14, 24*5)

irr_perc_day = [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.39, 0.45, 0.59, 0.75, 0.88, 0.97,\
                1.00, 0.98, 0.90, 0.77, 0.62, 0.47, 0.39, 0.00, 0.00, 0.00, 0.00, 0.00]

irr_perc = np.array(irr_perc_day*5)

# Solar Irradiance
with st.sidebar:
    st.write("Solar Irradiance Variables")
irr = st.sidebar.slider('Heat Flux (W)', 0, 2000, 1000)
perc = st.sidebar.slider('Percent Cloud Cover', 0.0, 1.0, 0.5)



#irr = 1500
#perc = 1.0 # Percent clouds
red = 1.0 * (1.0-(.75* perc))
#1500*red*.39
#irr_ = irr_perc * irr * red
#[irr_perc[7]*irr*red]*100

size = 1000
dists = np.zeros((size, int(len(x))))

#1500*.25*.39

#unif_low = .25 * irr
#unif_high = 1 * irr
#np.random.seed(123)
#unif = np.random.uniform(low=unif_low, high=unif_high, size=size)

for i, val in enumerate(irr_perc):
    if val != 0.00:
        ideal = [val * irr * red]
        #ideals = np.random.normal(ideal, scale=10, size=1000)
        ideals = 1000 * ideal

        decay_per = (i)/int(len(x))
        num_uniform = int(decay_per*size)
        num_ideal = size - num_uniform
        
        ideal_ = np.random.choice(ideals, num_ideal, replace=True)
        
        unif_low = val*irr*.25
        unif_high = val*irr*1
        np.random.seed(123)
        #unif = np.random.uniform(low=unif_low, high=unif_high, size=size)
        unif = np.random.normal(loc=ideal, scale=100, size=size)
        unif_ = np.random.choice(unif, num_uniform, replace=True)

        #samples = unif
        #samples = np.concatenate(ideal_, unif_)
        samples = list(ideal_) + list(unif_) #Clean this up
        samples = np.array(samples)
        samples[samples<0] = 0.0
        dists[:,i] = samples 

t = list(x)*1000
dists = dists.reshape(-1)


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


import plotly.express as px
fig2 = px.histogram(x=samples,nbins=10)
st.plotly_chart(fig2)