import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm
from utils import DLR, temp_data, gen_normal, gen_uniform, wind_direction_data

st.title('Probabilistic Weather - Sensitivity')
#st.markdown('This is some text')



### Sidebar Sliders
with st.sidebar:
    st.write("Wind Direction Variables")
dir_decay = st.sidebar.slider('Temporal Decay', 0.0, 1.0, 1.0)
dir_deg = st.sidebar.slider('Degrees From Transmission Line Axis', 0, 360, 45)
dir_kap = st.sidebar.slider('Kappa', 0.5, 10.0, 2.0)
dir_temp_decay = np.linspace(0.0, dir_decay, 5)












### Wind Direction
# Genearte Data
df_pol_0, d_0 = wind_direction_data(deg=dir_deg, kap=dir_kap, size=1000,perc_unif=dir_temp_decay[0])
df_pol_1, d_1 = wind_direction_data(deg=dir_deg, kap=dir_kap, size=1000,perc_unif=dir_temp_decay[1])
df_pol_2, d_2 = wind_direction_data(deg=dir_deg, kap=dir_kap, size=1000,perc_unif=dir_temp_decay[2])
df_pol_3, d_3 = wind_direction_data(deg=dir_deg, kap=dir_kap, size=1000,perc_unif=dir_temp_decay[3])
df_pol_4, d_4 = wind_direction_data(deg=dir_deg, kap=dir_kap, size=1000,perc_unif=dir_temp_decay[4])

x = np.linspace(1.5*np.pi, 11.5*3.14, 24*5)
d_dir = np.zeros((1000, int(len(x)))) # 1000, 120

# For DLR Calc
for i in range(d_dir.shape[1]):
    if i <24:
        d_dir[:,i] = d_0
    elif i >= 24 & i < 48:
        d_dir[:,i] = d_1
    elif i >= 48 & i < 72:
        d_dir[:,i] = d_2
    elif i >= 72 & i < 96:
        d_dir[:,i] = d_3
    elif i >= 96:
        d_dir[:,i] = d_4

# Figure
fig_wind = make_subplots(rows=1, cols=5, specs=[[{'type': 'polar'},
                                                {'type': 'polar'},
                                                {'type': 'polar'},
                                                {'type': 'polar'},
                                                {'type': 'polar'}]])

fig_wind.update_layout(height=280, width=900, title_text="Wind Direction")

max_step = df_pol_0['Percent'].max()+0.05
fig_wind.update_polars(radialaxis = dict(range=[0, max_step],nticks=5),
                       angularaxis = dict(direction='clockwise'))

fig_wind.add_trace(go.Barpolar(r=df_pol_0['Percent'], theta=df_pol_0['wind_dir'], name='Day 1', showlegend=True), row=1, col=1)
fig_wind.add_trace(go.Barpolar(r=df_pol_1['Percent'], theta=df_pol_1['wind_dir'], name='Day 2', showlegend=True), row=1, col=2)
fig_wind.add_trace(go.Barpolar(r=df_pol_2['Percent'], theta=df_pol_2['wind_dir'], name='Day 3', showlegend=True), row=1, col=3)
fig_wind.add_trace(go.Barpolar(r=df_pol_3['Percent'], theta=df_pol_3['wind_dir'], name='Day 4', showlegend=True), row=1, col=4)
fig_wind.add_trace(go.Barpolar(r=df_pol_4['Percent'], theta=df_pol_4['wind_dir'], name='Day 5', showlegend=True), row=1, col=5)


st.plotly_chart(fig_wind)