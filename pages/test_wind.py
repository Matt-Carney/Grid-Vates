import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm
from utils import DLR, temp_data, gen_normal, gen_uniform


import plotly.express as px
  


a = DLR(wind_speed=10, wind_angle=0, ambient_temp=20, eff_rad_heat_flux=1000)
#a.ampacity()


with st.sidebar:
    st.write("Wind Direction Variables")
deg = st.sidebar.slider('Degrees off Transmission Line', 0, 360, 45)

#deg = 40
rad = deg * (np.pi/180)
#kap = 1

np.random.seed(123)
vm = np.random.vonmises(mu=rad, kappa=4, size=100) # -pi to pi
vm_2pi = np.mod(vm, 2*np.pi) # 0 - 2 pi
vm_deg = vm_2pi * (180/np.pi) # degrees

pol_bins = np.linspace(0,360,17)
bin_val = np.digitize(vm_deg, bins= pol_bins)

pol_val = []
for i in bin_val:
    temp = pol_bins[i]
    pol_val.append(temp)


df_pol = pd.DataFrame({'wind_dir': pol_val})
df_pol = df_pol.groupby(['wind_dir']).size().reset_index().rename(columns={0: 'count'})
df_pol['Percent'] = df_pol['count']/df_pol['count'].sum()


df = px.data.wind()
fig = px.bar_polar(df_pol, r="Percent", theta="wind_dir",
                   color="Percent", template="plotly_dark", range_color=[0.0,1.0],
                   #color_discrete_sequence= px.colors.sequential.Plasma_r)
                    color_continuous_scale='turbo', title='Day 0')



#fig = go.Figure()
#fig.add_trace(go.Barpolar(
#    r=vm_deg))
st.plotly_chart(fig)

#s = np.random.vonmises(0, 400, 1000)
#plt.hist(vm_deg, 20, density=True)
#plt.show()

#df.head()