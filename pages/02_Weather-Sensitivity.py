import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm
from utils import DLR, temp_data, gen_normal, gen_uniform, wind_direction_data, solar_irr_data, velocity_temp__data

st.title('Weather - Sensitivity')
st.markdown('Experiemnt with the weather forecast variables on the left and observe how it changes the DLR forecast!')

with st.expander("Here's what's going on..."):
    st.markdown("For each of the weather forecasts (Temperature, Solar Irradiance, Wind Velocity, Wind Direction) there are\
            two groups of variables to experiment with - temporal decay and the initial distribution parameters.\
            At t = 0, 1,000 random samples are generated from an initial distribution determined by the initial distriution parameters.\
            Additionally, at t = 0 an end-state distribution of 1,000 is also generated. For every time step n\
            an aggregate distribution is collated by seleting (n * temporal decay) samples of the end-state distribution and\
            ((1000-n) * temporal decay) of the initial dribution. **Thus, a temporal decay of 0.0 will persist the initial distibution through the time space\
            and a temporal decay of 1.0 will linearly transition completly from the initial distribution at t=0 to the end-state distribution\
            at the end of day 5 (t=120).** A temporal decay of 0.5 will linearly transition from the initial at t=0 to 50% of the end state\
            distribution at the end of day 5 (t=120). And so on...")
    st.markdown("The initial and end state distributions are as follows:")
    st.markdown("**Temperature** : Normal(mean, std) -> Uniform(1%, 99% of Normal(mean, std))")
    st.markdown("**Solar Irradiance** : Heat Flux is deterministic since it can b e calculated by the Earth's sphericity and orbital pattern.\
                The maximum reduction of Solar Irradiance with 100% cloud cover is 0.75, thus the end-state distribution is Normal(ideal*cloud cover*0.75)")
    st.markdown("**Wind Velocity** : Normal(mean, std) -> Uniform(1%, 99% of Normal(mean, std))")
    st.markdown("**Wind Direction** : vonMises(mean, Kappa) -> Uniform(1%, 99% of vonMises(mean, Kappa))\
                Note that a vonMises distribution is a circular normal distribution")
    st.markdown("**Wind Direction** : vonMises(mean, Kappa) -> Uniform(1%, 99% of vonMises(mean, Kappa))\
                Note that a vonMises distribution is a circular normal distribution")
    st.markdown("Finally, for each time step t, the DLR is determined by randomly sampling a temperature, solar irradiance,\
                wind velocity and wind direction value 1,000 times and calculated using the IEEE 738-2006 methodology.")

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



### Sidebar Sliders
# Temperature
with st.sidebar:
    st.write("Temperature Variables")
temp_decay = st.sidebar.slider('Temperature Temporal Decay', 0.0, 1.0, 1.0)
temp_mean = st.sidebar.slider('Temperature (C)', 1, 50, 15)
temp_sd = st.sidebar.slider('Temperature Init. Standard Deviation', 0.1, 5.0, 2.0)

# Solar Irradiance
with st.sidebar:
    st.write("Solar Irradiance Variables")
sol_decay = st.sidebar.slider('Solar Temporal Decay', 0.0, 1.0, 1.0)
sol_irr = st.sidebar.slider('Heat Flux (W/m^2)', 0, 2000, 1000)
sol_perc = st.sidebar.slider('Percent Cloud Cover', 0.0, 1.0, 0.5)

# Wind Velocity
with st.sidebar:
    st.write("Wind Velocity Variables")
vel_decay = st.sidebar.slider('Wind Velocity Temporal Decay', 0.0, 1.0, 1.0)
vel_mean = st.sidebar.slider('Wind Velocity (m/s)', 1, 20, 9)
vel_sd = st.sidebar.slider('Wind Velocity Init. Standard Deviation', 0.1, 5.0, 2.0)

# Wind Direction
with st.sidebar:
    st.write("Wind Direction Variables")
dir_decay = st.sidebar.slider('Wind Dir. Temporal Decay', 0.0, 1.0, 1.0)
dir_deg = st.sidebar.slider('Degrees From Transmission Line Axis', 0, 360, 45)
dir_kap = st.sidebar.slider('Kappa', 0.5, 10.0, 2.0)


### Temperature
# Generate Data
d_temp, t = velocity_temp__data(mu=temp_mean, sigma=temp_sd, decay=temp_decay, type='temp')
t_new = t*1000
d_temp_new = d_temp.reshape(-1)

# Plot
fig_temp = go.Figure(go.Histogram2d(
        x = t_new,
        y = d_temp_new,
        histnorm='percent',
        colorscale='turbo',
        autobinx=False,
        xbins= dict(size= .264),
        autobiny=False,
        ybins= dict(size = .264)
        ))


fig_temp.update_xaxes(tickangle=-90,
                  tickvals = np.linspace(1.5*np.pi, 11.5*3.14, 11),
                  ticktext = t_text
                  )
fig_temp.update_layout(height=400, width=900, title_text="Temperature",
                       xaxis_title='Time',
                    yaxis_title='Temperature (C)')




### Solar Irradiance
# Generate Data
d_sol, t = solar_irr_data(sol_irr=sol_irr, sol_perc=sol_perc, decay=sol_decay)
d_sol_new = d_sol.reshape(-1)

# Plot
my_colorsc=[[0.0, 'rgb(231, 236, 239)'],
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

fig_sol = go.Figure(go.Histogram2d(
        x = t,
        y = d_sol_new,
        histnorm='percent',
        colorscale='turbo',
        autobinx=False,
        xbins= dict(size= .264),
        autobiny=False,
        ybins= dict(size = 40)
        ))

fig_sol.update_layout(height=400, width=900, title_text="Solar Irradiance",
                      xaxis_title='Time',
                    yaxis_title='Heat Flux (W/m^2)',)

fig_sol.update_xaxes(tickangle=-90,
                  tickvals = np.linspace(1.5*np.pi, 11.5*3.14, 11),
                  ticktext = t_text
                  )



### Wind Velocity
# Generate Data
d_vel, t = velocity_temp__data(mu=vel_mean, sigma=vel_sd, decay=vel_decay, type='vel')
t_new = t*1000
d_vel_new = d_vel.reshape(-1)

# Plot
fig_vel = go.Figure(go.Histogram2d(
        x = t_new,
        y = d_vel_new,
        histnorm='percent',
        colorscale='turbo',
        autobinx=False,
        xbins= dict(size= .264),
        autobiny=False,
        ybins= dict(size = .264)
        ))


fig_vel.update_layout(height=400, width=900, title_text="Wind Velocity",
                                        xaxis_title='Time',
                  yaxis_title='Velocity (m/s)')

fig_vel.update_xaxes(tickangle=-90,
                  tickvals = np.linspace(1.5*np.pi, 11.5*3.14, 11),
                  ticktext = t_text)



### Wind Direction
# Genearte Data
dir_temp_decay = np.linspace(0.0, dir_decay, 5)
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

d_dir_new = d_dir.reshape(-1)

# Plot
fig_wind = make_subplots(rows=1, cols=5, specs=[[{'type': 'polar'},
                                                {'type': 'polar'},
                                                {'type': 'polar'},
                                                {'type': 'polar'},
                                                {'type': 'polar'}]])

fig_wind.update_layout(height=280, width=900, title_text="Wind Direction",
                       xaxis_title='Time',
                    yaxis_title='Degrees',)

max_step = df_pol_0['Percent'].max()+0.05
fig_wind.update_polars(radialaxis = dict(range=[0, max_step],nticks=5),
                       angularaxis = dict(direction='clockwise'))

fig_wind.add_trace(go.Barpolar(r=df_pol_0['Percent'], theta=df_pol_0['wind_dir'], name='Day 1', showlegend=True), row=1, col=1)
fig_wind.add_trace(go.Barpolar(r=df_pol_1['Percent'], theta=df_pol_1['wind_dir'], name='Day 2', showlegend=True), row=1, col=2)
fig_wind.add_trace(go.Barpolar(r=df_pol_2['Percent'], theta=df_pol_2['wind_dir'], name='Day 3', showlegend=True), row=1, col=3)
fig_wind.add_trace(go.Barpolar(r=df_pol_3['Percent'], theta=df_pol_3['wind_dir'], name='Day 4', showlegend=True), row=1, col=4)
fig_wind.add_trace(go.Barpolar(r=df_pol_4['Percent'], theta=df_pol_4['wind_dir'], name='Day 5', showlegend=True), row=1, col=5)



num_rand = 1000

DLR_data = np.zeros((1000,120))

for i in range(d_vel.shape[1]):
    DLR_list = []
    for j in range(num_rand):
        rand_vel = np.random.choice(d_vel[:,i])
        rand_dir = np.random.choice(d_dir[:,i])
        rand_temp = np.random.choice(d_temp[:,i])
        rand_sol = np.random.choice(d_sol[:,i])
        #a = DLR(wind_speed=5, wind_angle=90, ambient_temp=20, eff_rad_heat_flux=1000)
        #d.append(a.ampacity())
        _dlr = DLR(wind_speed=rand_vel, wind_angle=rand_dir, ambient_temp=rand_temp, eff_rad_heat_flux=rand_sol)
        DLR_temp = _dlr.ampacity()
        DLR_list.append(DLR_temp)
    DLR_data[:,i] = DLR_list


DLR_data_new = DLR_data.reshape(-1)

# Plot
fig_DLR = go.Figure(go.Histogram2d(
        x = t_new,
        y = DLR_data_new,
        histnorm='percent',
        colorscale='turbo',
        autobinx=False,
        xbins= dict(size= .264),
        #autobiny=False,
        #ybins= dict(size = .264)
        ))


fig_DLR.update_xaxes(tickangle=-90,
                  tickvals = np.linspace(1.5*np.pi, 11.5*3.14, 11),
                  ticktext = t_text, 
                  )
fig_DLR.update_layout(height=400, width=900, title_text="DLR Forecast",
                    xaxis_title='Time',
                    yaxis_title='Rating (Amps)',)

# Plot All Figures
st.plotly_chart(fig_DLR)
hide = """
<style>
ul.streamlit-expander {
    border: 0 !important;
</style>
"""

st.markdown(hide, unsafe_allow_html=True)
with st.expander("Expand for Temperature"):
    st.plotly_chart(fig_temp)
with st.expander("Expand for Solar Irradiance"):
    st.plotly_chart(fig_sol)
with st.expander("Expand for Wind Velocity"):
    st.plotly_chart(fig_vel)
with st.expander("Expand for Wind Direction"):
    st.plotly_chart(fig_wind)

