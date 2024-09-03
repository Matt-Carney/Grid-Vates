import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm
from utils import DLR, plot_fig,generate_synthetic_temp_truncated, generate_synthetic_wind_direction,generate_synthetic_solar_irradiance, generate_DLR, process_plot_timestep_stats

st.title('Weather - Sensitivity')
st.markdown('Experiemnt with the weather forecast variables on the left and observe how it changes the DLR forecast!')

with st.expander("Here's what's going on..."):
    st.markdown("For each of the weather forecasts (Temperature, Solar Irradiance, Wind Velocity, Wind Direction) there are\
            two groups of variables to experiment with - temporal decay and the initial distribution parameters.\
            At t = 0, 1,000 random samples are generated from an initial distribution determined by the initial distribution parameters.\
            Additionally, at t = 0 an end-state distribution of 1,000 samples is also generated. For every time step n,\
            an aggregate distribution is collated by selecting (n * temporal decay) samples of the end-state distribution and\
            ((1000-n) * temporal decay) of the initial dribution. **Thus, a temporal decay of 0.0 will persist the initial distribution through the time space\
            and a temporal decay of 1.0 will linearly transition completely from the initial distribution at t=0 to the end-state distribution\
            at the end of day 5 (t=120).** A temporal decay of 0.5 will linearly transition from the initial at t=0 to 50% of the end state\
            distribution at the end of day 5 (t=120). And so on...")
    st.markdown("The initial and end state distributions are as follows:")
    st.markdown("**Temperature** : Normal(mean, std) -> Uniform(1%, 99% of Normal(mean, std))")
    st.markdown("**Solar Irradiance** : Heat Flux is deterministic since it can b e calculated by the Earth's sphericity and orbital pattern.\
                The maximum reduction of Solar Irradiance with 100% cloud cover is 0.75, thus the end-state distribution is Normal(ideal x cloud cover x 0.75)")
    st.markdown("**Wind Velocity** : Normal(mean, std) -> Uniform(1%, 99% of Normal(mean, std))")
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
temp_mean = st.sidebar.slider('Temperature (C)', 1, 50, 15)
temp_init_std_dev = st.sidebar.slider('Temperature Initial Standard Deviation', 0.1, 5.0, 1.0)
temp_uncertainty_growth = st.sidebar.slider('Temperature Uncertainty Growth', 0.0, 0.05, 0.01)

# Solar Irradiance
with st.sidebar:
    st.write("Solar Irradiance Variables")
sol_irr = st.sidebar.slider('Heat Flux (W/m^2)', 0, 2000, 1000)
sol_cloud_cover = st.sidebar.slider('Solar Initial Cloud Cover', 0.0, 1.0, 0.5)
sol_cloud_cover_std_dev = st.sidebar.slider('Cloud Cover Initial Standard Deviation', 0.1, 5.0, 1.0)
sol_uncertainty_growth = st.sidebar.slider('Solar Uncertainty Growth', 0.0, 1.0, 0.01)
#sol_perc = st.sidebar.slider('Percent Cloud Cover', 0.0, 1.0, 0.5)

# Wind Velocity
with st.sidebar:
    st.write("Wind Velocity Variables")
vel_mean = st.sidebar.slider('Wind Velocity (m/s)', 1, 20, 9)
vel_init_std_dev = st.sidebar.slider('Wind Velocity Init. Standard Deviation', 0.1, 5.0, 1.0)
#vel_decay = st.sidebar.slider('Wind Velocity Temporal Decay', 0.0, 1.0, 1.0)
vel_uncertainty_growth = st.sidebar.slider('Wind Velocity Uncertainty Growth', 0.0, 0.05, 0.01)

# Wind Direction
with st.sidebar:
    st.write("Wind Direction Variables")
#dir_decay = st.sidebar.slider('Wind Dir. Temporal Decay', 0.0, 1.0, 1.0)
#dir_deg = st.sidebar.slider('Degrees From Transmission Line Axis', 0, 360, 45)
dir_init = st.sidebar.slider('Wind Direction (deg)', 0.0, 360.0, 90.0)
dir_kappa = st.sidebar.slider('Kappa', 0.5, 20.0, 15.0)
dir_shift_prob = st.sidebar.slider('Shift Probability', 0.0, 1.0, 0.1)
dir_max_shift = st.sidebar.slider('Max Shift (deg)', 0.0, 180.0, 15.0)

# Wind Direction
with st.sidebar:
    st.write("DLR Stats Fig")
time_step = st.sidebar.slider("Select Time Step", 0, 119, 0)
early_stopping = st.sidebar.checkbox("Enable Early Stopping", value=True)

### Temperature
x_temp, y_temp = generate_synthetic_temp_truncated(mean=temp_mean,
                                         initial_std_dev=temp_init_std_dev, 
                                         uncertainty_growth=temp_uncertainty_growth,
                                         type='temp')

fig_temp = plot_fig(x=x_temp, y=y_temp, title = 'Temperature', y_axis_title = 'Temperature (C)')


# ### Solar Irradiance
x_sol, y_sol = generate_synthetic_solar_irradiance(sol_irr=sol_irr,
                                                   cloud_cover=sol_cloud_cover,
                                                   cloud_cover_init_std=sol_cloud_cover_std_dev,
                                                   cloud_cover_uncertainty_growth=sol_uncertainty_growth)  


fig_sol = plot_fig(x=x_sol, y=y_sol, title = 'Solar Irradiance',
                   y_axis_title = 'Solar Irradiance (W/m^2)')


### Wind Velocity
x_vel, y_vel = generate_synthetic_temp_truncated(mean=vel_mean,
                                         initial_std_dev=vel_init_std_dev, 
                                         uncertainty_growth=vel_uncertainty_growth,
                                         type='vel')

fig_vel = plot_fig(x=x_vel, y=y_vel, title = 'Wind Velocity', y_axis_title = 'Wind Velocity (m/s)')



### Wind Direction
x_dir, y_dir = generate_synthetic_wind_direction(start_direction=dir_init,kappa = dir_kappa, shift_prob = dir_shift_prob, max_shift = dir_max_shift)
fig_dir = plot_fig(x=x_dir, y=y_dir, title = 'Wind Direction', y_axis_title = 'Wind Direction (m/s)')



### DLR
x_DLR, y_DLR = generate_DLR(y_temp=y_temp, y_sol=y_sol, y_vel=y_vel, y_dir=y_dir, n_samples=1000, n_steps=120)
fig_DLR = plot_fig(x=x_DLR, y=y_DLR, title = 'DLR', y_axis_title = 'Rating (Amps)')
# num_rand = 1000

### DLR Monte Carlo
fig_mc, stop_index = process_plot_timestep_stats(y_DLR, tolerance=0.01, consecutive_stable=50, time_step=time_step, early_stopping=early_stopping)



# DLR_data = np.zeros((1000,120))

# for i in range(y_vel.shape[1]):
#     DLR_list = []
#     for j in range(num_rand):
#         rand_vel = np.random.choice(y_vel[:,i])
#         rand_dir = np.random.choice(d_dir[:,i])
#         rand_temp = np.random.choice(y_temp[:,i])
#         rand_sol = np.random.choice(d_sol[:,i])
#         _dlr = DLR(wind_speed=rand_vel, wind_angle=rand_dir, ambient_temp=rand_temp, eff_rad_heat_flux=rand_sol)
#         DLR_temp = _dlr.ampacity()
#         DLR_list.append(DLR_temp)
#     DLR_data[:,i] = DLR_list


# DLR_data_new = DLR_data.reshape(-1)

# # Plot
# fig_DLR = go.Figure(go.Histogram2d(
#         x = t_new,
#         y = DLR_data_new,
#         histnorm='percent',
#         colorscale='turbo',
#         autobinx=False,
#         xbins= dict(size= .264),
#         #autobiny=False,
#         #ybins= dict(size = .264)
#         ))


# fig_DLR.update_xaxes(tickangle=-90,
#                   tickvals = np.linspace(1.5*np.pi, 11.5*3.14, 11),
#                   ticktext = t_text, 
#                   )
# fig_DLR.update_layout(height=400, width=900, title_text="DLR Forecast",
#                     xaxis_title='Time',
#                     yaxis_title='Rating (Amps)',)

# # Plot All Figures
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
    st.plotly_chart(fig_dir)
with st.expander("Expand for DLR Timestep Monte Carlo"):
    st.plotly_chart(fig_mc)

