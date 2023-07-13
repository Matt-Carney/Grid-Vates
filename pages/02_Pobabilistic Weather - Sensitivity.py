import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm

st.title('Probabilistic Weather - Sensitivity')
st.markdown('This is some text')

#unif_bound = norm.ppf([0.136, 0.864], loc=20, scale=1)
#unif_bound[0]

def gen_normal(mu=5, sigma=0.1, num=10000):
    return np.random.normal(loc=mu, scale=sigma, size=num)

def gen_uniform(low=0, high=1.0, size=10000):
    return np.random.uniform(low=low, high=high, size=size)





def contour_data(mu = 20, sigma=3, time_steps=60, deacay=1.0, size=10000, bins=10, is_sin = False):
    x = np.arange(time_steps)
    z = np.zeros((int(bins), int(time_steps)))


    if is_sin == True:
        high = mu+3
        low = mu-3
        amp = (high-low)/2
        ave = (high+low)/2
        x = np.arange(0., 14*np.pi, (1/3))
        vals = np.sin(x) * amp + ave

        time_steps = int(len(x))
        z = np.zeros((int(bins), time_steps))



    for i in range(time_steps):
        if is_sin == True:
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


        # Determine y values of bins
        if i == 0: 
            hist = np.histogram(samples, bins)
            ###y_out = hist[1][:-1] # Need to remove last bin edge
            y = hist[1]
            #y = np.array([15, 20, 25, 30, 35, 40])
            z_temp = hist[0]/len(samples)
            #z_temp = z_temp.reshape(z_temp.shape[0],1)

        # Apply y values
        else:
            hist = np.histogram(samples, bins = y)
            z_temp = hist[0]/len(samples)
            #z_temp = z_temp.reshape(z_temp.shape[0],1)

        z[:,i] = z_temp
    
    z = z.round(decimals=2)*2
    
    return x, y, z

with st.sidebar:
    st.write("Wind Velocity Variables")

# Sidebar
vel_mean = st.sidebar.slider('Velocity', 24, 30, 27)
vel_sd = st.sidebar.slider('Init. Standard Deviation', 0.1, 5.0, 2.0)


# Contour scale dict
scale_dict = contours=dict(
                start=0.0,
                end=.9,
                size=.1)

left, middle, right = st.columns((1, 20, 1))

color_names = ['0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '100']

my_colorsc=[[0.0, 'rgb(221, 236, 239)'],
            #[0.0, 'rgb(221, 236, 239'], 
            [0.1, 'rgb(60, 90, 205)'],
            #[0.2, 'rgb(60, 90, 205)'],
            [0.2, 'rgb(62, 155, 254)'],
            #[0.3, 'rgb(65, 148, 254)'],
            [0.3, 'rgb(24, 214, 203)'], 
            #[0.4, 'rgb(29, 204, 217)'],
            [0.4, 'rgb(70, 247, 131)'],
            #[0.5, 'rgb(45, 240, 156)'],
            [0.5, 'rgb(162, 252, 60)'],
            #[0.6, 'rgb(129, 254, 82)'],
            [0.6, 'rgb(225, 220, 55)'], 
            #[0.7, 'rgb(195, 240, 51)'],
            [0.7, 'rgb(253, 165, 49)'],
            #[0.8, 'rgb(242, 200, 58)'], 
            [0.8, 'rgb(239, 90, 17)'], 
            #[0.9, 'rgb(253, 143, 40)'],
            [0.9, 'rgb(196, 37, 2)'],
            [1, 'rgb(122, 4, 2)']]

turbo = ['rgb(255, 255, 255)', 'rgb(68, 90, 205)', 'rgb(57, 163, 251)', 'rgb(24, 225, 187)', 'rgb(107, 253, 99)', 'rgb(195, 240, 51)', 'rgb(249, 188, 57)', 'rgb(246, 108, 25)', 'rgb(203, 42, 3)', 'rgb(122, 4, 2)']
# DLR Forecast
#st.subheader('DLR Forecast')
x_DLR, y_DLR, z_DLR = contour_data(mu = 25, sigma=2, is_sin=True)
#z_DLR = z_DLR.round(decimals=2)*2
#z1 = z1*2
fig_DLR = go.Figure(data =
     go.Contour(x = x_DLR, y = y_DLR, z = z_DLR,
               colorscale=my_colorsc, contours=scale_dict))
#fig2.update_layout(xaxis_range=[1,1.2])
fig_DLR.update_layout(yaxis_range=[0, 40], title_text="DLR Forecast")
#with middle:
st.plotly_chart(fig_DLR, use_container_width=True)



#st.subheader('Environmental Forecast')

# Wind Velocity
x_vel, y_vel, z_vel = contour_data(mu=vel_mean, sigma=vel_sd, is_sin=True)

# Wind Direction
x_dir, y_dir, z_dir = contour_data(mu=15, sigma=1, is_sin=True)

# Temperature
x_temp, y_temp, z_temp = contour_data(mu=10, sigma=1, is_sin=True)

# Solar Irradiance
x_irr, y_irr, z_irr = contour_data(mu=12, sigma=1, is_sin=True)


fig_env = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Wind Velocity', 'Wind Direction', 'Temperature', 'Solar Irradiance'))

fig_env.add_trace(go.Contour(x = x_vel, y = y_vel, z = z_vel, colorscale=my_colorsc, contours=scale_dict, showscale=False), row=1, col=1)
fig_env.add_trace(go.Contour(x = x_dir, y = y_dir, z = z_dir, colorscale=my_colorsc, contours=scale_dict, showscale=False), row=1, col=2)
fig_env.add_trace(go.Contour(x = x_temp, y = y_temp, z = z_temp, colorscale=my_colorsc, contours=scale_dict, showscale=False), row=2, col=1)
fig_env.add_trace(go.Contour(x = x_irr, y = y_irr, z = z_irr, colorscale=my_colorsc, contours=scale_dict, showscale=True), row=2, col=2)

fig_env.update_layout(height=700, width=900,
                  title_text="Environmental Forecasts")

st.plotly_chart(fig_env)



