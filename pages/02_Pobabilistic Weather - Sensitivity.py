import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm
from utils import DLR, temp_data, gen_normal, gen_uniform

st.title('Probabilistic Weather - Sensitivity')
#st.markdown('This is some text')

#unif_bound = norm.ppf([0.136, 0.864], loc=20, scale=1)
#unif_bound[0]

def gen_normal(mu=5, sigma=0.1, num=10000):
    return np.random.normal(loc=mu, scale=sigma, size=num)

def gen_uniform(low=0, high=1.0, size=10000):
    return np.random.uniform(low=low, high=high, size=size)


def contour_data(mu = 20, sigma=1, time_fraction=(1/24), deacay=1.0, size=1000, bins=10, type = 'sine'):
    #x = np.arange(time_steps)
    x = np.arange(0., 10*np.pi, time_fraction)    
    time_steps = int(len(x))
    #z = np.zeros((int(bins), time_steps))

    dists = np.zeros((size, int(time_steps))) 


    if type == 'sine':
        high = mu+3
        low = mu-3
        amp = (high-low)/2
        ave = (high+low)/2
        vals = np.sin(x) * amp + ave

    elif type == 'radial':
        vals = [mu for x in range(time_steps)]

    for i in range(time_steps):
        if type == 'sine':
            mu = vals[i]
        elif type == 'linear':
            mu = vals[i]
        # Uniform/Normal split
        decay_per = (i+1)/time_steps
        num_uniform = int(decay_per*size)
        num_normal = size - num_uniform

        #num_normal = size
        #num_uniform = 0

        # Gernerate data
        if type == 'sine':
            norm_ = gen_normal(mu, sigma, num_normal)
        elif type == 'radial':
            norm_ = gen_normal(mu, sigma, num_normal)
            #norm_ = np.random.vonmises(mu, sigma, num_normal)
            norm_ = abs(norm_)

        
        unif_bound = norm.ppf([0.1, 0.99], loc=mu, scale=sigma) # Set uniform distribution to 1 and 99th percentile
        unif = gen_uniform(unif_bound[0], unif_bound[1], num_uniform)

        samples = list(norm_) + list(unif)
        dists[:,i] = samples 


    _1, _99 = np.percentile(dists, [1, 99], axis=0)
    min = _1.min()
    max = _99.max()
    ext = (max+min)/bins
    #min = min - ext
    #max = max + ext

    bin_val = list(np.linspace(min, max, bins)) # creates list of len(bins), actual bins are bins-1
    ext_val = bin_val[-1] + ext
    bin_val.append(ext_val)

    y = bin_val
    z = np.zeros(((bins), time_steps)) 

    # For every distribution, create histogram, replate z_init
    for i in range(z.shape[1]):
        hist = np.histogram(dists[:,i], bins=bin_val) # dists = (10000, 132)
        #hist = hist/num_bins
        z[:,i] = hist[0]/(len(samples))

    
    z = z.round(decimals=2)*2
    
    return x, y, z, dists

### Sidebar
# Wind Velocity
with st.sidebar:
    st.write("Wind Velocity Variables (m/s)")
vel_mean = st.sidebar.slider('Velocity', 1, 20, 9)
vel_sd = st.sidebar.slider('Velocity Init. Standard Deviation', 0.1, 5.0, 2.0)

# Wind Direction
with st.sidebar:
    st.write("Wind Direction Variables (deg)")
dir_mean = st.sidebar.slider('Angle', 1, 90, 45)
dir_sd = st.sidebar.slider('Angle Init. Standard Deviation', 0.1, 5.0, 2.0)

# Wind Direction
with st.sidebar:
    st.write("Temperature Variables")
temp_mean = st.sidebar.slider('Degrees (C)', 1, 50, 15)
temp_sd = st.sidebar.slider('Deg Init. Standard Deviation', 0.1, 5.0, 2.0)


# Solar Irradiance
with st.sidebar:
    st.write("Solar Irradiance Variables")
irr_mean = st.sidebar.slider('Heat Flux (W)', 0, 3000, 1500)
irr_sd = st.sidebar.slider('Irradiance Init. Standard Deviation', 0.1, 5.0, 2.0)


# Contour scale dict
scale_dict = contours=dict(
                start=0.0,
                end=.9,
                size=.1)


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

line_param = {'width': 0}
contour_param = {'start': 0.1, 'end': .9, 'size': .1}

# DLR Forecast
#st.subheader('DLR Forecast')
#x_DLR, y_DLR, z_DLR, d_DLR = contour_data(mu = 25, sigma=1, type='sine')

#fig_DLR = go.Figure(data =
#     go.Contour(x = x_DLR, y = y_DLR, z = z_DLR,
#               colorscale=my_colorsc, contours = contour_param, hoverinfo='skip', connectgaps=True))

#fig_DLR.update_layout(height=450, width=900,yaxis_range=[0, 40], title_text="DLR Forecast")
#st.plotly_chart(fig_DLR)



#st.subheader('Environmental Forecast')
# Wind Velocity
x_vel, y_vel, z_vel, d_vel = contour_data(mu=vel_mean, sigma=vel_sd, type='sine')

# Wind Direction
x_dir, y_dir, z_dir, d_dir = contour_data(mu=dir_mean, sigma=dir_sd, type='radial')

# Temperature
x_temp, y_temp, z_temp, d_temp = contour_data(mu=temp_mean, sigma=temp_sd, type='sine')

# Solar Irradiance
x_irr, y_irr, z_irr, d_irr = contour_data(mu=irr_mean, sigma=irr_sd, type='sine')

#d_vel.shape
#x_vel.shape[0]

num_rand = 100
x_DLR_ = []
y_DLR_ = []
for i, val in enumerate(x_vel):
    rand_vel = np.random.choice(d_vel[i], replace=True, size=num_rand)
    rand_dir = np.random.choice(d_dir[i], replace=True, size=num_rand)
    rand_temp = np.random.choice(d_temp[i], replace=True, size=num_rand)
    rand_irr = np.random.choice(d_irr[i], replace=True, size=num_rand)
    x_ = [val for n in range(num_rand)]
    x_DLR_ = x_DLR_ + x_ 
    for j, val_2 in enumerate(rand_vel):
        _dlr = DLR(wind_speed=rand_vel[j], wind_angle=rand_dir[j], ambient_temp=rand_temp[j], eff_rad_heat_flux=rand_irr[j])
        y_DLR_.append(_dlr.ampacity())



fig_DLR = go.Figure(go.Histogram2dContour(
        x = x_DLR_,
        y = y_DLR_,
        histnorm='percent',
        colorscale = my_colorsc,
        contours = contour_param
        #autobinx=False,
        #xbins= dict(size= 0.3),
        #autobiny=False,
        #ybins= dict(size = 0.3)
        ))
fig_DLR.update_layout(height=450, width=900,yaxis_range=[0, 5000], title_text="DLR Forecast")
st.plotly_chart(fig_DLR)


fig_env = make_subplots(
    rows=4, cols=1, shared_xaxes=True,
    subplot_titles=('Wind Velocity', 'Wind Direction', 'Temperature', 'Solar Irradiance'))

fig_env.add_trace(go.Contour(x = x_vel, y = y_vel, z = z_vel, colorscale=my_colorsc, contours = contour_param, hoverinfo='skip', showscale=False), row=1, col=1)
fig_env.add_trace(go.Contour(x = x_dir, y = y_dir, z = z_dir, colorscale=my_colorsc, contours = contour_param, hoverinfo='skip', showscale=False), row=2, col=1)
fig_env.add_trace(go.Contour(x = x_temp, y = y_temp, z = z_temp, colorscale=my_colorsc, contours = contour_param, hoverinfo='skip', showscale=False), row=3, col=1)
fig_env.add_trace(go.Contour(x = x_irr, y = y_irr, z = z_irr, colorscale=my_colorsc, contours = contour_param, hoverinfo='skip', showscale=True), row=4, col=1)

fig_env.update_layout(height=700, width=900,
                  title_text="Environmental Forecasts",
                  yaxis_range=[0,30], yaxis_title='Velocity (m/s)',
                  yaxis2_range=[0,90],yaxis2_title='Angle (deg)',
                  yaxis3_range=[-20,60], yaxis3_title='Degrees (C)',
                  yaxis4_title='Heat Flux (W)') #yaxis4_range=[0,3000], 


st.plotly_chart(fig_env)


