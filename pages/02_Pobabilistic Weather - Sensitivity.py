import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import norm



unif_bound = norm.ppf([0.136, 0.864], loc=20, scale=1)
unif_bound[0]

def gen_normal(mu=5, sigma=0.1, num=10000):
    return np.random.normal(loc=mu, scale=sigma, size=num)

def gen_uniform(low=0, high=1.0, size=10000):
    return np.random.uniform(low=low, high=high, size=size)


#def contour_data(sigma=20, time_steps=60, deacay=1.0, size=1000):

mu = 20
sigma = 5
time_steps = 60 
decay = 1.0
size = 10000
bins=20

x = np.arange(time_steps)
z = np.zeros((int(bins), int(time_steps)))

for i in range(time_steps):
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
        y_out = hist[1][:-1] # Need to remove last bin edge
        y = hist[1]
        z_temp = hist[0]/len(samples)
        #z_temp = z_temp.reshape(z_temp.shape[0],1)

    # Apply y values
    else:
        hist = np.histogram(samples, bins = y)
        z_temp = hist[0]/len(samples)
        #z_temp = z_temp.reshape(z_temp.shape[0],1)

    z[:,i] = z_temp


fig3 = go.Figure(data =
     go.Contour(x = x, y = y, z = z,
               colorscale='jet'
               ))
#fig2.update_layout(xaxis_range=[1,1.2])
fig3.update_layout(yaxis_range=[0, 40])
st.plotly_chart(fig3)



for i in range(10):
    if i == 0:
        y = 14
        print('do something')
    else:
        print(y)

#y_1 = gen_normal(sigma=0.1, num = 10000)
y_1 = gen_uniform(4, 6, 10000)
x_1 = [1 for x in range(len(y_1))]
#y_2 = gen_normal(mu=5.1, sigma=0.3,num=10000)
y_2 = gen_uniform(4, 6, 10000)
x_2 = [1.1 for x in range(len(y_1))]
#y_3 = gen_normal(mu=5,sigma=0.5, num=10000)
y_3 = gen_uniform(4, 6, 10000)
x_3 = [1.2 for x in range(len(y_1))]
y = np.array([y_1, y_2, y_3]).flatten('F')
x = np.array([x_1, x_2, x_3]).flatten('F')
df = pd.DataFrame({'y': y, 'x': x})
#fig = px.density_contour(df, x="x", y="y")
#fig.update_traces(contours_coloring="fill", contours_showlabels = True)



#hist = np.histogram(y)
#hist[0]/len(y)
#hist[1]


fig = go.Figure(go.Histogram2dContour(
        x = x,
        y = y,
        colorscale = 'jet',
        ))
fig.update_layout(xaxis_range=[1,1.2])
st.plotly_chart(fig)

hist_1 = np.histogram(y_1, bins=20)
z_1 = hist_1[0]/len(y_1)
y__1 = hist_1[1]

hist_2 = np.histogram(y_2, bins=y__1)
z_2 = hist_2[0]/len(y_2)
y__2 = hist_2[1]

hist_3 = np.histogram(y_3, bins=y__1)
z_3 = hist_3[0]/len(y_3)
y__3 = hist_3[1]

x = [1, 2, 3]

zz = np.array([z_1, z_2, z_3])
zzT = zz.T

zzT.shape

fig2 = go.Figure(data =
     go.Contour(x = x, y = y__1, z = zzT,
               colorscale='jet'
               ))
#fig2.update_layout(xaxis_range=[1,1.2])
#fig2.update_layout(yaxis_range=[4.96, 5.03])
st.plotly_chart(fig2)





#np.random.seed(123)
mean = st.sidebar.slider('temp_high', 1, 10, 5)
variance = st.sidebar.slider('variance', 0.0, 5.0, .1)
mean1 = st.sidebar.slider('temp_low', int(0), int(mean), int(mean/2))
variance1 = st.sidebar.slider('variance1', 0.0, 5.0, .1)
mean2 = st.sidebar.slider('mean2', 0, 10, 5)
variance2 = st.sidebar.slider('variance2', 0.0, 5.0, .1)



norm = gen_normal(mu=mean, sigma=variance)



st.subheader('Wind Forecast')
fig, ax = plt.subplots()
ax.set_facecolor('#e7ecef')
ax.set_xlim(-20,20)
ax.hist(norm, bins=20)
#hist_values = np.histogram(data[DATE_COLUMN].dt.hour, bins=24, range=(0,24))[0]
#plt.hist(norm)
st.pyplot(fig)