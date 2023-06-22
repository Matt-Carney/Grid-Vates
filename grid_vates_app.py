import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


st.title('Dynamic Line Rating Explorer')


#np.random.seed(123)
mean = st.slider('mean', 0, 10, 5)
variance = st.slider('variance', 0.0, 5.0, .1)


def gen_normal(mu=5, sigma=0.1, num=500):
    return np.random.normal(loc=mu, scale=sigma, size=num)

norm = gen_normal(mu=mean, sigma=variance)

st.write('Slider va', mean)

new_val = mean + 5

st.write('new value', norm.mean())

st.subheader('Temperature Forecast')
fig, ax = plt.subplots()
ax.hist(norm, bins=20)
#hist_values = np.histogram(data[DATE_COLUMN].dt.hour, bins=24, range=(0,24))[0]
#plt.hist(norm)
st.pyplot(fig)


#np.random.seed(123)
mean_2 = st.sidebar.slider('mean_2', 0, 10, 5)
variance_2 = st.sidebar.slider('variance_2', 0.0, 5.0, .1)

norm_2 = gen_normal(mu=mean_2, sigma=variance_2)

st.subheader('Wind Forecast')
fig2, ax2 = plt.subplots()
ax2.set_facecolor('#e7ecef')
ax2.set_xlim(-20,20)
ax2.hist(norm_2, bins=20)
#hist_values = np.histogram(data[DATE_COLUMN].dt.hour, bins=24, range=(0,24))[0]
#plt.hist(norm)
st.pyplot(fig2)