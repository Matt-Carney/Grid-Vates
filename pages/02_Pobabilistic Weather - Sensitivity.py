import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#np.random.seed(123)
mean = st.sidebar.slider('mean', 0, 10, 5)
variance = st.sidebar.slider('variance', 0.0, 5.0, .1)
mean1 = st.sidebar.slider('mean1', 0, 10, 5)
variance1 = st.sidebar.slider('variance1', 0.0, 5.0, .1)
mean2 = st.sidebar.slider('mean2', 0, 10, 5)
variance2 = st.sidebar.slider('variance2', 0.0, 5.0, .1)

def gen_normal(mu=5, sigma=0.1, num=500):
    return np.random.normal(loc=mu, scale=sigma, size=num)

norm = gen_normal(mu=mean, sigma=variance)



st.subheader('Wind Forecast')
fig, ax = plt.subplots()
ax.set_facecolor('#e7ecef')
ax.set_xlim(-20,20)
ax.hist(norm, bins=20)
#hist_values = np.histogram(data[DATE_COLUMN].dt.hour, bins=24, range=(0,24))[0]
#plt.hist(norm)
st.pyplot(fig)