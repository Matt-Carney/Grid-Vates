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

