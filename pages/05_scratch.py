import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import norm

def gen_normal(mu=5, sigma=0.1, num=10000):
    return np.random.normal(loc=mu, scale=sigma, size=num)

def gen_uniform(low=0, high=1.0, size=10000):
    return np.random.uniform(low=low, high=high, size=size)

p = np.arange(0.1,1,.1)
#p
n_0 = norm.ppf(p, loc=25, scale=.1)
n_1= norm.ppf(p, loc=25, scale=5)
n_0
n_1

# Creating 2-D grid of features
#[X, Y] = np.meshgrid(feature_x, feature_y)
 
#Z = np.cos(X / 2) + np.sin(Y / 4)
 
fig = go.Figure(data =
     go.Contour(x = feature_x, y = feature_y, z = v,
               colorscale='jet'
               ))



new  = test['labs'].value_counts()
test = test.iloc[:,0]
test = test.to_numpy()
pd.cut(test,5)


test.shape

len(x)








p_mod = [round(1.0 - x, 2) if x >0.5 else round(x, 2) for x in p]
p_mod

y_ = n_0
z = p_mod
p_mod
test = np

feature_x = np.arange(0, 10, 1)
feature_y = np.arange(25, 30, 1)


v = np.array([[.1, .1, .1, .1, .1, .1, .1, .1, .1, .1], 
     [.2, .2, .2, .2, .2, .2, .2, .2, .2, .2 ], 
     [.5, .5, .5, .5, .5, .5, .5, .5, .5, .5,],
     [.4, .4, .4, .4, .4, .4, .4, .4, .4, .4],
     [.1, .1, .1, .1, .1, .1, .1, .1, .1, .1,]])

#v.shape










b = gen_uniform(22, 25)
b
np.percentile(b, 0.5)


a = gen_normal()
p_30 = np.percentile(a, 0.3)
pp_30 = norm.ppf(.3, loc=5, scale=0.1)
p_30
pp_30

fig = go.Figure()

fig.add_trace(go.Scatter(x=[1, 2, 3, 4, 5], y=[7, 7, 7, 7, 7], fill='tozeroy', fillcolor='purple',
                    mode='none', line_shape='spline' # override default markers+lines
                    ))

fig.add_trace(go.Scatter(x=[1, 2, 3, 4], y=[6, 5, 6, 5], fill='tozeroy', fillcolor='red',
                    mode= 'none', line_shape='spline'))

fig.add_trace(go.Scatter(x=[1, 2, 3], y=[3, 4, 3], fill='tozeroy', fillcolor='green',
                    mode='none', line_shape='spline' # override default markers+lines
                    ))
fig.add_trace(go.Scatter(x=[1, 2, 3], y=[2, 1, 2], fill='tozeroy', fillcolor='red',
                    mode='none', line_shape='spline' # override default markers+lines
                    ))
fig.add_trace(go.Scatter(x=[1, 2, 3, 4], y=[1, 1, 1, 1], fill='tozeroy', fillcolor='purple',
                    mode='none', line_shape='spline'  # override default markers+lines
                    ))
