import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import norm

p = np.arange(0.1,1,.1)

#n_0 = norm.ppf(p, loc=25, scale=4)
#n_1= norm.ppf(p, loc=25, scale=0.ZZ)



def gen_normal(mu=5, sigma=0.1, num=500):
    return np.random.normal(loc=mu, scale=sigma, size=num)

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


# x = np.array([1, 2, 3, 4, 5])
# y = np.array([1, 3, 2, 3, 1])



# fig = go.Figure()

# fig.add_trace(go.Scatter(x=x, y=y + 5, name="spline",fill='tozeroy', fillcolor='purple',
#                     #text=["tweak line smoothness<br>with 'smoothing' in line object"],
#                     #hoverinfo='text+name',
#                     line_shape='spline'))








st.plotly_chart(fig)



y_1 = gen_normal(sigma=0.5, num = 10000)
x_1 = [1 for x in range(len(y_1))]
y_2 = gen_normal(mu=5.1, sigma=0.5,num=10000)
x_2 = [1.1 for x in range(len(y_1))]
y_3 = gen_normal(mu=5,sigma=0.5, num=10000)
x_3 = [1.2 for x in range(len(y_1))]
y = np.array([y_1, y_2, y_3]).flatten('F')
x = np.array([x_1, x_2, x_3]).flatten('F')
df = pd.DataFrame({'y': y, 'x': x})
#fig = px.density_contour(df, x="x", y="y")
#fig.update_traces(contours_coloring="fill", contours_showlabels = True)

len(x)
fig = go.Figure(go.Histogram2dContour(
        x = x,
        y = y,
        colorscale = 'jet'
        ))
fig.update_layout(xaxis_range=[1,1.2])
st.plotly_chart(fig)







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