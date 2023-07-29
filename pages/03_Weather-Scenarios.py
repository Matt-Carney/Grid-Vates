import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm
from utils import DLR, temp_data, gen_normal, gen_uniform



st.title('Weather - Scenarios')
st.markdown('Determine the ampacity profile based on the enivronmental conditions below.')


x = np.linspace(1.5*np.pi, 11.5*3.14, 24*5)

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



# Profile 1 - hot, high wind velocity, consistent wind angle about 45 deg perpindicualt to the transmission line
temp_1 = temp_data(mu=30, delta=2, steps=x)
w_vel_1 = [16 for i in x]
w_ang_1 = [45 for i in x]
dlr_1 = []

for i, _ in enumerate(x):
    dlr_temp = DLR(wind_speed=w_vel_1[i], wind_angle=w_ang_1[i], ambient_temp=temp_1[i], eff_rad_heat_flux=1000)
    dlr_1.append(dlr_temp.ampacity())    

prof_1 = {'shape': 'spline', 'color': 'blue', 'width': 2}


# Profile 2 - Cool, moderate wind velocity, inconsistent wind angle to the transmission line
temp_2 = temp_data(mu=10, delta=2, steps=x)
w_vel_2 = [8 for i in x]
w_ang_2 = gen_uniform(30, 60, len(x))
dlr_2 = []

for i, _ in enumerate(x):
    dlr_temp = DLR(wind_speed=w_vel_2[i], wind_angle=w_ang_2[i], ambient_temp=temp_2[i], eff_rad_heat_flux=1000)
    dlr_2.append(dlr_temp.ampacity())    

prof_2 = {'shape': 'spline', 'color': 'green', 'width': 2}


# Profile 3 - Cool, moderate wind velocity, consistent wind angle about 45 deg perpindicualt to the transmission line
temp_3 = temp_data(mu=10, delta=2, steps=x)
w_vel_3 = [8 for i in x]
w_ang_3 = [45 for i in x]
dlr_3 = []

for i, _ in enumerate(x):
    dlr_temp = DLR(wind_speed=w_vel_3[i], wind_angle=w_ang_3[i], ambient_temp=temp_3[i], eff_rad_heat_flux=1000)
    dlr_3.append(dlr_temp.ampacity())    

prof_3 = {'shape': 'spline', 'color': 'red', 'width': 2}


# Profile 4 - cool, low wind velocity, consistent wind angle about 45 deg perpindicualt to the transmission line
temp_4 = temp_data(mu=10, delta=2, steps=x)
w_vel_4 = [2 for i in x]
w_ang_4 = [45 for i in x]
dlr_4 = []

for i, _ in enumerate(x):
    dlr_temp = DLR(wind_speed=w_vel_4[i], wind_angle=w_ang_4[i], ambient_temp=temp_4[i], eff_rad_heat_flux=1000)
    dlr_4.append(dlr_temp.ampacity())    

prof_4 = {'shape': 'spline', 'color': 'purple', 'width': 2}


fig_0 = go.Figure()
fig_0.add_trace(go.Scatter(x=x, y=dlr_1, line=prof_1, name='Profile 1', showlegend=True))
fig_0.add_trace(go.Scatter(x=x, y=dlr_2, line=prof_2, name='Profile 2', showlegend=True))
fig_0.add_trace(go.Scatter(x=x, y=dlr_3, line=prof_3, name='Profile 3', showlegend=True))
fig_0.add_trace(go.Scatter(x=x, y=dlr_4, line=prof_4, name='Profile 4', showlegend=True))

fig_0.update_layout(#title='Ampacity Profiles',
                   xaxis_title='Time',
                   yaxis_title='Ratings (Amps)', yaxis_range=[1000,3000], height=500, width=680)

fig_0.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True,tickangle=-90,
                  tickvals = np.linspace(1.5*np.pi, 11.5*3.14, 11),
                  ticktext = t_text)
fig_0.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)

st.plotly_chart(fig_0)



with st.expander("Cool temperature, moderate wind velocity, inconsistent wind angle (30-60 deg) to the transmission line"):
    st.write("Profile 2! The extreme fluctuations are being driven by the inconsistent wind angle")
    fig_1 = go.Figure()
    fig_1.add_trace(go.Scatter(x=x, y=dlr_1, line=prof_1, name='Profile 1', showlegend=True))
    fig_1.add_trace(go.Scatter(x=x, y=dlr_2, line={'shape': 'spline', 'color': 'green', 'width': 4}, name='Profile 2', showlegend=True))    
    fig_1.add_trace(go.Scatter(x=x, y=dlr_3, line=prof_3, name='Profile 3', showlegend=True))
    fig_1.add_trace(go.Scatter(x=x, y=dlr_4, line=prof_4, name='Profile 4', showlegend=True))

    fig_1.update_layout(#title='Ampacity Profiles',
                    xaxis_title='Time',
                    yaxis_title='Rating (Amps)', yaxis_range=[1000,3000], height=500, width=680)

    fig_1.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True,\
                                         tickvals = np.linspace(1.5*np.pi, 11.5*3.14, 11),
                                            ticktext = t_text, tickangle=270)
    fig_1.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
        
    st.plotly_chart(fig_1)


with st.expander("Cool temperature, moderate wind velocity, consistent wind angle (45 deg) to the transmission line"):
    st.write("Profile 3! The temperature and wind velocity are the same as Profile 2\
              but the wind angle is consistent at the average of Profile 2.")
    fig_2 = fig_0
    fig_2 = go.Figure()
    fig_2.add_trace(go.Scatter(x=x, y=dlr_1, line=prof_1, name='Profile 1', showlegend=True))
    fig_2.add_trace(go.Scatter(x=x, y=dlr_2, line=prof_2, name='Profile 2', showlegend=True))
    fig_2.add_trace(go.Scatter(x=x, y=dlr_3, line={'shape': 'spline', 'color': 'red', 'width': 6}, name='Profile 3', showlegend=True))
    fig_2.add_trace(go.Scatter(x=x, y=dlr_4, line=prof_4, name='Profile 4', showlegend=True))
    fig_2.update_layout(#title='Ampacity Profiles',
                xaxis_title='Time',
                yaxis_title='Rating (Amps)', yaxis_range=[1000,3000], height=500, width=680)

    fig_2.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True,
                       tickvals = np.linspace(1.5*np.pi, 11.5*3.14, 11),
                                            ticktext = t_text, tickangle=270)
    fig_2.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
    
    st.plotly_chart(fig_2)


with st.expander("Cool temperature, low wind velocity, consistent wind angle (45 deg) to the transmission line"):
    st.write("Profile 4! While the ambient temperature is cooler, the lower wind velocity reduces the overall amapcity.")

    fig_3 = go.Figure()
    fig_3.add_trace(go.Scatter(x=x, y=dlr_1, line=prof_1, name='Profile 1', showlegend=True))
    fig_3.add_trace(go.Scatter(x=x, y=dlr_2, line=prof_2, name='Profile 2', showlegend=True))
    fig_3.add_trace(go.Scatter(x=x, y=dlr_3, line=prof_3, name='Profile 3', showlegend=True))   
    fig_3.add_trace(go.Scatter(x=x, y=dlr_4, line={'shape': 'spline', 'color': 'purple', 'width': 6}, name='Profile 4', showlegend=True))
    
    fig_3.update_layout(#title='Ampacity Profiles',
            xaxis_title='Time',
            yaxis_title='Rating (Amps)', yaxis_range=[1000,3000], height=500, width=680)

    fig_3.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True,
                       tickvals = np.linspace(1.5*np.pi, 11.5*3.14, 11),
                                            ticktext = t_text, tickangle=270)
    fig_3.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
    
    st.plotly_chart(fig_3)




with st.expander("Hot temperature, high wind velocity, consistent wind angle (45 deg) to the transmission line"):
    st.write("Profile 1! While the ambient temperature is hotter, the higher wind velocity increases the overall amapcity.")

    fig_4 = go.Figure()
    fig_4.add_trace(go.Scatter(x=x, y=dlr_1, line = {'shape': 'spline', 'color': 'blue', 'width': 6}, name='Profile 1', showlegend=True))
    fig_4.add_trace(go.Scatter(x=x, y=dlr_2, line=prof_2, name='Profile 2', showlegend=True))
    fig_4.add_trace(go.Scatter(x=x, y=dlr_3, line=prof_3, name='Profile 3', showlegend=True))
    fig_4.add_trace(go.Scatter(x=x, y=dlr_4, line=prof_4, name='Profile 4', showlegend=True))

    fig_4.update_layout(#title='Ampacity Profiles',
                    xaxis_title='Time',
                    yaxis_title='Rating (Amps)', yaxis_range=[1000,3000], height=500, width=680)

    fig_4.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True,
                       tickvals = np.linspace(1.5*np.pi, 11.5*3.14, 11),
                                            ticktext = t_text, tickangle=270)
    fig_4.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)

    st.plotly_chart(fig_4)
