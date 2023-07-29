import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


st.title('Benefits of DLR')

st.subheader('What are Dynamic Line Ratings?')
st.markdown("Sustaining energy throughout the grid is by no account a trivial task. Operators are constantly balancing the supply \
            and demand for the energy that powers everything from industry to household appliances, signaling to power plants when \
            more power is needed and maintaining the power grid’s electrical flow to transmission lines and distribution network.")

st.markdown("There are many safety concerns that need to be considered. For transmission lines, the thermal limits drive line capacity and both maximum sag \
            (as the conductor temperature increases it sags which can  result in contact in nearby objects) and maximum conductor temperature (safe operating limits \
            from a materials perspective) need to be considered. For the sake of the simulations in Grid Vātēs we will only be considering the latter.")

st.markdown("Environmental conditions such as ambient temperature, wind speed, wind direction and solar irradiance impact the line rating (maximum current). \
            In order to calculate the rating for a given span, there are three main approaches to determining how to calculate the values for the abovementioned environmental conditions.")

st.markdown(" - **Static:** Based on seasonal historical worst cast ambient conditions.")
st.markdown(" - **Ambient Adjusted:** The granularity of ambient temperature is adjusted to daily or hourly (other environmental conditions are the same as static).")
st.markdown(" - **Dynamic:** Uses real-time conditions, based on field information from sensors.")

st.write('')

st.subheader('What are the benefits of DLR?')
st.markdown("The value of DLR is well established.")
st.markdown(" - LineVision, a climate tech company that provides a DLR hardware/software solution, has been working with U.S. \
            as well as European utilities and indicates that DLRs can reveal “anywhere between 15 and 40 percent of additional capacity”[2].")
st.markdown(" - The Texas utility Oncor and the New York Power Authority found average real-time  transmission capacity of at least 30 percent greater than static ratings.")
st.markdown(" - The Belgian grid operator Elia indicates the 5 to 50 percent increase in rating significantly improves market dispatch cost-effectiveness[2].")
st.markdown("By allowing transmission lines to operate closer to their true capacity (and still within a safe safety margin), \
            DLR help reduce the need for costly infrastructure upgrades, reduce grid congestion and facilitate better integration of renewable energy sources[3]. \
            Of particular interest is the symbiotic relationship between wind energy and DLR. As you can explore with the Weather-Sensitivty tab, \
            the line rating increases when the wind speed increases due to the enhanced cooling. Given that there will be additional power supplied \
            from the increased wind speed this allows the increase in power to be integrated in the grid and prevent curtailment[2].")
st.write('')

st.subheader('How are they calculated?')
st.markdown("The ampacity (ie. max current or 'Rating') is calculated from teh IEEE Standard 738-2006 - Steady State Heat Balance[4]")

st.write('')
IEEE_SS = Image.open('IEEE_Steady_State_DLR.png')
st.image(IEEE_SS, width=200)
st.markdown("Where:")
st.markdown(" - qc = convected heat loss rate per unit length")
st.markdown(" - qr = radiated heat loss per unit length")
st.markdown(" - qs = heat gain rate from sun")
st.markdown(" - R(Tc) = AC resistance of conductor at temperature, T")

### Bullet and bold example 
# st.markdown(" - **Bullet Bold Example** Your one-stop-shop to learn about what Dynmic Line Ratings are and why they're important.")


#st.write("""This is an example \n
#of writing text \n
#on multiple lines \n
#""")


