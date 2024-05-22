import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


st.set_page_config(page_title="Grid Vātēs")

#st.set_page_config(layout="wide")

st.title('Welcome to Grid Vātēs!')
st.header('A Dynamic Line Rating Educational Sandbox')
#st.subheader('Vātēs is a Latin term used to refer to a prophet, seer, or poet;\
#            someone believed to possess the ability to predict the future')


#st.write('Vātēs is a Latin term used to refer to a prophet, seer, or poet;\
#            someone believed to possess the ability to predict the future')


st.write('<font size="4">Vātēs is a Latin term used to refer to a prophet, seer, or poet; \
         someone believed to possess the ability to predict the future[1].\
         The goal of this web app is to transform you into a DLR Vātēs!\
         A summary of each tab, which can be accessed through the left pane, is listed below.</font>', unsafe_allow_html=True)

st.write('')
st.subheader('Benefits of DLR')
st.markdown("Your one-stop-shop to learn about Dynamic Line Ratings and why they're important.")

st.subheader('Weather - Sensitivity')
st.markdown("This is the experimental sandbox ya'll have been asking for.\
            As you may be aware, the DLR rating is influenced by ambient environmental conditions\
            such as temperature, solar irradiance, wind speed and wind direction. Accordingly, a DLR forecast\
            is a 'forecast of forecasts' of said environmental conditions.\
            Experiment with different environmental condition parameters and see how they affect the DLR forecast.")


st.subheader('Weather - Scenarios')
st.markdown("This is a knowledge check after your experimentation on the Weather-Sensitivity tab.\
            Here, you're presented with several rating profiles and are asked to select\
            the corresponding environmental conditions.")


st.write('')
st.write('')
st.markdown('Please feel free to reach out with any questions or feedback - mcarney31@gatech.com')


st.write('')
st.write('')
vates = Image.open('DALLE_vates_image.png')
st.image(vates)
#with st.expander("Expand this drop down to see a painting of the original Grid Vātēs himself!"):
#    st.write('Just kidding, but thanks DALL-E!')
#    st.image(phil)









