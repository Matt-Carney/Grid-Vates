import pandas as pd
import numpy as np



class DLR:
    
    # Conductor AC resistance
    T_c = 100
    T_low = 25
    T_high = 75
    R_low = 7.283e-5
    R_high = 8.688e-5

    def __init__(self, wind_speed):

        self.wind_speed = wind_speed

    def resistance():
        R_Tc = (DLR.R_high - DLR.R_low)/(DLR.T_high - DLR.T_low)*(DLR.T_c-DLR.T_low) + DLR.R_low
        return R_Tc


a = DLR
a.resistance()
