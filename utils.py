import pandas as pd
import numpy as np
import math


q_c1, q_c2 = 0, 0
math.pi

class DLR:
    """
    Steps:
    1.) Calculate T_film variable calculated
    2.) Fluid properties, u_f, p_f, k_f calculated
    3.) 
    """
    
    
    def __init__(self, wind_speed, wind_angle, ambient_temp):
        # These could be class variables, but access slightly faster as per-instance variables
        self.T_c = 100 # Max allowable temp, celcius
        self.D = 28.11 # Conductor outside diameter, mm
        self.elevation = 0

        self.E = 0.5 # Emissivity 

        # Conductor AC resistance
        self.T_low = 25
        self.T_high = 75
        self.R_low = 7.283e-5
        self.R_high = 8.688e-5

        # Variables
        self.w_speed = wind_speed
        self.w_angle = wind_angle * (math.pi/180) # Convert degres to radians   
        self.T_amb = ambient_temp # Ambient temp, celcius

        self.T_f = (self.T_c + self.T_amb)/2 # Film temperature


    def fluid_propertires(self,):
        # Dynamic viscosity, Pa-s, Table 2
        u_f = 5.47e-8 * self.T_f + 1.72e-5 

        # Termal conductivity of air, W/(m-C), Table 2
        if self.elevation == 0:
            p_f = -3.44e-3 * self.T_f + 1.27e0
        elif self.elevation == 1000:
            p_f = -3.05e-3 * self.T_f + 1.13e0
        elif self.elevation == 2000:
            p_f = -2.70e-3 * self.T_f + 9.99e-1
        elif self.elevation == 4000:
            p_f = -2.09e-3 * self.T_f + 7.74e-1

        # Termal conductivity of air, W/(m-C), Table 2
        k_f = 7.45e-5 * self.T_f + 2.42e-2 

        return u_f, p_f, k_f

    def wind_direction_factor(self):
        Kang = 1.194 - math.cos(self.wind_angle) + 0.194 * math.cos(2*self.wind_angle) + 0.368 * math.sin(2*self.wind_angle)
        return Kang


    def convection_heat_loss(self, p_f, u_f, k_f, Kang):
        # Natural convection (no wind)
        q_nat = 0.02050 * p_f**0.5 * self.D**0.75 * (self.T_c - self.T_amb)**1.25 # W/m

        # Forced convection heat loss
        if self.wind_speed > 0: 
            # low wind speed
            q_c1 = (1.01 + 0.0372*((self.D * p_f * self.w_speed)/(u_f))**0.52) * k_f * Kang * (self.T_c - self.T_a)
            
            # high wind speed
            q_c2 = (0.0119*((self.D * p_f * self.w_speed)/(u_f))**0.6) * k_f * Kang * (self.T_c - self.T_a)
        else:
            q_c1, q_c2 = 0, 0
                
        return max(q_nat, q_c1, q_c2) # always use max

    def radiated_heat_loss(self):
        q_r = 0.0178 * self.D * self.E * (((self.T_c + 273)/100)**4 - ((self.T_amb + 273)/100)**4)
        return q_r        


    def resistance(self):
        R_Tc = (self.R_high - self.R_low)/(self.T_high - self.T_low)*(self.T_c-self.T_low) + self.R_low
        return R_Tc

    def runner(self):
        u_f, p_f, k_f = self.fluid_propertires() # fluid properties
        Kang = self.wind_direction_factor() # Wind direction factor
        q_c = self.convection_heat_loss(p_f, u_f, k_f, Kang)
        q_r = self.radiated_heat_loss()

        R_Tc = self.resistance() 




a = DLR
a.head_loss()
a.resistance()
