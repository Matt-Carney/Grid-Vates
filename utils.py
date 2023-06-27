import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt


high = 30
low = 20
amp = (high-low)/2
ave = (high+low)/2
len(t)
t = np.arange(3, 27)
vals = np.sin(t) * amp + ave

fig, ax = plt.subplots()
ax.plot(t, vals, 'green')
plt.show()
plt.close()
vals

len(t)
len(vals)

time = np.arange(0., 10., 0.2)
velocity = np.sin(time) * 10
distance = np.zeros_like(time, dtype=float)

fig, ax = plt.subplots()
#ax1.set_ylabel("distance (m)")
#ax1.set_xlabel("time")
ax.plot(t, vals, "blue")

plt.show()






class DLR:
    """
    Returns the DLR Ampacity, I, based on IEEE 738-2006
    Steps:
    1.) Calculate film temperature: T_film 
    2.) Calculate fuid properties: u_f, p_f, k_f
    3.) Calculate wind direction factor: K_ang
    4.) Calculaate convection heat loss
    5.) Calculate radiated heat loss
    6.) Calculate solar head gain
    7.) DLR Ampactity

    Args:
    wind_speed: m/s
    wing_angle: deg from line
    ambient_temp: deg C
    eff_rad_had_flux: W/m^s, already compensated for effective angle of incidence of the sun's rays
 
    """
    
    
    def __init__(self, wind_speed, wind_angle, ambient_temp, eff_rad_heat_flux):
        # These could be class variables, but access slightly faster as per-instance variables
        self.T_c = 100 # Max allowable temp, celcius
        self.D = 28.11 # Conductor outside diameter, mm
        self.elevation = 0
        self.Ksolar = 1.0 
        self.E = 0.5 # Emissivity 
        self.alpha = 0.5 # solar absorptivity

        # Conductor AC resistance
        self.T_low = 25
        self.T_high = 75
        self.R_low = 7.283e-5
        self.R_high = 8.688e-5

        # Variables
        self.w_speed = wind_speed
        self.w_angle = wind_angle * (math.pi/180) # Convert degres to radians   
        self.T_amb = ambient_temp # Ambient temp, celcius
        self.Q_ef = eff_rad_heat_flux

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
        K_ang = 1.194 - math.cos(self.w_angle) + 0.194 * math.cos(2*self.w_angle) + 0.368 * math.sin(2*self.w_angle)
        return K_ang


    def convection_heat_loss(self, p_f, u_f, k_f, K_ang): 
        # Natural convection (no wind)
        q_nat = 0.02050 * p_f**0.5 * self.D**0.75 * (self.T_c - self.T_amb)**1.25 # W/m

        # Forced convection heat loss
        if self.w_speed > 0: 
            # low wind speed
            q_c1 = (1.01 + 0.0372*((self.D * p_f * self.w_speed)/(u_f))**0.52) * k_f * K_ang * (self.T_c - self.T_amb)
            
            # high wind speed
            q_c2 = (0.0119*((self.D * p_f * self.w_speed)/(u_f))**0.6) * k_f * K_ang * (self.T_c - self.T_amb)
        else:
            q_c1, q_c2 = 0, 0
                
        return max(q_nat, q_c1, q_c2) # always use max

    def radiated_heat_loss(self): #qr
        q_r = 0.0178 * self.D * self.E * (((self.T_c + 273)/100)**4 - ((self.T_amb + 273)/100)**4)
        return q_r        

    def solar_heat_gain(self): 
        self.Q_ef = self.Q_ef * self.Ksolar # Elevation correction factor
        q_s = self.alpha * self.Q_ef * self.D/1000
        return q_s

    def resistance(self):
        R_Tc = (self.R_high - self.R_low)/(self.T_high - self.T_low)*(self.T_c-self.T_low) + self.R_low
        return R_Tc

    def ampacity(self):
        u_f, p_f, k_f = self.fluid_propertires() # fluid properties
        K_ang = self.wind_direction_factor() # Wind direction factor
        q_c = self.convection_heat_loss(p_f, u_f, k_f, K_ang)
        q_r = self.radiated_heat_loss()
        q_s = self.solar_heat_gain()
        R_Tc = self.resistance()

        I = math.sqrt((q_c + q_r - q_s)/R_Tc)

        return I
        


a = DLR(wind_speed=5, wind_angle=90, ambient_temp=20, eff_rad_heat_flux=1000)
d = []
for i in range(1000):
    a = DLR(wind_speed=5, wind_angle=90, ambient_temp=20, eff_rad_heat_flux=1000)
    d.append(a.ampacity())

a.ampacity()
