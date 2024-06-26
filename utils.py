import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import norm


def temp_data(mu=20, delta=2, steps=10):
    high = mu + delta
    low = mu - delta
    amp = (high-low)/2
    ave = (high+low)/2
    temp = np.sin(steps) * amp + ave
    return temp

def gen_uniform(low=0, high=1.0, size=10000):
    return np.random.uniform(low=low, high=high, size=size)

def gen_normal(mu=5, sigma=0.1, num=10000):
    return np.random.normal(loc=mu, scale=sigma, size=num)


def contour_data(mu = 20, sigma=3, time_steps=60, deacay=1.0, size=10000, bins=20, is_sin = False):
    x = np.arange(time_steps)
    z = np.zeros((int(bins), int(time_steps)))


    if is_sin == True:
        high = mu+10
        low = mu-10
        amp = (high-low)/2
        ave = (high+low)/2
        t = np.arange(time_steps)
        vals = np.sin(t) * amp + ave

    for i in range(time_steps):
        if is_sin == True:
            mu = vals[i]
        # Uniform/Normal split
        decay_per = (i+1)/time_steps
        num_uniform = int(decay_per*size)
        num_normal = size - num_uniform

        # Gernerate data
        norm_ = gen_normal(mu, sigma, num_normal)
        unif_bound = norm.ppf([0.01, 0.999], loc=mu, scale=sigma) # Set uniform distribution to 95th
        unif = gen_uniform(unif_bound[0], unif_bound[1], num_uniform)

        samples = list(norm_) + list(unif)


        # Determine y values of bins
        if i == 0: 
            hist = np.histogram(samples, bins)
            y_out = hist[1][:-1] # Need to remove last bin edge
            y = hist[1]
            z_temp = hist[0]/len(samples)
            #z_temp = z_temp.reshape(z_temp.shape[0],1)

        # Apply y values
        else:
            hist = np.histogram(samples, bins = y)
            z_temp = hist[0]/len(samples)
            #z_temp = z_temp.reshape(z_temp.shape[0],1)

        z[:,i] = z_temp
    
    return x, y, z



def velocity_temp__data(mu = 20, sigma=1, decay=1.0, size=1000, type='temp'):
    x = np.linspace(1.5*np.pi, 11.5*3.14, 24*5)
    time_steps = int(len(x))


    dists = np.zeros((size, int(time_steps))) 

    high = mu+3
    low = mu-3
    amp = (high-low)/2
    ave = (high+low)/2
    vals = np.sin(x) * amp + ave

    #norm_init = gen_normal(mu, sigma, size)
    #unif_bound = norm.ppf([0.1, 0.99], loc=mu, scale=sigma)
    #unif_init = gen_uniform(unif_bound[0], unif_bound[1], size)

    nums = []
    for i in range(time_steps):
        #if type == 'sine':
        mu = vals[i]
        
        decay_per = (i)/time_steps
        num_uniform = int(decay_per*size*decay)
        num_normal = size - num_uniform

        #if type == 'sine':
        np.random.seed(123)
        norm_0 = gen_normal(mu, sigma, num_normal)
        if type == 'temp':
            norm_0 = norm_0
        elif type == 'vel':
            norm_0 = np.abs(norm_0)
        unif_bound = norm.ppf([0.001, 0.9999], loc=mu, scale=sigma)
        
        if type == 'temp':
            unif_0 = gen_uniform(unif_bound[0], unif_bound[1], num_uniform)
        elif type == 'vel':
            unif_0 = gen_uniform(max(unif_bound[0], 0), unif_bound[1], num_uniform)
        
        #if rand == 'index':
        norm_ = norm_0[0:num_normal]
        unif_ = unif_0[0:num_uniform]
        #elif rand == 'choice':
        #    norm_ = np.random.choice(norm_0, num_normal, replace=False)
        #    unif_ = np.random.choice(unif_0, num_uniform, replace=False)
        samples = list(norm_) + list(unif_)
        dists[:,i] = samples 
        #dists[:,i] = norm_init
        dists = np.array(dists)
        
        nums.append([num_normal, num_uniform])
        
    #dists[dists<=0] = 0.0
    return dists, list(x)








def solar_irr_data(sol_irr = 1000, sol_perc = 0.5, decay = 1.0):
    x = np.linspace(1.5*np.pi, 11.5*3.14, 24*5)
    size = 1000
    dists = np.zeros((size, int(len(x))))

    irr_perc_day = [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.39, 0.45, 0.59, 0.75, 0.88, 0.97,\
                    1.00, 0.98, 0.90, 0.77, 0.62, 0.47, 0.39, 0.00, 0.00, 0.00, 0.00, 0.00]

    irr_perc = np.array(irr_perc_day*5)

    red = 1.0 * (1.0-(.75* sol_perc))

    for i, val in enumerate(irr_perc):
        if val != 0.00:
            ideal = [val * sol_irr * red]
            #ideals = np.random.normal(ideal, scale=10, size=1000)
            ideals = 1000 * ideal

            decay_per = (i)/int(len(x))
            num_uniform = int(decay_per * size * decay)
            num_ideal = size - num_uniform
            
            ideal_ = np.random.choice(ideals, num_ideal, replace=True)
            
            #unif_low = val*sol_irr*.25
            #unif_high = val*sol_irr*1
            np.random.seed(123)
            #unif = np.random.uniform(low=unif_low, high=unif_high, size=size)
            unif = np.random.normal(loc=ideal, scale=100, size=size)
            np.random.seed(123)
            unif_ = np.random.choice(unif, num_uniform, replace=True)

            #samples = unif
            #samples = np.concatenate(ideal_, unif_)
            samples = list(ideal_) + list(unif_) #Clean this up
            samples = np.array(samples)
            samples[samples<0] = 0.0
            dists[:,i] = samples 

    t = list(x)*1000


    return dists, t






def wind_direction_data(deg = 45, kap = 2, size = 1000, perc_unif = 0.5):
    rad = deg * (np.pi/180)

    np.random.seed(123)
    vm_init = np.random.vonmises(mu=rad, kappa=kap, size=size) # -pi to pi
    unif_init = gen_uniform(0, 360, size) # 0 - 360 degrees

    num_unif = int(size*perc_unif)
    num_vm = size - num_unif

    vm = vm_init[0:num_vm]
    unif = unif_init[0:num_unif]

    vm_2pi = np.mod(vm, 2*np.pi) # 0 - 2 pi
    vm_deg = vm_2pi * (180/np.pi) # 0 - 360 degrees
    
    agg_dist = np.concatenate((vm_deg, unif))
    #agg_dist = vm_deg

    pol_bins = np.linspace(0,360,17)
    bin_val = np.digitize(agg_dist, bins= pol_bins)

    pol_val = []
    for i in bin_val:
        temp = pol_bins[i]
        pol_val.append(temp)


    df_pol = pd.DataFrame({'wind_dir': pol_val})
    df_pol = df_pol.groupby(['wind_dir']).size().reset_index().rename(columns={0: 'count'})
    df_pol['Percent'] = df_pol['count']/df_pol['count'].sum()

    return df_pol, agg_dist


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
        q_c = self.convection_heat_loss(p_f, u_f, k_f, K_ang) # loss due to convection
        q_r = self.radiated_heat_loss() # radiated loss
        q_s = self.solar_heat_gain() # heat gain from the sun
        R_Tc = self.resistance() # AC resistance 

        I = math.sqrt((q_c + q_r - q_s)/R_Tc)

        return I
        


a = DLR(wind_speed=10, wind_angle=90, ambient_temp=20, eff_rad_heat_flux=1000)
a.ampacity()




d = []
for i in range(1000):
    a = DLR(wind_speed=5, wind_angle=90, ambient_temp=20, eff_rad_heat_flux=1000)
    d.append(a.ampacity())


