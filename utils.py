import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import truncnorm
import plotly.graph_objects as go
from scipy.stats import vonmises


def plot_fig(x, y, title='Placeholder', x_axis_title='Time', y_axis_title='Temperature (C)', 
             y_min=None, y_max=None, n_bins_x=120, n_bins_y=60):
    """
    Creates a 2D histogram plot of temperature or velocity data over time.

    Parameters:
    x (np.array): Time values, repeated for each corresponding y value.
    y (np.array): Temperature or velocity values, flattened 2D array.
    title (str): Title of the plot.
    x_axis_title (str): Title for the x-axis.
    y_axis_title (str): Title for the y-axis.
    y_min (float): Optional minimum y-axis limit.
    y_max (float): Optional maximum y-axis limit.
    n_bins_x (int): Number of bins for x-axis.
    n_bins_y (int): Number of bins for y-axis.

    Returns:
    plotly.graph_objects.Figure: A 2D histogram figure object.
    """
    
    # Plot
    t_text = ['Day 0 00:00', 'Day 0 12:00', 'Day 1 00:00', 'Day 1 12:00', 'Day 2 00:00',
              'Day 2 12:00', 'Day 3 00:00', 'Day 3 12:00', 'Day 4 00:00', 'Day 4 12:00', 'Day 5 00:00']

    y = y.flatten()
    
    # Calculate bin ranges
    x_range = [np.min(x), np.max(x)]
    y_range = [np.min(y), np.max(y)] if y_min is None or y_max is None else [y_min, y_max]
    
    fig = go.Figure(go.Histogram2d(
            x = x,
            y = y,
            histnorm='percent',
            colorscale='turbo',
            autobinx=False,
            xbins=dict(start=x_range[0], end=x_range[1], size=(x_range[1]-x_range[0])/n_bins_x),
            autobiny=False,
            ybins=dict(start=y_range[0], end=y_range[1], size=(y_range[1]-y_range[0])/n_bins_y)
            ))

    fig.update_xaxes(tickangle=-90,
                    tickvals = np.linspace(1.5*np.pi, 11.5*np.pi, 11),
                    ticktext = t_text
                    )
    
    # Update y-axis range if y_min and y_max are provided
    if y_min is not None and y_max is not None:
        fig.update_yaxes(range=[y_min, y_max])

    fig.update_layout(height=400, width=900, title_text=title,
                      xaxis_title=x_axis_title,
                      yaxis_title=y_axis_title)

    return fig


def generate_synthetic_temp_truncated(n_steps=120, mean=15,sample_size=1000,
                                      temp_range=6, 
                                      initial_std_dev=1, 
                                      uncertainty_growth=0.01, 
                                      random_walk_scale=0.1,
                                      type='temp'):
    
    """
    Generates synthetic temperature or velocity data with a truncated normal distribution.

    Parameters:
    n_steps (int): Number of time steps.
    mean (float): Mean temperature or velocity.
    sample_size (int): Number of samples per time step.
    temp_range (float): Range of temperature fluctuation.
    inital_std_dev (float): Initial standard deviation of temperature or velocity.
    uncertainty_growth (float): Growth rate of uncertainty over time.
    random_walk_scale (float): Scale of random walk component.
    type (str): 'temp' for temperature or 'vel' for velocity data.


    1.) Time steps and repetition:

    There are 120 time steps (n_steps = 120).
    Each time step has 1000 samples (sample_size = 1000).
    np.repeat(x, sample_size) repeats each time value 1000 times in place.
    This results in an array of 120,000 elements (120 * 1000).


    2.) Structure of x_plot:

    [t1, t1, t1, ...(1000 times), t2, t2, t2, ...(1000 times), ..., t120, t120, t120, ...(1000 times)]
    Total length: 120,000


    3.) Structure of y_plot before flattening:

    2D array with shape (120, 1000)
    Each row corresponds to a time step
    Each column in a row is a sample for that time step


    4.) Flattening y_plot (in plotting function):

    When we flatten y_plot, it concatenates all rows end-to-end.
    Result: [y1_1, y1_2, ..., y1_1000, y2_1, y2_2, ..., y2_1000, ..., y120_1, y120_2, ..., y120_1000]
    Total length: 120,000


    5.) Alignment of x_plot and flattened y_plot:

    Each x value in x_plot correctly corresponds to its y value in the flattened y_plot.
    For example, all 1000 samples for the first time step (t1) align with the first 1000 y values.    
        
    Returns:
    tuple: (x_plot, y_plot) where x_plot is repeated time values and y_plot is a 2D array of generated data.
    """

    x = np.linspace(1.5*np.pi, 11.5*np.pi, n_steps)
    
    # Base temperature pattern
    daily_pattern = np.sin(x) * temp_range / 2
    y = mean + daily_pattern 
    
    # Random walk component
    random_walk = np.cumsum(np.random.normal(0, random_walk_scale, n_steps))
    y += random_walk
    
    # Generate probabilistic data with truncated normal distribution
    data = []
    for i in range(n_steps):
        #uncertainty = initial_uncertainty + uncertainty_growth * i
        # Uncertainty/error propagation, used when combining two independent sources of uncertainty
        # Simply adding could over estimate the total error
        # Think pythagorean theorem
        total_std = np.sqrt(initial_std_dev**2 + (uncertainty_growth * i)**2)
    
        if type == 'temp':
            a, b = -3, 3  # ±3 standard deviations for temperature
        elif type == 'vel':
            a = max(-3, -y[i] / total_std) # Lower bound: max of -3 std dev or 0
            b = 3  # 0 to +3 standard deviations for velocity (to keep it non-negative)

        samples = truncnorm.rvs(a, b, loc=y[i], scale=total_std, size=sample_size)
        data.append(samples)

    # Format data for output
    x_plot = np.repeat(x, sample_size)
    y_plot = np.array(data)
    #y_plot = np.array(data).flatten() # Reshape data in plot_fig
    
    return x_plot, y_plot


def generate_synthetic_wind_direction(n_steps=120, sample_size=1000, start_direction=90, kappa=2, 
                                      shift_prob=0.1, max_shift=90):
    """
    Generates synthetic wind direction data.

    Parameters:
    n_steps (int): Number of time steps.
    sample_size (int): Number of samples per time step.
    kappa (float): Concentration parameter for von Mises distribution.
    shift_prob (float): Probability of a sudden shift in wind direction.
    max_shift (float): Maximum amount of sudden shift (in degrees).
    start_direction (float): Starting wind direction (in degrees).

    Returns:
    tuple: (x_plot, y_plot) where x_plot is repeated time values and y_plot is a 2D array of generated data in degrees.
    """
    
    # Convert max_shift and start_direction from degrees to radians
    max_shift_rad = np.radians(max_shift)
    current_direction = np.radians(start_direction)
    
    x = np.linspace(1.5*np.pi, 11.5*np.pi, n_steps)
    
    data = []
    for _ in range(n_steps):
        # Decide if there's a sudden shift
        if np.random.random() < shift_prob:
            shift = np.random.uniform(-max_shift_rad, max_shift_rad)
            current_direction = (current_direction + shift) % (2*np.pi)
        
        # Generate samples from von Mises distribution
        samples = vonmises.rvs(kappa, loc=current_direction, size=sample_size)
        
        # Ensure all values are in [0, 2π)
        samples = samples % (2*np.pi)
        
        # Convert samples to degrees
        samples_deg = np.degrees(samples)
        
        data.append(samples_deg)
    
    # Format data for output
    x_plot = np.repeat(x, sample_size)
    y_plot = np.array(data)
    
    return x_plot, y_plot


def generate_synthetic_solar_irradiance(n_steps=120, sample_size=1000, sol_irr=1000, 
                                        cloud_cover=0.5, cloud_cover_init_std=0.1, 
                                        cloud_cover_uncertainty_growth=0.01):
    """
    Generates synthetic solar irradiance data based on specified requirements.

    Parameters:
    n_steps (int): Number of time steps.
    sample_size (int): Number of samples per time step.
    sol_irr (float): Maximum solar irradiance (W/m^2)
    cloud_cover (float): Mean cloud cover (0-1)
    cloud_cover_init_std (float): Initial standard deviation of cloud cover
    cloud_cover_uncertainty_growth (float): Growth rate of cloud cover uncertainty

    Returns:
    tuple: (x_plot, y_plot) where x_plot is repeated time values and y_plot is a 2D array of irradiance data.
    """
    x = np.linspace(1.5*np.pi, 11.5*np.pi, n_steps)
    
    # Base irradiance pattern (using sine function for day/night cycle)
    base_irradiance = sol_irr * np.maximum(0, np.sin(x))
    
    # Generate probabilistic data with truncated normal distribution
    data = []
    for i in range(n_steps):
        # Calculate cloud cover standard deviation
        total_std = np.sqrt(cloud_cover_init_std**2 + (cloud_cover_uncertainty_growth * i)**2)
        
        # Generate cloud cover from truncated normal distribution
        a, b = (0 - cloud_cover) / total_std, (1 - cloud_cover) / total_std
        cloud_cover_sample = truncnorm.rvs(a, b, loc=cloud_cover, scale=total_std, size=sample_size)
        cloud_cover_sample = np.clip(cloud_cover_sample, 0, 1)  # Ensure values are between 0 and 1

        # Calculate irradiance
        clear_sky_irr = base_irradiance[i]
        direct = clear_sky_irr * (1 - cloud_cover_sample)
        diffuse = clear_sky_irr * cloud_cover_sample * 0.5  # Assume 50% of blocked light becomes diffuse

        # Combine components
        total_irradiance = direct + diffuse

        # Ensure non-negative values and cap at sol_irr
        samples = np.clip(total_irradiance, 0, sol_irr)
        data.append(samples)

    # Format data for output
    x_plot = np.repeat(x, sample_size)
    y_plot = np.array(data)
    
    return x_plot, y_plot



def generate_DLR(y_temp, y_sol, y_vel, y_dir, n_samples=1000, n_steps=120):
    """
    Randomly sample from each dataset, process with DLR, and format for plotting.

    Parameters:
    y_temp, y_sol, y_vel, y_dir (np.array): Input data arrays
    n_samples (int): Number of samples to take for each time step
    x_time (np.array): Array of time values (assuming 120 unique time points)

    Returns:
    tuple: (x_plot, y_plot) where x_plot is repeated time values and y_plot is a 2D array of processed data
    """
    
    x = np.linspace(1.5*np.pi, 11.5*np.pi, n_steps)
    
    # Reshape input data to (120, 1000) if not already in this shape
    datasets = [y_temp, y_sol, y_vel, y_dir]
    datasets = [data.reshape(120, 1000) if data.shape != (120, 1000) else data for data in datasets]

    # Initialize output array
    output = np.zeros((120, n_samples))

    # Process each time step
    for i in range(120):
        for j in range(n_samples):
            # Randomly sample one value from each dataset for this time step
            temp = np.random.choice(datasets[0][i])
            sol = np.random.choice(datasets[1][i])
            vel = np.random.choice(datasets[2][i])
            dir = np.random.choice(datasets[3][i])

            # Instantiate DLR and calculate ampacity
            dlr_instance = DLR(wind_speed=vel, wind_angle=dir, ambient_temp=temp, eff_rad_heat_flux=sol)  # Assuming these are the correct parameters for DLR
            output[i, j] = dlr_instance.ampacity()

    # Format x_plot
    x_plot = np.repeat(x, n_samples)

    # Flatten y_plot for consistency with original format
    y_plot = output.flatten()

    return x_plot, y_plot



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
        


#a = DLR(wind_speed=10, wind_angle=90, ambient_temp=20, eff_rad_heat_flux=1000)
#a.ampacity()

#d = []
#for i in range(1000):
#    a = DLR(wind_speed=5, wind_angle=90, ambient_temp=20, eff_rad_heat_flux=1000)
#    d.append(a.ampacity())


