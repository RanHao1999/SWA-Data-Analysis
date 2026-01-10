"""
Author = @RanHao
Starting Date = 2024.10.09
In this Python script, I will define all the functions that I will use in the main scripts.
"""

import os, sys
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import scipy
import scipy.interpolate
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
#import plasmapy
import math
import pickle

import scipy.constants as const
import astropy.units as u
from matplotlib.collections import LineCollection


def resample_to_target_time(dataframe, target_tstart, target_tend, target_dt, keys, units=None):
    # The function for resampling the data to the target time resolution.
    # Turn the index of the dataframe to numbers according to their gap to the first point (time index).
    # Unit of dt: seconds
    buffer = timedelta(seconds=target_dt)
    dataframe = dataframe.loc[target_tstart: target_tend]
    time = dataframe.index
    time_index_in_num = [(time[i] - time[0]).total_seconds() for i in range(len(time))]

    # Derive the time index numbers of the target time range.
    target_time_gap = (target_tend - target_tstart).total_seconds()
    point_num = int(target_time_gap / target_dt)
    target_start_index = (target_tstart - time[0]).total_seconds()
    target_end_index = (target_tend - time[0]).total_seconds()

    target_timeindex_in_num = np.linspace(target_start_index, target_end_index, point_num + 1)


    res_dict = {}
    for i in range(len(keys)):
        values = dataframe[keys[i]].values
        f = scipy.interpolate.interp1d(time_index_in_num, values)
        y_new = f(target_timeindex_in_num)
        res_dict[keys[i]] = y_new

    target_timestamps = [time[0] + timedelta(seconds=x) for x in target_timeindex_in_num]
    res_df = pd.DataFrame(res_dict, index=target_timestamps)

    return res_df

def butter_lowpass_filter(data, cutoff, fs, order=5):
    # The butter low pass filter
    def butter_lowpass(cutoff, fs, order=5):
        return butter(order, cutoff, fs=fs, btype='low')

    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def magnitude(vector):
    # calculate the magnitude of a vector
    return math.sqrt(sum(pow(element, 2) for element in vector))


def fieldAlignedCoordinates(Bx, By, Bz):
    '''
    INPUTS:
         Bx, By, Bz = rank1 arrays of magnetic field measurements in SRF frame
    '''
    Bmag = np.sqrt(Bx**2 + By**2 + Bz**2)

    # Define field-aligned vector
    Nx = Bx/Bmag
    Ny = By/Bmag
    Nz = Bz/Bmag

    # Make up some unit vector
    if np.isscalar(Nx):
        Rx = 0
        Ry = 1.
        Rz = 0
    else:
        Rx = np.zeros(Nx.len())
        Ry = np.ones(len(Nx))
        Rz = np.zeros(len(Nx))

    # Find some vector perpendicular to field NxR 
    TEMP_Px = ( Ny * Rz ) - ( Nz * Ry )  # P = NxR
    TEMP_Py = ( Nz * Rx ) - ( Nx * Rz )  # This is temporary in case we choose a vector R that is not unitary
    TEMP_Pz = ( Nx * Ry ) - ( Ny * Rx )


    Pmag = np.sqrt( TEMP_Px**2 + TEMP_Py**2 + TEMP_Pz**2 ) #Have to normalize, since previous definition does not imply unitarity, just orthogonality
  
    Px = TEMP_Px / Pmag # for R=(0,1,0), NxR = P ~= RTN_N
    Py = TEMP_Py / Pmag
    Pz = TEMP_Pz / Pmag


    Qx = ( Pz * Ny ) - ( Py * Nz )   # N x P
    Qy = ( Px * Nz ) - ( Pz * Nx )  
    Qz = ( Py * Nx ) - ( Px * Ny )  

    return(Nx, Ny, Nz, Px, Py, Pz, Qx, Qy, Qz)


def rotateVectorIntoFieldAligned(Ax, Ay, Az, Nx, Ny, Nz, Px, Py, Pz, Qx, Qy, Qz):
    # For some Vector A in the SAME COORDINATE SYSTEM AS THE ORIGINAL B-FIELD VECTOR:

    An = (Ax * Nx) + (Ay * Ny) + (Az * Nz)  # A dot N = A_parallel
    Ap = (Ax * Px) + (Ay * Py) + (Az * Pz)  # A dot P = A_perp (~RTN_N (+/- depending on B), perpendicular to s/c y)
    Aq = (Ax * Qx) + (Ay * Qy) + (Az * Qz)  # 

    return(An, Ap, Aq)

def low_pass_filter(data, cutoff_frequency, sampling_rate):
    """
    Apply a simple low-pass filter to a 1D array of data.

    Parameters:
        data (list or numpy array): The input data to filter.
        cutoff_frequency (float): The cutoff frequency of the filter in Hz.
        sampling_rate (float): The sampling rate of the data in Hz.

    Returns:
        numpy array: The filtered data.
    """
    import numpy as np
    
    # Calculate the RC time constant
    dt = 1 / sampling_rate
    alpha = dt / (dt + (1 / (2 * np.pi * cutoff_frequency)))
    
    # Apply the low-pass filter
    filtered_data = np.zeros_like(data)
    filtered_data[0] = data[0]  # Initialize with the first value
    for i in range(1, len(data)):
        filtered_data[i] = alpha * data[i] + (1 - alpha) * filtered_data[i - 1]
    
    return filtered_data

def PTensor2T(PTensor, Density, B_field, k2eV):
    k_B = 1.38064852e-23
    TTensor = PTensor / k_B / Density 

    B_magnitude = np.sqrt(np.sum(B_field**2))
    b_hat = B_field / B_magnitude

    # T_para
    T_para = np.dot(b_hat, np.dot(TTensor, b_hat))
    T_trace = np.trace(TTensor)
    T_perp = (T_trace - T_para) / 2 # K
    # K to eV
    T_para = T_para * k2eV
    T_perp = T_perp * k2eV
    
    return T_para, T_perp

def read_pickle(file):
    with open(file, 'rb') as f:
        return pickle.load(f)
    
def save_pickle(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def OneParticleNoiseLevel(count_cdffile, vdffile):

    # For all the measurement of the day
    vdf = vdffile['vdf'][...] # Extract all the vdf
    vdf_epochs = np.array(vdffile['Epoch'], dtype="datetime64[ms]") # Extract all the epochs, convert to np.array for quicker search.
    counts = count_cdffile['COUNTS'][...] # Extract all the counts
    count_epochs = np.array(count_cdffile['Epoch'], dtype="datetime64[ms]") # Extract all the epochs

    adjusted_vdf_times = vdf_epochs - np.timedelta64(500, 'ms') # Adjust the vdf times to match the counts times
    match_indices = np.searchsorted(count_epochs, adjusted_vdf_times) # Find the indices of the counts that match the vdf times
    counts = counts[match_indices] # Get the counts that match the vdf times

    noise_level = np.divide(vdf, counts, out=np.zeros_like(vdf, dtype=float), where=counts != 0) # Calculate the noise level
    noise_level[np.isnan(noise_level)] = 0 # Drop the NaN values and replace them with 0.
    one_particle_noise_level = np.nanpercentile(noise_level, 99.99, axis=0)

    return one_particle_noise_level

def resample_cdf(cdf_file, variable_names, dt, tstart, tend, time_key='Epoch'):
    """
    This function interpolates the data in the cdf file to a target time grid, and returns a dataframe.
    """
    time = cdf_file[time_key][...]
    df = pd.DataFrame({'time': time})
    df.set_index('time', inplace=True)

    for key in variable_names:
        data = cdf_file[key][...]
        if data.ndim > 1:
            for i in range(data.shape[-1]):
                df[f"{key}_{i}"] = data[..., i]
        else:
            df[key] = data

    # Resample with specified dt and interpolate
    df_filtered = df[(df.index >= tstart) & (df.index <= tend)]
    
    df_resampled = df_filtered.resample(dt).mean().interpolate('time')

    df_resampled = df_resampled.loc[tstart: tend]  # clip to exact time range
    df_resampled.index.name = 'time'

    # Optional: Clip again just in case resample generated anything slightly outside
    df_resampled = df_resampled[(df_resampled.index >= tstart) & (df_resampled.index <= tend)]

    return df_resampled

def average_in_interval(SO_data, time_string, variable, date):
    """
    Returns the average of a variable in SO_data over a time interval defined by time_string.

    Parameters:
    - SO_data: pandas DataFrame with datetime index
    - time_string: str in the format "HHMMSSToHHMMSS"
    - variable: name of the column to average
    - date: date of the interval, format 'YYYY-MM-DD'

    Returns:
    - mean value of the variable over the interval
    """
    # Parse the time string
    start_str, end_str = time_string.split("To")
    start_time = pd.to_datetime(f"{date} {start_str[:2]}:{start_str[2:4]}:{start_str[4:]}")
    end_time = pd.to_datetime(f"{date} {end_str[:2]}:{end_str[2:4]}:{end_str[4:]}")

    # Slice the data using a mask
    mask = (SO_data.index >= start_time) & (SO_data.index <= end_time)
    data_in_range = SO_data.loc[mask]

    # Return the average of the variable
    return data_in_range[variable].mean()

def sound_speed(Tp, Ta, Te, Np, Na, Ne):
    """
    Calculate the sound speed (c_s) for a plasma with protons, alphas, and electrons.
    
    Parameters:
    - Tp : float
        Proton temperature (in eV)
    - Ta : float
        Alpha particle temperature (in eV)
    - Te : float
        Electron temperature (in eV)
    - Np : float
        Proton number density (in m^-3)
    - Na : float
        Alpha particle number density (in m^-3)
    - Ne : float
        Electron number density (in m^-3)

    Returns:
    - c_s : float
        Sound speed (in m/s)
    """

    # Physical constants
    eV_to_J = 1.602176634e-19  # Joules per eV
    mp = 1.6726219e-27         # Proton mass, kg
    me = 9.10938356e-31        # Electron mass, kg
    ma = 6.644657e-27          # Alpha particle mass, kg

    # Gamma factors
    gamma_e = 1.0  # Electrons isothermal
    gamma_i = 5.0 / 3.0  # Ions adiabatic

    # Mass density (rho)
    rho = Np * mp + Na * ma + Ne * me  # kg/m^3

    # Pressure contributions (in Pascals)
    Pp = Np * Tp * eV_to_J  # Proton pressure
    Pa = Na * Ta * eV_to_J  # Alpha pressure
    Pe = Ne * Te * eV_to_J  # Electron pressure

    # Total pressure with gamma factors
    total_pressure = gamma_e * Pe + gamma_i * (Pp + Pa)

    # Sound speed
    cs = (total_pressure / rho) ** 0.5  # in m/s

    return cs

def plot_signed_segments(ax, x, y, linestyle='solid', label=None, linewidth=2):
    """
    Plot continuous |y| vs x, coloring by the sign of y. 
    Breaks into segments where sign flips and uses ax.plot for full control.

    Parameters:
        ax        : matplotlib axis
        x, y      : arrays of equal length
        linestyle : 'solid', 'dashed', 'dotted', 'dashdot'
        label     : optional legend label (only shown on first segment)
        linewidth : line width
    """
    x = np.asarray(x)
    y = np.asarray(y)
    abs_y = np.abs(y)

    style_map = {
        'solid': '-',
        'dashed': '--',
        'dotted': ':',
        'dashdot': '-.'
    }
    line_style = style_map.get(linestyle, '-')

    # Helper: split into segments of the same sign
    def split_by_sign(x, y):
        segments = []
        start_idx = 0
        for i in range(1, len(y)):
            if np.sign(y[i]) != np.sign(y[i - 1]):
                segments.append((x[start_idx:i+1], y[start_idx:i+1]))
                start_idx = i
        segments.append((x[start_idx:], y[start_idx:]))  # last one
        return segments

    # Segment and plot
    segments = split_by_sign(x, y)
    for i, (x_seg, y_seg) in enumerate(segments):
        color = 'blue' if y_seg[0] >= 0 else 'red'
        ax.plot(
            x_seg, np.abs(y_seg),
            color=color,
            linestyle=line_style,
            linewidth=linewidth,
            label=label if i == 0 and label is not None else None
        )
        
    ax.annotate('+', xy=(0.1, 0.9), xycoords='axes fraction', ha='center', color='blue')
    ax.annotate('-', xy=(0.2, 0.9), xycoords='axes fraction', ha='center', color='red')
    

def cal_overlap(protons, alphas):
    """
    Calculate the overlap parameter between proton and alpha VDFs.
    Overlap parameter is defined as the integral of the minimum of the two VDFs over velocity space.

    Parameters:
    - protons: SolarWind object for protons
    - alphas: SolarWind object for alphas

    Returns:
    - overlap_param: float, the overlap parameter
    """
    vdf_proton = protons.get_vdf()
    vdf_alpha = alphas.get_vdf() / 4.0

    vdf_proton_1d = np.sum(vdf_proton, axis=(0, 1))
    vdf_alpha_1d = np.sum(vdf_alpha, axis=(0, 1))

    min_vdf = np.minimum(vdf_proton_1d, vdf_alpha_1d)
    overlap = np.sum(min_vdf) / np.sum(vdf_alpha_1d)
    
    return overlap

def angle_between_vectors(V1, V2):
    """
    Compute the angle between two vectors (or arrays of vectors) in radians.
    
    Parameters
    ----------
    B : array_like
        Magnetic field vector(s), shape (..., 3)
    Vd : array_like
        Drift velocity vector(s), shape (..., 3)
    
    Returns
    -------
    theta : ndarray
        Angle(s) between V1 and V2 in radians, shape (...)
    """
    V1 = np.asarray(V1)
    V2 = np.asarray(V2)

    # Dot product along last axis
    dot = np.sum(V1 * V2, axis=-1)
    
    # Norms
    V1_norm = np.linalg.norm(V1, axis=-1)
    V2_norm = np.linalg.norm(V2, axis=-1)

    # Avoid division by zero
    denom = V1_norm * V2_norm
    # Where denom == 0, set angle to NaN
    cos_theta = np.zeros_like(dot, dtype=float)
    valid = denom > 0
    cos_theta[valid] = dot[valid] / denom[valid]
    
    # Numerical safety: clip into [-1, 1]
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    theta = np.arccos(cos_theta)
    return theta
