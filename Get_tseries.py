# In this small script, we extract the time series of ion moments from the GMM separated results.
# From tstart to tend

# import packages
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from scipy.ndimage import gaussian_filter
from SolarWindPack import *
from Funcs import *

from datetime import datetime


def _isValidTimeStr(time_str):
    try:
        datetime.strptime(time_str, "%H%M%S")
        return True
    except ValueError:
        return False
    
def times_inbetween(tstart, tend, ion_hhmmss_list=None):
    if not ion_hhmmss_list:
        print("[times_inbetween] No time list provided.")
        return None

    # sort so we know the bounds
    ion_hhmmss_list = sorted(ion_hhmmss_list)
    tmin = datetime.strptime(ion_hhmmss_list[0], "%H%M%S")
    tmax = datetime.strptime(ion_hhmmss_list[-1], "%H%M%S")

    ts = datetime.strptime(tstart, "%H%M%S")
    te = datetime.strptime(tend, "%H%M%S")

    if ts < tmin:
        print(f"[times_inbetween] tstart {tstart} < min, using {ion_hhmmss_list[0]}")
        ts = tmin
    if te > tmax:
        print(f"[times_inbetween] tend {tend} > max, using {ion_hhmmss_list[-1]}")
        te = tmax

    if ts > te:
        print("[times_inbetween] start > end after clamp, returning [].")
        return []

    return [t for t in ion_hhmmss_list
            if ts <= datetime.strptime(t, "%H%M%S") <= te]


def ReadMoments(Moments_Path):
    moments_dict = {}
    try:
        with open(Moments_Path, 'r') as file:
            for line in file:
                key, value = line.split(':')
                key = key.strip()
                value = value.strip()
                try:
                    # Try to convert the value to a float
                    value = float(value)
                except ValueError:
                    # If it fails, keep it as a string
                    pass
                moments_dict[key] = value
    except UnicodeDecodeError:
        print(f"Failed to read file: {Moments_Path}")
    return moments_dict

def concatenate_moments(yymmdd, ion_hhmmss_list):
    data = []
    times = []
    for hhmmss in ion_hhmmss_list:
        moment_dict = ReadMoments(f'result/SO/{yymmdd}/Particles/Ions/{hhmmss}/moments.txt')
        if not moment_dict:
            continue
        datetime_obj = datetime.strptime(yymmdd + ' ' + hhmmss, "%Y%m%d %H%M%S")
        times.append(datetime_obj)
        data.append(moment_dict)
    df = pd.DataFrame(data, index=times)
    return df

def read_moments_csv(csv_path):
    moments_df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    return moments_df


def main():
    yymmdd = '20220302'
    tstart = '000001'
    tend = '235857'

    ion_hhmmss_list = sorted([time_str for time_str in os.listdir(f'result/SO/{yymmdd}/Particles/Ions/') 
                            if _isValidTimeStr(time_str) and os.path.exists(f'result/SO/{yymmdd}/Particles/Ions/{time_str}/moments.txt')])

    times_in_between = times_inbetween(tstart, tend, ion_hhmmss_list)

    moments_df = concatenate_moments(yymmdd, times_in_between)

    # save to a csv file
    moments_df.to_csv(f'result/SO/{yymmdd}/moments_{tstart}_to_{tend}.csv')

    return 0

if __name__ == "__main__":
    main()




