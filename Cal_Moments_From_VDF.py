"""
Author: Hao Ran
Institution: UCL / MSSL
Email: hao.ran.24@ucl.ac.uk
GitHub: @RanHao1999
Created: 2026-05-01

Description:
    Read saved proton and alpha VDF files, calculate plasma moments,
    and store the results in per-day HDF5 output files.
"""

# library imports
import concurrent.futures
import gc
import bisect
import os
import time

import pandas as pd
import numpy as np
import h5py
from spacepy import pycdf
from sunpy.net import Fido
from sunpy.net import attrs as a
import sunpy_soar

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

# Local application imports
from Funcs import *
from SolarWindPack import *
from Constrain_Funcs import *

# Constant things
Proton_example = read_pickle('Miscellany/Protons.pkl')
Alpha_example = read_pickle('Miscellany/Alphas.pkl')

vel_proton = Proton_example.grid['velocity']
vel_alpha = Alpha_example.grid['velocity']
elevation = Proton_example.grid['elevation']
azimuth = Proton_example.grid['azimuth']


def _reconstruct_vdf(fh, idx, VDF_SHAPE=(11, 9, 96)):
    ptr = fh['ptr']
    start = ptr[idx]
    end = ptr[idx + 1]

    vdf = np.zeros(VDF_SHAPE, dtype=np.float32)
    vdf[fh['i'][start:end], fh['j'][start:end], fh['k'][start:end]] = fh['value'][start:end]
    return vdf


def _iter_dates(start_date, end_date):
    start = pd.to_datetime(start_date, format='%Y%m%d')
    end = pd.to_datetime(end_date, format='%Y%m%d')

    if end < start:
        raise ValueError('EndDate must be greater than or equal to StartDate.')

    return [
        (start + pd.Timedelta(days=offset)).strftime('%Y%m%d')
        for offset in range((end - start).days + 1)
    ]


def _filter_dates_with_vdfs(dates):
    """
    Filter a list of date strings to only those that have VDF files present
    in result/SO/VDFs. Both Proton and Alpha VDF files must exist.

    Parameters
    ----------
    dates : list of str
        Dates in 'YYYYMMDD' format.

    Returns
    -------
    list of str
        Dates for which the corresponding VDF files exist.
    """
    available = []
    for d in dates:
        pfn = f'result/SO/VDFs/Proton_vdf_{d}.h5'
        afn = f'result/SO/VDFs/Alpha_vdf_{d}.h5'
        if os.path.exists(pfn) and os.path.exists(afn):
            available.append(d)
        else:
            print(f"Skipping {d}: missing VDF file(s) ({'proton' if not os.path.exists(pfn) else ''}{' ' if (not os.path.exists(pfn) and not os.path.exists(afn)) else ''}{'alpha' if not os.path.exists(afn) else ''})")
    return available


def _write_moments(moment_file, moments_by_key):
    os.makedirs(os.path.dirname(moment_file), exist_ok=True)

    if isinstance(moments_by_key, pd.DataFrame):
        moments_frame = moments_by_key.copy()
    else:
        moments_frame = pd.DataFrame(moments_by_key)

    if 'Time' in moments_frame.columns:
        moments_frame['Time'] = pd.to_datetime(moments_frame['Time'])
        moments_frame = moments_frame.set_index('Time')
    elif isinstance(moments_frame.index, pd.DatetimeIndex):
        if moments_frame.index.tz is not None:
            moments_frame.index = moments_frame.index.tz_localize(None)
        moments_frame.index.name = 'Time'
    else:
        raise ValueError("Moments data must contain a 'Time' column or have a DatetimeIndex.")

    moments_frame = moments_frame.sort_index()
    moments_frame.to_csv(moment_file, index=True, date_format='%Y-%m-%d %H:%M:%S')


def ensure_mag_file(date_str):
    """
    Ensure MAG file exists for the given date. If it doesn't exist in data/SO/MAG/,
    download it from SOAR.
    
    Parameters
    ----------
    date_str : str
        Date string in format 'YYYYMMDD'
    
    Returns
    -------
    str
        Path to the MAG file (only if file exists)
        
    Raises
    ------
    FileNotFoundError
        If the MAG file cannot be found or downloaded
    """
    # Create directory if it doesn't exist
    mag_dir = 'data/SO/MAG'
    os.makedirs(mag_dir, exist_ok=True)
    
    # Expected filename
    filename = f'solo_l2_mag-srf-normal_{date_str}_v01.cdf'
    mag_file = os.path.join(mag_dir, filename)
    
    # Check if file exists
    if os.path.exists(mag_file):
        print(f"MAG file exists for {date_str}")
        return mag_file
    
    # If not, download it
    print(f"Downloading MAG file for {date_str}...")
    
    # Convert date format YYYYMMDD to YYYY-MM-DD
    date_formatted = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
    
    MAG = a.Instrument('MAG')
    time = a.Time(date_formatted, date_formatted)
    product = a.soar.Product("mag-srf-normal")
    level = a.Level(2)
    
    try:
        query = Fido.search(MAG & time & product & level)
        if len(query[0]) > 0:
            Fido.fetch(query, path=mag_dir)
            print(f"Successfully downloaded MAG file for {date_str}")
            
            # Verify file exists after download
            if not os.path.exists(mag_file):
                # Check if any .cdf file was downloaded (in case naming differs)
                cdf_files = [f for f in os.listdir(mag_dir) if f.endswith('.cdf') and date_str in f]
                if cdf_files:
                    mag_file = os.path.join(mag_dir, cdf_files[0])
                    print(f"Using downloaded file: {cdf_files[0]}")
                else:
                    raise FileNotFoundError(f"MAG file for {date_str} was not found after download")
        else:
            raise FileNotFoundError(f"No MAG data found for {date_str} on SOAR")
    except FileNotFoundError as e:
        raise
    except Exception as e:
        raise FileNotFoundError(f"Error downloading MAG file for {date_str}: {e}")
    
    if not os.path.exists(mag_file):
        raise FileNotFoundError(f"MAG file for {date_str} not found at {mag_file}")
    
    return mag_file


def cal_moments(Protons, Alphas):

    Time = pd.to_datetime(Protons.time)
    # mag field
    magF_SRF = Protons.magfield
    (Nx, Ny, Nz, Px, Py, Pz, Qx, Qy, Qz) = fieldAlignedCoordinates(magF_SRF[0], magF_SRF[1], magF_SRF[2])

    # Moments.
    Nproton = cal_density_Spherical(Protons)
    Nalpha = cal_density_Spherical(Alphas)

    # Velocity
    Valpha = cal_bulk_velocity_Spherical(Alphas)
    Vproton = cal_bulk_velocity_Spherical(Protons)

    Vproton_Baliged = rotateVectorIntoFieldAligned(Vproton[0], Vproton[1], Vproton[2], Nx, Ny, Nz, Px, Py, Pz, Qx, Qy, Qz)
    VprotonPara, VprotonPerp = Vproton_Baliged[0], np.sqrt(Vproton_Baliged[1]**2 + Vproton_Baliged[2]**2)
    
    Valpha_Baligned = rotateVectorIntoFieldAligned(Valpha[0], Valpha[1], Valpha[2], Nx, Ny, Nz, Px, Py, Pz, Qx, Qy, Qz)
    ValphaPara, ValphaPerp = Valpha_Baligned[0], np.sqrt(Valpha_Baligned[1]**2 + Valpha_Baligned[2]**2)

    # Temperature
    TparaAlpha, TperpAlpha = Temperature_para_perp(Alphas)
    TparaProton, TperpProton = Temperature_para_perp(Protons)

    # VA
    B_magnitude = np.sqrt(np.sum(magF_SRF**2)) * 1e-9  #T
    mu0 = 4 * np.pi * 1e-7 # N/A^2
    mp = 1.67*1e-27 # kg
    ma = 4 * mp # kg
    dens_p = Nproton
    dens_a = Nalpha

    VA = B_magnitude / np.sqrt(mu0 * (dens_p * mp + dens_a * ma)) / 1000.0 #  km/s

    overlap_alpha_ref, overlap_proton_ref = cal_overlap(Protons, Alphas)

    moment_features = cal_moment_plausibility_features(Protons, Alphas)
    flag = moment_features['moment_plausibility_flag']

    Moments = {
            "Bsrf0": magF_SRF[0],
            "Bsrf1": magF_SRF[1],
            "Bsrf2": magF_SRF[2],
            "Vsrf0_Proton": Vproton[0] / 1e3,
            "Vsrf1_Proton": Vproton[1] / 1e3,
            "Vsrf2_Proton": Vproton[2] / 1e3,
            "Vpara_Proton": VprotonPara / 1e3,
            "Vperp1_Proton": Vproton_Baliged[1] / 1e3,
            "Vperp2_Proton": Vproton_Baliged[2] / 1e3,
            "Vsrf0_Alpha": Valpha[0] / 1e3,
            "Vsrf1_Alpha": Valpha[1] / 1e3,
            "Vsrf2_Alpha": Valpha[2] / 1e3,
            "Vpara_Alpha": ValphaPara / 1e3,
            "Vperp1_Alpha": Valpha_Baligned[1] / 1e3,
            "Vperp2_Alpha": Valpha_Baligned[2] / 1e3,
            "Nproton": Nproton / 1e6,
            "Nalpha": Nalpha / 1e6,
            "VprotonPara": VprotonPara / 1e3,
            "VprotonPerp": VprotonPerp / 1e3,
            "ValphaPara": ValphaPara / 1e3,
            "ValphaPerp": ValphaPerp / 1e3,
            "VA": VA,
            "TparaProton": TparaProton,
            "TperpProton": TperpProton,
            "TparaAlpha": TparaAlpha,
            "TperpAlpha": TperpAlpha,
            "overlap_alpha_ref": overlap_alpha_ref,
            "overlap_proton_ref": overlap_proton_ref,
            "Time": Time,
            "moment_plausibility_flag": flag
    }

    return Moments

def Read_One_Day(Date):
    """Read one day of data and calculate the moments, and save to a moment file"""

    day_start = time.perf_counter()

    proton_vdf_file = f'result/SO/VDFs/Proton_vdf_{Date}.h5'
    alpha_vdf_file = f'result/SO/VDFs/Alpha_vdf_{Date}.h5'
    moment_file = f'result/SO/Moments/Moments_{Date}.csv'
    one_particle_noise_file = f'result/SO/Noise_Level/one_particle_noise_level_{Date}.npz'
    # Skip if moments already computed
    if os.path.exists(moment_file):
        print(f"Moment file already exists for {Date}, skipping processing.")
        return 0

    # Skip if VDF files are missing (safety check)
    if not (os.path.exists(proton_vdf_file) and os.path.exists(alpha_vdf_file)):
        print(f"VDF file(s) missing for {Date}, skipping processing.")
        return 0
    
    # Ensure MAG file exists
    try:
        mag_file = ensure_mag_file(Date)
    except FileNotFoundError as e:
        print(f"Cannot process date {Date}: {e}")
        raise
    
    # Noise level
    loaded_data = np.load(one_particle_noise_file)
    one_particle_noise_level = loaded_data['noise_level']

    moments_rows = []

    with h5py.File(proton_vdf_file, 'r') as fp, h5py.File(alpha_vdf_file, 'r') as fa:
        sample_count = len(fp['ptr']) - 1
        mag_cdffile = pycdf.CDF(mag_file)

        try:
            mag_b_srf = mag_cdffile['B_SRF']
            mag_epoch = mag_cdffile['EPOCH']

            for idx in range(sample_count):
                sample_time = pd.to_datetime(fp['time'][idx])
                start_time = sample_time - pd.Timedelta(seconds=0.5)
                end_time = sample_time + pd.Timedelta(seconds=0.5)

                tslice_start = bisect.bisect_left(mag_epoch, start_time)
                tslice_end = bisect.bisect_right(mag_epoch, end_time)
                magF_SRF = np.asarray(mag_b_srf[tslice_start:tslice_end].mean(axis=0))

                protons = SolarWindParticle(
                    species='proton',
                    time=sample_time,
                    magfield=magF_SRF,
                    grid=[elevation, azimuth, vel_proton],
                    coord_type='Spherical',
                )
                p_vdf = _reconstruct_vdf(fp, idx)
                p_vdf[p_vdf < one_particle_noise_level] = 0.0
                protons.set_vdf(p_vdf)

                alphas = SolarWindParticle(
                    species='alpha',
                    time=sample_time,
                    magfield=magF_SRF,
                    grid=[elevation, azimuth, vel_alpha],
                    coord_type='Spherical',
                )
                a_vdf = _reconstruct_vdf(fa, idx)
                a_vdf[a_vdf < one_particle_noise_level] = 0.0
                alphas.set_vdf(a_vdf)

                moments = cal_moments(protons, alphas)

                moments_rows.append(moments)

                del protons, alphas, moments, magF_SRF

        finally:
            mag_cdffile.close()
    
    print(f"Calculated moments for {len(moments_rows)} samples on {Date}. Adding spacecraft spherical coordinates...")
    moments_rows = add_sc_spherical_coords_to_moments(moments_rows, 
                                                    timestamps_npy_path='../Alpha_Analysis/data/SO/From_Harry/timestamps.npy',
                                                    sph_coords_npz_path='../Alpha_Analysis/data/SO/From_Harry/solo_mag_spherical.npz')

    print(f"Writing moments to file for {Date}...")
    _write_moments(moment_file, moments_rows)
    del moments_rows
    gc.collect()

    day_elapsed = time.perf_counter() - day_start
    print(f"Finished processing date: {Date} in {day_elapsed:.2f} s")

    return 0


def main():
    """Main execution function."""
    run_start = time.perf_counter()

    StartDate = '20230603'
    EndDate = '20231016'

    date_range = _iter_dates(StartDate, EndDate)
    # Remove dates that do not have VDF files to avoid wasted work on data-gap days
    date_range = _filter_dates_with_vdfs(date_range)
    max_workers = 20
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        if len(date_range) == 0:
            print("No available dates with VDF files in the requested range. Exiting.")
            return 0

        future_to_date = {executor.submit(Read_One_Day, date): date for date in date_range}

        for future in concurrent.futures.as_completed(future_to_date):
            date = future_to_date[future]
            try:
                future.result()
            except Exception as exc:
                print(f"Failed processing date {date}: {exc}")
                raise

    run_elapsed = time.perf_counter() - run_start
    print(f"Total runtime: {run_elapsed:.2f} s")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())