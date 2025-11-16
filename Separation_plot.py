import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
from scipy.ndimage import gaussian_filter
from SolarWindPack import *
from Funcs import *

from datetime import datetime
from matplotlib.dates import DateFormatter
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import concurrent.futures
import gc

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


def log10_1D_dist(vel, vdf):
    def log10_vdf(vdf):
        vdf = np.array(vdf)  # Ensure vdf is a NumPy array
        mask = vdf > 0  # Create a mask for positive values
        result = np.zeros_like(vdf)  # Initialize an array of zeros
        result[mask] = np.log10(vdf[mask])  # Compute log10 only for positive values
        return result
    y = log10_vdf(np.sum(vdf, axis=(0, 1)))
    mask = y != 0
    x = vel[mask]
    y = y[mask]
    return x, y

def log10_1D_dist(vel, vdf):
    def log10_vdf(vdf):
        vdf = np.array(vdf)  # Ensure vdf is a NumPy array
        mask = vdf > 0  # Create a mask for positive values
        result = np.zeros_like(vdf)  # Initialize an array of zeros
        result[mask] = np.log10(vdf[mask])  # Compute log10 only for positive values
        return result
    y = log10_vdf(np.sum(vdf, axis=(0, 1)))
    mask = y != 0
    x = vel[mask]
    y = y[mask]
    return x, y

def plot_separation(moments_df, yymmdd, idx, hhmmss):

    file_dir = f'result/SO/{yymmdd}/Particles/Ions/{hhmmss}/'

    Protons = read_pickle(file_dir + 'Protons.pkl')
    Alphas = read_pickle(file_dir + 'Alphas.pkl')


    # Proton and Alpha VDFs
    file_dir = f'result/SO/{yymmdd}/Particles/Ions/{hhmmss}/'

    Protons = read_pickle(file_dir + 'Protons.pkl')
    Alphas = read_pickle(file_dir + 'Alphas.pkl')

    vel = Protons.grid['velocity']
    theta = Protons.grid['elevation']
    phi = Protons.grid['azimuth']
    magfield = Protons.magfield

    core_vdf = Protons.get_vdf('core')
    beam_vdf = Protons.get_vdf('beam')
    proton_vdf = Protons.get_vdf()
    alpha_vdf = Alphas.get_vdf()

    BulkVelocity = cal_bulk_velocity_Spherical(Protons)
    AlphaVelocity = cal_bulk_velocity_Spherical(Alphas)

    (Nx, Ny, Nz, Px, Py, Pz, Qx, Qy, Qz) = fieldAlignedCoordinates(magfield[0], magfield[1], magfield[2])
    (V_para_bulk, V_perp1_bulk, V_perp2_bulk) = rotateVectorIntoFieldAligned(BulkVelocity[0], BulkVelocity[1], BulkVelocity[2], Nx, Ny, Nz, Px, Py, Pz, Qx, Qy, Qz)
    V_perp_bulk = np.sqrt(V_perp1_bulk**2 + V_perp2_bulk**2)
    (V_para_alpha, V_perp1_alpha, V_perp2_alpha) = rotateVectorIntoFieldAligned(AlphaVelocity[0], AlphaVelocity[1], AlphaVelocity[2], Nx, Ny, Nz, Px, Py, Pz, Qx, Qy, Qz)
    V_perp_alpha = np.sqrt(V_perp1_alpha**2 + V_perp2_alpha**2)


    # Moments data
    times = moments_df.index
    B_magnitudes =np.sqrt(moments_df['Bsrf0'] ** 2 + moments_df['Bsrf1'] ** 2 + moments_df['Bsrf2'] ** 2)
    magfields = moments_df[['Bsrf0', 'Bsrf1', 'Bsrf2']].values

    ProtonVelocities = moments_df[['Vp_SRF_0', 'Vp_SRF_1', 'Vp_SRF_2']].values
    AlphaVelocities = moments_df[['Va_SRF_0', 'Vp_SRF_1', 'Vp_SRF_2']].values 

    ProtonVelocity_magnitudes = np.sqrt(ProtonVelocities[:, 0]**2 + ProtonVelocities[:, 1]**2 + ProtonVelocities[:, 2]**2)
    AlphaVelocity_magnitudes = np.sqrt(AlphaVelocities[:, 0]**2 + AlphaVelocities[:, 1]**2 + AlphaVelocities[:, 2]**2)

    Vap_magnitudes = moments_df['Vap']
    VA = moments_df['VA']

    ProtonDensities = moments_df['Np']
    AlphaDensities = moments_df['Nalpha']

    TProton_para = moments_df['TparaPcore']
    TProton_perp = moments_df['TperpPcore']
    TAlpha_para = moments_df['TparaAlpha']
    TAlpha_perp = moments_df['TperpAlpha']

    # Smooth some data.
    smoothed_Va = gaussian_filter(AlphaVelocity_magnitudes, sigma=2)
    smoothed_Vap = gaussian_filter(Vap_magnitudes, sigma=2)
    smoothed_Nap = gaussian_filter(AlphaDensities / ProtonDensities, sigma=2)
    smoothed_Ta = gaussian_filter(TAlpha_para + 2 * TAlpha_perp, sigma=2)
    smoothed_TAaniso = gaussian_filter(TAlpha_perp / TAlpha_para, sigma=2)
    smoothed_Na = gaussian_filter(AlphaDensities, sigma=2)

    # Some plot settings
    dateform=DateFormatter('%H:%M')

    plt.rcParams.update({'font.size': 15})
    colors_use = ['#23486A', '#F0A04B', '#A94A4A', '#889E73', '#A37903FF']


    # Plot here
    fig = plt.figure(figsize=(22, 10))
    gs = GridSpec(2, 4, figure=fig)

    ax_big = fig.add_subplot(gs[:, :2])
    plt.setp(ax_big.get_xticklabels(), visible=False)
    plt.setp(ax_big.get_yticklabels(), visible=False)
    inner_gs = GridSpecFromSubplotSpec(7, 1, wspace=0, hspace=0.1, subplot_spec=gs[:, :2])
    sub_axes = []
    for i in range(7):
        ax = fig.add_subplot(inner_gs[i])
        sub_axes.append(ax)

    sub_axes[0].plot(times, B_magnitudes, color=colors_use[0], linewidth=0.8, label='|B|')
    sub_axes[0].plot(times, -magfields[:, 0], color=colors_use[1], linewidth=0.8, label='Br')
    sub_axes[0].plot(times, -magfields[:, 1], color=colors_use[2], linewidth=0.8, label='Bt')
    sub_axes[0].plot(times, magfields[:, 2], color=colors_use[3], linewidth=0.8, label='Bn')
    sub_axes[0].axvline(times[idx], color='red', linestyle='--')
    sub_axes[0].set_ylabel('B [nT]')
    sub_axes[0].legend(frameon=False, loc='upper right', fontsize=12)
    plt.setp(sub_axes[0].get_xticklabels(), visible=False)

    sub_axes[1].plot(times, ProtonVelocity_magnitudes / 1e3, color=colors_use[0], linewidth=0.8, label='Vp')
    sub_axes[1].plot(times, AlphaVelocity_magnitudes / 1e3, color=colors_use[1], linewidth=0.8, label='Va')
    sub_axes[1].plot(times, smoothed_Va / 1e3, color='red', linewidth=0.8)
    sub_axes[1].axvline(times[idx], color='red', linestyle='--')
    sub_axes[1].set_ylabel('V [km/s]')
    sub_axes[1].legend(frameon=False, loc='upper right', fontsize=12)
    plt.setp(sub_axes[1].get_xticklabels(), visible=False)

    sub_axes[2].plot(times, Vap_magnitudes, color=colors_use[0], linewidth=0.8, label='Vap')
    sub_axes[2].plot(times, smoothed_Vap, color='red', linewidth=0.8)
    sub_axes[2].plot(times, VA, color=colors_use[1], linewidth=0.8, label='VA')
    sub_axes[2].axvline(times[idx], color='red', linestyle='--')
    sub_axes[2].set_ylabel('V [km/s]')
    sub_axes[2].legend(frameon=False, loc='upper right', fontsize=12)
    plt.setp(sub_axes[2].get_xticklabels(), visible=False)

    sub_axes[3].plot(times, ProtonDensities, color=colors_use[0], linewidth=0.8, label='Np')
    sub_axes[3].axvline(times[idx], color='red', linestyle='--')
    sub_axes[3].set_ylabel('Np [cm$^{-3}$]', color=colors_use[0])
    plt.setp(sub_axes[3].get_xticklabels(), visible=False)

    sub_axes[4].plot(times, AlphaDensities / ProtonDensities * 100, color=colors_use[0], linewidth=0.8, label='Na/Np')
    sub_axes[4].set_ylim(0, 7)
    sub_axes[4].axvline(times[idx], color='red', linestyle='--')
    sub_axes[4].set_ylabel('Na/Np [%]')
    plt.setp(sub_axes[4].get_xticklabels(), visible=False)

    sub_axes[5].plot(times, (TProton_para + 2 * TProton_perp) / 3, color=colors_use[0], linewidth=0.8, label='Tp')
    sub_axes[5].plot(times, (TAlpha_para + 2 * TAlpha_perp) / 3 / 4.0, color=colors_use[1], linewidth=0.8, label='Ta / 4')
    sub_axes[5].axvline(times[idx], color='red', linestyle='--')
    sub_axes[5].set_ylabel('T [eV]')
    sub_axes[5].legend(frameon=False, loc='upper right', fontsize=12)
    sub_axes[5].set_ylim(10, 30)
    plt.setp(sub_axes[5].get_xticklabels(), visible=False)

    sub_axes[6].plot(times, TProton_perp / TProton_para, color=colors_use[0], linewidth=0.8, label='p')
    sub_axes[6].plot(times, TAlpha_perp / TAlpha_para, color=colors_use[1], linewidth=0.8, label='a')
    sub_axes[6].plot(times, smoothed_TAaniso, color='red', linewidth=0.8)
    sub_axes[6].axvline(times[idx], color='red', linestyle='--')
    sub_axes[6].set_ylabel(r'$T_{\perp} / T_{\parallel}$')
    sub_axes[6].set_xlabel('Time [UT]')
    sub_axes[6].set_ylim(0.5, 2.0)
    sub_axes[6].legend(frameon=False, loc='upper right', fontsize=12)

    min_vdf = np.minimum(proton_vdf, alpha_vdf)
    overlap = np.sum(min_vdf / np.sum(alpha_vdf))

    ax1 = fig.add_subplot(gs[0, 2])
    x, y = log10_1D_dist(vel / 1e3, proton_vdf + alpha_vdf)
    ax1.plot(x, y, label='Total', color='black')
    ax1.scatter(x, y, s=20, color='black', marker='s')
    x, y = log10_1D_dist(vel / 1e3, core_vdf)
    ax1.plot(x, y, label='Core', color='red')
    ax1.scatter(x, y, s=20, color='red')
    x, y = log10_1D_dist(vel / 1e3, beam_vdf)
    ax1.plot(x, y, label='Beam', color='blue')
    ax1.scatter(x, y, s=20, color='blue')
    x, y = log10_1D_dist(vel / 1e3, alpha_vdf)
    ax1.plot(x, y, label='Alpha', color='green')
    ax1.scatter(x, y, s=20, color='green')
    ax1.set_ylim(-12, -6)
    ax1.set_title(str(hhmmss) + ' VDF')
    ax1.set_ylabel('log10(VDF)')
    ax1.set_xlabel('Velocity (km/s)')
    ax1.legend(frameon=False)

    ax2 = fig.add_subplot(gs[0, 3])
    x, y = log10_1D_dist(vel / 1e3, proton_vdf + alpha_vdf)
    ax2.plot(x, y, label='Total', color='black')
    ax2.scatter(x, y, s=20, color='black', marker='s')
    x, y = log10_1D_dist(vel / 1e3, proton_vdf)
    ax2.plot(x, y, label='Proton', color='red')
    ax2.scatter(x, y, s=20, color='red')
    x, y = log10_1D_dist(vel / 1e3, alpha_vdf)
    ax2.plot(x, y, label='Alpha', color='green')
    ax2.scatter(x, y, s=20, color='green')
    min_vdf = np.minimum(proton_vdf, alpha_vdf)
    overlap = np.sum(min_vdf) / np.sum(alpha_vdf)
    percentage = f"{overlap * 100:.2f}%"
    ax2.text(0.50, 0.3, "Overlap " + percentage, transform=ax2.transAxes, fontsize=12, ha='right', va='top')
    ax2.set_ylim(-12, -6)
    ax2.set_title(str(hhmmss) + ' VDF')
    ax2.set_xlabel('Velocity (km/s)')
    ax2.legend(frameon=False)

    # Remove the one-particle-noise-level.
    loaded_data = np.load(f'result/SO/{yymmdd}/one_particle_noise_level.npz')
    one_particle_noise_level = loaded_data['noise_level']

    pcore_vdf = Protons.get_vdf('core')
    pbeam_vdf = Protons.get_vdf('beam')
    alpha_vdf = Alphas.get_vdf()

    pcore_vdf[pcore_vdf < one_particle_noise_level] = 0
    pbeam_vdf[pbeam_vdf < one_particle_noise_level] = 0
    alpha_vdf[alpha_vdf < one_particle_noise_level] = 0

    Protons.set_vdf(pcore_vdf, component='core')
    Protons.set_vdf(pbeam_vdf, component='beam')
    Alphas.set_vdf(alpha_vdf)

    # Ion density.
    Ion_density = (cal_density_Spherical(Protons) + 2 * cal_density_Spherical(Alphas)) / 1e6  # in cm^-3.

    # To Field-aligned 
    TMatrix = np.array([[-1, 0, 0], 
                        [0, 1, 0], 
                        [0, 0, -1]])    # This matrix basically transfers measurements into SRF coordinate. Please refer to the data tutorial for details.
                        
    V_bulk_SRF = cal_bulk_velocity_Spherical(Protons)

    Protons_FieldAligned = transferToFieldAligned(Protons, transfer_Matrix=TMatrix, VPbulk_SRF=V_bulk_SRF)

    # Alphas are a bit more complex. We need to consider the sqrt(2).
    Alphas_FieldAligned = transferToFieldAligned(Alphas, transfer_Matrix=TMatrix, VPbulk_SRF=V_bulk_SRF)

    vpara_p = Protons_FieldAligned.grid['Vpara']
    vperp_p = np.sqrt(Protons_FieldAligned.grid['Vperp1'] ** 2 + Protons_FieldAligned.grid['Vperp2'] ** 2)
    vdf_p = Protons_FieldAligned.get_vdf()
    mask_p = vdf_p > 0
    vpara_p = vpara_p[mask_p]
    vperp_p = vperp_p[mask_p]
    vdf_p = vdf_p[mask_p]

    ax3 = fig.add_subplot(gs[1, 2])
    cs1 = ax3.scatter(vpara_p / 1e3, vperp_p / 1e3, c=np.log10(vdf_p), cmap='viridis', s=10)
    ax3.set_xlabel('$V_{para}$ [km/s]')
    ax3.set_ylabel('$V_{perp}$ [km/s]')
    cbar1 = fig.colorbar(cs1, ax=ax3)
    ax3.text(
        0.95, 0.95,              # x and y position in axes fraction (95% of the width and height)
        "V [km/s]: " + str(round(V_para_bulk / 1e3, 2)) + '  ' + str(round(V_perp_bulk / 1e3, 2)),  # Text to display
        transform=ax3.transAxes,  # Use axes fraction coordinates
        fontsize=12,             # Font size
        ha='right',              # Horizontal alignment
        va='top'                 # Vertical alignment
    )

    vpara_a = Alphas_FieldAligned.grid['Vpara']
    vperp_a = np.sqrt(Alphas_FieldAligned.grid['Vperp1'] ** 2 + Alphas_FieldAligned.grid['Vperp2'] ** 2)
    vdf_a = Alphas_FieldAligned.get_vdf()
    mask_a = vdf_a > 0
    vpara_a = vpara_a[mask_a]
    vperp_a = vperp_a[mask_a]
    vdf_a = vdf_a[mask_a]

    ax4 = fig.add_subplot(gs[1, 3])
    cs2 = ax4.scatter(vpara_a / 1e3, vperp_a / 1e3, c=np.log10(vdf_a), cmap='viridis', s=10)
    ax4.set_xlabel('$V_{para}$ [km/s]')
    cbar2 = fig.colorbar(cs2, ax=ax4, label='log$_{10}$(VDF)')
    ax4.text(
        0.95, 0.95,              # x and y position in axes fraction (95% of the width and height)
        "V [km/s]: " + str(round(V_para_alpha / 1e3, 2)) + '  ' + str(round(V_perp_alpha / 1e3, 2)),  # Text to display
        transform=ax4.transAxes,  # Use axes fraction coordinates
        fontsize=12,             # Font size
        ha='right',              # Horizontal alignment
        va='top'                 # Vertical alignment
    )

    plt.subplots_adjust(wspace=0.3)
    plt.savefig(f'result/SO/{yymmdd}/Particles/Ions/{hhmmss}/Separation.png', dpi=300)
    plt.close('all')

    del Protons, Alphas, core_vdf, beam_vdf, proton_vdf, alpha_vdf
    print('Done: ', hhmmss)
    gc.collect()

    return 0



def main(): 

    yymmdd = '20220302'
    tstart = '043000'
    tend = '053000'


    ion_hhmmss_list = sorted([time_str for time_str in os.listdir(f'result/SO/{yymmdd}/Particles/Ions/') 
                            if _isValidTimeStr(time_str) and os.path.exists(f'result/SO/{yymmdd}/Particles/Ions/{time_str}/moments.txt')])
    times_in_between = times_inbetween(tstart, tend, ion_hhmmss_list)

    moments_df = pd.read_csv(f'result/SO/{yymmdd}/moments_{tstart}_to_{tend}.csv', index_col=0, parse_dates=True)

    with concurrent.futures.ProcessPoolExecutor(max_workers=12) as executor:
        results = [executor.submit(plot_separation, moments_df, yymmdd, idx, hhmmss) for idx, hhmmss in enumerate(times_in_between)]
        for f in concurrent.futures.as_completed(results):
            print(f.result())


    return 0

if __name__ == "__main__":
    main()
