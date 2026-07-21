import os
import sys
import numpy as np
import pandas as pd
import h5py

from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta

os.chdir(os.path.dirname(os.path.abspath(__file__)))


BASE_PATH = "result/SO"
VDF_SHAPE = (11, 9, 96)

# === CONFIG ===
DAY = '20240507'
# === END CONFIG ===


def process_one_day(day, species="Protons"):
    print(f"Processing {species} {day}...")

    ion_path = os.path.join(BASE_PATH, day, "Particles/Ions_auto_resample")
    if not os.path.exists(ion_path):
        print(f"{day} missing")
        return

    save_path = f'result/SO/VDFs/'
    os.makedirs(save_path, exist_ok=True)

    outname = f"result/SO/VDFs/{species[:-1]}_vdf_{day}.h5"

    if os.path.exists(outname):
        os.remove(outname)
        print(f"Removed existing file {outname}")

    with h5py.File(outname, "w") as f:

        # create extendable datasets
        f.create_dataset("time", shape=(0,), maxshape=(None,),
                         dtype="int64", compression="lzf")

        f.create_dataset("value", shape=(0,), maxshape=(None,),
                         dtype="float64", compression="lzf")

        f.create_dataset("i", shape=(0,), maxshape=(None,),
                         dtype="uint8", compression="lzf")
        f.create_dataset("j", shape=(0,), maxshape=(None,),
                         dtype="uint8", compression="lzf")
        f.create_dataset("k", shape=(0,), maxshape=(None,),
                         dtype="uint8", compression="lzf")

        f.create_dataset("ptr", shape=(1,), maxshape=(None,),
                         dtype="int64", compression="lzf")
        f["ptr"][0] = 0

        for hhmmss in sorted(os.listdir(ion_path)):

            pkl = os.path.join(ion_path, hhmmss, f"{species}.pkl")
            if not os.path.exists(pkl):
                continue

            obj = pd.read_pickle(pkl)

            t_ns = np.datetime64(obj.time, 'ns').astype('int64')
            vdf = obj.get_vdf()

            nz = np.nonzero(vdf)
            values = vdf[nz].astype(np.float64)

            count = len(values)

            # append time
            tds = f["time"]
            tds.resize(tds.shape[0] + 1, axis=0)
            tds[-1] = t_ns

            # append sparse bins
            for name, arr in zip(
                ["value", "i", "j", "k"],
                [values,
                 nz[0].astype(np.uint8),
                 nz[1].astype(np.uint8),
                 nz[2].astype(np.uint8)]
            ):
                ds = f[name]
                old = ds.shape[0]
                ds.resize(old + count, axis=0)
                ds[old:] = arr

            # update pointer
            ptr = f["ptr"]
            ptr.resize(ptr.shape[0] + 1, axis=0)
            ptr[-1] = ptr[-2] + count

    print(f"Finished {species} {day}")


def date_range(start, end):
    d0 = datetime.strptime(start, "%Y%m%d")
    d1 = datetime.strptime(end, "%Y%m%d")

    d = d0
    while d <= d1:
        yield d.strftime("%Y%m%d")
        d += timedelta(days=1)

def main():
    """Save sparse VDFs to HDF5 for a single day (Protons + Alphas)."""
    process_one_day(DAY, "Protons")
    process_one_day(DAY, "Alphas")
    return 0


if __name__ == "__main__":
    main()