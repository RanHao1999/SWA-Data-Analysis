"""
Make a m ovie of all the separation results.

author: Hao Ran @RanHao1999, hao.ran.24@ucl.ac.uk
at: Mullard Space Science Laboratory, University College London
"""

import os
from datetime import datetime
import imageio
import sys

os.chdir(sys.path[0])

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

def main():

    tstart_llst = [datetime(2022, 3, 2, 4, 30, 0)]
    tend_lst = [datetime(2022, 3, 2, 5, 30, 0)]

    for tstart, tend in zip(tstart_llst, tend_lst):
        yymmdd = tstart.strftime("%Y%m%d")
        tstart_str = tstart.strftime("%H%M%S")
        tend_str = tend.strftime("%H%M%S")
        hhmmss_str = tstart.strftime("%H%M%S") + "To" + tend.strftime("%H%M%S")

        ion_hhmmss_list = sorted([time_str for time_str in os.listdir(f'result/SO/{yymmdd}/Particles/Ions/') 
                        if _isValidTimeStr(time_str) and os.path.exists(f'result/SO/{yymmdd}/Particles/Ions/{time_str}/Separation.png')])
        times_inbetween_list = times_inbetween(tstart_str, tend_str, ion_hhmmss_list)
        
        image_files = [f'result/SO/{yymmdd}/Particles/Ions/{hhmmss}/Separation.png' for hhmmss in times_inbetween_list]

        #output_path = f'result/SO/{yymmdd}/4ALPS/{hhmmss_str}/Seperation_Movie_{hhmmss_str}.mp4'
        output_path = f'result/SO/{yymmdd}/Seperation_Movie_{hhmmss_str}.mp4'

        with imageio.get_writer(output_path, fps=15) as writer:
            for filename in image_files:
                image = imageio.imread(filename)
                writer.append_data(image)

        print(f"Movie saved as {output_path}")

    return 0

if __name__ == "__main__":
    main()