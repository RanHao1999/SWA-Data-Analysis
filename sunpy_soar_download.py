"""
Author: Hao Ran
Institution: UCL / MSSL
Email: hao.ran.24@ucl.ac.uk
GitHub: @RanHao1999
Created: 2025-12-11

Description:
    This script is designed to download SOAR data from a specified source and save it to a local directory.
"""

# library imports
from sunpy.net import Fido
from sunpy.net import attrs as a
import sunpy_soar   

import sunpy.net.attrs as a

import os
import sys
from datetime import datetime, timedelta

os.chdir(sys.path[0])

def days_between(start_date: str, end_date: str):
    """
    Return a list of YYYY-MM-DD strings for all days between start_date and end_date (inclusive).
    """
    # Convert to datetime.date
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()

    if end < start:
        raise ValueError("end_date must not be earlier than start_date.")

    delta = end - start
    return [(start + timedelta(days=i)).isoformat() for i in range(delta.days + 1)]


def main():
    """Main execution function."""
    time_start = "2023-09-26"
    time_end = "2023-09-27"

    days_in_between = days_between(time_start, time_end)

    for day in days_in_between:
        day_in_path = day.replace('-', '')
        data_path = f'data/SO/{day_in_path}/'
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        
        SWA = a.Instrument('SWA')
        MAG = a.Instrument('MAG')

        time = a.Time(day, day)
        
        # grnd_mom
        product = a.soar.Product("swa-pas-grnd-mom")
        level = a.Level(2)
        query = Fido.search(SWA & time & product & level)
        if len(query[0]) > 0:
            if os.path.exists(data_path + str(query[0]['Filename'][0])) is False:
                Fido.fetch(query, path=data_path)
        else:
            print(f"No SWA grnd_mom data found for {day}")

        # vdf
        product = a.soar.Product("swa-pas-vdf")
        level = a.Level(2)
        query = Fido.search(SWA & time & product & level)
        if len(query[0]) > 0:
            if os.path.exists(data_path + str(query[0]['Filename'][0])) is False:
                Fido.fetch(query, path=data_path)
        else:
            print(f"No SWA vdf data found for {day}")

        # count
        product = a.soar.Product("swa-pas-3d")
        level = a.Level(1)
        query = Fido.search(SWA & time & product & level)
        if len(query[0]) > 0:
            if os.path.exists(data_path + str(query[0]['Filename'][0])) is False:
                Fido.fetch(query, path=data_path)
        else:
            print(f"No SWA count data found for {day}")
        
        # Mag
        product = a.soar.Product("mag-srf-normal")
        level = a.Level(2)
        query = Fido.search(MAG & time & product & level)
        if len(query[0]) > 0:
            if os.path.exists(data_path + str(query[0]['Filename'][0])) is False:
                Fido.fetch(query, path=data_path)
        else:
            print(f"No MAG data found for {day}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())