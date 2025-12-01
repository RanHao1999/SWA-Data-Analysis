"""
Created on Tue Jul 8 10:13:33 2025

@author: CI
"""
"INFO FROM https://git.ias.u-psud.fr/spice/data-analysis-club/-/blob/main/20231205-soar-tap/tap-demo.ipynb"


import os
import requests
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
# print(os.getcwd())

start_total = time.time()

# === Configuration ===
soar_end_point = "https://soar.esac.esa.int/soar-sl-tap"

file_prefixes = [
    "solo_L2_swa-pas-grnd-mom",
    "solo_L2_swa-pas-vdf",
    "solo_L1_swa-pas-3d",
    "solo_L2_mag-srf-normal"
]

versions = ["V03", "V02", "V01"]
start_date = datetime(2022, 3, 25)
end_date = datetime(2022, 3, 25)  # inclusive

# === Prepare Directories ===
# os.makedirs(project_dir, exist_ok=True)
# os.chdir(project_dir)


def download_file(file_prefix, date_obj):
    date_str = date_obj.strftime('%Y%m%d')
    save_folder = os.path.join("data/SO", date_str)
    os.makedirs(save_folder, exist_ok=True)

    for version in ["V03", "V02", "V01"]:
        versioned_filename = f"{file_prefix}_{date_str}_{version}.cdf"
        base_filename = f"{file_prefix}_{date_str}.cdf"
        save_path = os.path.join(save_folder, base_filename)

        print(f"➡️ Attempting download: {versioned_filename}")

        adql_query = f"""
            SELECT filepath, filename 
            FROM soar.v_sc_repository_file 
            WHERE filename = '{versioned_filename}'
        """.strip()

        payload = {
            'retrieval_type': 'ALL_PRODUCTS',
            'QUERY': adql_query
        }

        try:
            r = requests.get(f'{soar_end_point}/data', params=payload)
            r.raise_for_status()

            if r.content:
                with open(save_path, "wb") as f:
                    f.write(r.content)
                print(f"✅ Saved as: {save_path}")
                return
        except Exception as e:
            print(f"❌ Error downloading {versioned_filename}: {e}")

    print(f"⚠️ No valid file found for {file_prefix} on {date_str}")

# === Threaded Execution ===
tasks = []
with ThreadPoolExecutor(max_workers=8) as executor:
    current_date = start_date
    while current_date <= end_date:
        for prefix in file_prefixes:
            tasks.append(executor.submit(download_file, prefix, current_date))
        current_date += timedelta(days=1)

    for future in as_completed(tasks):
        future.result()

total_time = time.time() - start_total
print(f"\n🚀 Total script runtime: {total_time:.2f} seconds.")
