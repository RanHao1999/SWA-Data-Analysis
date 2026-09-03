#!/usr/bin/env python3
"""Download the local Solar Orbiter SPICE kernels used by this project."""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path
from urllib.request import urlopen


BASE_URL = "https://spiftp.esac.esa.int/data/SPICE/SOLAR-ORBITER/kernels"
KERNELS = {
    "fk/solo_ANC_soc-sci-fk_V08.tf": "solo_ANC_soc-sci-fk_V08.tf",
    "lsk/naif0012.tls": "naif0012.tls",
    "pck/pck00010.tpc": "pck00010.tpc",
    "sclk/solo_ANC_soc-sclk_20250803_V01.tsc": "solo_ANC_soc-sclk_20250803_V01.tsc",
    "spk/de421.bsp": "de421.bsp",
    (
        "spk/solo_ANC_soc-orbit_20200210-20301120_L022_V1_00464_V02.bsp"
    ): "solo_ANC_soc-orbit_20200210-20301120_L022_V1_00464_V02.bsp",
}


def download_kernels(destination: Path, force: bool = False, dry_run: bool = False) -> None:
    if not dry_run:
        destination.mkdir(parents=True, exist_ok=True)

    for relative_url, filename in KERNELS.items():
        target = destination / filename
        if target.exists() and not force:
            print(f"SKIP  {filename} (already exists)")
            continue
        if dry_run:
            print(f"WOULD DOWNLOAD  {filename}")
            continue

        url = f"{BASE_URL}/{relative_url}"
        print(f"DOWNLOAD  {filename}")
        with urlopen(url, timeout=120) as response:
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=destination, prefix=f".{filename}.", delete=False
            ) as temporary:
                temporary_path = Path(temporary.name)
                shutil.copyfileobj(response, temporary)
        temporary_path.replace(target)

    print(f"SPICE kernels are in: {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path(__file__).parent / "data/SO/SPICE",
        help="directory where the kernels will be stored",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="redownload and replace existing kernels",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="show what would be downloaded without changing files",
    )
    args = parser.parse_args()
    download_kernels(args.destination, args.force, args.dry_run)


if __name__ == "__main__":
    main()
