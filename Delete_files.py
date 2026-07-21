# After running the week, let's delete the data and results to save space.
# We will keep the code for reference, but we will not run it again.

import os
import sys
import subprocess
import time
from datetime import datetime, timedelta

os.chdir(sys.path[0])

# === CONFIG ===
DAY_BEGIN = '20240507'
DAY_END = '20240507'
# === END CONFIG ===

def days_inbetween(day_start, day_end):
    # day_start and day_end are in 'yymmdd' format, str
    # return the list of days in 'yymmdd' format in between, including day_start and day_end
    ds = datetime.strptime(day_start, "%Y%m%d")
    de = datetime.strptime(day_end, "%Y%m%d")

    if ds > de:
        return []

    days = []
    current = ds
    while current <= de:
        days.append(current.strftime("%Y%m%d"))
        current += timedelta(days=1)
    return days


def _kill_holders(path):
    """Find and kill any process with open files under *path*.

    Without this, NFS will "silly rename" files that are still open when we
    ``rm -rf``, leaving behind undeletable ``.nfsXXXX`` stubs that pin the
    parent directory.
    """
    if not os.path.exists(path):
        return
    try:
        result = subprocess.run(
            ["fuser", path],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return
        pids = result.stdout.strip().split()
        print(f"  [fuser] processes holding files in {path}: {pids}")
        for sig in [15, 9]:   # SIGTERM then SIGKILL
            for pid in list(pids):
                try:
                    os.kill(int(pid), sig)
                except ProcessLookupError:
                    pids.remove(pid)
            if pids:
                time.sleep(1.5)
    except Exception as e:
        print(f"  [fuser] non-fatal error: {e}")


def nuke_dir(path):
    """Safely remove *path*, handling NFS silly-rename stubs."""
    if not os.path.exists(path):
        return

    # Give any exiting processes a moment to release file handles.
    time.sleep(0.5)

    # Evict any process still holding files here.
    _kill_holders(path)

    # First attempt.
    subprocess.run(["rm", "-rf", path], check=False)

    if not os.path.exists(path):
        print(f"Removed {path}")
        return

    # If the directory survived, check for .nfs stubs.
    try:
        entries = os.listdir(path)
    except PermissionError:
        entries = []
    nfs_files = [e for e in entries if e.startswith('.nfs')]

    if nfs_files:
        print(f"  Found {len(nfs_files)} .nfs stub(s) in {path} — "
              f"a process still holds these files open.")
        # The .nfs stubs lock the directory.  Try to find the process via
        # the stub files themselves and kill it.
        for nf in nfs_files:
            stub = os.path.join(path, nf)
            try:
                result = subprocess.run(
                    ["fuser", stub],
                    capture_output=True, text=True, timeout=10,
                )
                if result.stdout.strip():
                    for pid in result.stdout.strip().split():
                        try:
                            os.kill(int(pid), 9)
                            print(f"  Killed PID {pid} holding {nf}")
                        except ProcessLookupError:
                            pass
            except Exception:
                pass
        time.sleep(1)

    # Retry removal.  .nfs stubs auto-disappear once the holding process
    # releases (or is killed), so this usually succeeds on the second try.
    subprocess.run(["rm", "-rf", path], check=False)

    if os.path.exists(path):
        print(f"  WARNING: could not fully remove {path} — "
              f"try manually after the pipeline finishes.")
    else:
        print(f"Removed {path}")

def main():
    days = days_inbetween(DAY_BEGIN, DAY_END)

    # Keep the one_particle_noise level files.
    for day in days:
        if os.path.exists(f"result/SO/{day}/one_particle_noise_level.npz"):
            os.rename(f"result/SO/{day}/one_particle_noise_level.npz", f"result/SO/Noise_Level/one_particle_noise_level_{day}.npz")
            print(f"Moved one_particle_noise_level.npz for {day}")

    for day in days:
        nuke_dir(f"data/SO/{day}")
        nuke_dir(f"result/SO/{day}")

    return 0

if __name__ == "__main__":
    main()