#!/usr/bin/env python3
"""
Pipeline Controller — master orchestrator for Alpha Processing.

This script edits config blocks inside each sub-script and executes them
as independent subprocesses.  Data passes between steps via the filesystem
(data/SO/{day}/ → result/SO/{day}/ → result/SO/VDFs/).

Workflow per day:
  1. Download SOAR data      (sunpy_soar_download.py)
  2. GMM auto-parallelised    (gmm_auto_parallelised.py)
  3. Save sparse VDFs to HDF5 (Save_vdfs.py)
  4. Delete raw + intermediate (Delete_files.py)

Usage:
    conda activate research_env
    python /disk/plasma/hr2/Alpha_Processing/pipeline_controller.py

Author: Hao Ran
Created: 2025-06-30
Updated: 2026-07-23 — removed ThreadPoolExecutor for background downloads
          (Popen is already non-blocking; forking from a thread was causing
          asyncio signal-handler errors and hung subprocesses on exit).
"""

import os
import re
import sys
import time
import subprocess
from datetime import datetime, timedelta

# Work relative to this script's own directory.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)


# =========================================================================
# Config update helpers
# =========================================================================

def _format_config_value(key, value, original_line):
    """Produce a 'KEY = <formatted_value>' line, preserving indentation."""
    indent = original_line[:len(original_line) - len(original_line.lstrip())]

    if isinstance(value, bool):
        return f"{indent}{key} = {value}\n"
    elif isinstance(value, (int, float)):
        return f"{indent}{key} = {value}\n"
    elif value is None:
        return f"{indent}{key} = None\n"
    elif isinstance(value, str):
        escaped = value.replace("'", "\\'")
        return f"{indent}{key} = '{escaped}'\n"
    else:
        return f"{indent}{key} = {value!r}\n"


def update_config(script_name, **kwargs):
    """Edit the # === CONFIG === block in *script_name* in-place.

    Each kwarg key is matched against config-block lines of the form
    ``KEY = ...``.  Only the first matching line per key is replaced.
    Raises ValueError if the script has no CONFIG block.
    """
    path = os.path.join(SCRIPT_DIR, script_name)
    with open(path) as f:
        lines = f.readlines()

    in_config = False
    block_found = False
    updated = set()
    new_lines = []

    for line in lines:
        stripped = line.strip()

        if stripped == '# === CONFIG ===':
            in_config = True
            block_found = True
            new_lines.append(line)
            continue
        if stripped == '# === END CONFIG ===':
            in_config = False
            new_lines.append(line)
            continue

        if in_config:
            matched = False
            for key, value in kwargs.items():
                if re.match(r'^\s*' + re.escape(key) + r'\s*=', line):
                    new_lines.append(_format_config_value(key, value, line))
                    updated.add(key)
                    matched = True
                    break
            if not matched:
                new_lines.append(line)
        else:
            new_lines.append(line)

    if not block_found:
        raise ValueError(f"{script_name}: no # === CONFIG === block found")

    missing = set(kwargs) - updated
    if missing:
        print(f"  [WARN] {script_name}: keys not found in config: {missing}")

    with open(path, 'w') as f:
        f.writelines(new_lines)


# =========================================================================
# Subprocess launchers
# =========================================================================

def _blas_free_env():
    """Return an environment dict that pins BLAS/OpenMP to 1 thread.

    Without this, each worker inside a ProcessPoolExecutor also tries to
    grab all cores for its own BLAS calls, causing
    N_processes × N_cores oversubscription.
    """
    env = os.environ.copy()
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
        env[var] = "1"
    return env


def run_script(script_name, desc, timeout=None, **cfg):
    """Update config and run *script_name* synchronously.

    Parameters
    ----------
    timeout : float or None
        Seconds before the subprocess is killed.  ``None`` = wait forever.
        On timeout the subprocess is killed (SIGTERM then SIGKILL) and the
        function returns a non-zero exit code.

    Returns the subprocess exit code.
    """
    print(f"\n  [{desc}] Starting..." + (f" (timeout={timeout}s)" if timeout else ""))
    update_config(script_name, **cfg)
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            cwd=SCRIPT_DIR,
            env=_blas_free_env(),
            timeout=timeout,
        )
        rc = result.returncode
    except subprocess.TimeoutExpired:
        print(f"  [{desc}] TIMEOUT after {timeout}s — subprocess killed.")
        rc = 1
    if rc != 0:
        print(f"  [{desc}] FAILED (exit code {rc})")
    else:
        print(f"  [{desc}] OK")
    return rc


def _start_download(yymmdd, label="bg"):
    """Launch sunpy_soar_download.py in the background, return the Popen object.

    ``Popen`` is non-blocking — the download runs concurrently while the
    caller carries on with other work.  Call ``_finish_download()`` later
    to wait for it.
    """
    iso = _to_iso_date(yymmdd)
    print(f"\n  ▶ Pre-fetching {yymmdd} ({label})")
    update_config("sunpy_soar_download.py", TIME_START=iso, TIME_END=iso)
    return subprocess.Popen(
        [sys.executable, "sunpy_soar_download.py"],
        cwd=SCRIPT_DIR,
        env=_blas_free_env(),
    )


def _finish_download(proc, yymmdd, timeout=3600):
    """Wait for a background-download *proc* to complete.

    Kills the subprocess and returns ``False`` if it exceeds *timeout*
    seconds or exits with a non-zero code.  Returns ``True`` on success.
    Safe to call on an already-exited process (``.wait()`` returns immediately).
    """
    if proc is None:
        return True
    try:
        ret = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"  [WARN] Download for {yymmdd} timed out "
              f"after {timeout}s — killing.")
        proc.kill()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            pass  # process is unkillable; abandon it
        return False
    ok = (ret == 0)
    if not ok:
        print(f"  [WARN] Download for {yymmdd} failed (exit {ret}).")
    return ok


# =========================================================================
# Date utilities
# =========================================================================

def days_between(t_start, t_end):
    """Return list of YYYYMMDD strings between two datetimes (inclusive)."""
    d0 = t_start.date() if isinstance(t_start, datetime) else t_start
    d1 = t_end.date() if isinstance(t_end, datetime) else t_end
    if d0 > d1:
        return []
    result = []
    current = d0
    while current <= d1:
        result.append(current.strftime("%Y%m%d"))
        current += timedelta(days=1)
    return result


def _to_iso_date(yymmdd):
    """Convert 'YYYYMMDD' → 'YYYY-MM-DD' (format sunpy_soar_download expects)."""
    return f"{yymmdd[:4]}-{yymmdd[4:6]}-{yymmdd[6:8]}"


def _data_ready(yymmdd):
    """Return True if all required CDF files exist for *yymmdd*."""
    data_dir = os.path.join(SCRIPT_DIR, "data", "SO", yymmdd)
    if not os.path.isdir(data_dir):
        return False
    files = os.listdir(data_dir)
    required = ['pas-vdf', 'pas-grnd-mom', 'pas-3d', 'mag-srf-normal']
    for pattern in required:
        if not any(pattern in f and not f.startswith('._') for f in files):
            return False
    return True

# =========================================================================
# Main pipeline
# =========================================================================

def main():
    # =====================================================================
    # USER SETTINGS — edit these
    # =====================================================================
    PIPELINE_TSTART = datetime(2024, 7, 17, 12, 0, 0)
    PIPELINE_TEND   = datetime(2024, 8, 7, 23, 59, 59)

    DT_WANTED   = 4.0   # desired output cadence (s)
    N_PROCESSES = 20    # parallel workers for GMM fitting
    _DELETE     = True  # False → keep raw data & intermediate products (skip cleanup)
    # =====================================================================

    days = days_between(PIPELINE_TSTART, PIPELINE_TEND)
    if not days:
        print("No days to process — check PIPELINE_TSTART/PIPELINE_TEND.")
        return 1

    total_tstart = time.time()
    print(f"Pipeline: {len(days)} day(s)  |  "
          f"{PIPELINE_TSTART} → {PIPELINE_TEND}")
    print(f"Parameters:  dt={DT_WANTED}s, workers={N_PROCESSES}")
    print(f"{'='*60}")

    # ------------------------------------------------------------------
    # Pre-download the very first day (must finish before processing)
    # ------------------------------------------------------------------
    print(f"\n{'─'*60}")
    print(f"Pre-downloading day 1: {days[0]}")
    run_script(
        "sunpy_soar_download.py",
        f"download {days[0]}",
        TIME_START=_to_iso_date(days[0]),
        TIME_END=_to_iso_date(days[0]),
        timeout=3600,           # 1 h — SOAR can be slow
    )
    # Note: download may fail (datagap / server error).  _data_ready()
    # will catch it on the first iteration and skip the day.

    next_dl_proc = None   # Popen handle for the background download
    next_dl_day = None     # which day is being pre-fetched

    for i, day in enumerate(days):
        print(f"\n{'='*60}")
        print(f"DAY {i+1}/{len(days)}: {day}")
        print(f"{'='*60}")

        # --------------------------------------------------------------
        # Finish the background download that was kicked off during the
        # previous iteration.  It has had the entire GMM run (hours) to
        # complete, so normally this returns immediately.
        # --------------------------------------------------------------
        if next_dl_proc is not None and next_dl_day == day:
            print(f"\n  ⏳ Finishing pre-fetched download for {day}...")
            ok = _finish_download(next_dl_proc, day, timeout=3600)
            next_dl_proc = None
            if not ok:
                print(f"  ⚠ Pre-fetch of {day} failed — "
                      f"re-downloading synchronously.")
                rc = run_script(
                    "sunpy_soar_download.py",
                    f"download {day} (retry)",
                    timeout=3600,
                    TIME_START=_to_iso_date(day),
                    TIME_END=_to_iso_date(day),
                )
                if rc != 0:
                    print(f"  [SKIP] {day}: download failed after retry — "
                          f"moving to next day.")
                    # No next_dl_proc to clean up; just continue to next day.
                    continue

        # --------------------------------------------------------------
        # Kick off download of NEXT day in background (runs while GMM
        # processes the current day).  Must happen BEFORE the data-ready
        # check so a datagap skip doesn't stall the pre-fetch pipeline
        # (otherwise next_dl_proc stays None forever and every subsequent
        # day cascades as "data missing").
        # --------------------------------------------------------------
        if i + 1 < len(days):
            nxt = days[i + 1]
            next_dl_proc = _start_download(nxt, label=f"pre-fetch for day {i+2}")
            next_dl_day = nxt

        # --------------------------------------------------------------
        # Skip day if required data files are still missing (datagap)
        # --------------------------------------------------------------
        if not _data_ready(day):
            print(f"\n  [SKIP] {day}: data missing (datagap) "
                  f"— moving to next day.")
            # next_dl_proc was just set above for the next day, so it
            # will be correctly collected at the top of the next iteration.
            continue

        # --------------------------------------------------------------
        # GMM process current day (foreground)
        # --------------------------------------------------------------
        start_iso = (PIPELINE_TSTART.strftime("%Y-%m-%d %H:%M:%S")
                     if i == 0 else _to_iso_date(day) + " 00:00:00")
        end_iso   = _to_iso_date(day) + " 23:59:59"

        rc = run_script(
            "gmm_auto_parallelised.py",
            f"GMM {day}",
            timeout=14400,   # 4 h — safety net against hung workers
            YYMMDD=day,
            T_START_ISO=start_iso,
            T_END_ISO=end_iso,
            DT_WANTED=DT_WANTED,
            N_PROCESSES=N_PROCESSES,
            _PLOT=False,
        )

        if rc != 0:
            print(f"  [{day}] GMM failed — skipping VDF save and cleanup.")
            # next_dl_proc is still running; it will be collected at the
            # start of the next iteration (or abandoned at end of loop).
            continue

        # --------------------------------------------------------------
        # Save VDFs + cleanup (skip if _DELETE=False)
        # --------------------------------------------------------------
        if _DELETE:
            run_script(
                "Save_vdfs.py",
                f"save VDFs {day}",
                DAY=day,
            )

            run_script(
                "Delete_files.py",
                f"cleanup {day}",
                DAY_BEGIN=day,
                DAY_END=day,
            )
        else:
            print(f"  [{day}] _DELETE=False — keeping raw data & intermediates.")

    # Clean up any remaining background download (should already be done).
    if next_dl_proc is not None:
        print(f"\n  Cleaning up leftover download for {next_dl_day}...")
        _finish_download(next_dl_proc, next_dl_day, timeout=600)
        next_dl_proc = None

    total_tend = time.time()
    elapsed = total_tend - total_tstart
    print(f"\n{'='*60}")
    print(f"Pipeline finished.  Total time: {elapsed/60:.1f} min "
          f"({elapsed/3600:.2f} hr)")
    print(f"{'='*60}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
