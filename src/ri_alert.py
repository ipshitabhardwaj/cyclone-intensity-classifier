"""
Rapid Intensification (RI) Alert.

Standard definition (Kaplan & DeMaria, 2003; used by NHC and widely adopted
across basins including the North Indian Ocean literature we cited in the
Solution Overview doc): a tropical cyclone is undergoing rapid intensification
if its maximum sustained wind increases by >= 30 knots in any 24-hour window.

This module works on two kinds of input:
  1. A best-track CSV (ground truth) -- lets us verify the algorithm against
     a real, documented RI event (Amphan) with no model involved.
  2. A time-ordered list of MODEL-PREDICTED wind speeds for a storm (once we
     have a real image sequence, e.g. from the MOSDAC order) -- same function,
     so the alert logic doesn't change between "checking history" and
     "live forecasting", only the source of the wind-speed series does.

Usage:
    python src/ri_alert.py --besttrack data/besttrack/amphan_2020.csv
"""
import argparse
import csv
from datetime import datetime

RI_THRESHOLD_KT = 30.0
RI_WINDOW_HOURS = 24.0


def find_ri_events(times, winds_kt, threshold_kt=RI_THRESHOLD_KT, window_hours=RI_WINDOW_HOURS):
    """Given parallel lists of datetimes and wind speeds (knots), return every
    (start_time, end_time, start_kt, end_kt, delta_kt) window where wind rose
    by >= threshold_kt within <= window_hours. Not just adjacent points -- it
    checks every pair, since RI can straddle several best-track points.
    """
    events = []
    n = len(times)
    for i in range(n):
        for j in range(i + 1, n):
            dt_hours = (times[j] - times[i]).total_seconds() / 3600.0
            if dt_hours > window_hours:
                break
            delta = winds_kt[j] - winds_kt[i]
            if delta >= threshold_kt:
                events.append((times[i], times[j], winds_kt[i], winds_kt[j], delta))
    return events


def merge_overlapping(events):
    """Collapse a run of overlapping RI windows into one alert per continuous
    intensifying episode, reporting its full extent (earliest start, latest
    end reached during that same episode, max delta observed)."""
    if not events:
        return []
    events = sorted(events, key=lambda e: e[0])
    merged = [list(events[0])]
    for start, end, w0, w1, delta in events[1:]:
        last = merged[-1]
        if start <= last[1]:  # overlaps the ongoing episode
            if end > last[1]:
                last[1] = end
                last[3] = w1
            last[4] = max(last[4], w1 - last[2])
        else:
            merged.append([start, end, w0, w1, delta])
    return merged


def load_series(besttrack_csv):
    times, winds = [], []
    with open(besttrack_csv, newline="") as f:
        for r in csv.DictReader(f):
            hm = r["time_utc"].zfill(4)
            times.append(datetime.strptime(r["date"] + hm, "%Y-%m-%d%H%M"))
            winds.append(float(r["wind_kt"]))
    return times, winds


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--besttrack", required=True)
    args = ap.parse_args()

    times, winds = load_series(args.besttrack)
    events = find_ri_events(times, winds)
    episodes = merge_overlapping(events)

    print(f"Loaded {len(times)} points, {times[0]} -> {times[-1]}")
    print(f"RI threshold: >= {RI_THRESHOLD_KT:.0f} kt increase within {RI_WINDOW_HOURS:.0f}h "
          f"(Kaplan & DeMaria 2003)\n")
    if not episodes:
        print("No RI episode detected.")
    for start, end, w0, w1, delta in episodes:
        hours = (end - start).total_seconds() / 3600.0
        print(f"RI ALERT: {start} ({w0:.0f}kt) -> {end} ({w1:.0f}kt) "
              f"= +{delta:.0f}kt over {hours:.0f}h")
