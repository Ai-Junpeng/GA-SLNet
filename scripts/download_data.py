"""Download the public AIS daily dumps used in this study.

Source: NOAA MarineCadastre AIS (US coastal waters, 2020)
        https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2020/

The raw dumps are NOT redistributed in this repository because a single day is
several hundred MB. Run this script once, then run the preprocessing step.

Usage
-----
    python scripts/download_data.py                 # all three dates
    python scripts/download_data.py --date 2020-01-01
"""
import argparse
import os
import sys
import urllib.request

DATES = ["2020-01-01", "2020-07-01", "2020-12-01"]
BASE = "https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2020"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, "data", "raw")


def fetch(date, out_dir=RAW_DIR):
    os.makedirs(out_dir, exist_ok=True)
    name = f"AIS_{date.replace('-', '_')}.zip"
    url = f"{BASE}/{name}"
    dst = os.path.join(out_dir, name)

    if os.path.exists(dst) and os.path.getsize(dst) > 10_000_000:
        print(f"[skip] {name} already present ({os.path.getsize(dst)/1e6:.0f} MB)")
        return dst

    print(f"[get ] {url}")
    tmp = dst + ".part"
    try:
        with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
            total = int(r.headers.get("Content-Length") or 0)
            got = 0
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                got += len(chunk)
                if total:
                    pct = got * 100 / total
                    sys.stdout.write(f"\r       {got/1e6:7.1f}/{total/1e6:.0f} MB  {pct:5.1f}%")
                    sys.stdout.flush()
        os.replace(tmp, dst)
        print(f"\n[ok  ] {name}  {os.path.getsize(dst)/1e6:.0f} MB")
        return dst
    except Exception as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        print(f"\n[fail] {name}: {e}")
        print("       Download manually from:")
        print(f"       {url}")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", action="append", default=None,
                    help="YYYY-MM-DD, repeatable; default = all three study dates")
    a = ap.parse_args()
    for d in (a.date or DATES):
        fetch(d)
    print(f"\nRaw files in: {RAW_DIR}")
    print("Next: python scripts/preprocess.py")


if __name__ == "__main__":
    main()
