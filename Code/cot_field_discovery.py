"""
cot_field_discovery.py — List ALL available ICE fields for KC COT symbols
=========================================================================
Run this once to discover exact field names, then update cot_mapping.csv.

Output: prints all fields to console + saves to cot_fields_raw.txt
"""

import icepython as ice
from pathlib import Path
from datetime import datetime, timedelta

CODE_DIR = Path(__file__).parent
OUT_FILE = CODE_DIR / "cot_fields_raw.txt"

SYMBOLS = [
    "KC #COMB-CFTC",
    "KC #FUT-CFTC",
]

START = (datetime.today() - timedelta(weeks=20)).strftime("%Y-%m-%d")
END   = datetime.today().strftime("%Y-%m-%d")

lines = []

for sym in SYMBOLS:
    header = f"\n{'='*70}\n  {sym}\n{'='*70}"
    print(header)
    lines.append(header)

    try:
        fields = ice.get_fields(sym)
        result = f"  {len(fields)} fields found:\n"
        for f in fields:
            result += f"    {f}\n"
        print(result)
        lines.append(result)
    except Exception as e:
        err = f"  get_fields failed: {e}\n  Trying get_field_list..."
        print(err)
        lines.append(err)

        try:
            fields = ice.get_field_list(sym)
            result = f"  {len(fields)} fields found:\n"
            for f in fields:
                result += f"    {f}\n"
            print(result)
            lines.append(result)
        except Exception as e2:
            err2 = f"  get_field_list also failed: {e2}"
            print(err2)
            lines.append(err2)

            # Fallback: fetch a known field and inspect the response structure
            fallback = f"  Falling back to timeseries inspect..."
            print(fallback)
            lines.append(fallback)
            try:
                data = ice.get_timeseries(sym, ["Open Interest All Close"],
                                          granularity="D", start_date=START, end_date=END)
                rows = list(data)
                info = f"  Sample timeseries rows[0]: {rows[0] if rows else 'empty'}"
                print(info)
                lines.append(info)
            except Exception as e3:
                err3 = f"  Timeseries also failed: {e3}"
                print(err3)
                lines.append(err3)

OUT_FILE.write_text("\n".join(lines))
print(f"\nOutput saved to {OUT_FILE}")
