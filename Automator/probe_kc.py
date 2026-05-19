import icepython as ice
from concurrent.futures import ThreadPoolExecutor

sym   = "KC #FUT-CFTC"
start = "2026-04-01"
end   = "2026-05-19"
field = "Open Interest All Close"

print(f"Probing {sym} | {start} -> {end}")

def _call():
    data = ice.get_timeseries(sym, [field], granularity="D", start_date=start, end_date=end)
    return list(data)

with ThreadPoolExecutor(max_workers=1) as ex:
    fut = ex.submit(_call)
    try:
        rows = fut.result(timeout=60)
        print(f"Got {len(rows)} rows (including header)")
        for r in rows[:5]:
            print(r)
        if len(rows) > 5:
            print("...")
            for r in rows[-3:]:
                print(r)
    except TimeoutError:
        print("TIMED OUT after 60s — ICE not responding for this symbol")
