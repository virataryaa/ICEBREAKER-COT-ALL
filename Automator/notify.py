"""
notify.py — COT_ALL Automator email summary
Called by run.bat after ingest + git push.
Usage: python notify.py <status> <git_status>
  status     : ok | error
  git_status : pushed | skipped | failed
"""

import sys
import datetime
import pandas as pd
from pathlib import Path

TO_EMAIL = "virat.arya@etgworld.com"
DB_DIR   = Path(r"C:\Users\virat.arya\ETG\SoftsDatabase - Documents\Database\Hardmine\ICEBREAKER\COT_ALL\Database")

CIT_FILE        = DB_DIR / "cot_cit.parquet"
DISAGG_FO_FILE  = DB_DIR / "cot_disagg_futopt.parquet"
DISAGG_F_FILE   = DB_DIR / "cot_disagg_fut.parquet"

status     = sys.argv[1] if len(sys.argv) > 1 else "ok"
git_status = sys.argv[2] if len(sys.argv) > 2 else "unknown"
run_dt     = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
today      = datetime.date.today().strftime("%Y-%m-%d")


def parquet_summary(path: Path, label: str, group_by=None) -> str:
    if not path.exists():
        return f"  {label}: FILE NOT FOUND\n"
    group_by = group_by or ["Commodity"]
    df = pd.read_parquet(path)
    df["Date"] = pd.to_datetime(df["Date"])
    lines = [f"  {label}:"]
    for keys, grp in df.groupby(group_by):
        key_str = "  ".join(str(k) for k in (keys if isinstance(keys, tuple) else (keys,)))
        lines.append(f"    {key_str:<25}  {len(grp):>5} rows   {grp['Date'].min().date()} -> {grp['Date'].max().date()}")
    return "\n".join(lines) + "\n"


def send_outlook_email(subject: str, body: str):
    try:
        import win32com.client
        outlook      = win32com.client.Dispatch("Outlook.Application")
        mail         = outlook.CreateItem(0)
        mail.To      = TO_EMAIL
        mail.Subject = subject
        mail.Body    = body
        mail.Send()
        print(f"  Email sent -> {TO_EMAIL}")
    except Exception as e:
        print(f"  Email failed: {e}")


ok  = status == "ok"
tag = "[OK]" if ok else "[ERROR]"
subject = f"{tag} ICEBREAKER-COT_ALL — {today}"

git_line = {
    "pushed":  "GitHub  : Pushed successfully",
    "skipped": "GitHub  : No changes — push skipped",
    "failed":  "GitHub  : PUSH FAILED",
}.get(git_status, f"GitHub  : {git_status}")

body = f"""ICEBREAKER COT_ALL — Weekly Update
Run time : {run_dt}
Status   : {"OK" if ok else "ERROR — ingest failed, check run_log.txt"}
{git_line}

{"=" * 55}
DATABASE SUMMARY
{"=" * 55}
{parquet_summary(CIT_FILE,       "CIT       (KC / CC / SB / CT)")}
{parquet_summary(DISAGG_FO_FILE, "DISAGG F&O (All/Old/Other)", group_by=["Commodity", "Crop"])}
{parquet_summary(DISAGG_F_FILE,  "DISAGG Fut (All/Old/Other)", group_by=["Commodity", "Crop"])}
{"=" * 55}
Log: C:\\Users\\virat.arya\\ETG\\SoftsDatabase - Documents\\Database\\Hardmine\\ICEBREAKER\\COT_ALL\\Automator\\run_log.txt
"""

print(body)
send_outlook_email(subject, body)
