import pandas as pd
import numpy as np
import sys
sys.path.insert(0, r'C:\Users\virat.arya\ETG\SoftsDatabase - Documents\Database\Hardmine\ICEBREAKER\Rollex\Code')
from rollex_builder import generate_contract_table, COMMODITY_CONFIG, BDAY_CAL, START_YEAR

df = pd.read_parquet(r'C:\Users\virat.arya\ETG\SoftsDatabase - Documents\Database\Hardmine\ICEBREAKER\Rollex\Database\rollex_KC.parquet')
df = df.sort_index()
df['c1_chg'] = df['c1'].diff()

ct = generate_contract_table('KC', START_YEAR, pd.Timestamp.today().year + 1)
ct['FND'] = pd.to_datetime(ct['FND']).dt.normalize()
ct['LTD'] = pd.to_datetime(ct['LTD']).dt.normalize()

# Only check contracts whose FND falls within our price history
min_date = df.index.min()
max_date = df.index.max()
ct = ct[(ct['FND'] >= min_date) & (ct['FND'] <= max_date)].copy()

print(f"Checking {len(ct)} KC contracts with FND in {min_date.date()} -> {max_date.date()}\n")
print(f"{'Contract':<12} {'FND':<14} {'LTD':<14} {'c1 on FND-1':>12} {'c1 on FND':>12} {'Jump':>8} {'Pct':>7}  {'Verdict'}")
print('-' * 90)

fnd_hit = 0
for _, row in ct.iterrows():
    fnd = row['FND']
    contract = row.get('Contract', row.get('Month', '?'))

    # Find closest available trading days
    fnd_data = df[df.index <= fnd]
    if len(fnd_data) == 0:
        continue
    fnd_day = fnd_data.index[-1]

    pre_data = df[df.index < fnd_day]
    if len(pre_data) == 0:
        continue
    pre_day = pre_data.index[-1]

    c1_pre = df.loc[pre_day, 'c1']
    c1_fnd = df.loc[fnd_day, 'c1']
    jump   = c1_fnd - c1_pre
    pct    = (jump / c1_pre) * 100 if c1_pre != 0 else 0

    # A jump > 0.5 cents on FND day almost certainly signals a contract roll
    is_roll = abs(jump) > 0.5
    if is_roll:
        fnd_hit += 1
    verdict = "ROLL" if is_roll else "no move"

    print(f"{str(contract):<12} {str(fnd.date()):<14} {str(row['LTD'].date()):<14} "
          f"{c1_pre:>12.2f} {c1_fnd:>12.2f} {jump:>+8.2f} {pct:>+6.2f}%  {verdict}")

print(f"\n{'='*90}")
print(f"Contracts with FND roll signal: {fnd_hit}/{len(ct)}")
print(f"(A 'ROLL' means c1 moved > 0.5c on FND day, consistent with ICE switching front month)")
