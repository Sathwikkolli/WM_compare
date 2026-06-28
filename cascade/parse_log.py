"""
parse_log.py  --  rebuild cascade CSVs from a (possibly truncated) run log.

If a run is killed (e.g. SLURM time limit) before run_cascade.py writes its CSVs,
the per-config results are still in the stdout log. This scrapes them back into
cascade_interference.csv and cascade_robustness.csv so make_report.py can run on
whatever completed.

Usage:
  python parse_log.py ~/wm_compare/cascade_52460189.log
  python make_report.py
"""
import os, re, csv, sys

BASE = os.path.expanduser('~/wm_compare')
OUT  = os.path.join(BASE, 'cascade_out'); os.makedirs(OUT, exist_ok=True)
CODE2NAME = {'AU': 'AudioSeal', 'AW': 'AWARE', 'TI': 'Timbre'}

cfg_re    = re.compile(r'^####\s+CONFIG\s+(\S+)')
stage_re  = re.compile(r'stage(\d+)\s+after\s+\+(\w+)\s*\|\s*detect\s+(\w+)\s*->\s*bit_acc=([\d.]+|ERR)\s+conf=([\d.]+|ERR|\w+)')
attack_re = re.compile(r'attack\s+(\w+)\s*\|\s*(.*?)\s*(?:pesq=([\d.]+|None))?\s*$')
pair_re   = re.compile(r'(AU|AW|TI)=([\d.]+|ERR)')

def main():
    if len(sys.argv) < 2:
        print('usage: python parse_log.py <logfile>'); sys.exit(1)
    inter, robust = [], []
    cur = None
    for line in open(sys.argv[1], errors='ignore'):
        m = cfg_re.search(line)
        if m:
            cur = m.group(1); continue
        if cur is None:
            continue
        m = stage_re.search(line)
        if m:
            stage, embedded, detected, acc, conf = m.groups()
            inter.append([cur, stage, embedded, detected, conf, acc]); continue
        m = attack_re.search(line)
        if m:
            attack, body, pesq = m.groups()
            for code, acc in pair_re.findall(body):
                robust.append([cur, attack, CODE2NAME[code], '', acc, '', pesq or ''])
    with open(os.path.join(OUT, 'cascade_interference.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['config','stage','embedded','detected_wm','conf','bit_acc']); w.writerows(inter)
    with open(os.path.join(OUT, 'cascade_robustness.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['config','attack','watermark','conf','bit_acc','bits','pesq']); w.writerows(robust)
    cfgs = sorted({r[0] for r in inter})
    print(f'parsed {len(cfgs)} configs: {cfgs}')
    print(f'  {len(inter)} interference rows, {len(robust)} robustness rows')
    print('wrote cascade_interference.csv + cascade_robustness.csv -> run: python make_report.py')

if __name__ == '__main__':
    main()
