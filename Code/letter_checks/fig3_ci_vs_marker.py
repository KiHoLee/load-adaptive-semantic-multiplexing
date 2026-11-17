# Reads data/general_numbers.md section C.0 (generated from stored results) and the axis geometry of Code/fig_rev.py
# (ONE_AW = 2.4 in, ONE_AH = 0.75 ONE_AW, ylim -5..64, COL_MS = 4 pt). Prints the largest CI half-width in pt.
import re, sys
t = open(sys.argv[1]).read()
sec = t[t.index('### C.0'):t.index('### C.1 ')]
hw = []
for line in sec.splitlines():
    if not line.startswith('| U = '): continue
    for m in re.finditer(r'\+([\d.]+) % \[\+([\d.]+), \+([\d.]+)\]', line):
        v, lo, hi = map(float, m.groups()); hw.append((max(v - lo, hi - v), line.split('|')[1].strip(), v))
AH_pt = 0.75 * 2.4 * 72; span = 64 - (-5); ppt = AH_pt / span
mx = max(hw)
print(f'axis {AH_pt:.1f} pt for {span} pp -> {ppt:.3f} pt per pp; largest CI half-width {mx[0]:.2f} pp ({mx[0]*ppt:.2f} pt) at {mx[1]}, value {mx[2]}; marker radius 2.0 pt')
