import os, sys, re, glob
src, dst, cap = sys.argv[1], sys.argv[2], int(sys.argv[3])
RULES = {
 'bend': r'bend_', 'displace_tilt': r'displace_tilt', 'guitar': r'guitar', 'kick': r'kick',
 'squat': r'squat', 'punch': r'punch', 'pour': r'pour', 'stairs_climbing': r'upstairs|downstairs|stair',
 'push_recovery': r'push_recovery|push_from|RecoveryStepping', 'tennis': r'tennis',
 'turn': r'(?i)turn', 'walk': r'(?i)walk', 'wipe': r'wipe',
}
stairs = re.compile(RULES['stairs_climbing'])
files = sorted(glob.glob(os.path.join(src, '*.npz')))
for task, pat in RULES.items():
    rx = re.compile(pat); d = os.path.join(dst, task); os.makedirs(d, exist_ok=True)
    sel = [f for f in files if rx.search(os.path.basename(f)) and not (task == 'walk' and stairs.search(os.path.basename(f)))][:cap]
    for f in sel: os.symlink(f, os.path.join(d, os.path.basename(f)))
    print(task, len(sel))
