'''Constants related to phasespace markers.'''

# labels for markers reported by the phasespace setup. abbreviations:
# mc = metacarpal
# mt = metatarsal
# tN = thoracic vertebra #N
# psis = posterior superior iliac spine
# asis = anterior superior iliac spine
MARKER_LABELS = [
    'r-head-back',   # 00
    'r-head-front',
    'l-head-front',
    'l-head-back',
    'r-head-mid',
    'l-head-mid',    # 05
    'r-collar',
    'r-shoulder',
    'r-elbow',
    'r-wrist',
    'r-fing-pinky',  # 10
    'r-fing-ring',
    'r-fing-middle',
    'r-fing-index',
    'r-mc5',
    'r-mc2',         # 15
    'r-mc1',
    'r-thumb',
    'l-collar',
    'l-shoulder',
    'l-elbow',       # 20
    'l-wrist',
    'l-fing-pinky',
    'l-fing-ring',
    'l-fing-middle',
    'l-fing-index',  # 25
    'l-mc5',
    'l-mc3',
    'l-mc1',
    'l-thumb',
    'abdomen',       # 30
    'sternum',
    't3',
    't9',
    'l-psis',
    'r-psis',        # 35
    'r-asis',
    'r-knee',
    'r-shin',
    'r-ankle',
    'r-heel',        # 40
    'r-mt5',
    'r-mt1',
    'l-asis',
    'l-knee',
    'l-shin',        # 45
    'l-ankle',
    'l-heel',
    'l-mt5',
    'l-mt1',
    ]


# a class for holding constants mapping marker labels to numbers.
class MARKERS:
    pass

for i, l in enumerate(MARKER_LABELS):
    setattr(MARKERS, l.replace('-', '_').upper(), i)
