'''Constants related to phasespace markers.'''

# labels for markers reported by the phasespace setup.
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
    'r-mc-outer',
    'r-mc-inner',    # 15
    'r-thumb-base',
    'r-thumb-tip',
    'l-collar',
    'l-shoulder',
    'l-elbow',       # 20
    'l-wrist',
    'l-fing-pinky',
    'l-fing-ring',
    'l-fing-middle',
    'l-fing-index',  # 25
    'l-mc-outer',
    'l-mc-inner',
    'l-thumb-base',
    'l-thumb-tip',
    'abdomen',       # 30
    'sternum',
    't3',
    't9',
    'l-ilium',
    'r-ilium',       # 35
    'r-hip',
    'r-knee',
    'r-shin',
    'r-ankle',
    'r-heel',        # 40
    'r-mt-outer',
    'r-mt-inner',
    'l-hip',
    'l-knee',
    'l-shin',        # 45
    'l-ankle',
    'l-heel',
    'l-mt-outer',
    'l-mt-inner',
    ]


# a class for holding constants mapping marker labels to numbers.
class MARKERS:
    pass

for i, l in enumerate(MARKER_LABELS):
    setattr(MARKERS, l.replace('-', '_').upper(), i)
