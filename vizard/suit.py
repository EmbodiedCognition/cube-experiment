'''Constants related to phasespace markers.'''

# labels for markers reported by the phasespace setup.
MARKER_LABELS = [
    '00-r-head-back',
    '01-r-head-front',
    '02-l-head-front',
    '03-l-head-back',
    '04-r-head-mid',
    '05-l-head-mid',
    '06-r-collar',
    '07-r-shoulder',
    '08-r-elbow',
    '09-r-wrist',
    '10-r-fing-pinky',
    '11-r-fing-ring',
    '12-r-fing-middle',
    '13-r-fing-index',
    '14-r-mc-outer',
    '15-r-mc-inner',
    '16-r-thumb-base',
    '17-r-thumb-tip',
    '18-l-collar',
    '19-l-shoulder',
    '20-l-elbow',
    '21-l-wrist',
    '22-l-fing-pinky',
    '23-l-fing-ring',
    '24-l-fing-middle',
    '25-l-fing-index',
    '26-l-mc-outer',
    '27-l-mc-inner',
    '28-l-thumb-base',
    '29-l-thumb-tip',
    '30-abdomen',
    '31-sternum',
    '32-t3',
    '33-t9',
    '34-l-ilium',
    '35-r-ilium',
    '36-r-hip',
    '37-r-knee',
    '38-r-shin',
    '39-r-ankle',
    '40-r-heel',
    '41-r-mt-outer',
    '42-r-mt-inner',
    '43-l-hip',
    '44-l-knee',
    '45-l-shin',
    '46-l-ankle',
    '47-l-heel',
    '48-l-mt-outer',
    '49-l-mt-inner',
    ]


# a class for holding constants mapping marker labels to numbers.
class MARKERS:
    pass

for l in MARKER_LABELS:
    setattr(MARKERS, l[3:].replace('-', '_').upper(), int(l[:2]))
