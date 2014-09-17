import os
import time
import viz

for f in os.listdir('C:/Program Files/WorldViz/Vizard4/Resources/'):
    if f.lower().endswith('.wav'):
        print f
        a = viz.addAudio(f)
        a.play()
        time.sleep(3)
        a.stop()
