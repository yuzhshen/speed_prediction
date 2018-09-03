import os

for e in range(1,21):
    for f in ['models']:
        os.system('python model.py '+str(e)+' '+f)