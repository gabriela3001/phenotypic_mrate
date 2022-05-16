import numpy as np
import subprocess
from sklearn.model_selection import ParameterGrid

landscape_vals = [1,3,5]
param_vals = [2,3]
noise_vals = [0,15,29]

script_path = 'script_complex_adaptation.py'
cmd = 'bsub -q new-short -R "rusage[mem=6000]" python ' + script_path

for l in landscape_vals:
    for p in param_vals:
        for n in noise_vals:
            _cmd = ' '.join([cmd, str(p), str(n), str(l)])
            subprocess.run(_cmd, shell=True)