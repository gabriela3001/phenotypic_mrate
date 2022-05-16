import numpy as np
import subprocess
from sklearn.model_selection import ParameterGrid

landscape_vals = [1,3,5]
param_vals = [2,3]

script_path = 'script_complex_adaptation.py'
cmd = 'bsub -q new-short -R "rusage[mem=6000]" python ' + script_path

for l in landscape_vals:
    for p in param_vals:
        _cmd = ' '.join([cmd, str(l), str(p)])
        subprocess.run(_cmd, shell=True)