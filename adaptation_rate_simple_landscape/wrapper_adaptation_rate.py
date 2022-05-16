import numpy as np
import subprocess
from sklearn.model_selection import ParameterGrid

noisevals = np.arange(30)

script_path = 'rate_adaptation_script.py'
cmd = 'bsub -q new-long -R "rusage[mem=6000]" python ' + script_path

for noise in noisevals:
	_cmd = ' '.join([cmd, str(noise)])
	subprocess.run(_cmd, shell=True)
