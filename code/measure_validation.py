"""Measure complete fresh prediction + independent scoring wall time (validation)."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root=Path(__file__).resolve().parent
env=os.environ.copy()
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    env[key]='4'
commands=[
    [sys.executable,'predict.py','--method','submission.py','--checkpoint','runs/v2/model.npz',
     '--split','validation','--output','runs/v2/validation-predictions.json'],
    [sys.executable,'score_predictions.py','--split','validation',
     '--predictions','runs/v2/validation-predictions.json','--output','runs/v2/validation-score.json'],
]
start=time.perf_counter()
for command in commands:
    subprocess.run(command,cwd=root,env=env,check=True)
result={'split':'validation','fresh_prediction_plus_scoring_seconds':time.perf_counter()-start,
        'note':'Validation only; does not certify reference-machine test eligibility.'}
(root/'runs/v2/validation-wall-time.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
