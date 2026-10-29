"""Run isolated ablations, then official validation prediction/scoring.
Run from the course code/ directory. Never reads test inputs or references.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--experiment',choices=['all','no_embeddings','no_prior'],default='all')
    p.add_argument('--output-root',type=Path,default=Path('runs/ablations'))
    args=p.parse_args()
    os.chdir(ROOT)
    env=os.environ.copy()
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
        env[key]='4'
    cases={'no_embeddings':'--no-embeddings','no_prior':'--no-vote-prior'}
    selected=list(cases) if args.experiment=='all' else [args.experiment]
    # Check all destinations before starting so reruns never overwrite a model.
    for name in selected:
        if (args.output_root/name/'model.npz').exists():
            p.error(f'{args.output_root/name}/model.npz exists; choose a new --output-root.')
    rows=[]
    for name in selected:
        out=args.output_root/name
        out.mkdir(parents=True,exist_ok=True)
        train=[sys.executable,'ablation_train.py',cases[name],'--seed','17',
               '--run-dir',str(out)]
        print('\nTRAINING '+name,flush=True)
        with (out/'console-log.txt').open('w',encoding='utf-8') as log:
            proc=subprocess.Popen(train,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                  text=True,encoding='utf-8',errors='replace',env=env)
            for line in proc.stdout:
                print(line,end='',flush=True); log.write(line)
            if proc.wait()!=0:
                raise RuntimeError(f'Training failed. See {out}/console-log.txt')
        t=time.perf_counter()
        subprocess.run([sys.executable,'predict.py','--method','ablation_embedding.py',
                        '--checkpoint',str(out/'model.npz'),'--split','validation',
                        '--output',str(out/'validation-predictions.json')],env=env,check=True)
        subprocess.run([sys.executable,'score_predictions.py','--split','validation',
                        '--predictions',str(out/'validation-predictions.json'),
                        '--output',str(out/'validation-score.json')],env=env,check=True)
        elapsed=time.perf_counter()-t
        score=json.loads((out/'validation-score.json').read_text())
        summary=json.loads((out/'training-summary.json').read_text())
        row={'experiment':name,'correct':score['correct'],'n':score['n'],
             'accuracy':score['accuracy'],'best_epoch':summary['best_epoch'],
             'alpha':summary['selected_alpha'],'training_seconds':summary['training_seconds'],
             'checkpoint_bytes':summary['checkpoint_bytes'],
             'fresh_validation_prediction_plus_scoring_seconds':elapsed}
        (out/'experiment-result.json').write_text(json.dumps(row,indent=2),encoding='utf-8')
        rows.append(row)
    path=args.output_root/('summary-'+args.experiment+'.json')
    path.write_text(json.dumps(rows,indent=2),encoding='utf-8')
    print('\nABLATION RESULTS\n'+json.dumps(rows,indent=2),flush=True)

if __name__=='__main__':
    main()
