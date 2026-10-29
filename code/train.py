"""Groupwise MLP verifier: train-only fitting, validation checkpoint selection.
Only NumPy is required. Test references are never loaded.
"""
import os
# Set before importing NumPy. No multi-process training or inference.
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key] = '4'
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
import numpy as np
from common import DATA, load_inputs, read_jsonl, parse_answer
from embedding_method import FEATURE_VERSION, make_features, model_scores, select


def labels(bank, split):
    if split not in ('train','validation'):
        raise ValueError('Training cannot load test references')
    refs = read_jsonl(DATA/f'gsm8k_{split}.jsonl')
    if len(refs)!=len(bank):
        raise ValueError('Reference count mismatch')
    result=[]
    for row, ref in zip(bank,refs):
        if row['id']!=ref['id'] or row['question']!=ref['question']:
            raise ValueError('Reference alignment mismatch')
        gold = parse_answer(ref['answer'])
        if gold is None:
            raise ValueError('Unparseable reference')
        y = [float(parse_answer(c['text'])==gold) for c in row['candidates']]
        if not any(y):
            raise ValueError('No correct candidate')
        result.append(y)
    return np.asarray(result,np.float32)


def accuracy(y, choices):
    return int(y[np.arange(len(y)),choices].sum())


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--epochs',type=int,default=60)
    p.add_argument('--hidden',type=int,default=64)
    p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--lr',type=float,default=0.001)
    p.add_argument('--weight-decay',type=float,default=0.001)
    p.add_argument('--dropout',type=float,default=0.15)
    p.add_argument('--seed',type=int,default=17)
    p.add_argument('--run-dir',type=Path,default=Path('runs/v2'))
    args=p.parse_args()
    if args.epochs<1 or args.hidden<1 or args.batch_size<1 or not 0<=args.dropout<1:
        p.error('Invalid training configuration')
    args.run_dir.mkdir(parents=True,exist_ok=True)
    if (args.run_dir/'model.npz').exists():
        p.error('Run directory already contains model.npz; use a new --run-dir')
    start=time.perf_counter()
    tb,ta=load_inputs('train'); vb,va=load_inputs('validation')
    y=labels(tb,'train'); vy=labels(vb,'validation')
    tx,tv=make_features(tb,ta['embeddings']); vx,vv=make_features(vb,va['embeddings'])
    # Statistics are fitted on training inputs only.
    mean=tx.mean(axis=(0,1)); scale=np.maximum(tx.std(axis=(0,1)),0.05)
    x=np.clip((tx-mean)/scale,-8,8).astype(np.float32)
    rng=np.random.default_rng(args.seed); d=x.shape[-1]
    params={
        'w1':rng.normal(0,np.sqrt(2/d),(d,args.hidden)).astype(np.float32),
        'b1':np.zeros(args.hidden,np.float32),
        'w2':rng.normal(0,0.01,(args.hidden,1)).astype(np.float32),
        'b2':np.zeros(1,np.float32),
    }
    m={k:np.zeros_like(v) for k,v in params.items()}
    v={k:np.zeros_like(a) for k,a in params.items()}
    base=accuracy(vy,np.argmax(vv,axis=1))
    best=base; best_epoch=0; history=[]; step=0
    # Fixed, declared validation search. Do not expand based on test results.
    alphas=(0.0,0.5,1.0,2.0,4.0)
    def save(alpha,baseline):
        np.savez(args.run_dir/'model.npz',**params,mean=mean,scale=scale,
                 alpha=np.array(alpha,np.float32),baseline_only=np.array(baseline,np.int32),
                 feature_version=np.array(FEATURE_VERSION,np.int32))
    save(0,1)
    for epoch in range(1,args.epochs+1):
        order=rng.permutation(len(x)); total_loss=0.0
        for first in range(0,len(x),args.batch_size):
            idx=order[first:first+args.batch_size]; b=len(idx)
            xx=x[idx].reshape(-1,d); yy=y[idx]
            pre=xx@params['w1']+params['b1']; h=np.maximum(pre,0)
            mask=(rng.random(h.shape)>=args.dropout).astype(np.float32)/(1-args.dropout)
            hd=h*mask
            score=(hd@params['w2']+params['b2']).reshape(b,4)
            # Fixed training prior; validation chooses the final inference prior.
            score=score+np.log(np.maximum(tv[idx],0.1))
            exp=np.exp(score-score.max(axis=1,keepdims=True))
            prob=exp/exp.sum(axis=1,keepdims=True)
            mass=np.maximum((prob*yy).sum(axis=1,keepdims=True),1e-8)
            # -log(total probability assigned to ANY correct candidate).
            # All-correct groups naturally contribute zero gradient.
            ds=((prob-prob*yy/mass)/b).reshape(-1,1)
            total_loss+=float(-np.log(mass).sum())
            dh=(ds@params['w2'].T)*mask*(pre>0)
            grads={'w2':hd.T@ds,'b2':ds.sum(axis=0),
                   'w1':xx.T@dh,'b1':dh.sum(axis=0)}
            step+=1
            for k in params:
                g=grads[k]
                if k.startswith('w'):
                    g=g+args.weight_decay*params[k]
                m[k]=0.9*m[k]+0.1*g; v[k]=0.999*v[k]+0.001*g*g
                params[k]-=args.lr*(m[k]/(1-0.9**step))/(np.sqrt(v[k]/(1-0.999**step))+1e-8)
        if epoch==1 or epoch%5==0 or epoch==args.epochs:
            model=dict(params,mean=mean,scale=scale)
            scores=model_scores(vx,model)
            results=[accuracy(vy,select(scores,vv,a)) for a in alphas]
            j=int(np.argmax(results)); correct=results[j]
            record={'epoch':epoch,'train_group_loss':total_loss/len(x),
                    'validation_correct_by_alpha':dict(zip(map(str,alphas),results)),
                    'best_epoch_accuracy':100*correct/len(vy)}
            history.append(record); print(json.dumps(record),flush=True)
            if correct>best:
                best=correct; best_epoch=epoch; save(alphas[j],0)
    path=args.run_dir/'model.npz'
    with np.load(path,allow_pickle=False) as f:
        alpha=float(f['alpha'].item()); fallback=bool(f['baseline_only'].item())
    summary={'method':'groupwise_mlp_with_consensus','configuration':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
             'best_epoch':best_epoch,'baseline_validation_correct':base,
             'best_validation_correct':best,'validation_questions':len(vy),
             'best_validation_accuracy':100*best/len(vy),'selected_alpha':alpha,
             'baseline_only':fallback,'training_seconds':time.perf_counter()-start,
             'checkpoint_bytes':path.stat().st_size,'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
             'numpy_version':np.__version__,'python_version':platform.python_version(),
             'platform':platform.platform(),'processor':platform.processor(),
             'test_evaluated':False,'validation_alphas':alphas,'validation_interval':5}
    (args.run_dir/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
    (args.run_dir/'training-summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('Training complete:\n'+json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':
    main()
