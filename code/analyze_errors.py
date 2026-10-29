"""Read-only validation error analysis. Does not train or load test data."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
from common import DATA, PROTOCOL, load_bank, read_jsonl, parse_answer, validate_choices


def baseline(row):
    answers=[parse_answer(c['text']) for c in row['candidates']]
    counts=Counter(a for a in answers if a is not None)
    return next((i for i,a in enumerate(answers) if a is not None and counts[a]==max(counts.values())),0)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--predictions',type=Path,default=Path('runs/v2-local/validation-predictions.json'))
    p.add_argument('--output-dir',type=Path,default=Path('runs/error-analysis'))
    args=p.parse_args()
    bank=load_bank('validation'); refs=read_jsonl(DATA/'gsm8k_validation.jsonl')
    pred=json.loads(args.predictions.read_text(encoding='utf-8'))
    if pred.get('split')!='validation' or pred.get('protocol')!=PROTOCOL:
        raise ValueError('Only official validation predictions are accepted')
    validate_choices(bank,pred['ids'],pred['selected'])
    if len(refs)!=len(bank):
        raise ValueError('Reference count mismatch')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        p.error('Output directory is not empty; use a new --output-dir to preserve prior records')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    rows=[]; groups=Counter(); strata=defaultdict(lambda:[0,0,0]); errors=Counter()
    for row,ref,chosen in zip(bank,refs,pred['selected']):
        if row['id']!=ref['id'] or row['question']!=ref['question']:
            raise ValueError('Reference alignment mismatch')
        gold=parse_answer(ref['answer'])
        if gold is None: raise ValueError('Unparseable gold')
        answers=[parse_answer(c['text']) for c in row['candidates']]
        correct=[i for i,a in enumerate(answers) if a==gold]
        counts=Counter(a for a in answers if a is not None)
        largest=max(counts.values(),default=0)
        modes=[a for a,n in counts.items() if n==largest]
        b=baseline(row); bc=b in correct; mc=chosen in correct
        group=('both_correct' if bc and mc else 'corrected_by_v2' if mc else 'introduced_by_v2' if bc else 'both_wrong')
        groups[group]+=1
        pattern='+'.join(map(str,sorted(counts.values(),reverse=True))) or 'none'
        invalid=sum(a is None for a in answers)
        if invalid: pattern+=f' + {invalid} invalid'
        if not mc:
            if answers[chosen] is None: tag='selected_unparseable'
            elif len(modes)>1: tag='wrong_on_plurality_tie'
            elif modes and modes[0]==gold: tag='rejected_unique_correct_plurality'
            else: tag='unique_plurality_wrong'
            errors[tag]+=1
        else: tag=''
        for name in [f'answer_pattern={pattern}',f'correct_candidates={len(correct)}']:
            strata[name][0]+=1; strata[name][1]+=int(bc); strata[name][2]+=int(mc)
        rows.append({'id':row['id'],'question':row['question'],'gold_answer':gold,
                     'correct_indices':correct,'answers':answers,'answer_pattern':pattern,
                     'baseline_selected':b,'v2_selected':chosen,'baseline_correct':bc,
                     'v2_correct':mc,'comparison':group,'error_category':tag,
                     'candidates':row['candidates'],'reference_solution':ref['answer']})
    n=len(rows); bsum=sum(r['baseline_correct'] for r in rows); msum=sum(r['v2_correct'] for r in rows)
    summary={'split':'validation','n':n,'baseline_correct':bsum,'v2_correct':msum,
             'baseline_accuracy':100*bsum/n,'v2_accuracy':100*msum/n,
             'net_additional_correct':msum-bsum,
             'comparison_counts':{k:groups[k] for k in ('both_correct','corrected_by_v2','introduced_by_v2','both_wrong')},
             'v2_error_categories':dict(errors),
             'prediction_file':str(args.predictions),
             'prediction_sha256':hashlib.sha256(args.predictions.read_bytes()).hexdigest(),
             'warning':'Numeric-answer error categories are observable descriptors, not verified reasoning-error causes. Validation was used for model selection.',
             'test_evaluated':False}
    strata_rows=[{'stratum':k,'n':v[0],'baseline_correct':v[1],'v2_correct':v[2],
                 'baseline_accuracy':100*v[1]/v[0],'v2_accuracy':100*v[2]/v[0]} for k,v in sorted(strata.items())]
    def dump(name,value):
        (args.output_dir/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    dump('summary.json',summary); dump('strata.json',strata_rows)
    with (args.output_dir/'all-cases.jsonl').open('w',encoding='utf-8') as f:
        for r in rows: f.write(json.dumps(r,ensure_ascii=False)+'\n')
    fields=['id','gold_answer','answers','correct_indices','baseline_selected','v2_selected','comparison','answer_pattern','error_category','question']
    with (args.output_dir/'case-index.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for r in rows: w.writerow({k:r[k] for k in fields})
    text=['# Validation error analysis','',
          f'Baseline: {bsum}/{n} ({100*bsum/n:.2f}%). V2: {msum}/{n} ({100*msum/n:.2f}%).','',
          '## Prediction comparison','', '| Category | Questions |','|---|---:|']
    text.extend(f'| {k} | {groups[k]} |' for k in ('both_correct','corrected_by_v2','introduced_by_v2','both_wrong'))
    text+=['','## Answer-pattern and correct-candidate strata','',
           '| Stratum | N | Baseline accuracy | V2 accuracy |','|---|---:|---:|---:|']
    text.extend(f"| {r['stratum']} | {r['n']} | {r['baseline_accuracy']:.2f}% | {r['v2_accuracy']:.2f}% |" for r in strata_rows)
    text+=['','Categories describe final-answer patterns; they do not establish why a reasoning trace is wrong.',
           'Examples below are the first three in bank order per category, not representative random samples.',
           'All cases and all candidate texts are available in all-cases.jsonl.','']
    for group in ('introduced_by_v2','both_wrong','corrected_by_v2'):
        selected=[r for r in rows if r['comparison']==group]
        text += [f'## {group} ({len(selected)} total)','']
        for r in selected[:3]:
            text += [f"### {r['id']}",'',r['question'],'',
                     f"Gold: {r['gold_answer']}; baseline: {r['baseline_selected']}; V2: {r['v2_selected']}; correct indices: {r['correct_indices']}",'',
                     f"Pattern: {r['answer_pattern']}; error category: {r['error_category']}",'']
            for c in r['candidates']:
                text += [f"Candidate {c['index']}:",'','```text',c['text'],'```','']
    (args.output_dir/'analysis.md').write_text('\n'.join(text),encoding='utf-8')
    # Separate exhaustive review files keep the overview manageable.
    for group in ('introduced_by_v2','both_wrong','corrected_by_v2'):
        dump(group+'.json',[r for r in rows if r['comparison']==group])
    print(json.dumps(summary,indent=2,ensure_ascii=False))
    print('Saved analysis to '+str(args.output_dir))

if __name__=='__main__': main()
