"""V2 features and portable NumPy-only inference. No reference-file access."""
import re
from collections import Counter
from pathlib import Path
import numpy as np
from common import parse_answer

FEATURE_VERSION = 2

def make_features(bank, embeddings):
    """Independent and within-question embedding features plus observable evidence."""
    raw = np.asarray(embeddings, dtype=np.float32)
    if raw.shape != (len(bank), 4, 1024):
        raise ValueError('Expected embeddings of shape [N,4,1024]')
    norms = np.linalg.norm(raw, axis=-1, keepdims=True)
    x = raw / np.maximum(norms, 1e-8)
    similarities = x @ x.transpose(0, 2, 1)
    extra, votes, valid = [], [], []
    for i, row in enumerate(bank):
        texts = [c['text'] for c in row['candidates']]
        answers = [parse_answer(t) for t in texts]
        counts = Counter(a for a in answers if a is not None)
        vote = np.array([counts.get(a, 0) if a is not None else 0 for a in answers], dtype=np.float32)
        features = []
        for j, (text, answer) in enumerate(zip(texts, answers)):
            low = text.lower()
            same = np.array([answer is not None and a == answer for a in answers])
            other_same = same.copy(); other_same[j] = False
            different = np.array([a is not None and a != answer for a in answers])
            def sim_mean(mask):
                return float(similarities[i,j,mask].mean()) if mask.any() else 0.0
            features.append([
                float(answer is not None), vote[j]/4, vote.max()/4,
                len(counts)/4, float(sum(a is None for a in answers))/4,
                np.log1p(len(text)), np.log1p(len(text.split())),
                np.log1p(text.count('\n')), np.log1p(text.count('=')),
                np.log1p(len(re.findall(r'\d+',text))),
                float('####' in text), float('\\boxed' in text),
                float(any(s in low for s in ('wait','mistake','incorrect','reconsider'))),
                float(any(s in low for s in ('therefore','thus','hence'))),
                sim_mean(other_same), sim_mean(different),
                float((similarities[i,j].sum()-1)/3),
                float(np.log1p(norms[i,j,0])),
            ])
        features = np.asarray(features, dtype=np.float32)
        # Relative textual evidence, computed within the input question only.
        relative = features[:,5:10] - features[:,5:10].mean(axis=0,keepdims=True)
        extra.append(np.concatenate([features,relative],axis=1))
        votes.append(vote); valid.append([a is not None for a in answers])
    if not bank:
        return np.empty((0,4,2071),np.float32), np.empty((0,4),np.float32)
    extra = np.asarray(extra,np.float32)
    features = np.concatenate([x, x-x.mean(axis=1,keepdims=True), extra],axis=-1)
    return features, np.asarray(votes,np.float32)

def model_scores(x, model):
    x = np.clip((x-model['mean'])/model['scale'], -8,8)
    h = np.maximum(x @ model['w1'] + model['b1'], 0)
    return (h @ model['w2']).squeeze(-1) + model['b2'].item()

def select(scores, votes, alpha):
    # A positive prior favors corroborated final answers. Invalid parses get no votes.
    return np.argmax(scores + alpha*np.log(np.maximum(votes,0.1)),axis=1)

def predict(bank, arrays, checkpoint):
    if checkpoint is None:
        raise ValueError('Supply --checkpoint runs/v2/model.npz')
    with np.load(Path(checkpoint),allow_pickle=False) as f:
        model = {k:f[k] for k in f.files}
    if int(model['feature_version'].item()) != FEATURE_VERSION:
        raise ValueError('Feature version mismatch')
    x, votes = make_features(bank, arrays['embeddings'])
    if not bank:
        return []
    if int(model['baseline_only'].item()):
        choices = np.argmax(votes,axis=1)
    else:
        choices = select(model_scores(x,model),votes,float(model['alpha'].item()))
    return [int(i) for i in choices]
