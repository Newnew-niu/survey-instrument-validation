"""Section 7: can 8 items replace the 40-item questionnaire?"""
# Items are selected on one random half of the sample and evaluated on the other half; repeated 200 times.
import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
d = pd.read_csv(ROOT / 'outputs' / 'scored.csv')   # created by 02_scoring_segments.py
blocks = {k:[f'{k}{i}' for i in range(1,11)] for k in 'ABCD'}
items = sum(blocks.values(), [])
y = d.Level.values; total = d[items].sum(axis=1).values
alpha = lambda df: df.shape[1]/(df.shape[1]-1)*(1-df.var(ddof=1).sum()/df.sum(axis=1).var(ddof=1))

def pick(train):
    tot = train[items].sum(axis=1); chosen=[]
    for b,cols in blocks.items():
        citc = {c: np.corrcoef(train[c], tot-train[c])[0,1] for c in cols}
        chosen += sorted(citc, key=citc.get, reverse=True)[:2]
    return chosen

rng = np.random.default_rng(42); res=[]; freq=Counter()
for _ in range(200):
    idx = rng.permutation(len(d)); tr, te = idx[:77], idx[77:]
    sel = pick(d.iloc[tr]); freq.update(sel)
    s_tr = d.iloc[tr][sel].sum(axis=1).values.reshape(-1,1); s_te = d.iloc[te][sel].sum(axis=1).values.reshape(-1,1)
    clf = LogisticRegression(max_iter=2000).fit(s_tr, y[tr]); p = clf.predict(s_te)
    res.append(dict(r=np.corrcoef(s_te.ravel(), total[te])[0,1], alpha=alpha(d.iloc[te][sel]),
                    acc=(p==y[te]).mean(), kappa=cohen_kappa_score(p,y[te]),
                    base=(y[te]==Counter(y[tr]).most_common(1)[0][0]).mean()))
r = pd.DataFrame(res)
print(r.describe(percentiles=[.05,.95]).loc[['mean','5%','95%']].round(3).to_string())
print('most often selected:', freq.most_common(10))
