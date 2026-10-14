"""Data quality, item analysis, reliability, EFA and CFA (sections 3-4)."""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity
import semopy
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'survey_responses.csv'   # not in the public repo; available on request
OUT = ROOT / 'outputs'
d = pd.read_csv(DATA)
dims = {'MH':'A','SA':'B','AP':'C','SR':'D'}
items = {k:[f'{v}{i}' for i in range(1,11)] for k,v in dims.items()}
X = d[[c for v in items.values() for c in v]]

def alpha(df):
    k=df.shape[1]; return k/(k-1)*(1-df.var(ddof=1).sum()/df.sum(axis=1).var(ddof=1))

print('== 1. Data quality')
sd = X.std(axis=1)
print('straight-liners (row SD=0):', (sd==0).sum())
longstr = X.apply(lambda r: max(len(list(g)) for _,g in __import__('itertools').groupby(r)), axis=1)
print('longest identical run >=20:', (longstr>=20).sum(), ' max run:', longstr.max())
from scipy.stats import chi2
Xc = X - X.mean(); inv = np.linalg.pinv(np.cov(X.T)); md = np.einsum('ij,jk,ik->i', Xc, inv, Xc)
print('Mahalanobis outliers p<.001:', (md > chi2.ppf(.999, 40)).sum())
print('missing:', X.isna().sum().sum())

print('\n== 2. Reliability & item analysis')
for k,cols in items.items():
    sub = d[cols]; tot = sub.sum(axis=1)
    citc = [np.corrcoef(sub[c], tot-sub[c])[0,1] for c in cols]
    aid = [alpha(sub.drop(columns=c)) for c in cols]
    kmo = calculate_kmo(sub)[1]
    print(f'{k}: alpha={alpha(sub):.3f} KMO={kmo:.3f} CITC min={min(citc):.2f} ({cols[int(np.argmin(citc))]}) ; items where alpha-if-deleted > alpha:',
          [c for c,a in zip(cols,aid) if a>alpha(sub)])
print('overall alpha 40 items:', round(alpha(X),3), ' KMO overall:', round(calculate_kmo(X)[1],3),
      ' Bartlett p:', calculate_bartlett_sphericity(X)[1])
# discrimination: top/bottom 27% t-test per item
from scipy.stats import ttest_ind
tot = X.sum(axis=1); hi = tot>=tot.quantile(.73); lo = tot<=tot.quantile(.27)
cr = {c: ttest_ind(X.loc[hi,c], X.loc[lo,c]).statistic for c in X}
print('critical ratio min:', round(min(cr.values()),2), min(cr, key=cr.get))

print('\n== 3. EFA (40 items)')
ev = np.linalg.eigvalsh(np.corrcoef(X.T))[::-1]
print('eigenvalues top 6:', ev[:6].round(2), ' n>1:', (ev>1).sum(), ' var% first:', round(ev[0]/40*100,1))
# parallel analysis
rng=np.random.default_rng(0); sims=np.array([np.linalg.eigvalsh(np.corrcoef(rng.normal(size=X.shape).T))[::-1] for _ in range(200)])
print('parallel analysis factors to retain:', (ev > np.percentile(sims,95,axis=0)).sum())
fa = FactorAnalyzer(n_factors=4, rotation='oblimin', method='minres'); fa.fit(X)
L = pd.DataFrame(fa.loadings_, index=X.columns).round(2)
L['max_f']=L.iloc[:,:4].abs().idxmax(axis=1); L['max']=L.iloc[:,:4].abs().max(axis=1)
print(L.groupby(L.index.str[0])['max_f'].agg(lambda s: s.value_counts().to_dict()))
print('items max loading <.40:', L[L['max']<.4].index.tolist())
print('cumulative var 4 factors:', fa.get_factor_variance()[2][-1].round(3))
print('factor correlations:\n', pd.DataFrame(fa.phi_).round(2))
# Harman single factor
fa1 = FactorAnalyzer(n_factors=1, rotation=None); fa1.fit(X)
print('Harman single factor var %:', round(fa1.get_factor_variance()[1][0]*100,1))

print('\n== 4. CFA 4-factor')
desc = '\n'.join(f'{k} =~ ' + ' + '.join(c) for k,c in items.items())
m = semopy.Model(desc); m.fit(d)
st = semopy.calc_stats(m).T
print(st.loc[['DoF','chi2','chi2 p-value','CFI','TLI','GFI','RMSEA']].round(3).to_string())
print('chi2/df:', round(float(st.loc['chi2'].iloc[0])/float(st.loc['DoF'].iloc[0]),3))
est = m.inspect(std_est=True)
ld = est[(est.op=='~') & est['rval'].isin(items.keys())][['lval','rval','Est. Std']]
for k in items:
    lam = ld[ld.rval==k]['Est. Std'].astype(float)
    ave = (lam**2).mean(); cr_ = lam.sum()**2/(lam.sum()**2+(1-lam**2).sum())
    print(f'{k}: loadings {lam.min():.2f}-{lam.max():.2f} AVE={ave:.3f} CR={cr_:.3f} sqrtAVE={np.sqrt(ave):.3f}')
cov = est[(est.op=='~~') & est.lval.isin(items.keys()) & est.rval.isin(items.keys()) & (est.lval!=est.rval)]
print('latent correlations:\n', cov[['lval','rval','Est. Std']].to_string(index=False))
