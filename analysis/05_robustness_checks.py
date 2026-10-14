"""Robustness checks quoted in the portfolio (section 3 and Appendix B)."""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from pathlib import Path
from scipy.stats import chi2, ttest_ind, f_oneway
import semopy
ROOT = Path(__file__).resolve().parents[1]
d = pd.read_csv(ROOT / 'data' / 'survey_responses.csv')
blocks = {k: [f'{k}{i}' for i in range(1, 11)] for k in 'ABCD'}
items = sum(blocks.values(), [])
alpha = lambda df: df.shape[1]/(df.shape[1]-1)*(1-df.var(ddof=1).sum()/df.sum(axis=1).var(ddof=1))

print('== 1. Results without the 4 multivariate outliers (Mahalanobis p < .001)')
X = d[items]; Xc = X - X.mean()
md = np.einsum('ij,jk,ik->i', Xc, np.linalg.pinv(np.cov(X.T)), Xc)
e = d[md <= chi2.ppf(.999, 40)].copy()
print('n =', len(e))
print('alpha by block:', {k: round(float(alpha(e[c])), 3) for k, c in blocks.items()})
m = semopy.Model('\n'.join(f'{k} =~ ' + ' + '.join(c) for k, c in blocks.items())); m.fit(e)
st = semopy.calc_stats(m).T.iloc[:, 0]
print(f"CFA chi2/df={st['chi2']/st['DoF']:.3f} CFI={st['CFI']:.3f} RMSEA={st['RMSEA']:.3f}")
est = m.inspect(std_est=True)
lat = est[(est.op == '~~') & (est.lval != est.rval) & est.lval.isin(blocks) & est.rval.isin(blocks)]
print('latent correlations:', lat['Est. Std'].astype(float).round(2).tolist())
e['total'] = e[items].sum(axis=1)
print('gender p =', round(ttest_ind(*[g.total for _, g in e.groupby('gender')], equal_var=False).pvalue, 3),
      '| year p =', round(f_oneway(*[g.total for _, g in e.groupby('year')]).pvalue, 3))

print('\n== 2. GA-BP model inputs vs. target (Appendix B)')
s = pd.read_csv(ROOT / 'data' / 'scoring_memberships.csv')
M = s[['mu_good', 'mu_normal', 'mu_mild', 'mu_severe']].values
print('Taking the largest input membership reproduces the target level for',
      int((M.argmax(1) == s.level.values).sum()), 'of', len(s), 'cases')
