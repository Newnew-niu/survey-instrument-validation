"""Fuzzy scoring model, weight sensitivity, measurement-model comparison and segment tests (sections 4-6)."""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
import semopy
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'survey_responses.csv'   # not in the public repo; available on request
OUT = ROOT / 'outputs'
d = pd.read_csv(DATA)
LV_EN = ['Good','Normal','Mild','Severe']
DIM = {'MH':'A','SA':'B','AP':'C','SR':'D'}
cols = {k:[f'{v}{i}' for i in range(1,11)] for k,v in DIM.items()}

def trap(x,a,b,c,e,left=False,right=False):
    if left and x<=c: return 1.0 if x>=b or left else 0
    if right and x>=b: return 1.0
    if x<=a or x>=e: return 0.0
    if b<=x<=c: return 1.0
    return (x-a)/(b-a) if x<b else (e-x)/(e-c)
P=[(10,15,20,25),(20,25,30,35),(30,35,40,45),(40,45,50,50)]
def mu(x):  # x on 10-50 scale
    return np.array([trap(x,*P[0],left=True),trap(x,*P[1]),trap(x,*P[2]),trap(x,*P[3],right=True)])

def classify(w, level='item'):
    out=[]
    for _,r in d.iterrows():
        R=[]
        for k in DIM:
            if level=='item': R.append(np.mean([mu(10*r[c]) for c in cols[k]],axis=0))
            else: R.append(mu(r[cols[k]].sum()))
        out.append(int(np.argmax(np.array(w)@np.array(R))))
    return np.array(out)

W0=[.5,.2,.2,.1]
base=classify(W0,'total')
print('Level counts (Good/Normal/Mild/Severe):', np.bincount(base,minlength=4), '- thesis reported 64/21/67/2')

print('\n== Weight sensitivity (vs original 50/20/20/10)')
S={k:d[c].sum(axis=1) for k,c in cols.items()}; S=pd.DataFrame(S)
P_=S/S.sum(); E=-(P_*np.log(P_)).sum()/np.log(len(S)); ent=((1-E)/(1-E).sum()).values
schemes={'Equal 25/25/25/25':[.25]*4,'Entropy (data-driven)':list(ent),
         'MH-dominant 60/15/15/10':[.6,.15,.15,.1],'Reverse 10/20/20/50':[.1,.2,.2,.5],
         'Random Dirichlet (200 draws)':None}
from sklearn.metrics import cohen_kappa_score as kap
rows=[]
for n,w in schemes.items():
    if w is None:
        rng=np.random.default_rng(1); agr=[]
        for _ in range(1000):
            agr.append(np.mean(classify(rng.dirichlet([1]*4),'total')==base)) if _<200 else None
        agr=[a for a in agr if a is not None]
        rows.append((n,'—',f'{np.mean(agr):.3f} (min {np.min(agr):.3f})','—')); continue
    c=classify(w,'total'); rows.append((n,np.round(w,3).tolist(),f'{np.mean(c==base):.3f}',f'{kap(c,base):.3f}'))
print(pd.DataFrame(rows,columns=['scheme','weights','agreement','kappa']).to_string(index=False))
print('entropy weights:',ent.round(3))

print('\n== Model comparison')
spec={'1-factor':'G =~ '+' + '.join(sum(cols.values(),[])),
 '4-factor correlated':'\n'.join(f'{k} =~ '+' + '.join(v) for k,v in cols.items()),
 'Bifactor':'G =~ '+' + '.join(sum(cols.values(),[]))+'\n'+'\n'.join(f'{k} =~ '+' + '.join(v) for k,v in cols.items())
   +'\n'+'\n'.join(f'G ~~ 0*{k}' for k in cols)+'\n'+'\n'.join(f'{a} ~~ 0*{b}' for i,a in enumerate(cols) for b in list(cols)[i+1:])}
for n,s in spec.items():
    m=semopy.Model(s); m.fit(d); st=semopy.calc_stats(m).T.iloc[:,0]
    print(f"{n:22s} chi2/df={st['chi2']/st['DoF']:.3f} CFI={st['CFI']:.3f} TLI={st['TLI']:.3f} RMSEA={st['RMSEA']:.3f} AIC={st['AIC']:.1f} BIC={st['BIC']:.1f}")
    if n=='Bifactor':
        e=m.inspect(std_est=True); g=e[(e.op=='~')&(e.rval=='G')]['Est. Std'].astype(float)
        sp=e[(e.op=='~')&(e.rval!='G')&e.rval.isin(cols)]['Est. Std'].astype(float)
        lam=g.values; th=1-lam**2
        print('  general-factor loadings mean',lam.mean().round(2),' specific mean',sp.mean().round(2))

print('\n== Segments (total score 40-200; higher = more difficulty)')
from scipy import stats
d['Total']=S.sum(axis=1); d['Gender']=d.gender.map({1:'Male',2:'Female'}); d['Year']=d.year.map({1:'Y1',2:'Y2',3:'Y3'})
d['Level']=np.array(LV_EN)[base]
for g in ['Gender','Year']:
    print(d.groupby(g)[['Total']].agg(['mean','std','count']).round(1).to_string())
    grp=[s.Total.values for _,s in d.groupby(g)]
    if len(grp)==2:
        t=stats.ttest_ind(*grp,equal_var=False); a,b=grp; dd=(a.mean()-b.mean())/np.sqrt((a.var(ddof=1)+b.var(ddof=1))/2)
        print(f'  Welch t={t.statistic:.2f} p={t.pvalue:.3f} d={dd:.2f}')
    else:
        f=stats.f_oneway(*grp); kw=stats.kruskal(*grp)
        ss_b=sum(len(x)*(x.mean()-d.Total.mean())**2 for x in grp); eta=ss_b/((d.Total-d.Total.mean())**2).sum()
        print(f'  ANOVA F={f.statistic:.2f} p={f.pvalue:.3f} eta2={eta:.3f}; Kruskal p={kw.pvalue:.3f}')
    ct=pd.crosstab(d[g],d.Level)[[l for l in LV_EN if l in d.Level.unique()]]
    print(ct.to_string()); chi=stats.chi2_contingency(ct); n=ct.values.sum()
    print(f'  chi2={chi[0]:.2f} df={chi[2]} p={chi[1]:.3f} CramerV={np.sqrt(chi[0]/(n*(min(ct.shape)-1))):.3f} minExpected={chi[3].min():.1f}')
# Item-level: highest-endorsed items (% agree 4-5)
agree=(d[sum(cols.values(),[])]>=4).mean().sort_values(ascending=False)
print('\nTop 5 items %agree:', agree.head(5).round(2).to_dict()); print('Bottom 5:', agree.tail(5).round(2).to_dict())
d.to_csv(OUT/'scored.csv',index=False)
