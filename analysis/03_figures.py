"""Figures 1-4 used in the portfolio."""
import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / 'figures'
plt.rcParams.update({'font.family':'Segoe UI','font.size':9,'axes.edgecolor':'#b8b7b0','axes.labelcolor':'#52514e',
  'xtick.color':'#52514e','ytick.color':'#52514e','axes.spines.top':False,'axes.spines.right':False,
  'svg.fonttype':'none','axes.titlesize':10,'axes.titleweight':'semibold','axes.titlecolor':'#0b0b0b','axes.titlelocation':'left'})
BLUE, ORANGE, MUTED, GRID = '#2a78d6', '#eb6834', '#8a8983', '#e6e5df'
d = pd.read_csv(ROOT / 'outputs' / 'scored.csv')   # created by 02_scoring_segments.py

# Fig 1: total score distribution
fig, ax = plt.subplots(figsize=(6.4,2.5))
ax.hist(d.Total, bins=range(40,201,8), color=BLUE, edgecolor='white', linewidth=2)
ax.set_xlabel('Total score (40 = no difficulty reported, 200 = maximum)'); ax.set_ylabel('Respondents')
ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True); ax.set_xlim(40,200)
ax.annotate('Few respondents in the middle\n(only 20 of 154 score 100–139)', xy=(118,7), xytext=(92,24), fontsize=8, color='#52514e',
            arrowprops=dict(arrowstyle='-', color=MUTED, lw=.8))
fig.tight_layout(); fig.savefig(FIG / 'fig1_distribution.svg'); plt.close()

# Fig 2: weight sensitivity
rows=[('Original weights 50/20/20/10',100),('Mental-health heavy 60/15/15/10',96.8),('Equal 25/25/25/25',94.8),
      ('Entropy (data-driven)',94.2),('Random weights (mean of 200)',90.6),('Fully reversed 10/20/20/50',87.0)]
fig, ax = plt.subplots(figsize=(6.4,2.3))
lab=[r[0] for r in rows][::-1]; val=[r[1] for r in rows][::-1]
cols=[MUTED if l.startswith('Original') else BLUE for l in lab]
ax.barh(lab, val, color=cols, height=.6)
for i,v in enumerate(val): ax.text(v+.6, i, f'{v:.1f}%', va='center', fontsize=8, color='#0b0b0b')
ax.set_xlim(0,108); ax.set_xlabel('% of respondents keeping the same classification as with original weights')
ax.xaxis.grid(True, color=GRID); ax.set_axisbelow(True); ax.spines['left'].set_visible(False); ax.tick_params(axis='y', length=0)
fig.tight_layout(); fig.savefig(FIG / 'fig2_sensitivity.svg'); plt.close()

# Fig 3: segment means with 95% CI
grp=[('Male','Gender'),('Female','Gender'),('Y1','Year'),('Y2','Year'),('Y3','Year')]
names={'Male':'Male (n=71)','Female':'Female (n=83)','Y1':'Year 1 (n=44)','Y2':'Year 2 (n=56)','Y3':'Year 3 (n=54)'}
fig, ax = plt.subplots(figsize=(6.4,2.4))
ys=[4.6,3.6,2,1,0]
for y,(g,c) in zip(ys,grp):
    x=d[d[c]==g].Total; m=x.mean(); h=stats.t.ppf(.975,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x))
    ax.plot([m-h,m+h],[y,y],color=BLUE,lw=2,solid_capstyle='round'); ax.plot(m,y,'o',ms=8,color=BLUE,mec='white',mew=2)
    ax.text(m+h+1.5,y,f'{m:.0f}',va='center',fontsize=8)
ax.axvline(d.Total.mean(),color=MUTED,lw=1,ls=(0,(3,3))); ax.text(d.Total.mean()+.8,5.15,f'Overall mean {d.Total.mean():.0f}',fontsize=8,color='#52514e')
ax.set_yticks(ys,[names[g] for g,_ in grp]); ax.tick_params(axis='y',length=0); ax.spines['left'].set_visible(False)
ax.set_xlim(95,140); ax.set_ylim(-.6,5.5); ax.set_xlabel('Mean total score with 95% confidence interval')
ax.xaxis.grid(True,color=GRID); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(FIG / 'fig3_segments.svg'); plt.close()

# Fig 4: top-2-box agreement by item, one panel per block
labels = {
 'A1':'Always short of time','A2':'Forgets plans','A3':'Feels low','A4':'Avoids gatherings','A5':'Irritable, noise-sensitive',
 'A6':'Study feels tiresome','A7':'Worries about judgement','A8':'Withdraws from others','A9':'Exhausted after short study','A10':'Poor sleep',
 'B1':'Bothered if unwelcome','B2':'Wants acceptance','B3':'Monitors own behaviour','B4':'Needs someone nearby','B5':'Awkward with strangers',
 'B6':'Worries about friendships','B7':"Can't focus in noise",'B8':'Dislikes social events','B9':'Tense in crowds','B10':'Uneasy around teachers',
 'C1':'Heavy workload','C2':'Fewer publications','C3':'English / literature','C4':'Finds research dull','C5':'Hard to publish',
 'C6':'Feels less able','C7':'Grade competition','C8':'Project difficulty','C9':'Strict supervisor','C10':"Parents' expectations",
 'D1':'Ashamed of emotions','D2':'Needs approval','D3':'Sharing feelings a burden','D4':'Rejects own weaknesses','D5':'Popularity over self',
 'D6':'Hides feelings','D7':'Must meet expectations','D8':'No committed life goal','D9':'Feels unloved','D10':'Afraid of failure'}
titles = {'A':'Mental health','B':'Social adaptation','C':'Academic pressure','D':'Self-actualisation'}
fig, axes = plt.subplots(2, 2, figsize=(6.6, 4.6))
for ax, b in zip(axes.ravel(), 'ABCD'):
    cols = [f'{b}{i}' for i in range(1, 11)]
    s = ((d[cols] >= 4).mean() * 100).sort_values()
    ax.barh([labels[c] for c in s.index], s.values, color=BLUE, height=.62)
    for i, v in enumerate(s.values): ax.text(v + 1, i, f'{v:.0f}%', va='center', fontsize=7)
    ax.set_xlim(0, 60); ax.set_title(f'{titles[b]} · block mean {s.mean():.0f}%', fontsize=8.5)
    ax.tick_params(axis='y', labelsize=7, length=0); ax.tick_params(axis='x', labelsize=7)
    ax.spines['left'].set_visible(False); ax.xaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    ax.set_xticks([0, 20, 40, 60], ['0', '20', '40', '60%'])
fig.tight_layout(h_pad=1.2, w_pad=1.5); fig.savefig(FIG / 'fig4_items.svg'); plt.close()
