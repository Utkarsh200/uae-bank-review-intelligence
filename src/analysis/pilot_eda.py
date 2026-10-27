"""Pilot EDA and preliminary RQ1 / RQ4 analysis (interim report).

Run from the repository root:  python src/analysis/pilot_eda.py
Reads  data/processed/uae_bank_reviews.csv
Writes reports/figures/fig4..fig8.png and reports/pilot_results.json

NOTE: themes here come from a transparent keyword lexicon (English + Arabic), a rule-based
stand-in for the zero-shot LLM classifier that replaces it in the final report.
Requires: pandas numpy scipy statsmodels scikit-posthocs matplotlib
"""
import sys, re, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.formula.api as smf
import scikit_posthocs as sp
warnings.filterwarnings("ignore")
ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[2]
FIG = ROOT / "reports" / "figures"; FIG.mkdir(parents=True, exist_ok=True)
d=pd.read_csv(ROOT/'data'/'processed'/'uae_bank_reviews.csv',parse_dates=['review_date','reply_date'])
AR=re.compile(r'[؀-ۿݐ-ݿ]'); LAT=re.compile(r'[A-Za-z]')
ARZ=re.compile(r"\b(?:[a-z]*[2356789][a-z]+|[a-z]+[2356789][a-z]*)\b|\b(?:mashkala|mushkila|moshkela|shukran|shukran|tamam|kteer|ktir|kwayes|zain|walla|wallah|yalla|habibi|bas|mafi|mish|msh|inshallah|inshaallah|mu3|ana|enta|ya3ni|yani|lazem|leh|shu|shino|fi)\b",re.I)
EN=re.compile(r"\b(the|and|is|to|it|this|that|app|bank|not|good|very|great|my|me|for|with|i|you|can|cannot|working|please|account|login|can't|don't|worst)\b",re.I)
def lang(t):
    a=len(AR.findall(t)); l=len(LAT.findall(t))
    if a+l==0: return 'none'
    s=a/(a+l)
    if s>=.8: return 'ar'
    if s>.2: return 'mixed'
    en=len(EN.findall(t)); az=len(re.findall(r"\b[a-z]*[2356789][a-z]+\b",t.lower()))+len(re.findall(r"\b(mashkala|mushkila|moshkela|shukran|tamam|kteer|ktir|kwayes|walla|wallah|habibi|inshallah|ya3ni|yani|lazem|mafi|mish)\b",t.lower()))
    return 'arabizi' if az>=1 and az>=en/2 else 'en'
d['language']=d.review_text.map(lang)
d['review_length']=d.review_text.str.split().str.len()
d['month']=d.review_date.dt.to_period('M').astype(str)
d['replied']=d.bank_reply.notna().astype(int)
d['reply_lag_days']=(d.reply_date-d.review_date).dt.total_seconds()/86400
d['short']=(d.review_length<3)|(d.language=='none')


LEX={
'login':r"log ?in|sign ?in|password|otp|face ?id|biometric|finger ?print|locked|verification code|authenticat|logout|register device|تسجيل الدخول|دخول|كلمة المرور|كلمة السر|الرمز|رمز التحقق|بصمة|باسورد|الباسورد",
'transfers':r"transfer|payment|\bpay\b|send money|remittance|beneficiar|transaction|bill|declined|حول|تحويل|دفع|حوالة|عملية|عمليات|مدفوعات|سداد",
'cards':r"\bcards?\b|credit|debit|\bpin\b|activate|\bblock|limit|apple pay|atm|cvv|بطاقة|بطاقتي|البطاقة|كرت|كارد|فيزا",
'performance':r"crash|slow|\bbug|error|loading|freez|hang|not working|doesn.?t work|does not work|glitch|stuck|\blag|server|network|connection|won.?t open|not open|keeps|يعلق|بطيء|بطئ|خطأ|لا يعمل|ما يشتغل|لا يفتح|ما يفتح|هنج|يهنج|عطل|معلق|مشكلة|مشاكل|تحديث",
'onboarding':r"open(ing)? (an? )?account|new account|kyc|emirates id|\beid\b|onboard|sign ?up|salary|فتح حساب|حساب جديد|الهوية|هوية|توثيق",
'support':r"support|customer (service|care)|call ?cent|contact|complain|response|respond|helpline|\bchat\b|agent|no reply|خدمة العملاء|الدعم|شكوى|شكاوي|تواصل|اتصال|الرد",
'ui':r"design|interface|\bui\b|\bux\b|feature|user.?friendly|easy|simple|navigation|layout|dark mode|dashboard|option|smooth|تصميم|واجهة|ميزة|ميزات|خاصية|سهل|سهولة|صعب",
'fees':r"\bfees?\b|charge|interest|\brate|commission|penalt|expensive|deduct|رسوم|عمولة|فائدة|خصم|رسم"}
ORDER=['transfers','cards','login','performance','onboarding','support','fees','ui']
SEV={'login':.7,'transfers':.9,'cards':.8,'performance':.6,'onboarding':.6,'support':.5,'ui':.3,'fees':.4}
txt=d.review_text.str.lower()
cnt=pd.DataFrame({k:txt.str.count(v) for k,v in LEX.items()})
mx=cnt.max(axis=1)
# tie -> ORDER priority
theme=np.where(mx==0,'other',cnt[ORDER].idxmax(axis=1))
d['theme']=theme
res={}
res['n']=len(d); res['themed_share']=float((d.theme!='other').mean())
res['theme_counts']=d.theme.value_counts().to_dict()
# ---------- RQ1 ----------
low=d[(d.rating<=2)&(d.theme!='other')]
ct=pd.crosstab(low.theme,low.group)
chi2,p,dof,exp=stats.chi2_contingency(ct)
V=np.sqrt(chi2/(ct.values.sum()*(min(ct.shape)-1)))
res['rq1_chi']=dict(n=int(ct.values.sum()),chi2=float(chi2),dof=int(dof),p=float(p),V=float(V),min_expected=float(exp.min()),n_low_all=int((d.rating<=2).sum()),n_low_themed=int(len(low)))
res['rq1_ct_pct']=(ct/ct.sum()).round(3).to_dict()
cl=d[d.theme!='other']
groups=[g.rating.values for _,g in cl.groupby('theme')]
H,pk=stats.kruskal(*groups)
eps=H/(len(cl)-1)  # epsilon-squared
res['rq1_kw']=dict(H=float(H),p=float(pk),eps2=float(eps),n=int(len(cl)),k=len(groups))
res['theme_rating']=cl.groupby('theme').rating.agg(['count','mean','median']).round(2).to_dict('index')
res['theme_lowshare']=cl.groupby('theme').rating.apply(lambda s:(s<=2).mean()).round(3).to_dict()
dn=sp.posthoc_dunn(cl,val_col='rating',group_col='theme',p_adjust='bonferroni')
res['dunn_sig_pairs']=int(((dn<.05).values.sum()-len(dn))/2); res['dunn_pairs']=int(len(dn)*(len(dn)-1)/2)
# group-level KW of rating
gg=[g.rating.values for _,g in d.groupby('group')]
Hg,pg=stats.kruskal(*gg); res['rq1_kw_group']=dict(H=float(Hg),p=float(pg),eps2=float(Hg/(len(d)-1)))
res['group_rating']=d.groupby('group').rating.agg(['count','mean','median']).round(3).to_dict('index')
res['group_low']=d.groupby('group').rating.apply(lambda s:(s<=2).mean()).round(3).to_dict()
res['bank_rating']=d.groupby('bank').rating.agg(['count','mean']).round(2).to_dict('index')
# ---------- RQ4 pilot ----------
d['m']=d.review_date.dt.to_period('M')
dd=d[d.theme!='other'].copy()
bm=d.groupby(['bank','m']).agg(bank_n=('rating','size'),bank_mean=('rating','mean')).reset_index()
dd['late']=(dd.review_date.dt.day>dd.review_date.dt.days_in_month-14).astype(int)
cell=dd.groupby(['bank','theme','m']).agg(n=('rating','size'),rating_m=('rating','mean'),late=('late','mean')).reset_index()
cell=cell.merge(bm,on=['bank','m'])
cell['freq_share']=cell.n/cell.bank_n
cell['F']=cell.freq_share/cell.groupby(['bank','m']).freq_share.transform('max')
cell['S']=cell.theme.map(SEV)
cell['R']=(cell.late-cell.late.min())/(cell.late.max()-cell.late.min())
cell['I']=((cell.bank_mean-cell.rating_m).clip(lower=0))/4
W=dict(F=.3,S=.3,R=.2,I=.2)
cell['P']=sum(cell[k]*w for k,w in W.items())
nxt=cell[['bank','theme','m','n','rating_m']].copy(); nxt['m']=nxt.m-1
nxt=nxt.rename(columns={'n':'n_next','rating_m':'rating_next'})
pan=cell.merge(nxt,on=['bank','theme','m'])
pan=pan[(pan.n>=10)&(pan.n_next>=10)].copy()
pan['decline']=pan.rating_m-pan.rating_next
pan['series']=pan.bank+'|'+pan.theme
res['rq4_cells_possible']=int(11*24*8); res['rq4_cells_nonempty']=int(len(cell)); res['rq4_usable']=int(len(pan)); res['rq4_series']=int(pan.series.nunique())
rho,pr=stats.spearmanr(pan.P,pan.decline)
rhoF,_=stats.spearmanr(pan.freq_share,pan.decline)
# cluster bootstrap
rng=np.random.default_rng(42); ser=pan.series.unique(); grp={s:g for s,g in pan.groupby('series')}
bs=[]
for _ in range(2000):
    pick=rng.choice(ser,len(ser)); x=pd.concat([grp[s] for s in pick])
    bs.append(stats.spearmanr(x.P,x.decline)[0])
res['rq4_spearman']=dict(rho=float(rho),ci=[float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))],rho_freq=float(rhoF))
m1=smf.ols('decline ~ freq_share + rating_m',pan).fit(cov_type='cluster',cov_kwds={'groups':pd.factorize(pan.series)[0]})
m2=smf.ols('decline ~ freq_share + rating_m + P',pan).fit(cov_type='cluster',cov_kwds={'groups':pd.factorize(pan.series)[0]})
w=m2.t_test('P = 0')
res['rq4_reg']=dict(r2_1=float(m1.rsquared),r2_2=float(m2.rsquared),dR2=float(m2.rsquared-m1.rsquared),beta_P=float(m2.params['P']),se_P=float(m2.bse['P']),F=float(w.statistic[0][0]**2),p=float(m2.pvalues['P']),beta_rating_m=float(m2.params['rating_m']))
# sensitivity: equal weights, and freq-only
for nm,wt in {'equal':dict(F=.25,S=.25,R=.25,I=.25),'freq_heavy':dict(F=.5,S=.2,R=.15,I=.15)}.items():
    p_=sum(pan[k]*v for k,v in wt.items()); res['rq4_sens_'+nm]=float(stats.spearmanr(p_,pan.decline)[0])
# top priority example: latest month
last=cell[cell.m==cell.m.max()].sort_values('P',ascending=False).head(5)[['bank','theme','n','rating_m','P']].round(3)
res['rq4_top_last_month']=last.to_dict('records')
res['rq4_top_themes_overall']=cell.groupby('theme').P.mean().round(3).sort_values(ascending=False).to_dict()
# pilot summaries for status
res['replied']=float(d.replied.mean())

out_cols=[c for c in d.columns if c not in ('short',)]
d[out_cols].assign(m=d['m'].astype(str)).to_csv(ROOT/'data'/'processed'/'analysis_dataset.csv',index=False,encoding='utf-8-sig')
pan.assign(m=pan['m'].astype(str)).to_csv(ROOT/'data'/'processed'/'rq4_panel.csv',index=False,encoding='utf-8-sig')
json.dump(res,open(ROOT/'reports'/'pilot_results.json','w'),indent=1,default=str)

plt.rcParams.update({'font.family':'serif','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.25,'figure.dpi':200})

C={'Conventional':'#2F4B7C','Islamic':'#2E8B6F','Digital-only':'#C77D1A'}
# Fig4 monthly volume
mv=d.groupby(['month','group']).size().unstack()
fig,ax=plt.subplots(figsize=(6.5,3.2))
for g in C: ax.plot(range(len(mv)),mv[g],marker='o',ms=3,color=C[g],label=g)
ax.set_xticks(range(0,len(mv),3)); ax.set_xticklabels(mv.index[::3],rotation=0,fontsize=7.5)
ax.set_ylabel('Reviews per month'); ax.legend(frameon=False,fontsize=8); ax.set_yscale('log'); ax.set_xlabel('Month')
fig.tight_layout(); fig.savefig(FIG/'fig4.png'); plt.close()
# Fig5 rating distribution by group
rd=pd.crosstab(d.group,d.rating,normalize='index')*100
cols=['#B3402F','#D98A5B','#C9C9C9','#7FA7C9','#2F4B7C']
fig,ax=plt.subplots(figsize=(6.5,2.8)); left=np.zeros(3)
order=['Conventional','Islamic','Digital-only']
for r,c in zip(range(1,6),cols):
    v=rd.loc[order,r].values; ax.barh(order,v,left=left,color=c,label=f'{r} star')
    for i,(l,x) in enumerate(zip(left,v)):
        if x>6: ax.text(l+x/2,i,f'{x:.0f}%',ha='center',va='center',color='white' if r in (1,5) else 'black',fontsize=8)
    left+=v
ax.set_xlabel('Share of reviews (%)'); ax.legend(ncol=5,frameon=False,fontsize=8,loc='upper center',bbox_to_anchor=(.5,1.22)); ax.invert_yaxis(); ax.grid(False)
fig.tight_layout(); fig.savefig(FIG/'fig5.png'); plt.close()
# Fig6 language by group
lg=pd.crosstab(d.group,d.language,normalize='index')*100
fig,ax=plt.subplots(figsize=(6.5,2.8))
x=np.arange(3); w=.2
for i,(l,c) in enumerate([('en','#2F4B7C'),('ar','#2E8B6F'),('none','#9A9A9A'),('arabizi','#C77D1A')]):
    ax.bar(x+(i-1.5)*w,lg.loc[order,l],w,color=c,label={'en':'English','ar':'Arabic','none':'No letters (emoji/digits)','arabizi':'Arabizi'}[l])
ax.set_xticks(x); ax.set_xticklabels(order); ax.set_ylabel('Share of reviews (%)'); ax.legend(frameon=False,fontsize=7.5,ncol=4,loc='lower center',bbox_to_anchor=(.5,1.0))
for i,g in enumerate(order): ax.text(i-0.3,lg.loc[g,'en']+1.5,f"{lg.loc[g,'en']:.0f}%",fontsize=7.5,ha='center'); ax.text(i-0.1,lg.loc[g,'ar']+1.5,f"{lg.loc[g,'ar']:.1f}%",fontsize=7.5,ha='center')
fig.tight_layout(); fig.savefig(FIG/'fig6.png'); plt.close()
# Fig7 theme share among low-rated (<=2) by group
low=d[(d.rating<=2)&(d.theme!='other')]
ct=pd.crosstab(low.theme,low.group,normalize='columns')*100
th=['performance','transfers','support','login','cards','ui','fees','onboarding']
ct=ct.loc[th,order]
fig,ax=plt.subplots(figsize=(5.4,3.6)); im=ax.imshow(ct.values,cmap='Blues',aspect='auto')
ax.set_xticks(range(3)); ax.set_xticklabels(order); ax.set_yticks(range(len(th))); ax.set_yticklabels(th); ax.grid(False)
for i in range(len(th)):
    for j in range(3): ax.text(j,i,f'{ct.values[i,j]:.1f}',ha='center',va='center',color='white' if ct.values[i,j]>22 else 'black',fontsize=8)
ax.set_title('% of themed low-rated (1–2 stars) reviews',fontsize=8.5); fig.colorbar(im,fraction=.04); fig.tight_layout(); fig.savefig(FIG/'fig7.png'); plt.close()
# Fig8 mean rating by theme w CI
cl=d[d.theme!='other']; g=cl.groupby('theme').rating.agg(['mean','std','count']); g['ci']=1.96*g['std']/np.sqrt(g['count']); g=g.sort_values('mean')
fig,ax=plt.subplots(figsize=(6.2,3.2)); ax.barh(g.index,g['mean'],xerr=g.ci,color=['#2F4B7C' if t!='ui' else '#9A9A9A' for t in g.index],capsize=2)
for i,(t,r) in enumerate(g.iterrows()): ax.text(r['mean']+r.ci+.05,i,f"{r['mean']:.2f} (n={int(r['count']):,})",va='center',fontsize=7.5)
ax.set_xlim(0,5.6); ax.set_xlabel('Mean star rating (95% CI)'); ax.grid(axis='y',alpha=0)
fig.tight_layout(); fig.savefig(FIG/'fig8.png'); plt.close()
print('figures and results written')
