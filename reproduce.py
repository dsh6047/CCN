"""Numerical checks using released scores only. No images, models or network.

Run: python reproduce.py
Outputs computed_results.json and prints deviations from archived results.
"""
from pathlib import Path
import json
import numpy as np
from scipy import stats
from scipy.integrate import trapezoid
from sklearn.metrics import roc_auc_score

ROOT=Path(__file__).resolve().parent
def read(name): return json.loads((ROOT/'data'/name).read_text())
def load(name):
    with np.load(ROOT/'data'/name,allow_pickle=False) as z:
        return {k:z[k] for k in z.files}
def ccn(title_scores):
    """Candidate-specific leave-one-out top-two rival subtraction (last axis)."""
    v=np.asarray(title_scores,float)
    if v.shape[-1]<3:raise ValueError('CCN requires at least three candidates')
    return np.stack([v[...,i]-np.sort(np.delete(v,i,axis=-1),axis=-1)[...,-2:].mean(-1)
                     for i in range(v.shape[-1])],axis=-1)
def f1_at(y,s,t):
    a=s>=t;tp=np.sum(a & (y==1));fp=np.sum(a & (y==0));fn=np.sum(~a & (y==1))
    return float(2*tp/(2*tp+fp+fn)) if tp else 0.
def archived_rank_oracle(y,s):
    """Preserve the original Table 3 rank-scan implementation.

    It scans every sorted position, including positions inside tied scores.
    This differs from a tie-grouped threshold sweep; sort-version differences
    can affect the archived oracle. Zero-threshold metrics do not use this scan.
    """
    if y.sum()==0 or np.ptp(s)<1e-12:return 0.,float(s.max())
    o=np.argsort(-s);tp=np.cumsum(y[o]);pp=np.arange(1,len(y)+1)
    pr=tp/pp;rc=tp/y.sum()
    f=np.divide(2*pr*rc,pr+rc,out=np.zeros_like(pr),where=(pr+rc)>0)
    j=int(np.argmax(f));return float(f[j]),float(s[o[j]])
def pick(u,target):
    s=np.sort(u)[::-1];k=int(np.floor(target*len(s)))
    return float(s[0])+1e-9 if k<=0 else float(s[k-1])
def curve(gr,gu,ok):
    # Exact unique-threshold sweep; searchsorted is equivalent to strict >.
    ts=np.r_[np.inf,np.unique(np.r_[gr,gu])[::-1],-np.inf]
    f=(len(gu)-np.searchsorted(np.sort(gu),ts,side='right'))/len(gu)
    d=(len(gr)-np.searchsorted(np.sort(gr),ts,side='right'))/len(gr)
    correct=np.sort(gr[ok])
    c=(len(correct)-np.searchsorted(correct,ts,side='right'))/len(gr)
    order=np.argsort(f,kind='mergesort')
    return f[order],d[order],c[order]
def integral(y,x):return float(trapezoid(y,x))
def metrics(gr,gu,gc,ok,target=.2):
    f,d,c=curve(gr,gu,ok);t=pick(gc,target);a=gr>t
    return {'det_auc':float(roc_auc_score(np.r_[np.ones(len(gr)),np.zeros(len(gu))],np.r_[gr,gu])),
            'ccr_area':integral(c,f),'theta':t,'far':float((gu>t).mean()),
            'correct':float((a&ok).mean()),'wrong':float((a&~ok).mean()),'reject':float((~a).mean())}

def main():
    result={};checks=[]
    def check(label,actual,expected,tolerance=1e-10):
        error=abs(float(actual)-float(expected))
        checks.append({'check':label,'absolute_error':error,'tolerance':tolerance,'pass':error<=tolerance})

    vcat=read('verification_catalogs.json'); old3=read('table3_archived.json')
    tags={'psw_sscd':'webtoon','aihub41_sscd':'aihub41','manga109_sscd':'manga109'}
    modes=['absolute','ccn','asnorm_top2','asnorm_all']
    oldm={'absolute':'abs','ccn':'con_top2','asnorm_all':'asnorm_all'}
    result['verification']={}
    for name,meta in vcat.items():
        z=load(f'verification_{name}.npz'); per=[]
        for ti in np.unique(z['title_index']):
            mask=z['title_index']==ti; row={'title_id':meta['titles'][ti]['title_id']}
            for j,m in enumerate(modes):
                values=[]
                for seed in np.unique(z['seed'][mask]):
                    ix=mask & (z['seed']==seed); y=z['label'][ix];s=z['scores'][ix,j]
                    fo,t=archived_rank_oracle(y,s)
                    values.append([f1_at(y,s,0),fo,float(roc_auc_score(y,s)),t,float((s[y==0]>=t).mean()),float((s[y==0]>t).mean())])
                row[m]=dict(zip(['f1_at_0','f1_oracle','auc','theta','fpr_at_oracle_theta','fpr_strict_at_oracle_theta'],np.mean(values,axis=0).tolist()))
            per.append(row)
        summary={m:{k:float(np.mean([r[m][k] for r in per])) for k in per[0][m]} for m in modes}
        for m in modes:
            th=np.array([r[m]['theta'] for r in per]); summary[m].update(theta_min=float(th.min()),theta_max=float(th.max()),theta_sd=float(th.std()))
        result['verification'][name]={'summary':summary,'per_title':per}
        displayed2={'psw_sscd':[[.538,.211,.730],[.826,.044,.955],[.817,.036,.887],[.820,.041,.936]],
                    'psw_dinov2':[[.418,.482,.619],[.638,.136,.838],[.622,.110,.783],[.636,.133,.838]]}
        if name in displayed2:
            for m,expected in zip(modes,displayed2[name]):
                for field,value in zip(['f1_oracle','fpr_strict_at_oracle_theta','auc'],expected):
                    check(f'Table2/{name}/{m}/{field}/displayed_3dp',round(summary[m][field],3),value)
        if name in tags:
            for m,om in oldm.items():
                old=old3[tags[name]]['methods'][om]
                for k in ['f1_at_0','auc','f1_oracle','theta_min','theta_max','theta_sd']:
                    check(f'Table3/{name}/{m}/{k}',summary[m][k],old[k])
        print('Verified query-score aggregation:',name,flush=True)

    gate_names=['max','lse@1','lse@0.1','lse@0.05','lse@0.02','1 rival','2 rivals','all rivals']
    old4=read('table4_archived.json'); result['openset']={}
    old_keys=list(old4['a'])
    for di,name in enumerate(['psw','aihub','manga109']):
        z=load(f'openset_{name}.npz'); splits={}
        for split in np.unique(z['split']):
            si=z['split']==split
            r=si & (z['role']==0); u=si & (z['role']==2); c=si & (z['role']==1)
            ok=z['prediction'][r]==z['owner'][r]
            row={m:metrics(z['scores'][r,j],z['scores'][u,j],z['scores'][c,j],ok) for j,m in enumerate(gate_names)}
            splits[str(split)]=row
            if split==0:
                for m in gate_names:
                    expected={**old4['a'][old_keys[di]][m],**old4['b'][old_keys[di]][m]}
                    for k,v in row[m].items():check(f'Table4/{name}/{m}/{k}',v,expected[k])
        result['openset'][name]=splits
    print('Verified Table 4 and Table S1, including all eight gates.',flush=True)

    # Figure 6: same PSW/AI Hub scored rows and strictly held-out calibration.
    result['transfer']={};old6=read('figure6_archived.json')
    pairs=[('psw','aihub'),('aihub','aihub'),('aihub','psw'),('psw','psw')]
    for (source,dest),oldkey in zip(pairs,old6):
        s=load(f'openset_{source}.npz');d=load(f'openset_{dest}.npz')
        r=d['role']==0;u=d['role']==2;cal=s['role']==1;ok=d['prediction'][r]==d['owner'][r]
        row={}
        for j,m in [(0,'absolute'),(6,'CCN')]:
            row[m]=[]
            for target in [.1,.2,.3]:
                v=metrics(d['scores'][r,j],d['scores'][u,j],s['scores'][cal,j],ok,target)
                row[m].append({'target':target,**v})
                old=next(o for o in old6[oldkey][m]['ops'] if o['target']==target)
                for k in ['far','correct','wrong','reject']:check(f'Fig6/{source}/{dest}/{m}/{target}/{k}',v[k],old[k])
        result['transfer'][f'{source}_to_{dest}']=row

    z=load('figure5_scores.npz');old5=read('figure5_archived.json');f5={};grid=np.linspace(0,1,401)
    for j,m in [(0,'abs'),(1,'con')]:
        rows=[];det=[];ccr=[]
        for split in range(-1,30):
            si=z['split']==split;r=si&(z['role']==0);u=si&(z['role']==2)
            gr=z['scores'][r,j];gu=z['scores'][u,j];ok=z['owner'][r]==z['prediction'][r]
            f,d,c=curve(gr,gu,ok)
            au=float(roc_auc_score(np.r_[np.ones(len(gr)),np.zeros(len(gu))],np.r_[gr,gu]));ar=integral(c,f)
            rows.append({'split':split,'det_auc':au,'ccr_area':ar})
            if split==-1:
                check(f'Fig5/main/{m}/det_auc',au,old5['main'][m]['det_auc'])
                check(f'Fig5/main/{m}/ccr_area',ar,old5['main'][m]['ccr_area'])
            else:
                fu=np.unique(f)
                det.append(np.interp(grid,fu,[d[f==v].max() for v in fu]))
                ccr.append(np.interp(grid,fu,[c[f==v].max() for v in fu]))
        for field in ['det_auc','ccr_area']:
            check(f'Fig5/strat/{m}/{field}',np.median([r[field] for r in rows[1:]]),old5['strat'][m][field+'_med'])
        for label,arr in [('det',det),('ccr',ccr)]:
            for suff,val in [('',np.mean(arr,0)),('_q1',np.percentile(arr,25,axis=0)),('_q3',np.percentile(arr,75,axis=0))]:
                check(f'Fig5/{m}/{label+suff}/max_error',np.max(np.abs(val-old5['strat'][m][label+suff])),0)
        f5[m]=rows
    result['figure5']=f5

    result['table5']={}
    for name,methods in [('sscd',['frozen_abs','frozen_con','arcface','arcface_con']),
                         ('dinov3',['frozen_abs','frozen_con','arcface','arcface_con']),('logistic',['lr'])]:
        rows=read(f'table5_{name}.json')['rows']
        result['table5'][name]={m:{k:float(np.mean([r[f'{m}_{k}'] for r in rows])) for k in ['auc','oracle','blind','loss']} for m in methods}
        for m in methods:
            for r in rows:check(f'Table5/{name}/{m}/{r["split"]}/loss_identity',r[f'{m}_loss'],r[f'{m}_oracle']-r[f'{m}_blind'])

    result['geometry']={}
    for name,rows in read('geometry.json').items():
        f=np.array([r['f1_con'] for r in rows]);flag=np.array([r['flagged'] for r in rows]);margin=np.array([r['margin_conmax'] for r in rows])
        result['geometry'][name]={'flagged_mean_f1':float(f[flag].mean()),'unflagged_mean_f1':float(f[~flag].mean()),
                                  'mannwhitney_one_sided_p':float(stats.mannwhitneyu(f[flag],f[~flag],alternative='less').pvalue),
                                  'pearson_margin_abs_f1':float(stats.pearsonr(margin,[r['f1_abs'] for r in rows]).statistic)}
    g=read('growth_fit_inputs.json');x=np.r_[g['AIHub']['K'],g['Manga109']['K']];y=np.r_[g['AIHub']['theta'],g['Manga109']['theta']]
    a,b=np.linalg.lstsq(np.stack([np.ones(len(x)),-np.log(x)],axis=1),y,rcond=None)[0]
    predicted=a-b*np.log(x)
    result['joint_fit']={'a':float(a),'b':float(b),'r2':float(1-np.sum((y-predicted)**2)/np.sum((y-y.mean())**2)),
                         'zero_crossing':float(np.exp(a/b)),'residual_std':float(np.std(y-predicted))}
    check('Fig7/joint_fit/rounded_r2',round(result['joint_fit']['r2'],3),.919)
    check('Fig7/joint_fit/rounded_zero_crossing',round(result['joint_fit']['zero_crossing']),36)
    for name,keys in [('growth_verification',[('theta','median','q1','q3')]),
                      ('growth_far',[('far_absolute','abs_med','abs_q1','abs_q3'),('far_ccn','con_med','con_q1','con_q3')])]:
        z=load(name+'_orders.npz');old=read(name+'_archived.json')
        for arr,med,q1,q3 in keys:
            for key,val in [(med,np.median(z[arr],axis=0)),(q1,np.percentile(z[arr],25,axis=0)),(q3,np.percentile(z[arr],75,axis=0))]:
                check(f'Fig7/{name}/{key}/max_error',np.max(np.abs(val-old[key])),0)
    growth=read('growth_verification_archived.json');ks=np.asarray(growth['k']);res=np.asarray(growth['median'])-(a-b*np.log(ks))
    result['joint_fit'].update(growth_mean_abs_residual_within=float(np.abs(res[ks<=41]).mean()),
                              growth_mean_abs_residual_beyond=float(np.abs(res[ks>41]).mean()),
                              growth_max_abs_residual_beyond=float(np.abs(res[ks>41]).max()))
    result['checks']=checks
    (ROOT/'computed_results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    failed=[c for c in checks if not c['pass']]
    print(f'{len(checks)-len(failed)}/{len(checks)} numerical comparisons passed.')
    for c in failed:print(c)
    print('Computed results saved to computed_results.json')
    if failed:raise SystemExit(1)
if __name__=='__main__':main()
