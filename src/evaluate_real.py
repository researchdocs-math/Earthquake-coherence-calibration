"""Real-data calibration, fixed perturbations, and optical-reference evaluation."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
import argparse,json,time,csv
import numpy as np
from scipy.special import logit
from scipy.ndimage import binary_dilation
from sklearn.metrics import average_precision_score,roc_auc_score
from calibration import *
from data_utils import load_event,ROOT

def evaluate(name):
 start=time.perf_counter();a,c,meta,fit,cal,test,eligible,weights,extra=load_event(name);reg=region_grid(a.shape[1:]);nregions=4
 good=np.array([all((np.isfinite(x)[(reg==r)&eligible]).mean()>=.95 for r in range(4)) for x in a])
 if not good[-1]:raise ValueError('The event map fails the fixed regional coverage requirement.')
 cal_used=cal[good[cal]];test_used=test[good[test]]
 mu,scale=fit_scores(a[fit]);z=score(a,mu,scale);z[:,~eligible]=np.nan
 methods,info=fit_calibrators(z[fit],z[cal_used],reg,weights)
 rows=[]
 for method,th in methods.items():
  for i,m in zip(test_used,alarm_metrics(z[test_used],th,reg,weights)):
   rows.append(dict(event=name,score='robust',method=method,pair=c['pairs'][i],**m))
  eventmetric=alarm_metrics(z[-1:],th,reg,weights)[0];rows.append(dict(event=name,score='robust',method=method,pair=c['pair'],is_event=True,**eventmetric))
 # Two established coherence score families isolate calibration from scoring.
 base=np.nanmedian(a[fit],axis=0)
 auxiliary=[]
 for label,scores in [('difference',base-a),('log-ratio',np.log(np.maximum(base,1/255)/np.maximum(a,1/255)))]:
  scores[:,~eligible]=np.nan;mm,_=fit_calibrators(scores[fit],scores[cal_used],reg,weights)
  for method in ['Pooled','Global-map','Bonferroni','Regional-max']:
   met=alarm_metrics(scores[test_used],mm[method],reg,weights)
   auxiliary.append(dict(event=name,score=label,method=method,n=len(met),exceedance=np.mean([m['any_exceed'] for m in met]),area=np.mean([m['area'] for m in met])))
 # Perturb each held-out map with a 5-km-scale contiguous square, fixed RNG.
 rng=np.random.default_rng(80421);injections=[];hw=a.shape[1:];patchside=50
 for ii,i in enumerate(test_used):
  for rep in range(8):
   for attempt in range(1000):
    y=int(rng.integers(0,hw[0]-patchside));x=int(rng.integers(0,hw[1]-patchside));patch=(slice(y,y+patchside),slice(x,x+patchside))
    if (eligible[patch]&np.isfinite(a[i][patch])).mean()>.9:break
   else:continue
   for delta in [.05,.10,.20,.30]:
    changed=a[i][patch].copy();changed=np.maximum(changed-delta,1/255);zz=score(changed,mu[patch],scale[patch]);valid=eligible[patch]&np.isfinite(zz);w=weights[patch]*valid
    for method,th in methods.items():
     detection=(zz>th[reg[patch]])&valid;rec=float(w[detection].sum()/w.sum())
     injections.append(dict(event=name,pair=c['pairs'][i],rep=rep,delta=delta,method=method,recall=rec))
 out=ROOT/'results'/name;out.mkdir(exist_ok=True)
 for f,data in [('background',rows),('injections',injections),('score_ablation',auxiliary)]:
  (out/(f+'.json')).write_text(json.dumps(data,indent=2))
 summary=dict(event=name,frame=c['frame'],shape=list(a.shape),fit_indices=fit.tolist(),cal_indices=cal_used.tolist(),test_indices=test_used.tolist(),quality_rejected_cal=int((~good[cal]).sum()),quality_rejected_test=int((~good[test]).sum()),fit_dates=[c['pairs'][fit[0]],c['pairs'][fit[-1]]],cal_dates=[c['pairs'][cal_used[0]],c['pairs'][cal_used[-1]]],test_dates=[c['pairs'][test_used[0]],c['pairs'][test_used[-1]]],eligible_pixels=int(eligible.sum()),eligible_fraction=float(eligible.mean()),thresholds={k:v.tolist() for k,v in methods.items()},q=float(info['q']),event_q=float(np.nanmax((region_quantiles(z[-1:],reg,weights)[0]-info['a'])/info['b'])),transform=meta['transform'],elapsed_seconds=time.perf_counter()-start,**extra)
 (out/'summary.json').write_text(json.dumps(summary,indent=2))
 np.savez_compressed(out/'map_results.npz',pre_coherence=np.nanmedian(a[fit],axis=0),event_coherence=a[-1],event_score=z[-1],eligible=eligible,regions=reg,thresholds=methods['Regional-max'][reg],pooled_threshold=methods['Pooled'][reg],a=info['a'],b=info['b'],u=info['u'],mu=mu,scale=scale)
 # Sensitivity analyses use exactly the same fitting/calibration/test partitions.
 sensitivity=[]
 for side in [1,2,3,4]:
  rr=region_grid(a.shape[1:],side)
  for rho in [.01,.025,.05,.10]:
   # Each spatial region must have eligible pixels.
   if any(weights[rr==j].sum()==0 for j in range(side*side)):continue
   mm,ii=fit_calibrators(z[fit],z[cal_used],rr,weights,rho=rho)
   for method in ['Pooled','Global-map','Bonferroni','Regional-max']:
    met=alarm_metrics(z[test_used],mm[method],rr,weights,rho)
    sensitivity.append(dict(event=name,regions=side*side,rho=rho,method=method,n=len(met),exceedance=np.mean([m['any_exceed'] for m in met]),area=np.mean([m['area'] for m in met]),finite=bool(np.isfinite(mm[method]).all())))
 (out/'sensitivity.json').write_text(json.dumps(sensitivity,indent=2))
 summary['elapsed_seconds']=time.perf_counter()-start
 (out/'summary.json').write_text(json.dumps(summary,indent=2))
 print(name,'done',summary['eligible_pixels'],'eligible',len(cal_used),'cal',len(test_used),'test',round(time.perf_counter()-start,1),'s',flush=True)
 return summary

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--event',choices=['all','turkey','ridgecrest','morocco'],default='all');ap.add_argument('--workers',type=int,default=3);args=ap.parse_args()
 names=['turkey','ridgecrest','morocco'] if args.event=='all' else [args.event]
 if len(names)>1:
  from concurrent.futures import ProcessPoolExecutor
  with ProcessPoolExecutor(args.workers) as ex:list(ex.map(evaluate,names))
 else:evaluate(names[0])
if __name__=='__main__':main()
