"""Seeded independent Monte Carlo repetitions; no manuscript-generation code."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse,json,time,platform
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter
from calibration import corrected_quantile
ROOT=Path(__file__).resolve().parents[1]
SCENARIOS=['independent','spatial','common-shock','heterogeneous','temporal-AR','mean-shift']
METHODS=['Gaussian','Pooled','Regional-pool','Global-map','Bonferroni','Regional-max','Unscaled-max','Repaired-max']
def run_batch(args):
 scenario,start,count=args;rows=[]
 for rep in range(start,start+count):
  rng=np.random.default_rng(np.random.SeedSequence([20260923,SCENARIOS.index(scenario),rep]))
  nfit,ncal,h,w=24,39,32,32;t=nfit+ncal+1
  field=rng.standard_normal((t,h,w)).astype(np.float64)
  if scenario!='independent':
   sig=1.25;x=np.arange(-5,6);k=np.exp(-x*x/(2*sig*sig));k/=k.sum()
   field=gaussian_filter(field,(0,sig,sig),mode='wrap',truncate=4)/np.sum(k*k)
  if scenario in ['common-shock','heterogeneous','temporal-AR','mean-shift']:
   c=rng.normal(size=t)
   if scenario=='temporal-AR':
    for j in range(1,t):c[j]=.8*c[j-1]+.6*c[j]
   field=.8*field+.6*c[:,None,None]
  if scenario=='heterogeneous':
   field[:,:,:16]*=.65;field[:,:,16:]*=1.6
  if scenario=='mean-shift':field[-1]+=.8
  regions=np.stack([field[:,:16,:16].reshape(t,-1),field[:,:16,16:].reshape(t,-1),field[:,16:,:16].reshape(t,-1),field[:,16:,16:].reshape(t,-1)],axis=1)
  q=np.quantile(regions,.95,axis=2,method='inverted_cdf')
  a=np.median(q[:nfit],axis=0);b=np.maximum(1.4826*np.median(abs(q[:nfit]-a),axis=0),.25)
  qc=q[nfit:nfit+ncal];u=np.max((qc-a)/b,axis=1)
  cal=regions[nfit:nfit+ncal];test=regions[-1]
  pooled=np.quantile(cal,.95,method='inverted_cdf');globalq=np.quantile(cal.reshape(ncal,-1),.95,axis=1,method='inverted_cdf')
  th=[np.full(4,1.6448536269514722),np.full(4,pooled),np.quantile(cal,.95,axis=(0,2),method='inverted_cdf'),np.full(4,corrected_quantile(globalq)),np.array([corrected_quantile(qc[:,j],.1/4) for j in range(4)]),a+b*corrected_quantile(u),np.full(4,corrected_quantile(qc.max(axis=1))),a+b*corrected_quantile(u,corruption=2)]
  for method,threshold in zip(METHODS,th):
   frac=(test>threshold[:,None]).mean(axis=1)
   # A contiguous 8x8 signal patch in the first geographic region.
   signal=test[0].reshape(16,16).copy();signal[4:12,4:12]+=3
   recall=float((signal[4:12,4:12]>threshold[0]).mean())
   rows.append([scenario,rep,method,float(frac.mean()),float(frac.max()),int(frac.max()>.05),recall])
  # Worst downward corruption: replace the two largest map scores.
  corrupt=np.sort(u).copy();corrupt[-2:]=-1e6
  for label,m in [('corrupt-unrepaired',0),('corrupt-repaired',2)]:
   threshold=a+b*corrected_quantile(corrupt,corruption=m);frac=(test>threshold[:,None]).mean(axis=1)
   rows.append([scenario,rep,label,float(frac.mean()),float(frac.max()),int(frac.max()>.05),np.nan])
 return rows

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--repetitions',type=int,default=50000);ap.add_argument('--workers',type=int,default=8);args=ap.parse_args();start=time.perf_counter()
 import csv
 tasks=[(s,i,min(25,args.repetitions-i)) for s in SCENARIOS for i in range(0,args.repetitions,25)]
 import gzip,io
 out=ROOT/'results';out.mkdir(exist_ok=True)
 with gzip.GzipFile(filename=str(out/'monte_carlo.csv.gz'),mode='wb',mtime=0) as raw:
  with io.TextIOWrapper(raw,encoding='utf-8',newline='') as f:
   w=csv.writer(f);w.writerow(['scenario','replicate','method','alarm_area','max_region_area','any_exceed','injection_recall'])
   with ProcessPoolExecutor(args.workers) as ex:
    for i,result in enumerate(ex.map(run_batch,tasks)):
     w.writerows(result)
     if i%400==0:print('Monte Carlo batches',i+1,'/',len(tasks),flush=True)
 (out/'monte_carlo_runtime.json').write_text(json.dumps({'seconds':time.perf_counter()-start,'workers':args.workers,'repetitions_per_scenario':args.repetitions,'platform':platform.platform()},indent=2))
if __name__=='__main__':main()
