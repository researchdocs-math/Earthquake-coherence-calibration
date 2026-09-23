"""Numerical summaries; no document or artwork generation."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];RES=ROOT/'results'
def block_interval(x,seed=4419,repetitions=2000,block=4):
 x=np.asarray(x,dtype=float);n=len(x);rng=np.random.default_rng(seed)
 starts=rng.integers(n,size=(repetitions,int(np.ceil(n/block))))
 ids=((starts[:,:,None]+np.arange(block))%n).reshape(repetitions,-1)[:,:n]
 return np.quantile(x[ids].mean(axis=1),[.025,.975]).tolist()
def main():
 backgrounds=[];injections=[];neural=[]
 for event in ['ridgecrest','turkey','morocco']:
  d=pd.read_json(RES/event/'background.json');d=d[d.is_event!=True]
  for method,dd in d.groupby('method'):
   dd=dd.sort_values('pair');lo,hi=block_interval(dd.any_exceed)
   backgrounds.append(dict(event=event,method=method,n=len(dd),breaches=int(dd.any_exceed.sum()),rate=float(dd.any_exceed.mean()),mean_area=float(dd.area.mean()),block_low=lo,block_high=hi))
  d=pd.read_json(RES/event/'injections.json')
  for (method,delta),dd in d.groupby(['method','delta']):
   v=dd.groupby('pair').recall.mean().sort_index();lo,hi=block_interval(v)
   injections.append(dict(event=event,method=method,delta=float(delta),dates=len(v),patches=len(dd),mean=float(v.mean()),low=lo,high=hi))
  f=RES/event/'gru_background.json'
  if f.exists():
   d=pd.read_json(f)
   for method,dd in d.groupby('method'):neural.append(dict(event=event,method=method,n=len(dd),breaches=int(dd.any_exceed.sum()),rate=float(dd.any_exceed.mean()),mean_area=float(dd.area.mean())))
 for name,rows in [('background_summary',backgrounds),('injection_summary',injections),('gru_summary',neural)]:
  (RES/(name+'.json')).write_text(json.dumps(rows,indent=2));pd.DataFrame(rows).to_csv(RES/(name+'.csv'),index=False)
 mc=pd.read_csv(RES/'monte_carlo.csv.gz');out=mc.groupby(['scenario','method']).agg(repetitions=('any_exceed','size'),breaches=('any_exceed','sum'),rate=('any_exceed','mean'),area=('alarm_area','mean'),recall=('injection_recall','mean')).reset_index();out.to_csv(RES/'simulation_summary.csv',index=False)
 print(pd.DataFrame(backgrounds).query("method in ['Pooled','Global-map','Bonferroni','Regional-max']").to_string(index=False))
if __name__=='__main__':main()
