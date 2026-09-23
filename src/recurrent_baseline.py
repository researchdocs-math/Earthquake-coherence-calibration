"""Small published GRU architecture, fitted only to pre-calibration observations."""
import argparse,json,time,os
from pathlib import Path
import numpy as np
import torch
from scipy.special import logit
from vendor.rnn_model import RNN
from data_utils import ROOT,load_event
from calibration import region_grid,fit_calibrators,alarm_metrics

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--event',required=True);ap.add_argument('--epochs',type=int,default=60);ap.add_argument('--device',default='mps');ap.add_argument('--retrain',action='store_true');args=ap.parse_args()
 start=time.perf_counter();torch.set_num_threads(6);torch.manual_seed(271828);rng=np.random.default_rng(271828)
 device=torch.device(args.device if args.device!='mps' or torch.backends.mps.is_available() else 'cpu')
 a,c,meta,fit,cal,test,eligible,weights,extra=load_event(args.event);reg=region_grid(a.shape[1:])
 good=np.array([all(np.isfinite(x)[(reg==r)&eligible].mean()>=.95 for r in range(4)) for x in a]);cal=cal[good[cal]];test=test[good[test]]
 # Published baseline uses logit of squared coherence.
 yy=logit(np.clip(a*a,1e-4,1-1e-4)).reshape(len(a),-1).T
 candidates=np.flatnonzero(eligible.ravel()&np.isfinite(yy[:,:len(fit)]).all(axis=1));rng.shuffle(candidates)
 sample=candidates[:min(16384,len(candidates))];ntrain=int(.8*len(sample));tr=sample[:ntrain];val=sample[ntrain:]
 center=float(np.mean(yy[tr,:len(fit)]));spread=float(np.std(yy[tr,:len(fit)]));yy=(yy-center)/spread;yy=np.nan_to_num(yy).astype(np.float32)
 config=dict(data_dim=1,h_dim=32,rnn_dim=32,rnn_cell='gru',num_layers=1)
 model=RNN(config).to(device);opt=torch.optim.Adam(model.parameters(),lr=.001)
 out=ROOT/'results'/args.event;out.mkdir(exist_ok=True);checkpoint=out/'gru_checkpoint.pt';log=[]
 if not checkpoint.exists() or args.retrain:
  best=float('inf')
  for epoch in range(args.epochs):
   model.train();order=rng.permutation(tr);total=0
   for k in range(0,len(order),512):
    batch=torch.tensor(yy[order[k:k+512],:len(fit),None],device=device);mean,lv=model(batch);lv=torch.clamp(lv,-8,8);loss=.5*(((batch-mean)**2)*torch.exp(-lv)+lv).mean();opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10);opt.step();total+=float(loss.item())
   model.eval()
   with torch.no_grad():
    batch=torch.tensor(yy[val,:len(fit),None],device=device);mean,lv=model(batch);lv=torch.clamp(lv,-8,8);v=float((.5*(((batch-mean)**2)*torch.exp(-lv)+lv).mean()).item())
   log.append(dict(epoch=epoch+1,validation_nll=v))
   if v<best:best=v;torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},checkpoint)
   if epoch%10==0:print(args.event,'GRU',epoch+1,'/',args.epochs,'device',device,'validation',round(v,5),flush=True)
  (out/'gru_training.json').write_text(json.dumps(log,indent=2))
 model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True));model.eval();scores=np.full(yy.shape,np.nan,dtype=np.float32)
 # A frozen network forecasts each next observation; no calibration/test gradients.
 with torch.no_grad():
  for k in range(0,len(candidates),512):
   ids=candidates[k:k+512];batch=torch.tensor(yy[ids,:,None],device=device);mean,lv=model(batch);sc=(mean-batch)/torch.exp(.5*torch.clamp(lv,-8,8));scores[ids]=sc[:,:,0].cpu().numpy()
 z=scores.T.reshape(a.shape);z[~np.isfinite(a)]=np.nan;z[:,~eligible]=np.nan
 methods,info=fit_calibrators(z[fit],z[cal],reg,weights);rows=[]
 for method,th in methods.items():
  for i,m in zip(test,alarm_metrics(z[test],th,reg,weights)):rows.append(dict(event=args.event,method=method,pair=c['pairs'][i],**m))
 (out/'gru_background.json').write_text(json.dumps(rows,indent=2));np.savez_compressed(out/'gru_event.npz',score=z[-1],threshold=methods['Regional-max'][reg])
 (out/'gru_summary.json').write_text(json.dumps(dict(device=str(device),parameters=model.num_parameters(),epochs=args.epochs,training_pixels=len(tr),validation_pixels=len(val),seed=271828,center=center,spread=spread,seconds=time.perf_counter()-start,config=config),indent=2))
 print(args.event,'GRU complete',flush=True)
if __name__=='__main__':main()
