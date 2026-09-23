"""Map-level calibration of regional exceedance area. MIT License."""
import numpy as np
from scipy.special import logit

def corrected_quantile(x,alpha=0.1,corruption=0):
    x=np.asarray(x);n=len(x);k=int(np.ceil((n+1)*(1-alpha)-1e-12))+corruption
    return np.inf if k>n else float(np.partition(x,k-1)[k-1])

def weighted_quantile(x,w,q=.95):
    good=np.isfinite(x)&np.isfinite(w)&(w>0)
    if not good.any():return np.nan
    a=np.asarray(x)[good];b=np.asarray(w)[good];order=np.argsort(a);a=a[order];b=b[order]
    return float(a[min(np.searchsorted(np.cumsum(b),q*b.sum(),side='left'),len(a)-1)])

def coherence_transform(x):
    return logit(np.clip(x,1/255,254/255)).astype(np.float32)

def fit_scores(train):
    y=coherence_transform(train);y[~np.isfinite(train)]=np.nan
    mu=np.nanmedian(y,axis=0);scale=np.maximum(1.4826*np.nanmedian(np.abs(y-mu),axis=0),.15)
    return mu,scale

def score(x,mu,scale):return (mu-coherence_transform(x))/scale

def region_grid(shape,side=2):
    h,w=shape;yy,xx=np.indices(shape)
    return np.minimum(yy*side//h,side-1)*side+np.minimum(xx*side//w,side-1)

def region_quantiles(scores,region,weights,rho=.05):
    ans=[]
    for z in scores:
        ans.append([weighted_quantile(z[region==r],weights[region==r],1-rho) for r in range(int(region.max())+1)])
    return np.asarray(ans)

def fit_calibrators(train_z,cal_z,region,weights,rho=.05,alpha=.1):
    qfit=region_quantiles(train_z,region,weights,rho);qcal=region_quantiles(cal_z,region,weights,rho)
    a=np.nanmedian(qfit,axis=0);b=np.maximum(1.4826*np.nanmedian(np.abs(qfit-a),axis=0),.25)
    u=np.nanmax((qcal-a)/b,axis=1);q=corrected_quantile(u,alpha)
    globalq=np.asarray([weighted_quantile(z,weights,1-rho) for z in cal_z])
    pooled=weighted_quantile(cal_z.ravel(),np.broadcast_to(weights,cal_z.shape).ravel(),1-rho)
    regional_pool=np.asarray([weighted_quantile(cal_z[:,region==r].ravel(),np.broadcast_to(weights[region==r],cal_z[:,region==r].shape).ravel(),1-rho) for r in range(len(a))])
    methods={
      'Gaussian':np.full(len(a),1.6448536269514722),
      'Pooled':np.full(len(a),pooled),
      'Regional-pool':regional_pool,
      'Global-map':np.full(len(a),corrected_quantile(globalq,alpha)),
      'Bonferroni':np.asarray([corrected_quantile(qcal[:,r],alpha/len(a)) for r in range(len(a))]),
      'Regional-max':a+b*q,
      'Unscaled-max':np.full(len(a),corrected_quantile(np.nanmax(qcal,axis=1),alpha)),
      'Repaired-max':a+b*corrected_quantile(u,alpha,2)}
    return methods,dict(a=a,b=b,u=u,q=q,qcal=qcal,globalq=globalq)

def alarm_metrics(z,thresholds,region,weights,rho=.05):
    out=[]
    for s in z:
        valid=np.isfinite(s)&(weights>0);alarm=valid&(s>thresholds[region]);fr=[]
        for r in range(len(thresholds)):
            idx=valid&(region==r);den=weights[idx].sum();fr.append(float(weights[idx&alarm].sum()/den) if den>0 else np.nan)
        out.append(dict(area=float(weights[alarm].sum()/weights[valid].sum()),max_region_area=float(np.nanmax(fr)),any_exceed=int(np.nanmax(fr)>rho+1e-9),regions=fr))
    return out
