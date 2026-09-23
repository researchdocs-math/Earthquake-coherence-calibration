"""Load georeferenced coherence crops and enforce acquisition-separated splits."""
import json
from pathlib import Path
import numpy as np
from affine import Affine
from rasterio.warp import reproject,Resampling
ROOT=Path(__file__).resolve().parents[1]
def load_event(name):
 c=json.loads((ROOT/'config.json').read_text())[name];pairs=c['pairs'];base=ROOT/'data'/name
 missing=[x for x in pairs if not (base/(x+'.npz')).exists()]
 if missing:raise FileNotFoundError(f'{name}: {len(missing)} missing crops')
 ref=json.loads((base/(c['pair']+'.json')).read_text());tr=Affine(*ref['transform']);shape=tuple(ref['crop_shape']);arrays=[];reprojected=0
 for pair in pairs:
  a=np.load(base/(pair+'.npz'))['coherence'];m=json.loads((base/(pair+'.json')).read_text())
  if not np.allclose(m['transform'],ref['transform'],atol=1e-10,rtol=0) or a.shape!=shape:
   out=np.zeros(shape,dtype=np.uint8);reproject(a,out,src_transform=Affine(*m['transform']),src_crs='EPSG:4326',src_nodata=0,dst_transform=tr,dst_crs='EPSG:4326',dst_nodata=0,resampling=Resampling.nearest);a=out;reprojected+=1
  arrays.append(a)
 a=np.stack(arrays).astype(np.float32)/255;a[a==0]=np.nan
 nf=min(24,len(pairs[:-1])//4);fit=np.arange(nf);fit_end=max(pairs[i][9:] for i in fit)
 remaining=[i for i in range(nf,len(pairs)-1) if pairs[i][:8]>fit_end]
 nc=39 if len(remaining)>=59 else 29
 cal=np.array(remaining[:nc]);cal_end=max(pairs[i][9:] for i in cal)
 test=np.array([i for i in remaining[nc:] if pairs[i][:8]>cal_end])
 fitmedian=np.nanmedian(a[fit],axis=0)
 eligible=(np.isfinite(a[fit]).mean(axis=0)>=.9)&(fitmedian>=.35)
 yy=tr.f+(np.arange(shape[0])+.5)*tr.e;weights=np.broadcast_to(np.cos(np.deg2rad(yy))[:,None],shape).copy();weights[~eligible]=0
 return a,c,ref,fit,cal,test,eligible,weights,dict(reprojected=reprojected)
