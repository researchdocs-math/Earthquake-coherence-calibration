"""Independent optical reference at radar-grid support; no tuning on labels."""
import json,argparse
from pathlib import Path
from collections import Counter
import numpy as np
from shapely.geometry import shape
from affine import Affine
from sklearn.metrics import average_precision_score,roc_auc_score,precision_score,recall_score,f1_score
from data_utils import ROOT

def reference_grid(features,tr,gridshape,dy=0,dx=0,include_possible=False):
    pos=np.zeros(gridshape,int);neg=pos.copy();unk=pos.copy()
    for f in features:
        g=shape(f['geometry']);pt=g if g.geom_type=='Point' else g.representative_point();xx,yy=(~tr)*(pt.x,pt.y);r=int(np.floor(yy))+dy;c=int(np.floor(xx))+dx
        if 0<=r<gridshape[0] and 0<=c<gridshape[1]:
            label=f['properties'].get('damage_gra','')
            if label in ['Destroyed','Damaged'] or (include_possible and label=='Possibly damaged'):pos[r,c]+=1
            elif label=='No visible damage':neg[r,c]+=1
            else:unk[r,c]+=1
    lab=np.full(gridshape,-1,np.int8);lab[(neg>0)&(unk==0)&(pos==0)]=0;lab[pos>0]=1
    return lab

def metrics(y,s,pred):
    return dict(positive=int(y.sum()),negative=int((y==0).sum()),prevalence=float(y.mean()),average_precision=float(average_precision_score(y,s)),roc_auc=float(roc_auc_score(y,s)),precision=float(precision_score(y,pred,zero_division=0)),recall=float(recall_score(y,pred,zero_division=0)),f1=float(f1_score(y,pred,zero_division=0)),flagged=int(pred.sum()))
def main():
    out=ROOT/'results/turkey';s=json.loads((out/'summary.json').read_text());m=np.load(out/'map_results.npz');tr=Affine(*s['transform']);z=m['event_score'];eligible=m['eligible'];regions=m['regions'];rows=[]
    for version,folder in [('initial','copernicus'),('monitoring','copernicus_monit')]:
        f=next((ROOT/'data'/folder).rglob('*builtUp*.json'));features=json.loads(f.read_text())['features']
        for include_possible in [False,True]:
            for dy,dx in [(0,0),(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                labels=reference_grid(features,tr,z.shape,dy,dx,include_possible);valid=(labels>=0)&eligible&np.isfinite(z)
                if not include_possible and dx==0 and dy==0:np.savez_compressed(out/(version+'_labels.npz'),labels=labels,valid=valid)
                for method in ['Pooled','Global-map','Bonferroni','Regional-max','Unscaled-max','Repaired-max']:
                    th=np.array(s['thresholds'][method])[regions];margin=z-th;pred=z>th
                    # Regionally standardized continuous score is meaningful for the scaled rules.
                    continuous=(z-m['a'][regions])/m['b'][regions] if method in ['Regional-max','Repaired-max'] else z
                    met=metrics(labels[valid],continuous[valid],pred[valid]);rows.append(dict(version=version,include_possible=include_possible,dy=dy,dx=dx,method=method,reference_pixels=int((labels>=0).sum()),assessed_pixels=int(valid.sum()),**met))
    (out/'reference_metrics.json').write_text(json.dumps(rows,indent=2))
    print([x for x in rows if x['version']=='monitoring' and not x['include_possible'] and x['dx']==x['dy']==0],flush=True)
if __name__=='__main__':main()
