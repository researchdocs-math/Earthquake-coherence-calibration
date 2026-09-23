"""Compare regenerated numerical summaries with immutable reference values."""
from pathlib import Path
import json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def compare(a,b,path=''):
 if isinstance(a,dict):
  assert set(a)==set(b),path
  for k in a:compare(a[k],b[k],path+'/'+k)
 elif isinstance(a,list):
  assert len(a)==len(b),path
  for i,(x,y) in enumerate(zip(a,b)):compare(x,y,path+'/'+str(i))
 elif isinstance(a,(float,int)) and not isinstance(a,bool):
  assert np.isclose(a,b,rtol=1e-5,atol=1e-6,equal_nan=True),(path,a,b)
 else:assert a==b,(path,a,b)
def main():
 ref=json.loads((ROOT/'expected_metrics.json').read_text())
 for f,v in ref.items():compare(v,json.loads((ROOT/f).read_text()),f)
 print('All reference numerical summaries reproduced within tolerance.')
if __name__=='__main__':main()
