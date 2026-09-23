"""Verify the distributed satellite crops and public-reference inputs."""
from pathlib import Path
import hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def main():
 config=json.loads((ROOT/'config.json').read_text());n=0
 for event,c in config.items():
  for pair in c['pairs']:
   path=ROOT/'data'/event/(pair+'.npz');m=json.loads(path.with_suffix('.json').read_text())
   assert hashlib.sha256(path.read_bytes()).hexdigest()==m['sha256_crop'],str(path)
   a=np.load(path)['coherence'];assert a.dtype==np.uint8 and list(a.shape)==m['crop_shape'];n+=1
 extra=ROOT/'data/reference_checksums.json'
 if extra.exists():
  for name,sha in json.loads(extra.read_text()).items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha,name
 print(f'Validated {n} satellite crops and optical-reference checksums.')
if __name__=='__main__':main()
