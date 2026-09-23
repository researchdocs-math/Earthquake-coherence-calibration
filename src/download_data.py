"""Download and crop public LiCSAR coherence; no event labels enter selection."""
import os,json,re,hashlib,time,argparse,io,tifffile
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor,as_completed
import requests,numpy as np,rasterio
from rasterio.io import MemoryFile
from rasterio.windows import from_bounds
ROOT=Path(__file__).resolve().parents[1]
BASE='https://gws-access.jasmin.ac.uk/public/nceo_geohazards/LiCSAR_products/'
def get(url):
 for attempt in range(3):
  try:
   r=requests.get(url,timeout=(15,90));r.raise_for_status();return r
  except requests.RequestException:
   if attempt==2:raise
   time.sleep(1+attempt)
def retrieve(name,cfg,pair):
 out=ROOT/'data'/name;out.mkdir(parents=True,exist_ok=True)
 dest=out/(pair+'.npz');meta=out/(pair+'.json')
 if dest.exists() and meta.exists():return json.loads(meta.read_text())
 track=str(int(cfg['frame'][:3]));directory=BASE+track+'/'+cfg['frame']+'/interferograms/'+pair
 # Migrated directory indexes have absolute links to the public object storage.
 listing=get(directory).text
 links=re.findall(r'href=[\"\']([^\"\']+\.geo\.cc\.tif)[\"\']',listing)
 url=links[0] if links else directory+'/'+pair+'.geo.cc.tif'
 if not url.startswith('http'):url=directory+'/'+url
 # Retrieve TIFF metadata and only compressed strips intersecting the crop.
 def ranged(a,b):
  for attempt in range(4):
   try:
    rr=requests.get(url,headers={'Range':f'bytes={a}-{b}'},timeout=(15,40));rr.raise_for_status()
    if rr.status_code==206:
     assert len(rr.content)==b-a+1
     return rr.content
    if rr.status_code==200:return rr.content[a:b+1]
   except (requests.RequestException,AssertionError):
    if attempt==3:raise
    time.sleep(1)
 header=ranged(0,65535)
 with tifffile.TiffFile(io.BytesIO(header)) as tf:
  page=tf.pages[0];offs=list(page.dataoffsets);sizes=list(page.databytecounts);rps=page.rowsperstrip
  tie=page.tags['ModelTiepointTag'].value;scale=page.tags['ModelPixelScaleTag'].value
  from affine import Affine
  transform=Affine(scale[0],0,tie[3],0,-scale[1],tie[4])
  win=from_bounds(*cfg['bounds'],transform).round_offsets().round_lengths()
  row0=max(0,int(win.row_off));row1=min(page.shape[0],int(win.row_off+win.height))
  inds=list(range(row0//rps,(row1+rps-1)//rps))
  begin=min(offs[i] for i in inds);end=max(offs[i]+sizes[i] for i in inds)
  raw=bytearray(max(65536,max(o+c for o,c in zip(offs,sizes))))
  raw[:len(header)]=header
  ranges=[]
  for a in range(begin,end,262144):
   b=min(a+262144,end)-1;part=ranged(a,b);raw[a:b+1]=part
   ranges.append({'start':a,'end':b,'sha256':hashlib.sha256(part).hexdigest()})
 with MemoryFile(bytes(raw)) as mem:
  with mem.open() as ds:
   arr=ds.read(1,window=win,boundless=True,fill_value=0)
   transform=ds.window_transform(win)
   m=dict(event=name,pair=pair,url=url,source_access='HTTP range; metadata and intersecting compressed strips only',header_sha256=hashlib.sha256(header).hexdigest(),ranges=ranges,source_shape=[ds.height,ds.width],source_dtype=str(ds.dtypes[0]),nodata=ds.nodata,crs=str(ds.crs),transform=list(transform)[:6],crop_shape=list(arr.shape),zero_fraction=float((arr==0).mean()))
 np.savez_compressed(dest,coherence=arr)
 m['sha256_crop']=hashlib.sha256(dest.read_bytes()).hexdigest();meta.write_text(json.dumps(m,indent=2));return m

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=6);args=ap.parse_args()
 config=json.loads((ROOT/'config.json').read_text());tasks=[]
 for name,cfg in config.items():
  tasks.extend((name,cfg,x) for x in cfg['pairs']);print(name,len(cfg['pairs'])-1,'pre-event',flush=True)
 errors=[];records=[]
 with ThreadPoolExecutor(args.workers) as ex:
  futures={ex.submit(retrieve,*task):task for task in tasks}
  for n,f in enumerate(as_completed(futures),1):
   task=futures[f]
   try:records.append(f.result())
   except Exception as e:errors.append({'event':task[0],'pair':task[2],'error':str(e)})
   if n%5==0 or errors:print(n,'/',len(tasks),'errors',len(errors),flush=True)
 (ROOT/'data/manifest.json').write_text(json.dumps(sorted(records,key=lambda x:(x['event'],x['pair'])),indent=2))
 (ROOT/'data/download_errors.json').write_text(json.dumps(errors,indent=2))
 if errors:raise SystemExit('Some downloads failed; rerun to resume.')
if __name__=='__main__':main()
