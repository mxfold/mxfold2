import json,math,subprocess,time,sys
from pathlib import Path
import numpy as np
R=Path(sys.argv[1]); repo=Path(sys.argv[2]); binary=R/'src/contrafold'
data_root=repo/'qsub_logs/rnagym/rnagym-full-coverage-20261003-01/data'
mods=['2A3','DMS','1M7','NMIA','CMCT']
allowed={'DMS':'AC','CMCT':'GU'}
def load_packed(folder):
 import csv
 seqs=list(csv.DictReader((folder/'sequences.tsv').open(),delimiter='\t'))
 return seqs,{n:np.load(folder/(n+'.npy'),mmap_mode='r') for n in ['values','modifiers','sequence_offsets','observation_offsets']}
def pick_profiles(folder,probe,n=4):
 seqs,a=load_packed(folder); picks=[]
 for k,item in enumerate(seqs):
  s=item['sequence']
  if item['split']!='train' or not 35<=len(s)<=200 or set(s)-set('ACGU'):continue
  for o in range(int(a['sequence_offsets'][k]),int(a['sequence_offsets'][k+1])):
   if mods[int(a['modifiers'][o])]!=probe:continue
   l,r=map(int,a['observation_offsets'][o:o+2]);v=np.array(a['values'][l:r],float)
   mask=np.isfinite(v)&(v>-900)&np.array([b in allowed.get(probe,'ACGU') for b in s])
   if mask.sum()<8:continue
   picks.append((k,o,s,v,mask));break
  if len(picks)==n:return picks
 raise ValueError((folder,probe,len(picks)))
def write_profile(folder,name,s,v,mask,nd,channel,transform=True):
 folder.mkdir(parents=True,exist_ok=True);path=folder/name
 out=[]
 for i,b in enumerate(s):
  vals=[-1.]*nd
  if mask[i]:vals[channel]=float(np.logaddexp(0,v[i])) if transform else float(v[i])
  out.append(f"{i+1} {b} e{nd} "+' '.join(format(x,'.9g') for x in vals))
 path.write_text('\n'.join(out)+'\n');return path
a=[]
list_a=repo/'qsub_logs/rnagym/rnagym-full-coverage-20261003-01/TrainSetA.lst'
for name in list_a.read_text().splitlines():
 path=Path(name.split()[0]);rows=[x.split() for x in path.read_text().splitlines() if x and not x.startswith('#')]
 s=''.join(x[1] for x in rows).upper().replace('T','U');p=[int(x[2]) for x in rows]
 if not 35<=len(s)<=80 or set(s)-set('ACGU'):continue
 if any(j>i+1 and (j-(i+1)<=3 or s[i]+s[j-1] not in ['AU','UA','CG','GC','GU','UG']) for i,j in enumerate(p)):continue
 a.append(path)
 if len(a)==4:break
(R/'a.lst').write_text('\n'.join(map(str,a))+'\n')
results=[]
def run(label,args,timeout=90):
 d=R/label;d.mkdir(exist_ok=True);t=time.monotonic()
 proc=subprocess.run([str(binary),*map(str,args)],cwd=d,capture_output=True,text=True,timeout=timeout)
 (d/'stdout.log').write_text(proc.stdout);(d/'stderr.log').write_text(proc.stderr)
 final=d/'optimize.params.final';vals={}
 if final.exists():
  for l in final.read_text().splitlines():
   k,v=l.split();vals[k]=float(v)
 row={'label':label,'args':list(map(str,args)),'seconds':time.monotonic()-t,'returncode':proc.returncode,'params':len(vals),'finite':bool(vals) and all(math.isfinite(v) for v in vals.values()),'last_stderr':proc.stderr[-500:]}
 results.append(row);print(json.dumps(row),flush=True)
 return final,vals
base,bv=run('pretrain_A',['train','--numdatasources','1','--regularize','1','--maxiter','3','--examplefile',R/'a.lst'])
for probe in ['2A3','DMS']:
 for kind,folder in [('synth',repo/'qsub_logs/rnagym/trainsetb-synthetic-20261020/data'/probe),('real',data_root)]:
  picks=pick_profiles(folder,probe)
  paths=[write_profile(R/'profiles'/f'{kind}_{probe}',str(k)+'.bpseq',s,v,mask,1,0) for k,o,s,v,mask in picks]
  lst=R/(kind+'_'+probe+'.lst');lst.write_text('\n'.join(map(str,paths))+'\n')
  f,fv=run(kind+'_'+probe,['train','--numdatasources','1','--regularize','1','--maxiter','3','--hyperparam_data','0.5','--initweights',base,'--examplefile',lst])
  deltas={k:abs(fv[k]-bv[k]) for k in bv} if fv else {}
  results[-1]['changed_structure_params']=sum(v>1e-9 for k,v in deltas.items() if not k.startswith('log_score_evidence'))
  results[-1]['changed_emission_params']=sum(v>1e-9 for k,v in deltas.items() if k.startswith('log_score_evidence'))
  results[-1]['source_profiles']=[{'sequence_index':k,'observation_index':o,'length':len(s),'valid_bases':int(mask.sum())} for k,o,s,v,mask in picks]
  print(json.dumps({'label':results[-1]['label'],'changed_structure_params':results[-1]['changed_structure_params'],'changed_emission_params':results[-1]['changed_emission_params']}),flush=True)
(R/'smoke_results.json').write_text(json.dumps({'upstream_commit':'87b9aac55cee14fd562049d08f7b92d3131f10ce','transform':'Observed values only: softplus(r)=log(1+exp(r)); missing=-1','A_examples':list(map(str,a)),'results':results},indent=2)+'\n')

