import csv,json,math,subprocess,time,sys
from pathlib import Path
import numpy as np
R=Path(sys.argv[1]);repo=Path(sys.argv[2]);binary=R/'src/contrafold'
mods=['2A3','DMS','1M7','NMIA','CMCT']; allowed={'DMS':'AC','CMCT':'GU'}
results=[]
def profiles(folder,probe,label,channel):
 seqs=list(csv.DictReader((folder/'sequences.tsv').open(),delimiter='\t'))
 a={n:np.load(folder/(n+'.npy'),mmap_mode='r') for n in ['values','modifiers','sequence_offsets','observation_offsets']}
 candidates=[]
 for k,item in enumerate(seqs):
  s=item['sequence']
  if item['split']=='train' and set(s)<=set('ACGU'):
   candidates.append((len(s),k,s))
 for length,k,s in sorted(candidates):
  for o in range(int(a['sequence_offsets'][k]),int(a['sequence_offsets'][k+1])):
   if mods[int(a['modifiers'][o])]!=probe:continue
   l,r=map(int,a['observation_offsets'][o:o+2]);v=np.array(a['values'][l:r],float)
   mask=np.isfinite(v)&(v>-900)&np.array([b in allowed.get(probe,'ACGU') for b in s])
   if mask.sum()<8:continue
   path=R/(label+'.bpseq');out=[]
   for i,b in enumerate(s):
    vals=[-1.]*5
    if mask[i]:vals[channel]=float(np.logaddexp(0,v[i]))
    out.append(f"{i+1} {b} e5 "+' '.join(format(x,'.9g') for x in vals))
   path.write_text('\n'.join(out)+'\n')
   return path,{'probe':probe,'sequence_index':k,'observation_index':o,'length':len(s),'valid':int(mask.sum())}
 raise ValueError(probe)
def run(label,args,initial=None):
 d=R/label;d.mkdir(exist_ok=True);t=time.monotonic()
 proc=subprocess.run([str(binary),*map(str,args)],cwd=d,capture_output=True,text=True,timeout=90)
 (d/'stdout.log').write_text(proc.stdout);(d/'stderr.log').write_text(proc.stderr)
 final=d/'optimize.params.final';vals={}
 if final.exists():vals={k:float(v) for k,v in map(str.split,final.read_text().splitlines())}
 row={'label':label,'args':list(map(str,args)),'seconds':time.monotonic()-t,'returncode':proc.returncode,'params':len(vals),'finite':bool(vals) and all(math.isfinite(v) for v in vals.values()),'last_stderr':proc.stderr[-250:]}
 if initial:
  iv={k:float(v) for k,v in map(str.split,initial.read_text().splitlines())}
  row['changed_structure_params']=sum(abs(v-iv[k])>1e-9 for k,v in vals.items() if not k.startswith('log_score_evidence'))
  row['changed_emission_params']=sum(abs(v-iv[k])>1e-9 for k,v in vals.items() if k.startswith('log_score_evidence'))
 results.append(row); print(json.dumps(row),flush=True); return final
base=run('pretrain_A_5',['train','--numdatasources',5,'--regularize',1,'--maxiter',3,'--examplefile',R/'a.lst'])
paths=[];selection=[]
for probe in ['2A3','DMS']:
 p,m=profiles(repo/'qsub_logs/rnagym/trainsetb-synthetic-20261020/data'/probe,probe,'synth5_'+probe,mods.index(probe));paths.append(p);selection.append(m)
lst=R/'synth5.lst';lst.write_text('\n'.join(map(str,paths))+'\n')
synth=run('synth5',['train','--numdatasources',5,'--regularize',1,'--maxiter',2,'--hyperparam_data',.5,'--initweights',base,'--examplefile',lst],base)
paths=[]
for probe in mods:
 p,m=profiles(repo/'qsub_logs/rnagym/rnagym-full-coverage-20261003-01/data',probe,'real5_'+probe,mods.index(probe));paths.append(p);selection.append(m)
lst=R/'real5.lst';lst.write_text('\n'.join(map(str,paths))+'\n')
run('real5',['train','--numdatasources',5,'--regularize',1,'--maxiter',2,'--hyperparam_data',.5,'--initweights',base,'--examplefile',lst],base)
run('synth5_then_real5',['train','--numdatasources',5,'--regularize',1,'--maxiter',2,'--hyperparam_data',.5,'--initweights',synth,'--examplefile',lst],synth)
(R/'smoke5_results.json').write_text(json.dumps({'upstream_commit':'87b9aac55cee14fd562049d08f7b92d3131f10ce','selection':selection,'results':results},indent=2)+'\n')

