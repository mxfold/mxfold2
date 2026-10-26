"""Build figures and TeX tables only from the frozen report evidence."""
from pathlib import Path
import csv
import json
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
E=HERE/'evidence'; F=HERE/'figures'; F.mkdir(exist_ok=True)
def read(p):
    with p.open() as f:return list(csv.DictReader(f,delimiter='\t'))
rows=read(E/'synthetic_all_curves.tsv')
index={(r['name'],int(r['epoch'])):r for r in rows}
manifest=json.loads((E/'report_manifest.json').read_text())
zero_epoch=manifest.get('zero_replay_comparison_epoch',manifest['zero_replay_common_latest_epoch'])
methods=[('imle-rank','IMLE-Rank'),('shape-fy','SHAPE-fy'),('imle-nll','IMLE-NLL'),
         ('rnafold','RNAfold pseudo'),('mxfold2','MXfold2 pseudo'),('truth','TrainSetB truth')]
colors={'imle-rank':'#007f87','shape-fy':'#386bbe','imle-nll':'#db7629',
        'rnafold':'#9b91ad','mxfold2':'#58616b','truth':'#aa4664'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
 'axes.spines.right':False,'axes.labelcolor':'#24394a','text.color':'#24394a',
 'axes.titleweight':'bold','axes.titlesize':12,'axes.labelsize':11,
 'figure.facecolor':'white','savefig.facecolor':'white','pdf.fonttype':42,
 'axes.prop_cycle':plt.cycler(color=list(colors.values()))})
def name(method,probe,pct):
    if method=='truth':return f'truth-fy-replay{pct:02d}'
    if method in ('rnafold','mxfold2'):return f'{method}-{probe}-fy-replay{pct:02d}'
    return f'{method}-{probe}-replay{pct:02d}'
def row(method,probe,pct,epoch=100):return index[(name(method,probe,pct),epoch)]
def save(fig,stem):
    fig.savefig(F/(stem+'.pdf'),bbox_inches='tight',pad_inches=.06)
    fig.savefig(F/(stem+'.png'),dpi=165,bbox_inches='tight',pad_inches=.06)
    plt.close(fig)

real={k:read(E/f'real_{k}/evaluation_curve.tsv') for k in ('rank','fy','nll')}
fig,axs=plt.subplots(1,2,figsize=(11.4,3.45),layout='constrained')
for key,m,label in [('rank','imle-rank','IMLE-Rank'),('fy','shape-fy','SHAPE-fy'),('nll','imle-nll','IMLE-NLL')]:
    x=[int(r['epoch']) for r in real[key]]
    for ax,col in zip(axs,['TestSetB_F1','RNAgym_test_cluster_macro_spearman']):
        ax.plot(x,[float(r[col]) for r in real[key]],'-o',label=label,color=colors[m],lw=2,ms=5)
for ax,col,title in zip(axs,['TestSetB_F1','RNAgym_test_cluster_macro_spearman'],['Base-pair accuracy','Chemical consistency']):
    ax.axhline(float(real['rank'][0][col]),color='#8c9aa5',ls='--',lw=1)
    ax.set(xlabel='Additional epochs',title=title,xticks=range(4))
    ax.grid(axis='y',alpha=.18)
axs[0].set_ylabel('TestSetB mean F1');axs[1].set_ylabel('4-probe macro Spearman')
axs[0].set_ylim(.525,.625);axs[1].set_ylim(.474,.510)
axs[1].legend(loc='lower right',fontsize=9,frameon=False)
save(fig,'real_curves')

fig,axs=plt.subplots(1,2,figsize=(11.4,3.45),layout='constrained')
probes=['2A3','DMS','1M7','NMIA'];x=np.arange(4)
for j,(key,m,label) in enumerate([('baseline','truth','Baseline'),('rank','imle-rank','IMLE-Rank'),('fy','shape-fy','SHAPE-fy'),('nll','imle-nll','IMLE-NLL')]):
    o=json.loads((E/('real_rank/evaluation-epoch-0.json' if key=='baseline' else f'real_{key}/evaluation-epoch-3.json')).read_text())
    y=[o['chemical_probe_panel']['modifiers'][p]['cluster_macro_spearman'] for p in probes]
    axs[0].plot(x,y,'-o',label=label,color='#89939d' if key=='baseline' else colors[m],lw=1.6)
    if key!='baseline':
        b=json.loads((E/'real_rank/evaluation-epoch-0.json').read_text())
        dy=[v-b['chemical_probe_panel']['modifiers'][p]['cluster_macro_spearman'] for p,v in zip(probes,y)]
        axs[1].bar(x+(j-2)*.23,dy,.23,label=label,color=colors[m])
for ax in axs:
    ax.set_xticks(x,probes);ax.grid(axis='y',alpha=.18)
axs[0].set(ylabel='Spearman',title='Epoch 3: each assay');axs[0].legend(fontsize=8,frameon=False,loc='lower right')
axs[1].axhline(0,c='#a5acb3',lw=1);axs[1].set(ylabel='Change from baseline',title='NLL macro average includes NMIA (n=5)')
save(fig,'real_probe_details')

replay_percentages=[0,5,10,20,50]
for metric,stem,ylabel,ylim in [('TestSetB_F1','synthetic_replay_f1','TestSetB mean F1',(.25,.68)),
 ('RNAgym_test_cluster_macro_spearman','synthetic_replay_chem','4-probe macro Spearman',(.34,.50))]:
    fig,axs=plt.subplots(1,2,figsize=(11.4,3.6),layout='constrained')
    for ax,probe in zip(axs,['2A3','DMS']):
        for m,l in methods:
            ax.plot(replay_percentages,[float(row(m,probe,p)[metric]) for p in replay_percentages],'-o',color=colors[m],label=l,lw=1.8,ms=4)
        ax.axhline(.6092510770123509 if metric=='TestSetB_F1' else .4792824508976543,color='#89939d',ls='--',lw=1)
        ax.set(title=probe+' training',xlabel='TrainSetA replay (%)',ylabel=ylabel,xticks=replay_percentages,ylim=ylim)
        ax.grid(axis='y',alpha=.18)
    fig.legend(*axs[0].get_legend_handles_labels(),loc='outside lower center',ncol=3,fontsize=9,frameon=False)
    save(fig,stem)

fig,axs=plt.subplots(1,2,figsize=(11.4,3.6),layout='constrained')
for ax,probe in zip(axs,['2A3','DMS']):
    for m,l in methods:
        rr=[row(m,probe,20,ep) for ep in range(0,101,10)]
        ax.plot(range(0,101,10),[float(r['TestSetB_F1']) for r in rr],'-',color=colors[m],label=l,lw=1.8)
    ax.axhline(.6092510770123509,color='#89939d',ls='--',lw=1)
    ax.set(title=probe+' / replay 20%',xlabel='Additional epochs',ylabel='TestSetB mean F1',ylim=(.45,.67))
    ax.grid(axis='y',alpha=.18)
fig.legend(*axs[0].get_legend_handles_labels(),loc='outside lower center',ncol=3,fontsize=9,frameon=False)
save(fig,'synthetic_learning20')

fig,axs=plt.subplots(1,2,figsize=(11.4,3.8),layout='constrained')
for ax,probe in zip(axs,['2A3','DMS']):
    for m,l in methods:
        xx=[float(row(m,probe,p)['TestSetB_F1']) for p in replay_percentages]
        yy=[float(row(m,probe,p)['RNAgym_test_cluster_macro_spearman']) for p in replay_percentages]
        ax.scatter(xx,yy,c=colors[m],s=[20,32,45,64,90],label=l,alpha=.85,edgecolors='white',linewidths=.5)
    ax.axvline(.6092510770123509,color='#89939d',ls='--',lw=1)
    ax.axhline(.4792824508976543,color='#89939d',ls='--',lw=1)
    ax.set(title=probe+' training',xlabel='TestSetB mean F1',ylabel='4-probe macro Spearman')
    ax.grid(alpha=.12)
axs[0].set(xlim=(.56,.67),ylim=(.34,.50));axs[1].set(xlim=(.27,.67),ylim=(.34,.50))
fig.legend(*axs[0].get_legend_handles_labels(),loc='outside lower center',ncol=3,fontsize=9,frameon=False)
save(fig,'synthetic_tradeoff')

fig,axs=plt.subplots(1,2,figsize=(11.4,3.5),layout='constrained')
for p in replay_percentages:
    rr=[row('imle-nll','DMS',p,ep) for ep in range(0,101,10)]
    axs[0].plot(range(0,101,10),[float(r['TestSetB_F1']) for r in rr],'-o',ms=3,label=f'Replay {p}%')
axs[0].axhline(.6092510770123509,c='#89939d',ls='--',lw=1)
axs[0].set(xlabel='Additional epochs',ylabel='TestSetB mean F1',title='DMS / IMLE-NLL')
axs[0].legend(fontsize=9,frameon=False);axs[0].grid(axis='y',alpha=.18)
xx=[];yy=[]
for p in replay_percentages:
    group='synth_zero' if p==0 else 'synth_nll'
    dest=E/f'{group}/conditions/imle-nll-DMS-replay{p:02d}/epoch-100-training.json'
    obj=json.loads(dest.read_text());xx.append(obj['nll_raw_mean']);yy.append(float(row('imle-nll','DMS',p)['TestSetB_F1']))
    axs[1].annotate(f'{p}%',(xx[-1],yy[-1]),xytext=(5,3),textcoords='offset points',fontsize=10)
axs[1].scatter(xx,yy,s=65,c=colors['imle-nll'])
axs[1].set(xlabel='Epoch-100 training NLL (lower is better)',ylabel='TestSetB mean F1',title='Lower NLL does not imply higher F1')
axs[1].set_xlim(.225,.30);axs[1].grid(alpha=.18)
save(fig,'dms_nll_diagnosis')

fig,axs=plt.subplots(1,2,figsize=(11.4,3.5),layout='constrained')
for ax,probe in zip(axs,['2A3','DMS']):
    for m,l in methods[:3]:
        for p,ls in [(0,'-'),(20,'--')]:
            rr=[row(m,probe,p,ep) for ep in range(0,zero_epoch+1,10)]
            ax.plot(range(0,zero_epoch+1,10),[float(r['TestSetB_F1']) for r in rr],ls,color=colors[m],label=l+f' / {p}%',lw=1.8)
    ax.axhline(.6092510770123509,c='#89939d',ls=':',lw=1)
    ax.set(title=probe+' training',xlabel='Additional epochs',ylabel='TestSetB mean F1')
    ax.grid(axis='y',alpha=.18)
fig.legend(*axs[0].get_legend_handles_labels(),loc='outside lower center',ncol=3,fontsize=8,frameon=False)
save(fig,'zero_replay_final')

# Appendix: all six methods at replay 0%, with both target metrics.
for probe in ['2A3','DMS']:
    fig,axs=plt.subplots(1,2,figsize=(11.4,3.6),layout='constrained')
    for m,label in methods:
        rr=[row(m,probe,0,ep) for ep in range(0,zero_epoch+1,10)]
        for ax,metric in zip(axs,['TestSetB_F1','RNAgym_test_cluster_macro_spearman']):
            ax.plot(range(0,zero_epoch+1,10),[float(r[metric]) for r in rr],color=colors[m],label=label,lw=1.8)
    for ax,base in zip(axs,[.6092510770123509,.4792824508976543]):
        ax.axhline(base,color='#89939d',ls='--',lw=1)
        ax.set_xlabel('Additional epochs');ax.set_xticks([0,20,40,60,80,100]);ax.grid(axis='y',alpha=.18)
    axs[0].set(title=probe+' / replay 0%',ylabel='TestSetB mean F1')
    axs[1].set(title=probe+' / replay 0%',ylabel='4-probe macro Spearman')
    fig.legend(*axs[0].get_legend_handles_labels(),loc='outside lower center',ncol=3,fontsize=9,frameon=False)
    save(fig,f'zero_all_methods_{probe}')

# TeX numeric tables: no 2A3/DMS training-condition averaging.
def make_table(probe,metric,epoch,pcts,filename):
    values=np.array([[float(row(m,probe,p,epoch)[metric]) for p in pcts] for m,_ in methods])
    text=['\\begin{tabular}{l'+('r'*len(pcts))+'}','\\toprule',
          '手法 & '+' & '.join(f'{p}\\%' for p in pcts)+r' \\',r'\midrule']
    for i,(m,l) in enumerate(methods):
        cells=[]
        for j,v in enumerate(values[i]):
            cell=f'{v:.4f}'
            if v==values[:,j].max():cell=r'\textbf{\color{Teal}'+cell+'}'
            if metric=='TestSetB_F1' and v<.55:cell=r'\textcolor{Orange}{'+cell+'}'
            cells.append(cell)
        text.append(l+' & '+' & '.join(cells)+r' \\')
    text+=['\\bottomrule','\\end{tabular}']
    (HERE/filename).write_text('\n'.join(text))
for probe in ['2A3','DMS']:
    make_table(probe,'TestSetB_F1',100,replay_percentages,f'table_{probe}_f1.tex')
    make_table(probe,'RNAgym_test_cluster_macro_spearman',100,replay_percentages,f'table_{probe}_chem.tex')
    make_table(probe,'TestSetB_F1',zero_epoch,[0,5,10,20,50],f'table_{probe}_zero_f1.tex')
    make_table(probe,'RNAgym_test_cluster_macro_spearman',zero_epoch,[0,5,10,20,50],f'table_{probe}_zero_chem.tex')

baseline=json.loads((E/'real_rank/evaluation-epoch-0.json').read_text())
details=[(m,label,probe) for probe in ['2A3','DMS'] for m,label in methods[:5]]+[('truth','TrainSetB truth','共通')]
struct=['\\begin{tabular}{llrrrr}',r'\toprule',r'手法 & 訓練 & PPV & SEN & Test F1 & Train F1 \\',r'\midrule']
b=baseline['structure']
struct.append('baseline & --- & '+' & '.join(f'{v:.4f}' for v in [b['TestSetB']['mean_PPV'],b['TestSetB']['mean_SEN'],b['TestSetB']['mean_F1'],b['TrainSetB']['mean_F1']])+r' \\'+ '\n'+r'\midrule')
chem=['\\begin{tabular}{llrrrrr}',r'\toprule',r'手法 & 訓練 & 2A3 & DMS & 1M7 & NMIA & macro \\',r'\midrule']
basechem=baseline['chemical_probe_panel']
chem.append('baseline & --- & '+' & '.join(f'{v:.4f}' for v in [basechem['modifiers'][p]['cluster_macro_spearman'] for p in ['2A3','DMS','1M7','NMIA']]+[basechem['mean_cluster_macro_spearman']])+r' \\'+ '\n'+r'\midrule')
for m,label,probe in details:
    rr=row(m,'2A3' if probe=='共通' else probe,0,100)
    struct.append(label+' & '+probe+' & '+' & '.join(f'{float(rr[k]):.4f}' for k in ['TestSetB_PPV','TestSetB_SEN','TestSetB_F1','TrainSetB_training_F1'])+r' \\')
    chem.append(label+' & '+probe+' & '+' & '.join(f'{float(rr[k]):.4f}' for k in ['RNAgym_2A3_spearman','RNAgym_DMS_spearman','RNAgym_1M7_spearman','RNAgym_NMIA_spearman','RNAgym_test_cluster_macro_spearman'])+r' \\')
for filename,lines in [('table_zero_structure_details.tex',struct),('table_zero_probe_details.tex',chem)]:
    (HERE/filename).write_text('\n'.join(lines+[r'\bottomrule',r'\end{tabular}']))

generated=f'\\newcommand{{\\snapshot}}{{{manifest["snapshot_JST"][:16].replace("T"," ")} JST}}\n'
generated+=f'\\newcommand{{\\zeroepoch}}{{{zero_epoch}}}\n'
for method,tag in [('imle-rank','Rank'),('shape-fy','FY'),('imle-nll','NLL'),('truth','Truth')]:
    for probe,letter in [('2A3','A'),('DMS','D')]:
        for metric,suffix in [('TestSetB_F1','F'),('RNAgym_test_cluster_macro_spearman','R')]:
            value=float(row(method,probe,0,zero_epoch)[metric])
            generated+=f'\\newcommand{{\\zero{tag}{letter}{suffix}}}{{{value:.4f}}}\n'
(HERE/'generated.tex').write_text(generated)
print('Built figures and tables; replay-0 comparison epoch',zero_epoch)
