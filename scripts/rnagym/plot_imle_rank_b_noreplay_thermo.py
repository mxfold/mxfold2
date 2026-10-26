#!/usr/bin/env python3
"""Plot baseline-initialized B-direction chemical-only IMLE-Rank progress."""
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--root',type=Path,required=True)
    p.add_argument('--partial',action='store_true'); a=p.parse_args()
    filename='partial_summary.tsv' if a.partial else 'summary.tsv'
    with (a.root/filename).open() as f: rows=list(csv.DictReader(f,delimiter='\t'))
    fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    for ax,field,title,ylabel in [
        (axes[0],'TestSetB_f1','TestSetB (430 RNAs)','Macro base-pair F1'),
        (axes[1],'validation_f1','Structural validation','Macro base-pair F1'),
        (axes[2],'heldout_rank_loss','Independent chemical validation','Mann-Whitney rank loss')]:
        valid=[r for r in rows if r[field]!='']
        ax.plot([int(r['epoch']) for r in valid],[float(r[field]) for r in valid],color='#d55e00',marker='o' if field=='TestSetB_f1' else None,markersize=3)
        if valid and int(valid[0]['epoch'])==0: ax.axhline(float(valid[0][field]),color='gray',linestyle=':')
        ax.set(title=title,xlabel='Epoch',ylabel=ylabel,xlim=(0,100)); ax.grid(alpha=.25)
    suffix='_partial' if a.partial else ''
    fig.savefig(a.root/f'learning_curves{suffix}.png',dpi=180)
    fig.savefig(a.root/f'learning_curves{suffix}.pdf'); plt.close(fig)


if __name__=='__main__': main()
