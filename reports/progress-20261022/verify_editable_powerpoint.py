"""Audit the delivered OOXML package against the frozen epoch-100 evidence."""
from pathlib import Path
from io import BytesIO
import csv
import hashlib
import json
import posixpath
import re
import zipfile
from lxml import etree as ET
from pptx import Presentation

HERE=Path(__file__).resolve().parent
PPTX=HERE/'progress_report_20261022.pptx'
NS={'a':'http://schemas.openxmlformats.org/drawingml/2006/main',
    'p':'http://schemas.openxmlformats.org/presentationml/2006/main',
    'm':'http://schemas.openxmlformats.org/officeDocument/2006/math',
    'c':'http://schemas.openxmlformats.org/drawingml/2006/chart',
    'mc':'http://schemas.openxmlformats.org/markup-compatibility/2006',
    'r':'http://schemas.openxmlformats.org/package/2006/relationships'}
with (HERE/'evidence/comparison_epoch100_replay00_to50.tsv').open() as f:
    rows=list(csv.DictReader(f,delimiter='\t'))
assert len(rows)==55 and all(int(r['epoch'])==100 for r in rows)
lookup={r['name']:r for r in rows}
methods=['imle-rank','shape-fy','imle-nll','rnafold','mxfold2','truth']
replays=[0,5,10,20,50]
def name(m,probe,p):
    if m=='truth':return f'truth-fy-replay{p:02d}'
    if m in ('rnafold','mxfold2'):return f'{m}-{probe}-fy-replay{p:02d}'
    return f'{m}-{probe}-replay{p:02d}'

stats={'slides':0,'editable_tables':0,'office_math_expressions':0,
       'native_charts':0,'embedded_workbooks':0,'pictures':0,
       'epoch100_cells_verified':0,'epoch100_chart_points_verified':0}
with zipfile.ZipFile(PPTX) as z:
    assert z.testzip() is None
    names=set(z.namelist())
    for n in names:
        if n.endswith(('.xml','.rels')):ET.fromstring(z.read(n))
        if n.endswith('.rels'):
            root=ET.fromstring(z.read(n))
            base='' if n=='_rels/.rels' else n.split('/_rels/')[0]+'/'
            for r in root:
                if r.get('TargetMode')=='External':continue
                target=posixpath.normpath(base+r.get('Target'))
                if target.startswith('/'):target=target[1:]
                assert target in names,(n,target)
        if n.endswith('.xlsx'):
            stats['embedded_workbooks']+=1
            with zipfile.ZipFile(BytesIO(z.read(n))) as w:
                assert w.testzip() is None
                for part in w.namelist():
                    if part.endswith(('.xml','.rels')):ET.fromstring(w.read(part))
        if re.fullmatch(r'ppt/slides/slide\d+\.xml',n):
            root=ET.fromstring(z.read(n));stats['slides']+=1
            for key,tag in [('editable_tables','a:tbl'),('office_math_expressions','m:oMath'),('native_charts','c:chart'),('pictures','p:pic')]:
                stats[key]+=len(root.findall('.//'+tag,NS))
    assert stats['slides']==47
    assert stats['editable_tables']==18
    assert stats['native_charts']==stats['embedded_workbooks']==18
    assert stats['office_math_expressions']>=24
    assert stats['pictures']==0 and not any(n.startswith('ppt/media/') for n in names)
    # Numeric comparison tables: all methods × all five replay rates × two probes × two metrics.
    for page,probe,metric in [(18,'2A3','TestSetB_F1'),(19,'DMS','TestSetB_F1'),
        (20,'2A3','RNAgym_test_cluster_macro_spearman'),(21,'DMS','RNAgym_test_cluster_macro_spearman')]:
        table=ET.fromstring(z.read(f'ppt/slides/slide{page}.xml')).find('.//a:tbl',NS)
        for tr,m in zip(table.findall('a:tr',NS)[1:],methods):
            for tc,p in zip(tr.findall('a:tc',NS)[1:],replays):
                got=''.join(tc.itertext()).strip();want=f"{float(lookup[name(m,probe,p)][metric]):.4f}"
                assert got==want,(page,m,p,got,want)
                stats['epoch100_cells_verified']+=1
    # Native chart cache is checked against the independent final comparison TSV.
    for page,metric in [(22,'TestSetB_F1')]:
        root=ET.fromstring(z.read(f'ppt/slides/slide{page}.xml'))
        rels=ET.fromstring(z.read(f'ppt/slides/_rels/slide{page}.xml.rels'))
        links={r.get('Id'):r.get('Target') for r in rels}
        for ch,probe in zip(root.findall('.//c:chart',NS),['2A3','DMS']):
            rid=ch.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id')
            target=posixpath.normpath('ppt/slides/'+links[rid])
            chart=ET.fromstring(z.read(target))
            for se,m in zip(chart.findall('.//c:scatterChart/c:ser',NS),methods):
                xs=[float(t.text) for t in se.findall('c:xVal/c:numRef/c:numCache/c:pt/c:v',NS)]
                ys=[float(t.text) for t in se.findall('c:yVal/c:numRef/c:numCache/c:pt/c:v',NS)]
                assert xs==replays
                for p,y in zip(replays,ys):
                    want=float(lookup[name(m,probe,p)][metric]);assert abs(y-want)<1e-12
                    stats['epoch100_chart_points_verified']+=1
    # Math is structured OMML, including fractions, scripts and matrix column definitions.
    roots=[ET.fromstring(z.read(f'ppt/slides/slide{i}.xml')) for i in range(1,48)]
    for tag in ['m:f','m:sSub','m:sSup','m:mPr']:assert any(r.find('.//'+tag,NS) is not None for r in roots)
    for page,word in [(15,'合成反応性'),(30,'RNAgym')]:
        root=ET.fromstring(z.read(f'ppt/slides/slide{page}.xml'))
        assert word in ''.join(root.itertext())
assert len(Presentation(PPTX).slides)==47
stats['pptx_sha256']=hashlib.sha256(PPTX.read_bytes()).hexdigest()
stats['pptx_bytes']=PPTX.stat().st_size
stats['snapshot_JST']=json.loads((HERE/'evidence/report_manifest.json').read_text())['snapshot_JST']
stats['status']='passed'
(HERE/'editable_verification.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(stats,ensure_ascii=False,indent=2))
