"""Build a native, editable 47-slide PPTX from the frozen report.

Dependencies: python-pptx, Pillow, lxml, latex2mathml, mathml2omml.
All text, equations, tables, diagrams and charts are native Office objects.
Chart workbooks contain the plotted values; no slide or chart raster images.
"""
from pathlib import Path
import copy
import csv
import json
import math
import re
import zipfile
from io import BytesIO

from PIL import ImageFont
from lxml import etree as ET
from pptx import Presentation
from pptx.chart.data import XyChartData, BubbleChartData, CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_MARKER_STYLE, XL_TICK_MARK
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.xmlchemy import OxmlElement
from pptx.util import Inches, Pt
from latex2mathml.converter import convert as latex_to_mathml
import mathml2omml

HERE = Path(__file__).resolve().parent
E = HERE / 'evidence'
OUT = HERE / 'progress_report_20261022.pptx'
NS = {'a':'http://schemas.openxmlformats.org/drawingml/2006/main',
      'p':'http://schemas.openxmlformats.org/presentationml/2006/main',
      'm':'http://schemas.openxmlformats.org/officeDocument/2006/math',
      'a14':'http://schemas.microsoft.com/office/drawing/2010/main',
      'mc':'http://schemas.openxmlformats.org/markup-compatibility/2006',
      'c':'http://schemas.openxmlformats.org/drawingml/2006/chart'}
NAVY, TEAL, BLUE, ORANGE, MUTED, PALE = '173449','007F87','386BBE','C56A22','697C88','EEF5F6'
FONT = 'Yu Gothic'
FONT_FILE = Path('/home/7/up04167/.local/share/fonts/UDEVGothic-Regular.ttf')
if not FONT_FILE.exists():
    FONT_FILE = Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
COLORS = {'imle-rank':TEAL, 'shape-fy':BLUE, 'imle-nll':'DB7629',
          'rnafold':'9B91AD', 'mxfold2':'58616B', 'truth':'AA4664', 'baseline':'89939D'}
METHODS = [('imle-rank','IMLE-Rank'),('shape-fy','SHAPE-fy'),('imle-nll','IMLE-NLL'),
           ('rnafold','RNAfold pseudo'),('mxfold2','MXfold2 pseudo'),('truth','TrainSetB truth')]
REPLAYS = [0,5,10,20,50]
F1, RHO = 'TestSetB_F1', 'RNAgym_test_cluster_macro_spearman'
BASE = {F1: .6092510770123509, RHO: .4792824508976543}

def rgb(s): return RGBColor.from_string(s)
def el(tag, **attrs):
    prefix, local = tag.split(':')
    e = ET.Element('{%s}%s' % (NS[prefix], local))
    for k,v in attrs.items():
        if ':' in k:
            pre, name = k.split(':'); k = '{%s}%s' % (NS[pre], name)
        e.set(k, str(v))
    return e

generated = dict(re.findall(r'\\newcommand\{\\(\w+)\}\{([^}]*)\}', (HERE/'generated.tex').read_text()))
def expand(s):
    for k,v in generated.items(): s = re.sub(r'\\'+k+r'(?:\{\})?(?![A-Za-z])', lambda _:v, s)
    return s

def group(s, pos):
    assert s[pos] == '{', (s[pos:pos+40],pos)
    depth=1; i=pos+1
    while depth:
        if s[i]=='{' and (i==0 or s[i-1]!='\\'): depth+=1
        if s[i]=='}' and (i==0 or s[i-1]!='\\'): depth-=1
        i+=1
    return s[pos+1:i-1],i

def norm_math(s):
    s = expand(s).strip()
    s = s.replace(r'\I',r'\mathbf{1}').replace(r'\sg',r'\operatorname{stopgrad}')
    s = s.replace(r'\argmax',r'\operatorname{arg\,max}')
    s = re.sub(r'\\rm\s+([A-Za-z]+)',r'\\mathrm{\1}',s).replace(r'\text{-}',r'\mathrm{-}')
    s = re.sub(r'\\(Bigl|Bigr|bigl|bigr|big|Big|!)(?![A-Za-z])','',s)
    s = s.replace(r'\boxed',r'\mathord')
    return s

MATH_CACHE={}
def math_xml(s, size=19, color=NAVY):
    s=norm_math(s)
    if s not in MATH_CACHE:
        omml = mathml2omml.convert(latex_to_mathml(s))
        root = ET.fromstring(('<root xmlns:m="'+NS['m']+'">'+omml+'</root>').encode())[0]
        # TeX groups are transparent; avoid the converter's redundant box objects.
        for b in root.findall('.//m:box',NS)[::-1]:
            parent=b.getparent();idx=parent.index(b);content=b.find('m:e',NS)
            parent.remove(b)
            for child in list(content):parent.insert(idx,child);idx+=1
        for nary in root.findall('.//m:nary',NS):
            pr=nary.find('m:naryPr',NS)
            for limit in ('sub','sup'):
                part=nary.find('m:'+limit,NS)
                if part is None or len(part)==0:pr.append(el('m:'+limit+'Hide',**{'m:val':'1'}))
        for rad in root.findall('.//m:rad',NS):
            deg=rad.find('m:deg',NS)
            if deg is None or len(deg)==0:
                pr=rad.find('m:radPr',NS)
                if pr is None:pr=el('m:radPr');rad.insert(0,pr)
                pr.append(el('m:degHide',**{'m:val':'1'}))
        # mathml2omml omits matrix column descriptors. Explicit counts are
        # required by Office's matrix reader (including a two-column cases block).
        for matrix in root.findall('.//m:m',NS):
            ncols=max(len(row.findall('m:e',NS)) for row in matrix.findall('m:mr',NS))
            pr=el('m:mPr');mcs=el('m:mcs');mc=el('m:mc');mcpr=el('m:mcPr')
            mcpr.append(el('m:count',**{'m:val':ncols}));mc.append(mcpr);mcs.append(mc);pr.append(mcs);matrix.insert(0,pr)
        MATH_CACHE[s]=root
    root=copy.deepcopy(MATH_CACHE[s])
    # Office Math in DrawingML uses DrawingML run properties, not w:rPr.
    for r in root.findall('.//m:r',NS):
        rp=el('a:rPr',sz=round(size*100),lang='ja-JP')
        fill=el('a:solidFill');fill.append(el('a:srgbClr',val=color));rp.append(fill)
        rp.append(el('a:latin',typeface='Cambria Math'))
        rp.append(el('a:ea',typeface=FONT))
        t=r.find('m:t',NS); r.insert(list(r).index(t),rp)
    m=el('a14:m');m.append(root)
    return m

def plain_math(s):
    # This is only a width estimate and editable text fallback, never an image.
    try:
        root=math_xml(s)
        return ''.join(root.itertext())
    except Exception:
        return re.sub(r'\\([A-Za-z]+)',r'\1',s).replace('{','').replace('}','')

def math_spans(s):
    """Editable text spans for small inline maths and older-client fallbacks.

    Superscripts/subscripts use DrawingML baseline attributes. Standalone
    equations retain full structured Office Math in the primary representation.
    """
    root=math_xml(s)[0]
    def visit(node,scale=1,baseline=0):
        tag=ET.QName(node).localname
        def part(name,sc=scale,ba=baseline):
            child=node.find('m:'+name,NS)
            return [] if child is None else visit(child,sc,ba)
        if tag.endswith('Pr'):return []
        if tag=='r':return [(''.join(node.find('m:t',NS).itertext()),scale,baseline)]
        if tag in ('sSub','sSup','sSubSup'):
            out=part('e')
            if tag in ('sSub','sSubSup'):out+=part('sub',scale*.76,baseline-25000)
            if tag in ('sSup','sSubSup'):out+=part('sup',scale*.76,baseline+35000)
            return out
        if tag=='f':return [('(',scale,baseline)]+part('num')+[(')/(',scale,baseline)]+part('den')+[(')',scale,baseline)]
        if tag=='nary':
            chrnode=node.find('m:naryPr/m:chr',NS);symbol=chrnode.get('{'+NS['m']+'}val') if chrnode is not None else '∑'
            return [(symbol,scale,baseline)]+part('sub',scale*.76,baseline-25000)+part('sup',scale*.76,baseline+35000)+part('e')
        if tag=='rad':return [('√(',scale,baseline)]+part('e')+[(')',scale,baseline)]
        if tag=='acc':return part('e')+[('\u0302',scale,baseline)]
        if tag=='d':
            pr=node.find('m:dPr',NS)
            def delim(key,default):
                c=None if pr is None else pr.find('m:'+key,NS)
                return default if c is None else c.get('{'+NS['m']+'}val','')
            return [(delim('begChr','('),scale,baseline)]+part('e')+[(delim('endChr',')'),scale,baseline)]
        out=[]
        for child in node:out+=visit(child,scale,baseline)
        return out
    return visit(root)

def add_inline_math(p,s,size,color):
    for token,scale,baseline in math_spans(s):
        if not token:continue
        r=p.add_run();r.text=token;font_props(r,size*scale,color)
        r.font.name='Cambria Math'
        if baseline:r._r.get_or_add_rPr().set('baseline',str(baseline))

def simplify(s):
    s=expand(s)
    s=re.sub(r'\\(?:fontsize)\{[^}]*\}\{[^}]*\}','',s)
    s=re.sub(r'\\(?:vspace|hspace)\*?\{[^}]*\}',' ',s)
    s=re.sub(r'\\(?:small|scriptsize|tiny|large|Large|bfseries|selectfont|centering|raggedright|par|vfill)(?![A-Za-z])',' ',s)
    s=s.replace(r'\quad',' ').replace('~',' ').replace('---','—').replace('--','–')
    s=s.replace(r'\%', '%').replace(r'\_', '_').replace(r'\&','&')
    return s

def runs(s, bold=False, color=NAVY):
    """Return ('text'|'math', value, bold, color) from the small TeX vocabulary."""
    s=simplify(s); out=[]; i=0; buf=''
    def flush():
        nonlocal buf
        if buf:
            out.append(('text',re.sub(r'\s+',' ',buf).replace('\ufff0','\n'),bold,color));buf=''
    while i<len(s):
        if s[i]=='$':
            j=s.find('$',i+1)
            if j!=-1:flush();out.append(('math',s[i+1:j],bold,color));i=j+1;continue
        m=re.match(r'\\(key|textbf|textit|texttt|alert|textcolor|href|color)\b',s[i:])
        if m:
            flush();cmd=m[1];j=i+m.end()
            while j<len(s) and s[j].isspace():j+=1
            arg,j=group(s,j)
            if cmd in ('href','textcolor'):
                while s[j].isspace():j+=1
                arg2,j=group(s,j)
                out+=runs(arg2,bold,{'Teal':TEAL,'Muted':MUTED,'Orange':ORANGE}.get(arg,color) if cmd=='textcolor' else TEAL)
            elif cmd=='color':color={'Teal':TEAL,'Muted':MUTED,'Orange':ORANGE}.get(arg,color)
            else:out+=runs(arg,bold or cmd in ('key','textbf'),TEAL if cmd=='key' else ORANGE if cmd=='alert' else color)
            i=j;continue
        if s.startswith('\\\\',i):buf+='\ufff0';i+=2;continue
        if s[i] in '{}':i+=1;continue
        buf+=s[i];i+=1
    flush()
    return out

def plain(s): return ''.join(plain_math(v) if k=='math' else v for k,v,_,_ in runs(s)).strip()

def measure(s,w,size):
    f=ImageFont.truetype(str(FONT_FILE),round(size*2))
    # Leave room for the wider Latin glyphs of Office's Japanese font fallback.
    capacity=w*144*.88
    lines=0
    for p in plain(s).split('\n'):
        if not p.strip():continue
        n=1;used=0
        for c in p:
            cw=f.getlength(c)
            if used+cw>capacity: n+=1;used=0
            used+=cw
        lines+=n
    return max(1,lines)*size*1.43/72+.035

prs=Presentation(); prs.slide_width=Inches(13.333333);prs.slide_height=Inches(7.5)
prs.core_properties.title='化学プロービングを用いたMXfold2の学習と評価'
prs.core_properties.subject='TrainSetB: all 55 conditions at epoch 100; RNAgym: epoch 3'
prs.core_properties.author='Research progress report'
prs.core_properties.comments='Editable native text, Office Math, tables, charts and diagrams. Frozen evidence: 2026-10-22 14:07 JST.'
layout_log=[]; formula_log=[]

def box(slide,x,y,w,h,fill=PALE,line=None,radius=False):
    shape=slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE, Inches(x),Inches(y),Inches(w),Inches(h))
    shape.fill.solid();shape.fill.fore_color.rgb=rgb(fill)
    shape.line.fill.background() if line is None else None
    if line:shape.line.color.rgb=rgb(line);shape.line.width=Pt(1)
    return shape

def font_props(run,size,color=NAVY,bold=False):
    run.font.name=FONT;run.font.size=Pt(size);run.font.bold=bold;run.font.color.rgb=rgb(color)
    rp=run._r.get_or_add_rPr();rp.set('lang','ja-JP')
    for tag in ['a:ea','a:cs']:
        rp.append(el(tag,typeface=FONT))

def text(slide,s,x,y,w,h,size=18,color=NAVY,bold=False,align=None,margin=0):
    sh=slide.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h))
    tf=sh.text_frame;tf.clear();tf.word_wrap=True
    tf.margin_left=tf.margin_right=Inches(margin);tf.margin_top=Inches(.015);tf.margin_bottom=0
    p=tf.paragraphs[0];p.space_before=Pt(0);p.space_after=Pt(0)
    p.line_spacing=1.23
    if align is not None:p.alignment=align
    rp=p._p.get_or_add_pPr();dr=el('a:defRPr',sz=round(size*100),lang='ja-JP');dr.append(el('a:latin',typeface=FONT));dr.append(el('a:ea',typeface=FONT));rp.append(dr)
    has_math=False
    for k,v,b,c in runs(s,bold,color):
        if k=='math':
            add_inline_math(p,v,size,c)
        else:
            for j,line in enumerate(v.split('\n')):
                if j:p.add_line_break()
                r=p.add_run();r.text=line;font_props(r,size,c,b)
    if has_math:sh.name='MathText '+str(sh.shape_id)
    p._p.append(el('a:endParaRPr',sz=round(size*100),lang='ja-JP'))
    sh._element.find('p:nvSpPr/p:cNvPr',NS).set('descr',plain(s))
    return sh

def equation(slide,s,x,y,w,h,size=20):
    sh=slide.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h))
    sh.name='Equation '+str(sh.shape_id)
    tf=sh.text_frame;tf.margin_left=tf.margin_right=0;tf.margin_top=tf.margin_bottom=0;tf.word_wrap=False
    p=tf.paragraphs[0];p.alignment=PP_ALIGN.CENTER;p.line_spacing=1
    math=math_xml(s,size);omath=math[0];math.remove(omath)
    para=el('m:oMathPara');para.append(omath);math.append(para)
    p._p.append(math)
    p._p.append(el('a:endParaRPr',sz=round(size*100),lang='ja-JP'))
    sh._element.find('p:nvSpPr/p:cNvPr',NS).set('descr',norm_math(s))
    formula_log.append({'slide':len(prs.slides),'latex':norm_math(s),'size_pt':size})
    return sh

def table(slide, rows, x,y,w,h,size=17, widths=None):
    n=len(rows);nc=len(rows[0]);sh=slide.shapes.add_table(n,nc,Inches(x),Inches(y),Inches(w),Inches(h));t=sh.table
    if widths is None:
        # First column carries method labels; prose columns receive proportional widths.
        scores=[max(len(plain(row[j])) for row in rows) for j in range(nc)]
        weights=[max(5,min(42,v)) for v in scores]
        widths=[v/sum(weights) for v in weights]
    for col,f in zip(t.columns,widths):col.width=Inches(w*f)
    # Explicit white/pale cell fills, no dependency on a user's PowerPoint theme.
    for i,row in enumerate(rows):
        for j,s in enumerate(row):
            c=t.cell(i,j);c.margin_left=c.margin_right=Inches(.10);c.margin_top=c.margin_bottom=Inches(.07)
            c.vertical_anchor=MSO_ANCHOR.MIDDLE;c.fill.solid();c.fill.fore_color.rgb=rgb(NAVY if i==0 else 'F2F6F7' if i%2 else 'FFFFFF')
            tf=c.text_frame;tf.clear();tf.word_wrap=True;p=tf.paragraphs[0];p.line_spacing=1.12
            p.space_after=Pt(0);p.space_before=Pt(0)
            p.alignment=PP_ALIGN.LEFT if j==0 else PP_ALIGN.RIGHT if re.match(r'^[+$\-\d.]+',plain(s)) else PP_ALIGN.LEFT
            for k,v,b,co in runs(s,i==0,'FFFFFF' if i==0 else NAVY):
                if k=='math':add_inline_math(p,v,size,'FFFFFF' if i==0 else co)
                else:
                    for jj,line in enumerate(v.split('\n')):
                        if jj:p.add_line_break()
                        r=p.add_run();r.text=line;font_props(r,size,'FFFFFF' if i==0 else co,b)
            p._p.append(el('a:endParaRPr',sz=round(size*100),lang='ja-JP'))
            tcpr=c._tc.get_or_add_tcPr()
            for edge in ('lnL','lnR','lnT','lnB'):
                ln=OxmlElement('a:'+edge);ln.set('w','0');ln.append(OxmlElement('a:noFill'));tcpr.append(ln)
    return sh

def base_slide(title,num):
    s=prs.slides.add_slide(prs.slide_layouts[6]);s.background.fill.solid();s.background.fill.fore_color.rgb=rgb('FFFFFF')
    title_size=26 if len(plain(title))<40 else 24
    text(s,title,.58,.28,12.2,.60,title_size,bold=True)
    box(s,.58,1.00,12.16,.027,TEAL)
    text(s,'MXfold2 / Chemical probing progress',.58,7.10,9,.18,9,MUTED)
    text(s,'2026.10.22    '+str(num),10.85,7.10,1.9,.18,9,MUTED,align=PP_ALIGN.RIGHT)
    return s

def extract_cmd(s,cmd):
    m=re.search(r'\\'+cmd+r'\s*\{',s)
    if not m:return None,s
    v,end=group(s,m.end()-1)
    return v,s[:m.start()]+s[end:]

def parse_table(s):
    start=s.index(r'\begin{tabular}')+len(r'\begin{tabular}')
    _,start=group(s,start)
    body=s[start:s.index(r'\end{tabular}')]
    body=re.sub(r'\\(?:toprule|midrule|bottomrule)','',body)
    body=re.sub(r'\\\\\s*\[[^]]*\]',r'\\\\',body)
    rows=[[cell.strip() for cell in row.strip().split('&')] for row in body.split('\\\\') if row.strip()]
    return rows

def parse_blocks(s):
    s=re.sub(r'(?m)^\s*%.*$','',s)
    s=re.sub(r'\\input\{([^}]+)\}',lambda m:(HERE/m[1]).read_text(),s)
    s=re.sub(r'\\(?:small|scriptsize|tiny|centering|raggedright|par)(?![A-Za-z])',' ',s)
    s=re.sub(r'\\vspace\*?\{[^}]+\}',' ',s)
    result=[];i=0
    pats=[r'\\\[',r'\\begin\{tabular\}',r'\\begin\{itemize\}',r'\\begin\{block\}',r'\\take\{',r'\\begin\{columns\}']
    matcher=re.compile('|'.join(pats))
    while i<len(s):
        m=matcher.search(s,i)
        if not m:
            result.extend(('text',p.strip()) for p in re.split(r'\n\s*\n',s[i:]) if plain(p));break
        before=s[i:m.start()]
        result.extend(('text',p.strip()) for p in re.split(r'\n\s*\n',before) if plain(p))
        st=m.start()
        if s.startswith('\\[',st):
            end=s.index('\\]',st);result.append(('math',s[st+2:end].strip()));i=end+2
        elif s.startswith(r'\begin{tabular}',st):
            end=s.index(r'\end{tabular}',st)+len(r'\end{tabular}');result.append(('table',parse_table(s[st:end])));i=end
        elif s.startswith(r'\begin{itemize}',st):
            end=s.index(r'\end{itemize}',st)
            result.extend(('bullet',p.strip()) for p in s[st+len(r'\begin{itemize}'):end].split(r'\item') if plain(p))
            i=end+len(r'\end{itemize}')
        elif s.startswith(r'\begin{block}',st):
            title,j=group(s,st+len(r'\begin{block}'));end=s.index(r'\end{block}',j)
            result.append(('card',(title,parse_blocks(s[j:end]))));i=end+len(r'\end{block}')
        elif s.startswith(r'\take',st):
            v,j=group(s,st+len(r'\take'));result.append(('take',v));i=j
        else:
            j=s.index(r'\end{columns}',st);part=s[st:j]
            cols=re.split(r'\\column\{[^}]+\}',part)[1:]
            result.append(('columns',[parse_blocks(c) for c in cols]));i=j+len(r'\end{columns}')
    return result

def block_height(block,w,size):
    k,v=block
    if k in ('text','bullet'):return measure(v,w-(.22 if k=='bullet' else 0),size)+.09
    if k=='math':
        # Fraction/sum expressions need generous vertical leading.
        return max(.55,size/72*(2.3 if ('\\frac' in v or '\\sum' in v or '\\cases' in v) else 1.65))+.12
    if k=='table':
        nc=len(v[0]);weights=[max(5,min(42,max(len(plain(r[j])) for r in v))) for j in range(nc)]
        widths=[w*a/sum(weights) for a in weights]
        return sum(max(measure(c,widths[j]-.20,size*.88) for j,c in enumerate(row))+.08 for row in v)+.14
    if k=='take':return measure(v,w-.30,size*.94)+.24
    if k=='card':return measure(v[0],w-.32,size,)+sum(block_height(b,w-.32,size*.94) for b in v[1])+.30
    if k=='columns':return max(sum(block_height(b,(w-.32)/len(v),size) for b in col) for col in v)+.10
    raise ValueError(k)

def draw_blocks(slide,blocks,x,y,w,size):
    for k,v in blocks:
        h=block_height((k,v),w,size)
        if k=='text':text(slide,v,x,y,w,h-.09,size)
        elif k=='bullet':
            text(slide,'•',x,y,.20,.30,size,TEAL,bold=True)
            text(slide,v,x+.23,y,w-.23,h-.09,size)
        elif k=='math':
            # Wider formulae are composed at a smaller editable math size.
            n=len(plain_math(v));ms=min(size*1.07,23)
            if n>90:ms*=.82
            equation(slide,v,x,y+.025,w,h-.10,ms)
        elif k=='table':table(slide,v,x,y,w,h-.14,size*.88)
        elif k=='take':box(slide,x,y,w,h-.08);text(slide,v,x+.16,y+.08,w-.32,h-.24,size*.94,bold=True)
        elif k=='card':
            box(slide,x,y,w,h-.08)
            ht=measure(v[0],w-.32,size)+.08
            text(slide,v[0],x+.16,y+.10,w-.32,ht,size,TEAL,bold=True)
            draw_blocks(slide,v[1],x+.16,y+.12+ht,w-.32,size*.94)
        elif k=='columns':
            cw=(w-.32)/len(v)
            for i,col in enumerate(v):draw_blocks(slide,col,x+i*(cw+.32),y,cw,size)
        y+=h
    return y

def read_tsv(p):
    with p.open() as f:return list(csv.DictReader(f,delimiter='\t'))
data=read_tsv(E/'synthetic_all_curves.tsv');index={(r['name'],int(r['epoch'])):r for r in data}
real={k:read_tsv(E/f'real_{k}/evaluation_curve.tsv') for k in ('rank','fy','nll')}
def run_name(m,pct,probe):
    if m=='truth':return f'truth-fy-replay{pct:02d}'
    if m in ('rnafold','mxfold2'):return f'{m}-{probe}-fy-replay{pct:02d}'
    return f'{m}-{probe}-replay{pct:02d}'
def value(m,probe,pct,epoch,metric):return float(index[(run_name(m,pct,probe),epoch)][metric])

def axis_font(ax):
    ax.tick_labels.font.name=FONT;ax.tick_labels.font.size=Pt(10);ax.tick_labels.font.color.rgb=rgb(MUTED)
    ax.major_tick_mark=XL_TICK_MARK.NONE;ax.minor_tick_mark=XL_TICK_MARK.NONE
    ax.format.line.color.rgb=rgb('B6C5CC');ax.format.line.width=Pt(.6)
def chart_title(ch,title):
    ch.has_title=True;tf=ch.chart_title.text_frame;tf.clear();r=tf.paragraphs[0].add_run();r.text=title;font_props(r,14,NAVY,True)
def axis_title(ax,title):
    ax.has_title=True;tf=ax.axis_title.text_frame;tf.clear();r=tf.paragraphs[0].add_run();r.text=title;font_props(r,10,MUTED)

def xy_chart(slide,series,x,y,w,h,title,xtitle,ytitle,xlim=None,ylim=None,bubble=False,markers=False):
    d=XyChartData()
    for spec in series:
        se=d.add_series(spec['label'])
        for j,(xx,yy) in enumerate(zip(spec['x'],spec['y'])):
            se.add_data_point(xx,yy)
    typ=XL_CHART_TYPE.XY_SCATTER_LINES if bubble or markers else XL_CHART_TYPE.XY_SCATTER_LINES_NO_MARKERS
    ch=slide.shapes.add_chart(typ,Inches(x),Inches(y),Inches(w),Inches(h),d).chart
    ch.has_legend=False;chart_title(ch,title)
    for ax in (ch.category_axis,ch.value_axis):axis_font(ax)
    axis_title(ch.category_axis,xtitle);axis_title(ch.value_axis,ytitle)
    for ax,lim in ((ch.category_axis,xlim),(ch.value_axis,ylim)):
        if lim:ax.minimum_scale=lim[0];ax.maximum_scale=lim[1]
    ch.value_axis.tick_labels.number_format='0.00'
    ch.category_axis.tick_labels.number_format='0.00' if xlim and xlim[1]<1 else '0'
    ch.value_axis.has_major_gridlines=True;ch.value_axis.major_gridlines.format.line.color.rgb=rgb('E2E9EC');ch.value_axis.major_gridlines.format.line.width=Pt(.5)
    for se,spec in zip(ch.series,series):
        color=spec['color'];se.format.line.color.rgb=rgb(color);se.format.line.width=Pt(1.8)
        if spec.get('dash'):
            from pptx.enum.dml import MSO_LINE_DASH_STYLE
            se.format.line.dash_style=MSO_LINE_DASH_STYLE.DASH
        if bubble and not spec.get('guide'):
            se.format.line.fill.background()
            se.marker.style=XL_MARKER_STYLE.CIRCLE
            se.marker.format.fill.solid();se.marker.format.fill.fore_color.rgb=rgb(color)
            se.marker.format.line.color.rgb=rgb('FFFFFF');se.marker.format.line.width=Pt(.5)
            for pt,diam in zip(se.points,[4,5,6,8,10]):pt.marker.size=diam
        elif bubble:
            se.marker.style=XL_MARKER_STYLE.NONE
        elif markers:
            se.marker.style=XL_MARKER_STYLE.CIRCLE;se.marker.size=4
            se.marker.format.fill.solid();se.marker.format.fill.fore_color.rgb=rgb(color);se.marker.format.line.fill.background()
    return ch

def legend(slide,items,y=5.34,columns=3):
    for j,(label,color,dash) in enumerate(items):
        col=j%columns;row=j//columns;x=.85+col*(11.65/columns);yy=y+row*.25
        ln=slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,Inches(x),Inches(yy+.11),Inches(x+.31),Inches(yy+.11))
        ln.line.color.rgb=rgb(color);ln.line.width=Pt(2)
        if dash:
            from pptx.enum.dml import MSO_LINE_DASH_STYLE
            ln.line.dash_style=MSO_LINE_DASH_STYLE.DASH
        text(slide,label,x+.38,yy,11.65/columns-.43,.23,10)

def fig_slide(slide,stem):
    xx=[0,20,40,60,80,100]
    if stem in ('synthetic_replay_f1','synthetic_learning20','synthetic_tradeoff','zero_replay_final','zero_all_methods_2A3','zero_all_methods_DMS'):
        for side,probe in enumerate(['2A3','DMS'] if not stem.startswith('zero_all_methods') else [stem.rsplit('_',1)[-1]]*2):
            specs=[];metric=RHO if stem.startswith('zero_all_methods') and side else F1
            for m,label in METHODS[:3] if stem=='zero_replay_final' else METHODS:
                if stem=='synthetic_replay_f1':xs=REPLAYS;ys=[value(m,probe,p,100,F1) for p in REPLAYS]
                elif stem=='synthetic_tradeoff':xs=[value(m,probe,p,100,F1) for p in REPLAYS];ys=[value(m,probe,p,100,RHO) for p in REPLAYS]
                else:xs=list(range(0,101,10));ys=[value(m,probe,20 if stem=='synthetic_learning20' else 0,e,metric) for e in xs]
                specs.append(dict(label=label,x=xs,y=ys,color=COLORS[m],sizes=[20,32,45,64,90]))
                if stem=='zero_replay_final':specs.append(dict(label=label+' / 20%',x=xs,y=[value(m,probe,20,e,F1) for e in xs],color=COLORS[m],dash=True))
            xt='TestSetB mean F1' if stem=='synthetic_tradeoff' else 'TrainSetA replay (%)' if stem=='synthetic_replay_f1' else 'Additional epochs'
            yt='4-probe macro Spearman' if metric==RHO or stem=='synthetic_tradeoff' else 'TestSetB mean F1'
            xl=(.56,.67) if stem=='synthetic_tradeoff' and side==0 else (.27,.67) if stem=='synthetic_tradeoff' else (0,50) if stem=='synthetic_replay_f1' else (0,100)
            yl=(.34,.50) if stem=='synthetic_tradeoff' else (.25,.68) if stem=='synthetic_replay_f1' else (.45,.67) if stem=='synthetic_learning20' else (.27,.66) if metric==F1 else (.34,.50)
            if stem!='synthetic_tradeoff':specs.append(dict(label='Initial baseline',x=list(xl),y=[BASE[metric]]*2,color=COLORS['baseline'],dash=True))
            else:
                specs.extend([dict(label='Initial F1',x=[BASE[F1]]*2,y=[.34,.50],color=COLORS['baseline'],dash=True,guide=True),
                              dict(label='Initial chemical score',x=list(xl),y=[BASE[RHO]]*2,color=COLORS['baseline'],dash=True,guide=True)])
            xy_chart(slide,specs,.63+side*6.1,1.23,6.05,3.95,probe+' training' if not stem.startswith('zero_all_methods') else probe+' / replay 0%',xt,yt,xl,yl,bubble=stem=='synthetic_tradeoff',markers=stem=='synthetic_replay_f1')
        items=[(l,COLORS[m],False) for m,l in METHODS]
        if stem=='zero_replay_final':items=[(l+' / '+str(p)+'%',COLORS[m],p==20) for p in (0,20) for m,l in METHODS[:3]]
        legend(slide,items)
        if stem=='synthetic_tradeoff':text(slide,'Point size: replay 0 / 5 / 10 / 20 / 50%',.80,5.88,11.5,.23,10,MUTED)
    elif stem=='dms_nll_diagnosis':
        specs=[];palette=[TEAL,BLUE,'DB7629','9B91AD','AA4664']
        for p,color in zip(REPLAYS,palette):specs.append(dict(label=f'Replay {p}%',x=list(range(0,101,10)),y=[value('imle-nll','DMS',p,e,F1) for e in range(0,101,10)],color=color))
        specs.append(dict(label='Initial baseline',x=[0,100],y=[BASE[F1]]*2,color=COLORS['baseline'],dash=True))
        xy_chart(slide,specs,.63,1.23,6.05,3.95,'DMS / IMLE-NLL','Additional epochs','TestSetB mean F1',(0,100),(.25,.67))
        specs=[]
        for p,color in zip(REPLAYS,palette):
            group='synth_zero' if p==0 else 'synth_nll'
            o=json.loads((E/f'{group}/conditions/imle-nll-DMS-replay{p:02d}/epoch-100-training.json').read_text())
            specs.append(dict(label=f'{p}%',x=[o['nll_raw_mean']],y=[value('imle-nll','DMS',p,100,F1)],color=color))
        ch=xy_chart(slide,specs,6.73,1.23,6.05,3.95,'Epoch 100: NLL vs F1','Training NLL (lower is better)','TestSetB mean F1',(.225,.30),(.25,.67),markers=True)
        ch.category_axis.tick_labels.number_format='0.000'
        for se in ch.series:se.marker.size=8
        for se in ch.series:
            labels=el('c:dLbls');labels.append(el('c:dLblPos',val='r'))
            for tag in ('showLegendKey','showVal','showCatName','showSerName','showPercent','showBubbleSize'):
                labels.append(el('c:'+tag,val='1' if tag=='showSerName' else '0'))
            se._element.insert_element_before(labels,'c:trendline','c:errBars','c:xVal','c:yVal','c:smooth','c:extLst')
        legend(slide,[(f'Replay {p}%',c,False) for p,c in zip(REPLAYS,palette)])
    elif stem=='real_curves':
        for side,metric in enumerate([F1,RHO]):
            specs=[]
            for k,m,label in [('rank','imle-rank','IMLE-Rank'),('fy','shape-fy','SHAPE-fy'),('nll','imle-nll','IMLE-NLL')]:
                specs.append(dict(label=label,x=[int(r['epoch']) for r in real[k]],y=[float(r[metric]) for r in real[k]],color=COLORS[m]))
            specs.append(dict(label='Initial baseline',x=[0,3],y=[BASE[metric]]*2,color=COLORS['baseline'],dash=True))
            ch=xy_chart(slide,specs,.63+6.1*side,1.23,6.05,3.95,'Base-pair accuracy' if side==0 else 'Chemical consistency','Additional epochs','TestSetB mean F1' if side==0 else '4-probe macro Spearman',(0,3),(.525,.625) if side==0 else (.474,.510),markers=True)
            ch.category_axis.major_unit=1
        legend(slide,[(l,COLORS[m],False) for m,l in METHODS[:3]],columns=3)
    elif stem=='real_probe_details':
        probes=['2A3','DMS','1M7','NMIA'];base=json.loads((E/'real_rank/evaluation-epoch-0.json').read_text())['chemical_probe_panel']['modifiers']
        vals={}
        for k in ('rank','fy','nll'):vals[k]=json.loads((E/f'real_{k}/evaluation-epoch-3.json').read_text())['chemical_probe_panel']['modifiers']
        for side in range(2):
            d=CategoryChartData();d.categories=probes
            if side==0:d.add_series('Baseline',[base[p]['cluster_macro_spearman'] for p in probes])
            for k,m,label in [('rank','imle-rank','IMLE-Rank'),('fy','shape-fy','SHAPE-fy'),('nll','imle-nll','IMLE-NLL')]:d.add_series(label,[vals[k][p]['cluster_macro_spearman']-(base[p]['cluster_macro_spearman'] if side else 0) for p in probes])
            ch=slide.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED if side else XL_CHART_TYPE.LINE_MARKERS,Inches(.63+side*6.1),Inches(1.23),Inches(6.05),Inches(3.95),d).chart
            ch.has_legend=False;chart_title(ch,'Epoch 3: each assay' if side==0 else 'Change from baseline');axis_title(ch.value_axis,'Spearman' if side==0 else 'Δ Spearman')
            for ax in (ch.category_axis,ch.value_axis):axis_font(ax)
            ch.value_axis.tick_labels.number_format='0.00';ch.value_axis.has_major_gridlines=True;ch.value_axis.major_gridlines.format.line.color.rgb=rgb('E2E9EC')
            cols=[COLORS['baseline']]+[COLORS[m] for m,_ in METHODS[:3]] if side==0 else [COLORS[m] for m,_ in METHODS[:3]]
            for se,c in zip(ch.series,cols):
                se.format.line.color.rgb=rgb(c);se.format.fill.solid();se.format.fill.fore_color.rgb=rgb(c)
                if side:se.invert_if_negative=False
        legend(slide,[('Baseline',COLORS['baseline'],True)]+[(l,COLORS[m],False) for m,l in METHODS[:3]],columns=4)
    else:raise ValueError(stem)

def split_source(body):
    src,body=extract_cmd(body,'src')
    return src or '',body

tex=expand((HERE/'main.tex').read_text())
frames=re.findall(r'\\begin\{frame\}(.*?)\\end\{frame\}',tex,re.S)
assert len(frames)==47
note_sections=[]
for section in re.split(r'(?m)^## ',(HERE/'speaker_notes.md').read_text())[1:]:
    heading,_,content=section.partition('\n')
    indices=[]
    for match in re.finditer(r'(\d+)(?:[–-](\d+))?枚目',heading):
        indices.extend(range(int(match[1]),int(match[2] or match[1])+1))
    if indices:note_sections.append((set(indices),heading+'\n'+content.strip()))
for num,frame in enumerate(frames,1):
    if num==1:
        s=prs.slides.add_slide(prs.slide_layouts[6]);box(s,0,0,13.333,7.5,'FFFFFF')
        box(s,.68,.70,.10,5.98,TEAL)
        text(s,'RESEARCH PROGRESS / 2026.10.22',1.05,.80,11,.45,16,TEAL,bold=True)
        text(s,'化学プロービングを用いた\\\\MXfold2の学習と評価',1.05,1.78,11.4,1.6,34,bold=True)
        text(s,'TrainSetB合成反応性と実測RNAgymによる比較',1.05,3.65,11.3,.70,24)
        text(s,'IMLE-Rank / IMLE-NLL / SHAPE-fy / assisted folding擬似ラベル',1.05,4.72,11.3,.6,18)
        text(s,'対象スレッド：「TrainSetB SHAPE実験を追加」\\\\「化学プロービング手法の有効性を検討」\\\\データ固定：'+generated['snapshot'],1.05,5.76,11.4,.95,12,MUTED)
        s.notes_slide.notes_text_frame.text='\n\n'.join(content for ids,content in note_sections if num in ids)
        continue
    title,end=group(frame,0);body=frame[end:];src,body=split_source(body)
    s=base_slide(title,num)
    text(s,'出典：'+src,.60,6.74,12.1,.32,8.3,MUTED)
    fig=re.search(r'\\fig\{([^}]+)\}',body)
    if fig:
        fig_slide(s,fig[1]);body=body[:fig.start()]+body[fig.end():]
        blocks=parse_blocks(body);size=16.5
        # Figure explanation is kept below the legend.
        height=sum(block_height(b,12.08,size) for b in blocks)
        while height>.52 and size>13:
            size-=.25;height=sum(block_height(b,12.08,size) for b in blocks)
        yy=6.02 if fig[1]=='synthetic_tradeoff' else 5.98
        endy=draw_blocks(s,blocks,.63,yy,12.08,size)
        layout_log.append(dict(slide=num,size_pt=size,bottom=endy,type='chart'))
    elif num==30:
        # Native editable boxes and arrow connectors for the RNAgym split diagram.
        for x,y,w,h,label in [(3.5,1.25,6.3,.9,'RMDB_s40_train\\\\117,773配列 / 198,411測定'),(.88,2.72,5.5,.9,'今回のtrain\\\\112,815配列 / 189,858測定'),(6.98,2.72,5.5,.9,'今回のvalidation\\\\4,958配列 / 2,043 cluster')]:
            box(s,x,y,w,h,PALE,TEAL,True);text(s,label,x+.1,y+.12,w-.2,h-.24,18,align=PP_ALIGN.CENTER)
        for xx in [3.63,9.73]:
            ln=s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,Inches(6.65),Inches(2.15),Inches(xx),Inches(2.72));ln.line.color.rgb=rgb(TEAL);ln.line.width=Pt(1.4)
            end=OxmlElement('a:tailEnd');end.set('type','triangle');ln.line._get_or_add_ln().append(end)
        body=re.sub(r'\\begin\{center\}.*?\\end\{center\}','',body,flags=re.S)
        blocks=parse_blocks(body);draw_blocks(s,blocks,.63,3.95,12.08,18)
        layout_log.append(dict(slide=num,size_pt=18,bottom=6.5,type='diagram'))
    else:
        blocks=parse_blocks(body);size=19.5 if num<39 else 18
        if num in (41,42):size=15
        height=sum(block_height(b,12.08,size) for b in blocks)
        while height>5.32 and size>13:
            size-=.25;height=sum(block_height(b,12.08,size) for b in blocks)
        yy=1.20+max(0,(5.28-height)*.24)
        endy=draw_blocks(s,blocks,.63,yy,12.08,size)
        layout_log.append(dict(slide=num,size_pt=size,bottom=endy,type='content'))
    # Notes preserve the TeX text and original sources as a second editing aid.
    detailed='\n\n'.join(content for ids,content in note_sections if num in ids)
    s.notes_slide.notes_text_frame.text='Slide '+str(num)+' — '+plain(title)+'\n\n'+detailed+'\n\n出典：'+plain(src)+'\n\n数式・表の原稿：同梱 main.tex。詳しい解釈：speaker_notes.md。'

prs.save(OUT)

# Wrap math-containing shapes according to Microsoft MS-ODRAWXML §3.5.
# Fallbacks remain editable text shapes for clients without the a14 math extension.
# Modern PowerPoint selects the Choice and edits its Office Math directly.
with zipfile.ZipFile(OUT) as z:parts={n:z.read(n) for n in z.namelist()}
for n,b in list(parts.items()):
    if not re.match(r'ppt/slides/slide\d+\.xml$',n):continue
    root=ET.fromstring(b)
    for sp in root.findall('.//p:spTree/p:sp',NS):
        if sp.find('.//a14:m',NS) is None:continue
        fallback=copy.deepcopy(sp)
        for m in fallback.findall('.//a14:m',NS):
            parent=m.getparent();idx=parent.index(m)
            source=sp.find('p:nvSpPr/p:cNvPr',NS).get('descr')
            parent.remove(m)
            for token,scale,baseline in math_spans(source):
                r=el('a:r');rp=el('a:rPr',sz=round(1700*scale),baseline=baseline)
                rp.append(el('a:latin',typeface='Cambria Math'));r.append(rp)
                t=el('a:t');t.text=token;r.append(t);parent.insert(idx,r);idx+=1
        ac=ET.Element('{%s}AlternateContent'%NS['mc'],nsmap={'mc':NS['mc']})
        choice=ET.SubElement(ac,'{%s}Choice'%NS['mc'],nsmap={'a14':NS['a14']});choice.set('Requires','a14')
        fb=ET.SubElement(ac,'{%s}Fallback'%NS['mc']);fb.append(fallback)
        parent=sp.getparent();idx=parent.index(sp);parent.remove(sp);choice.append(sp);parent.insert(idx,ac)
    parts[n]=ET.tostring(root,xml_declaration=True,encoding='UTF-8',standalone=True)
with zipfile.ZipFile(OUT,'w',zipfile.ZIP_DEFLATED) as z:
    for n,b in parts.items():z.writestr(n,b)
(HERE/'editable_layout.json').write_text(json.dumps(layout_log,ensure_ascii=False,indent=2))
(HERE/'editable_equations.json').write_text(json.dumps(formula_log,ensure_ascii=False,indent=2))
print(OUT,OUT.stat().st_size,'bytes;',len(prs.slides),'slides;',len(formula_log),'display equations')
print('Minimum content font:',min(r['size_pt'] for r in layout_log),'pt')
print('Maximum content bottom:',round(max(r['bottom'] for r in layout_log),3),'in')
