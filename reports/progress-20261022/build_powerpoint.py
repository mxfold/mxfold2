"""Package rendered report pages as a standards-based, picture-slide PPTX."""
from pathlib import Path
import re
import sys
import zipfile
from html import escape
from xml.etree import ElementTree as ET
from PIL import Image

HERE=Path(__file__).resolve().parent
render=Path(sys.argv[1])
pages=sorted(render.glob('slide-*.png'))
assert len(pages)==47, f'Expected 47 rendered slides, got {len(pages)}'
titles=re.findall(r'\\begin\{frame\}(?:\[[^\]]*\])?\{([^\n]*)\}',(HERE/'main.tex').read_text())
titles=['化学プロービングを用いたMXfold2の学習と評価']+titles
assert len(titles)==len(pages)
out=HERE/'progress_report_20261022_raster.pptx'
NSP='http://schemas.openxmlformats.org/presentationml/2006/main'
NSA='http://schemas.openxmlformats.org/drawingml/2006/main'
NSR='http://schemas.openxmlformats.org/officeDocument/2006/relationships'
CX,CY=12192000,6858000
def rels(items):
    return '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'+''.join(f'<Relationship Id="{rid}" Type="{NSR}/{typ}" Target="{target}"/>' for rid,typ,target in items)+'</Relationships>'
def grp():
    return '<p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>'
files={}
def xml(path,content):
    if not content.startswith('<?xml'):content='<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'+content
    ET.fromstring(content)
    files[path]=content.encode('utf-8')
overrides=[('/ppt/presentation.xml','application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml'),
 ('/ppt/slideMasters/slideMaster1.xml','application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml'),
 ('/ppt/slideLayouts/slideLayout1.xml','application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml'),
 ('/ppt/theme/theme1.xml','application/vnd.openxmlformats-officedocument.theme+xml'),
 ('/docProps/core.xml','application/vnd.openxmlformats-package.core-properties+xml'),
 ('/docProps/app.xml','application/vnd.openxmlformats-officedocument.extended-properties+xml')]
overrides += [(f'/ppt/slides/slide{i}.xml','application/vnd.openxmlformats-officedocument.presentationml.slide+xml') for i in range(1,len(pages)+1)]
xml('[Content_Types].xml','<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Default Extension="png" ContentType="image/png"/>'+''.join(f'<Override PartName="{p}" ContentType="{c}"/>' for p,c in overrides)+'</Types>')
xml('_rels/.rels',rels([('rId1','officeDocument','ppt/presentation.xml'),('rId2','metadata/core-properties','docProps/core.xml'),('rId3','extended-properties','docProps/app.xml')]).replace(NSR+'/metadata/core-properties','http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties'))
xml('docProps/core.xml','<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"><dc:title>化学プロービングを用いたMXfold2の学習と評価</dc:title><dc:creator>Research progress report</dc:creator><dc:description>47 picture slides. Edit mathematical content in the accompanying main.tex; see speaker_notes.md for explanation and provenance.</dc:description><dcterms:created xsi:type="dcterms:W3CDTF">2026-10-22T02:12:48Z</dcterms:created></cp:coreProperties>')
xml('docProps/app.xml',f'<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes"><Application>Research progress report</Application><PresentationFormat>On-screen Show (16:9)</PresentationFormat><Slides>{len(pages)}</Slides><Notes>0</Notes></Properties>')
xml('ppt/presentation.xml',f'<p:presentation xmlns:p="{NSP}" xmlns:a="{NSA}" xmlns:r="{NSR}"><p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/></p:sldMasterIdLst><p:sldIdLst>'+''.join(f'<p:sldId id="{255+i}" r:id="rId{i+1}"/>' for i in range(1,len(pages)+1))+f'</p:sldIdLst><p:sldSz cx="{CX}" cy="{CY}"/><p:notesSz cx="6858000" cy="9144000"/></p:presentation>')
xml('ppt/_rels/presentation.xml.rels',rels([('rId1','slideMaster','slideMasters/slideMaster1.xml')]+[(f'rId{i+1}','slide',f'slides/slide{i}.xml') for i in range(1,len(pages)+1)]))
clr='<p:clrMap accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" bg1="lt1" bg2="lt2" folHlink="folHlink" hlink="hlink" tx1="dk1" tx2="dk2"/>'
xml('ppt/slideMasters/slideMaster1.xml',f'<p:sldMaster xmlns:p="{NSP}" xmlns:a="{NSA}" xmlns:r="{NSR}"><p:cSld><p:spTree>{grp()}</p:spTree></p:cSld>{clr}<p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst><p:txStyles><p:titleStyle/><p:bodyStyle/><p:otherStyle/></p:txStyles></p:sldMaster>')
xml('ppt/slideMasters/_rels/slideMaster1.xml.rels',rels([('rId1','slideLayout','../slideLayouts/slideLayout1.xml'),('rId2','theme','../theme/theme1.xml')]))
xml('ppt/slideLayouts/slideLayout1.xml',f'<p:sldLayout xmlns:p="{NSP}" xmlns:a="{NSA}" type="blank" preserve="1"><p:cSld name="Blank"><p:spTree>{grp()}</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sldLayout>')
xml('ppt/slideLayouts/_rels/slideLayout1.xml.rels',rels([('rId1','slideMaster','../slideMasters/slideMaster1.xml')]))
scheme=''.join(f'<a:{k}><a:srgbClr val="{v}"/></a:{k}>' for k,v in [('dk1','173449'),('lt1','FFFFFF'),('dk2','697C88'),('lt2','EEF5F6'),('accent1','007F87'),('accent2','386BBE'),('accent3','C56A22'),('accent4','AA4664'),('accent5','58616B'),('accent6','9B91AD'),('hlink','007F87'),('folHlink','386BBE')])
solid='<a:solidFill><a:schemeClr val="phClr"/></a:solidFill>'
lines=''.join(f'<a:ln w="{w}" cap="flat" cmpd="sng" algn="ctr">{solid}<a:prstDash val="solid"/><a:miter lim="800000"/><a:headEnd type="none" w="med" len="med"/><a:tailEnd type="none" w="med" len="med"/></a:ln>' for w in [6350,12700,19050])
xml('ppt/theme/theme1.xml',f'<a:theme xmlns:a="{NSA}" name="Research report"><a:themeElements><a:clrScheme name="Report">{scheme}</a:clrScheme><a:fontScheme name="Report"><a:majorFont><a:latin typeface="Arial"/><a:ea typeface="Noto Sans CJK JP"/><a:cs typeface=""/></a:majorFont><a:minorFont><a:latin typeface="Arial"/><a:ea typeface="Noto Sans CJK JP"/><a:cs typeface=""/></a:minorFont></a:fontScheme><a:fmtScheme name="Report"><a:fillStyleLst>{solid*3}</a:fillStyleLst><a:lnStyleLst>{lines}</a:lnStyleLst><a:effectStyleLst>'+('<a:effectStyle><a:effectLst/></a:effectStyle>'*3)+f'</a:effectStyleLst><a:bgFillStyleLst>{solid*3}</a:bgFillStyleLst></a:fmtScheme></a:themeElements><a:objectDefaults/><a:extraClrSchemeLst/></a:theme>')
for i,(page,title) in enumerate(zip(pages,titles),1):
    with Image.open(page) as im:assert abs(im.width/im.height-16/9)<.01
    label=escape(re.sub(r'\\[A-Za-z]+','',title).replace('{','').replace('}',''),quote=True)
    pic=f'<p:pic><p:nvPicPr><p:cNvPr id="2" name="Slide {i}" descr="{label}"/><p:cNvPicPr><a:picLocks noChangeAspect="1"/></p:cNvPicPr><p:nvPr/></p:nvPicPr><p:blipFill><a:blip r:embed="rId1"/><a:stretch><a:fillRect/></a:stretch></p:blipFill><p:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{CX}" cy="{CY}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr></p:pic>'
    xml(f'ppt/slides/slide{i}.xml',f'<p:sld xmlns:p="{NSP}" xmlns:a="{NSA}" xmlns:r="{NSR}"><p:cSld name="{label}"><p:spTree>{grp()}{pic}</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>')
    xml(f'ppt/slides/_rels/slide{i}.xml.rels',rels([('rId1','image',f'../media/slide{i}.png'),('rId2','slideLayout','../slideLayouts/slideLayout1.xml')]))
    files[f'ppt/media/slide{i}.png']=page.read_bytes()
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
    for p,content in files.items():z.writestr(p,content)
with zipfile.ZipFile(out) as z:
    assert z.testzip() is None
    for p in z.namelist():
        if p.endswith(('.xml','.rels')):ET.fromstring(z.read(p))
print(out, out.stat().st_size, 'bytes;',len(pages),'slides; all package XML parses')
