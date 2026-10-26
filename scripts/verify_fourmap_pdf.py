"""Inspect the built experiment design and render pages for visual QA."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import pymupdf


def verify(path, output):
    output.mkdir(parents=True,exist_ok=True)
    doc=pymupdf.open(path)
    texts=[page.get_text() for page in doc]
    all_text='\n'.join(texts)
    required=('Connectivity','Spatial thermal proxy','HPWL','legality','58.91','65','viridis','inferno','magma','Reds','coolwarm')
    missing=[term for term in required if term not in all_text]
    assert not missing, missing
    selected={0,1}
    selected.update(i for i,text in enumerate(texts)
                    if '四通道的具体计算' in text or 'Priority 的计算和表现' in text
                    or 'Legality 的计算和表现' in text)
    selected.update(i+1 for i in list(selected) if i+1<len(doc) and i>2)
    for i in sorted(selected):
        doc[i].get_pixmap(matrix=pymupdf.Matrix(1.3,1.3),alpha=False).save(output/f'page_{i+1:02d}.png')
    report={'pages':len(doc),'pdf_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'required_channel_terms_present':True,'rendered_pages':[i+1 for i in sorted(selected)],
        'note':'Text checks and page renders; visual review is performed separately.'}
    (output/'pdf_text_check.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pdf',type=Path)
    parser.add_argument('output',type=Path)
    a=parser.parse_args()
    verify(a.pdf,a.output)
