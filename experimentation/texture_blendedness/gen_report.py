"""
Build a self-contained local HTML report from outputs/comparison/scores_all.csv.
Open outputs/comparison/index.html in a browser to browse all images with every
method's score + the side-by-side panel. Sort by any method or by disagreement.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
COMP = HERE / "outputs" / "comparison"
CSV = COMP / "scores_all.csv"


def main():
    rows = list(csv.DictReader(open(CSV)))
    methods = [c for c in rows[0].keys()
               if c not in ("rank", "image", "panel", "source")]
    data = []
    for r in rows:
        scores = {m: float(r[m]) for m in methods if r[m] not in ("", "nan")}
        vals = list(scores.values())
        spread = (max(vals) - min(vals)) if vals else 0.0
        data.append({
            "image": r["image"],
            "panel": f"panels/{r['panel']}",
            "scores": scores,
            "mean": round(sum(vals) / len(vals), 1) if vals else 0,
            "spread": round(spread, 1),
        })

    html = """<!doctype html><html><head><meta charset="utf-8">
<title>Blendedness method comparison</title><style>
body{font-family:-apple-system,system-ui,sans-serif;margin:0;background:#111;color:#eee}
header{position:sticky;top:0;background:#1b1b1b;padding:12px 16px;border-bottom:1px solid #333;z-index:10}
h1{font-size:16px;margin:0 0 8px}
.controls{font-size:13px}
button{background:#2a2a2a;color:#ddd;border:1px solid #444;border-radius:6px;padding:5px 10px;margin-right:6px;cursor:pointer}
button.active{background:#3a6;color:#fff;border-color:#3a6}
.row{border-bottom:1px solid #2a2a2a;padding:10px 16px}
.meta{display:flex;gap:14px;align-items:center;font-size:13px;margin-bottom:6px;flex-wrap:wrap}
.img{font-weight:600;color:#fff;min-width:80px}
.pill{padding:2px 8px;border-radius:10px;font-variant-numeric:tabular-nums}
.spread{color:#f90}
img{width:100%;max-width:1600px;border-radius:6px;display:block}
.legend{font-size:12px;color:#999;margin-top:4px}
</style></head><body>
<header><h1>Blendedness method comparison — __N__ images</h1>
<div class="controls">Sort by: <span id="btns"></span></div>
<div class="legend">Green = blended (high) &middot; Red = unblended (low). "spread" = max&minus;min across methods (high = methods disagree).</div>
</header>
<div id="list"></div>
<script>
const DATA = __DATA__;
const METHODS = __METHODS__;
function color(v){const h=Math.round(v*1.2);return 'hsl('+h+',70%,32%)';}
function render(key){
  let d=[...DATA];
  if(key==='spread') d.sort((a,b)=>b.spread-a.spread);
  else if(key==='mean') d.sort((a,b)=>a.mean-b.mean);
  else d.sort((a,b)=>(a.scores[key]??999)-(b.scores[key]??999));
  const list=document.getElementById('list');list.innerHTML='';
  for(const r of d){
    const div=document.createElement('div');div.className='row';
    let pills=METHODS.map(m=>{const v=r.scores[m];
      return '<span class="pill" style="background:'+(v==null?'#333':color(v))+'">'+m+': '+(v==null?'\\u2014':v.toFixed(1))+'</span>';}).join(' ');
    div.innerHTML='<div class="meta"><span class="img">'+r.image+'</span>'+pills+
      '<span class="pill spread">spread '+r.spread+'</span></div>'+
      '<img loading="lazy" src="'+r.panel+'">';
    list.appendChild(div);
  }
}
const btns=document.getElementById('btns');
['spread','mean',...METHODS].forEach((k,i)=>{const b=document.createElement('button');
  b.textContent=k;b.onclick=()=>{document.querySelectorAll('#btns button').forEach(x=>x.classList.remove('active'));
  b.classList.add('active');render(k);};btns.appendChild(b);if(i===0)b.classList.add('active');});
render('spread');
</script></body></html>"""
    html = (html.replace("__N__", str(len(data)))
                .replace("__DATA__", json.dumps(data))
                .replace("__METHODS__", json.dumps(methods)))

    (COMP / "index.html").write_text(html)
    print(f"Wrote {COMP/'index.html'}  ({len(data)} images, methods: {methods})")


if __name__ == "__main__":
    main()
