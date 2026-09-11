"""Local browser tool for human-reviewed, GT-assisted point annotation.

Uses verified sequential frames; never changes GT, original sessions or masks.
The resulting JSON is a new experimental annotation set, not a pickle session.
"""
import argparse
import csv
import io
import json
import secrets
from datetime import datetime, timezone
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
import numpy as np
from PIL import Image
from recompute import gt_labels, indexed, sha, write_json
from verify_correspondence import fingerprint

IDS = [14,38,75,113]
COLORS = ['#ff6666','#36d9ef','#ffd447','#c58aff']


def bounds(block, size, total):
    if block < 0 or block*size >= total: raise ValueError('Invalid block')
    return block*size, min((block+1)*size,total)-1


def validate_points(points, gt):
    if len(points)!=4 or sorted(p['gt_id'] for p in points)!=IDS:
        raise ValueError('Place exactly one click for each of the four GT IDs')
    result=[]
    for p in points:
        x,y=float(p['x']),float(p['y'])
        if not np.isfinite([x,y]).all() or not (0<=x<gt.shape[1] and 0<=y<gt.shape[0]):
            raise ValueError('Click is outside image')
        if int(gt[int(y),int(x)])!=p['gt_id']:
            raise ValueError(f"Click for GT {p['gt_id']} is outside that GT mask; review or choose another frame")
        result.append(dict(gt_id=p['gt_id'],x=x,y=y,positive=True))
    return sorted(result,key=lambda p:p['gt_id'])


class Review:
    def __init__(self, source, prior, output):
        self.source=source.resolve(); self.output=output.resolve()
        verified=json.loads((source/'verification.json').read_text())
        if verified['status']!='verified': raise ValueError('Image source is not verified')
        self.total=verified['count']
        session=json.loads((prior/'session.json').read_text())
        self.size=session['block_size']
        self.num_blocks=(self.total+self.size-1)//self.size
        manifest=json.loads((prior/'manifest.json').read_text())
        self.gt_files=indexed(manifest['inputs']['gt'])
        with (source/'index.csv').open(newline='') as f:
            self.rows={int(r['video_index']):r for r in csv.DictReader(f)}
        if sorted(self.rows)!=list(range(self.total)): raise ValueError('Incomplete source index')
        self.output.mkdir(exist_ok=True)
        self.data=dict(schema_version=1,experiment='GT-assisted human reannotation',
            source_directory=str(self.source),source_verification_sha256=sha(source/'verification.json'),
            block_size=self.size,total_frames=self.total,gt_ids=IDS,
            label_policy='Numeric GT identities only; original descriptive names are not assumed.',
            blocks={},status='incomplete')
        if (self.output/'annotations.json').exists():
            saved=json.loads((self.output/'annotations.json').read_text())
            if any(saved.get(key)!=self.data[key] for key in ['source_directory','source_verification_sha256','block_size','total_frames','gt_ids']):
                raise ValueError('Cannot resume annotations from a different source')
            self.data=saved
        self.persist()

    def persist(self):
        self.data['status']='ready_for_integration' if len(self.data['blocks'])==self.num_blocks else 'incomplete'
        temporary=self.output/'annotations.tmp'
        write_json(temporary,self.data)
        temporary.replace(self.output/'annotations.json')

    @lru_cache(maxsize=2)
    def frame(self, index):
        import cv2
        row=self.rows[index]
        if not row['gt_frames_index']: raise ValueError('This video frame has no GT; choose another')
        # Numeric source filenames are constructed here, not taken from browser input.
        with Image.open(self.source/'frames'/f'{index:06d}.png') as image:
            rgb=np.asarray(image.convert('RGB')).copy()
        if fingerprint(rgb)!=row['rgb_sha256']: raise ValueError('Source pixels changed: refusing annotation')
        gt_index=int(row['gt_frames_index']); path=self.gt_files[gt_index]
        gt=gt_labels(path)
        if gt.shape!=rgb.shape[:2]: raise ValueError('GT and photograph dimensions differ')
        stats=[]
        for bird in IDS:
            mask=(gt==bird).astype(np.uint8)
            area=int(mask.sum())
            count,labels,components,centroids=cv2.connectedComponentsWithStats(mask,connectivity=8)
            sizes=components[1:,cv2.CC_STAT_AREA]
            stats.append(dict(gt_id=bird,area=area,components=count-1,
                largest_fraction=float(max(sizes)/area) if area else 0,
                touches_edge=bool(mask[0].any() or mask[-1].any() or mask[:,0].any() or mask[:,-1].any())))
        return rgb,gt,dict(video_frame=index,gt_frame=gt_index,width=rgb.shape[1],height=rgb.shape[0],
            masks=stats,other_ids=sorted(set(map(int,np.unique(gt)))-set(IDS)-{0}),gt_sha256=sha(path),
            rgb_sha256=row['rgb_sha256'])

    def metadata(self, block, frame):
        lo,hi=bounds(block,self.size,self.total)
        if not lo<=frame<=hi: raise ValueError(f'Choose video frame {lo} through {hi}')
        _,_,meta=self.frame(frame)
        return dict(meta,block=block,first=lo,last=hi,saved=self.data['blocks'].get(str(block)))

    def save(self, payload):
        block=int(payload['block']);frame=int(payload['frame'])
        meta=self.metadata(block,frame)
        if payload.get('reviewed') is not True:
            raise ValueError('Visually review all four GT identities before saving')
        _,gt,_=self.frame(frame)
        points=validate_points(payload['points'],gt)
        self.data['blocks'][str(block)]=dict(video_frame=frame,gt_frame=meta['gt_frame'],
            local_frame=frame-block*self.size,points=points,gt_sha256=meta['gt_sha256'],
            rgb_sha256=meta['rgb_sha256'],human_reviewed=True,
            reviewed_at=datetime.now(timezone.utc).isoformat())
        self.persist()
        return dict(saved_blocks=sorted(map(int,self.data['blocks'])),status=self.data['status'])


HTML = r'''<!doctype html><meta charset="utf-8"><title>Ruff annotation review</title>
<style>body{font:16px system-ui;margin:22px;background:#f3f5f7;color:#17232f}button,input,select{font:inherit;padding:7px;margin:4px}canvas{width:100%;height:auto;cursor:crosshair}main{display:grid;grid-template-columns:1fr 1fr;gap:12px}.panel{background:white;padding:10px}#message{white-space:pre-wrap;padding:10px;background:#fff}label{margin-right:12px} .help{max-width:1100px}button{cursor:pointer}</style>
<h1>Choose a clear frame and click four birds</h1>
<p class="help">GT-assisted experiment. Both panels use the same verified photograph. The coloured masks are GT, not predictions. Use numeric GT identities consistently; check that each mask covers the correct bird. Do not click a GT artefact. No old clicks are reused.</p>
<label>Block <select id="block"></select></label><label>Video frame <input id="frame" type="number"></label>
<button id="load">Inspect frame</button><button id="middle">Block midpoint</button>
<button id="back">−25 frames</button><button id="next">+25 frames</button><span id="range"></span>
<p id="status"></p><div id="targets"></div>
<p>Click inside the selected bird on either panel. Clicking again replaces that bird's point. Zoom your browser if needed; coordinates follow the image.</p>
<main><div class="panel">Verified photograph<canvas id="raw"></canvas></div><div class="panel">GT overlay (all four IDs)<canvas id="gt"></canvas></div></main>
<p><label><input type="checkbox" id="reviewed">I reviewed all four GT masks and identities, and my clicks are inside the intended birds.</label></p>
<button id="save">Save this block</button><button id="clear">Clear clicks</button>
<p id="message">Loading…</p><p class="help">Warnings are screening checks, not certification: this tool cannot decide whether GT identities are biologically correct or whether a frame is sharp enough. Choose another frame if uncertain. Saved blocks remain on disk; unsaved clicks are lost when changing frames. Keep the terminal running during review.</p>
<script>
const token='TOKEN', ids=[14,38,75,113], colors=['#ff6666','#36d9ef','#ffd447','#c58aff'];
const $=id=>document.getElementById(id);let config,meta,points={},pictures={},loaded=false,dirty=false;
async function api(path,body){const r=await fetch(path,body?{method:'POST',headers:{'Content-Type':'application/json','X-Review-Token':token},body:JSON.stringify(body)}:{});const d=await r.json();if(!r.ok)throw Error(d.error);return d;}
function message(s){$('message').textContent=s;}
function draw(){for(const id of ['raw','gt']){const c=$(id),ctx=c.getContext('2d');c.width=meta.width;c.height=meta.height;ctx.drawImage(pictures[id],0,0);ctx.lineWidth=4;ctx.font='28px sans-serif';for(const bird of ids){const p=points[bird];if(!p)continue;ctx.strokeStyle=colors[ids.indexOf(bird)];ctx.fillStyle=ctx.strokeStyle;ctx.beginPath();ctx.arc(p.x,p.y,12,0,Math.PI*2);ctx.stroke();ctx.fillText('GT '+bird,p.x+17,p.y);}}}
async function inspect(){try{if(dirty&&!confirm('Discard unsaved clicks?'))return;loaded=false;message('Checking image and GT…');const b=Number($('block').value),f=Number($('frame').value);meta=await api(`/api/frame?block=${b}&frame=${f}`);pictures={};await Promise.all(['raw','gt'].map(id=>new Promise((resolve,reject)=>{const img=new Image();img.onload=()=>{pictures[id]=img;resolve();};img.onerror=()=>reject(Error('Image failed to load'));img.src=`/image?frame=${f}&kind=${id}`;})));points={};if(meta.saved&&meta.saved.video_frame===f)for(const p of meta.saved.points)points[p.gt_id]=p;dirty=false;$('reviewed').checked=false;loaded=true;draw();$('range').textContent=`Allowed: ${meta.first}–${meta.last}`;$('status').textContent=`Video ${f} ↔ GT/FRAMES ${meta.gt_frame}. Block ${b}.`;message(meta.masks.map(m=>`GT ${m.gt_id}: ${m.area} pixels; ${m.components} components; largest component ${(100*m.largest_fraction).toFixed(1)}%${m.touches_edge?'; touches image edge':''}${m.area<50?'; WARNING: missing/tiny mask':''}`).join('\n')+(meta.other_ids.length?'\nOther GT IDs: '+meta.other_ids.join(', '):'')+'\nInspect birds, blur, overlapping masks and identities visually.');}catch(e){loaded=false;message(e.message);}}
function middle(){const b=Number($('block').value),lo=b*config.size,hi=Math.min(lo+config.size,config.total)-1;$('frame').value=Math.floor((lo+hi)/2);inspect();}
for(const id of ['raw','gt'])$(id).onclick=e=>{if(!loaded)return;const rect=$(id).getBoundingClientRect(),bird=Number(document.querySelector('input[name=bird]:checked').value);points[bird]={gt_id:bird,x:(e.clientX-rect.left)*meta.width/rect.width,y:(e.clientY-rect.top)*meta.height/rect.height};dirty=true;$('reviewed').checked=false;draw();};
$('save').onclick=async()=>{try{if(!loaded)throw Error('Inspect a frame first');const result=await api('/api/save',{block:meta.block,frame:meta.video_frame,points:Object.values(points),reviewed:$('reviewed').checked});dirty=false;message('Saved. Completed blocks: '+result.saved_blocks.join(', ')+'. '+result.status);}catch(e){message(e.message);}};
$('clear').onclick=()=>{points={};dirty=true;$('reviewed').checked=false;if(loaded)draw();};$('load').onclick=inspect;$('middle').onclick=middle;$('block').onchange=middle;
for(const [id,step]of [['back',-25],['next',25]])$(id).onclick=()=>{const lo=Number($('block').value)*config.size,hi=Math.min(lo+config.size,config.total)-1;$('frame').value=Math.max(lo,Math.min(hi,Number($('frame').value)+step));inspect();};
window.onbeforeunload=()=>dirty?'Unsaved clicks':undefined;
async function selectBlock(){try{config=await api('/api/config');const saved=config.saved[$('block').value];if(saved!==undefined){$('frame').value=saved;inspect();}else middle();}catch(e){message(e.message);}}
$('block').onchange=selectBlock;
(async()=>{try{config=await api('/api/config');for(let b=0;b<config.blocks;b++)$('block').add(new Option(b,b));ids.forEach((bird,i)=>{const label=document.createElement('label');label.style.color=colors[i];label.innerHTML=`<input type="radio" name="bird" value="${bird}" ${i===0?'checked':''}>GT ${bird}`;$('targets').append(label);});selectBlock();}catch(e){message(e.message);}})();
</script>'''


def serve(review, port):
    token=secrets.token_hex(24)
    class Handler(BaseHTTPRequestHandler):
        def respond(self, value, code=200, kind='application/json'):
            data=json.dumps(value).encode() if kind=='application/json' else value
            self.send_response(code);self.send_header('Content-Type',kind)
            self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(data)))
            self.end_headers();self.wfile.write(data)
        def do_GET(self):
            try:
                url=urlparse(self.path);q=parse_qs(url.query)
                if url.path=='/': return self.respond(HTML.replace('TOKEN',token).encode(),kind='text/html; charset=utf-8')
                if url.path=='/api/config':return self.respond(dict(size=review.size,total=review.total,blocks=review.num_blocks,
                    saved={b:r['video_frame'] for b,r in review.data['blocks'].items()}))
                if url.path=='/api/frame':return self.respond(review.metadata(int(q['block'][0]),int(q['frame'][0])))
                if url.path=='/image':
                    rgb,gt,_=review.frame(int(q['frame'][0]));rgb=rgb.copy()
                    if q['kind'][0]=='gt':
                        for bird,color in zip(IDS,COLORS):
                            tint=np.array([int(color[i:i+2],16) for i in [1,3,5]])
                            mask=gt==bird;rgb[mask]=(.55*rgb[mask]+.45*tint).astype(np.uint8)
                    data=io.BytesIO();Image.fromarray(rgb).save(data,format='PNG',compress_level=1)
                    return self.respond(data.getvalue(),kind='image/png')
                self.respond(dict(error='Not found'),404)
            except Exception as exc:self.respond(dict(error=str(exc)),400)
        def do_POST(self):
            try:
                if self.path!='/api/save' or self.headers.get('X-Review-Token')!=token:
                    return self.respond(dict(error='Invalid save request'),403)
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<10000:raise ValueError('Invalid request size')
                self.respond(review.save(json.loads(self.rfile.read(length))))
            except Exception as exc:self.respond(dict(error=str(exc)),400)
        def log_message(self,*args):pass
    server=HTTPServer(('127.0.0.1',port),Handler)
    print(f'Open http://127.0.0.1:{port} in the browser ON LUMEN.',flush=True)
    print(f'Annotations: {review.output}/annotations.json',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--prior',type=Path,required=True)
    parser.add_argument('--port',type=int,default=8765)
    parser.add_argument('--resume',type=Path,help='Existing reannotation result directory')
    args=parser.parse_args()
    root=Path(__file__).resolve().parent/'results'
    output=args.resume.resolve() if args.resume else root/('reannotation-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    if not output.is_relative_to(root) or output==root:
        raise ValueError('Review results must stay inside this checkout audit/results directory')
    if args.resume and not (output/'annotations.json').is_file():
        raise ValueError('Resume requires an existing annotations.json')
    serve(Review(args.source,args.prior,output),args.port)
