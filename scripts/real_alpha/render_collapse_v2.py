"""Collapsed pair comparison: single vs two SAE. Shared/single-only/two-only 3x3 grids."""
from __future__ import annotations
import base64,io,json,sys,heapq,gc
from pathlib import Path
import numpy as np,torch
from PIL import Image
from safetensors.torch import load_file
from scipy.optimize import linear_sum_assignment
sys.path.insert(0,str(Path(__file__).resolve().parent))
import _bootstrap
from src.datasets.cached_clip_pairs import CachedClipPairsDataset
from src.models.modeling_sae import TopKSAE,TwoSidedTopKSAE

def b64(pil,h=100):
    w,oh=pil.size;pil=pil.resize((int(w*h/oh),h),Image.LANCZOS)
    buf=io.BytesIO();pil.save(buf,format="JPEG",quality=85)
    return base64.b64encode(buf.getvalue()).decode()

fu=sys.argv[1];out_path=sys.argv[2]
one_dir=f"outputs/real_alpha_followup_{fu}/one_sae/final"
two_dir=f"outputs/real_alpha_followup_{fu}/two_sae/final"
dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load embeddings
ds=CachedClipPairsDataset("cache/clip_b32_coco",split="train",l2_normalize=True)
N=len(ds)
img=torch.stack([ds._image_dict[int(p[0])] for p in ds.pairs])
txt=torch.stack([ds._text_dict[f"{int(p[0])}_{int(p[1])}"] for p in ds.pairs])
train_pairs=json.load(open("cache/clip_b32_coco/splits.json"))["train"]
del ds._image_dict,ds._text_dict; gc.collect()
print(f"N={N}",flush=True)

# ===== PHASE 1: Single SAE — compute C, Hungarian, find self-matches =====
one_model=TopKSAE.from_pretrained(one_dir).to(dev).eval()
L=one_model.latent_size
si=np.zeros(L,dtype=np.float64);st=np.zeros(L,dtype=np.float64)
sii=np.zeros(L,dtype=np.float64);stt=np.zeros(L,dtype=np.float64)
sit=np.zeros((L,L),dtype=np.float64)
fi=np.zeros(L,dtype=np.int64);ft=np.zeros(L,dtype=np.int64);cnt=0
with torch.no_grad():
    for s in range(0,N,2048):
        zi=one_model(hidden_states=img[s:s+2048].unsqueeze(1).to(dev),return_dense_latents=True).dense_latents.squeeze(1).cpu().double().numpy()
        zt=one_model(hidden_states=txt[s:s+2048].unsqueeze(1).to(dev),return_dense_latents=True).dense_latents.squeeze(1).cpu().double().numpy()
        B=zi.shape[0];si+=zi.sum(0);st+=zt.sum(0);sii+=(zi*zi).sum(0);stt+=(zt*zt).sum(0)
        sit+=zi.T@zt;fi+=(zi>0).sum(0);ft+=(zt>0).sum(0);cnt+=B
mi=si/cnt;mt=st/cnt
C1=np.nan_to_num((sit/cnt-np.outer(mi,mt))/np.sqrt(np.clip((sii/cnt-mi**2)[:,None]*(stt/cnt-mt**2)[None,:],1e-16,None)),nan=0.0)
ai1=np.where(fi/cnt>0.001)[0];at1=np.where(ft/cnt>0.001)[0]
W1n=one_model.W_dec.detach().cpu().float().numpy();W1n=W1n/(np.linalg.norm(W1n,axis=1,keepdims=True)+1e-12)
r1,c1=linear_sum_assignment(-C1[np.ix_(ai1,at1)])
oi1=ai1[r1];oj1=at1[c1];cos1=(W1n[oi1]*W1n[oj1]).sum(1);Cm1=C1[np.ix_(ai1,at1)][r1,c1]
self_mask=oi1==oj1;self_idx=np.where(self_mask)[0]
print(f"Single self-match: {len(self_idx)}",flush=True)
# Free C matrix
del C1,sit,si,st,sii,stt,mi,mt,fi,ft; gc.collect()

# ===== PHASE 2: Two SAE — load cached C, Hungarian =====
two_d=Path(two_dir)
C2=np.load(two_d/"diagnostic_B_C_train.npy")
rates2=np.load(two_d/"diagnostic_B_firing_rates.npz")
ai2=np.where(rates2["rate_i"]>0.001)[0];at2=np.where(rates2["rate_t"]>0.001)[0]
sd2=load_file(two_d/"model.safetensors")
Wi=sd2["image_sae.W_dec"].float().numpy();Wt=sd2["text_sae.W_dec"].float().numpy()
Win=Wi/(np.linalg.norm(Wi,axis=1,keepdims=True)+1e-12);Wtn=Wt/(np.linalg.norm(Wt,axis=1,keepdims=True)+1e-12)
r2,c2=linear_sum_assignment(-C2[np.ix_(ai2,at2)])
oi2=ai2[r2];oj2=at2[c2];cos2=(Win[oi2]*Wtn[oj2]).sum(1);Cm2=C2[np.ix_(ai2,at2)][r2,c2]
two_c02=np.where(Cm2>=0.2)[0]
del C2,sd2,Wi,Wt,Win,Wtn,rates2; gc.collect()
print(f"Two C>=0.2: {len(two_c02)}",flush=True)

# ===== PHASE 3: Stream top-50 (GPU, both models) =====
two_model=TwoSidedTopKSAE.from_pretrained(two_dir).to(dev).eval()
# one_model still on GPU from phase 1
top_k=50
oh={i:[] for i in self_idx};th={i:[] for i in two_c02}
print("streaming...",flush=True)
with torch.no_grad():
    for s in range(0,N,1024):
        e=min(s+1024,N);hi=img[s:e].unsqueeze(1).to(dev);ht=txt[s:e].unsqueeze(1).to(dev)
        z1i=one_model(hidden_states=hi,return_dense_latents=True).dense_latents.squeeze(1).cpu().numpy()
        z1t=one_model(hidden_states=ht,return_dense_latents=True).dense_latents.squeeze(1).cpu().numpy()
        z2i=two_model.image_sae(hidden_states=hi,return_dense_latents=True).dense_latents.squeeze(1).cpu().numpy()
        z2t=two_model.text_sae(hidden_states=ht,return_dense_latents=True).dense_latents.squeeze(1).cpu().numpy()
        for pi in self_idx:
            li=int(oi1[pi]);sc=np.minimum(z1i[:,li],z1t[:,li])
            for bi in range(len(sc)):
                if len(oh[pi])<top_k:heapq.heappush(oh[pi],(sc[bi],s+bi))
                elif sc[bi]>oh[pi][0][0]:heapq.heapreplace(oh[pi],(sc[bi],s+bi))
        for pi in two_c02:
            li=int(oi2[pi]);lj=int(oj2[pi]);sc=np.minimum(z2i[:,li],z2t[:,lj])
            for bi in range(len(sc)):
                if len(th[pi])<top_k:heapq.heappush(th[pi],(sc[bi],s+bi))
                elif sc[bi]>th[pi][0][0]:heapq.heapreplace(th[pi],(sc[bi],s+bi))
        if (s//1024)%100==0:print(f"  batch {s//1024}/{(N+1023)//1024}",flush=True)

del one_model,two_model,img,txt; gc.collect(); torch.cuda.empty_cache()
print("streaming done",flush=True)

os_={pi:set(sid for _,sid in oh[pi]) for pi in self_idx}
ts_={pi:set(sid for _,sid in th[pi]) for pi in two_c02}

def uimgs(sids,n=9):
    seen=set();out=[]
    for sid in sorted(sids):
        iid=train_pairs[sid][0]
        if iid not in seen:seen.add(iid);out.append((str(iid),int(train_pairs[sid][1])))
        if len(out)>=n:break
    return out

concepts=[]
for pi in self_idx:
    best_j=0;best_p2=None
    for p2 in two_c02:
        inter=len(os_[pi]&ts_[p2]);union=len(os_[pi]|ts_[p2])
        j=inter/union if union>0 else 0
        if j>best_j:best_j=j;best_p2=p2
    shared=os_[pi]&ts_[best_p2] if best_p2 else set()
    s_only=os_[pi]-(ts_[best_p2] if best_p2 else set())
    t_only=(ts_[best_p2] if best_p2 else set())-os_[pi]
    concepts.append(dict(
        single_lat=int(oi1[pi]),C_one=float(Cm1[pi]),cos_one=float(cos1[pi]),
        two_pair=[int(oi2[best_p2]),int(oj2[best_p2])] if best_p2 else None,
        C_two=float(Cm2[best_p2]) if best_p2 else 0,cos_two=float(cos2[best_p2]) if best_p2 else 0,
        jaccard=best_j,n_shared=len(shared),
        imgs_shared=uimgs(shared),imgs_single=uimgs(s_only),imgs_two=uimgs(t_only)))

# ===== PHASE 4: Render HTML =====
print("loading COCO images...",flush=True)
from datasets import load_dataset
hf=load_dataset("namkha1032/coco-karpathy",split="train")
id2r={str(hf[i]["image_id"]):i for i in range(len(hf))}
print(f"id2r: {len(id2r)}",flush=True)

def grid(img_list,label):
    if not img_list:return f"<div class='section'><h5>{label} (none)</h5></div>"
    h=[f"<div class='section'><h5>{label}</h5><div class='grid'>"]
    for iid,ci in img_list:
        ri=id2r.get(iid)
        if ri is None:continue
        row=hf[ri];pil=row["image"].convert("RGB")
        cap=row["captions"][ci] if ci<len(row["captions"]) else ""
        h.append(f"<div class='cell'><img src='data:image/jpeg;base64,{b64(pil)}'>"
                 f"<div class='cap'>{cap[:60]}</div></div>")
    h.append("</div></div>");return "\n".join(h)

html=[
"<!DOCTYPE html><html><head><meta charset='utf-8'>",
"<style>",
"body{font-family:'Helvetica Neue',sans-serif;margin:20px;font-size:12px}",
".concept{border:1px solid #ddd;padding:10px;margin-bottom:20px;border-radius:6px;page-break-inside:avoid}",
".concept h3{margin:0 0 6px;font-size:14px}",
".meta{font-size:11px;color:#888}",
".section{margin-top:8px}",
".section h5{margin:4px 0;font-size:11px;color:#555}",
".grid{display:grid;grid-template-columns:repeat(3,1fr);gap:4px}",
".cell{text-align:center}",
".cell img{height:90px;border-radius:3px}",
".cap{font-size:9px;color:#999;max-width:160px;word-wrap:break-word}",
"</style></head><body>",
f"<h1>Collapsed pairs: Single vs Two SAE (followup_{fu})</h1>",
]
matched=[c for c in concepts if c["jaccard"]>=0.1]
matched.sort(key=lambda c:-c["jaccard"])
for i,c in enumerate(matched):
    html.append("<div class='concept'>")
    html.append(f"<h3>#{i+1} Jaccard={c['jaccard']:.3f}</h3>")
    html.append(f"<p class='meta'><b>Single</b>: latent #{c['single_lat']}, C={c['C_one']:.3f}, cos={c['cos_one']:.4f} (merged)</p>")
    if c["two_pair"]:
        html.append(f"<p class='meta'><b>Two</b>: Img #{c['two_pair'][0]} / Txt #{c['two_pair'][1]}, C={c['C_two']:.3f}, cos={c['cos_two']:.4f}</p>")
    html.append(grid(c["imgs_shared"],"Shared"))
    html.append(grid(c["imgs_single"],"Single-only"))
    html.append(grid(c["imgs_two"],"Two-only"))
    html.append("</div>")
html.append("</body></html>")
Path(out_path).write_text("\n".join(html),encoding="utf-8")
print(f"saved {out_path}",flush=True)
