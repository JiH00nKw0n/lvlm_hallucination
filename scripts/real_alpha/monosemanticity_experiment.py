"""Monosemanticity experiment: does collapse hurt MS?

Compares MS (MonoSemanticity score) of:
  - Single SAE collapsed latents (self-matched, cos=1)
  - Single SAE C>=0.2 latents (all Hungarian-alive)
  - Two SAE corresponding latents

--emb-type controls which embedding space is used for MS:
  clip  : CLIP embeddings (already cached, fast — but same space as SAE)
  dino  : DINOv2-base image + E5-base-v2 text (independent, slower)

Usage:
    python scripts/real_alpha/monosemanticity_experiment.py \
        --followup 2 --cache-dir cache/clip_b32_coco \
        --out outputs/real_alpha_followup_2/monosemanticity \
        --device cuda --emb-type clip
"""
from __future__ import annotations
import argparse,json,gc,sys
from pathlib import Path
import numpy as np,torch,torch.nn.functional as F
from tqdm import tqdm
from safetensors.torch import load_file
from scipy.optimize import linear_sum_assignment
sys.path.insert(0,str(Path(__file__).resolve().parent))
import _bootstrap
from src.datasets.cached_clip_pairs import CachedClipPairsDataset
from src.models.modeling_sae import TopKSAE,TwoSidedTopKSAE

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--followup",type=str,default="2")
    p.add_argument("--cache-dir",type=str,default="cache/clip_b32_coco")
    p.add_argument("--out",type=str,required=True)
    p.add_argument("--device",type=str,default="cuda")
    p.add_argument("--batch-size",type=int,default=128)
    p.add_argument("--c-thr",type=float,default=0.2)
    p.add_argument("--emb-type",type=str,default="clip",choices=["clip","dino"])
    return p.parse_args()

# ============ Phase 1: Embedding caches ============

def load_clip_caches(cache_dir):
    """Load pre-computed CLIP embeddings (same space as SAE — fast baseline)."""
    img_path=Path(cache_dir)/"image_embeddings.pt"
    txt_path=Path(cache_dir)/"text_embeddings.pt"
    print(f"loading CLIP image cache from {img_path}",flush=True)
    img_cache=torch.load(img_path,map_location="cpu")
    print(f"  {len(img_cache)} images",flush=True)
    print(f"loading CLIP text cache from {txt_path}",flush=True)
    txt_cache=torch.load(txt_path,map_location="cpu")
    print(f"  {len(txt_cache)} texts",flush=True)
    return img_cache,txt_cache

def build_img_emb_cache(cache_dir,device,batch_size=256):
    path=Path(cache_dir)/"dino_embeddings.pt"
    if path.exists():
        print(f"loading DINOv2 cache from {path}",flush=True)
        return torch.load(path,map_location="cpu")
    print("building DINOv2-base cache...",flush=True)
    from transformers import AutoImageProcessor,AutoModel
    from datasets import load_dataset
    proc=AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model=AutoModel.from_pretrained("facebook/dinov2-base").to(device).eval()
    hf=load_dataset("namkha1032/coco-karpathy",split="train")
    ids_only=hf.select_columns(["image_id"])
    id2row={}
    for i in range(len(ids_only)):
        iid=str(ids_only[i]["image_id"])
        if iid not in id2row: id2row[iid]=i
    unique_ids=list(id2row.keys())
    cache={}
    print(f"  unique images: {len(unique_ids)}, id2row: {len(id2row)}",flush=True)
    total_dino=(len(unique_ids)+batch_size-1)//batch_size
    for bi in range(total_dino):
        s=bi*batch_size
        if bi%10==0:print(f"  DINOv2 {bi}/{total_dino}",flush=True)
        batch_ids=unique_ids[s:s+batch_size]
        pils=[hf[id2row[iid]]["image"].convert("RGB") for iid in batch_ids]
        inputs=proc(images=pils,return_tensors="pt").to(device)
        with torch.no_grad():
            out=model(**inputs)
        cls=out.last_hidden_state[:,0,:].cpu().float()
        for i,iid in enumerate(batch_ids):
            cache[int(iid)]=cls[i]
    del model,proc; gc.collect(); torch.cuda.empty_cache()
    torch.save(cache,path)
    print(f"saved DINOv2 cache: {len(cache)} images",flush=True)
    return cache

def build_txt_emb_cache(cache_dir,device,batch_size=256):
    path=Path(cache_dir)/"e5_embeddings.pt"
    if path.exists():
        print(f"loading E5 cache from {path}",flush=True)
        return torch.load(path,map_location="cpu")
    print("building E5-base cache...",flush=True)
    from transformers import AutoTokenizer,AutoModel
    tok=AutoTokenizer.from_pretrained("intfloat/e5-base-v2")
    model=AutoModel.from_pretrained("intfloat/e5-base-v2").to(device).eval()
    splits=json.load(open(Path(cache_dir)/"splits.json"))
    from datasets import load_dataset
    hf_text=load_dataset("namkha1032/coco-karpathy",split="train").select_columns(["image_id","captions"])
    id2row_t={}
    for i in range(len(hf_text)):
        iid=str(hf_text[i]["image_id"])
        if iid not in id2row_t: id2row_t[iid]=i
    print(f"  E5 id2row built: {len(id2row_t)}",flush=True)
    cache={}
    pairs=splits["train"]
    texts=[]
    keys=[]
    for iid,ci in pairs:
        key=f"{iid}_{ci}"
        if key in cache:continue
        row=hf_text[id2row_t[str(iid)]]
        cap=row["captions"][int(ci)] if int(ci)<len(row["captions"]) else ""
        texts.append("query: "+cap)
        keys.append(key)
    print(f"  {len(texts)} texts to encode",flush=True)
    total_e5=(len(texts)+batch_size-1)//batch_size
    for bi in range(total_e5):
        s=bi*batch_size
        if bi%20==0:print(f"  E5 {bi}/{total_e5}",flush=True)
        batch_texts=texts[s:s+batch_size]
        batch_keys=keys[s:s+batch_size]
        inputs=tok(batch_texts,padding=True,truncation=True,max_length=128,return_tensors="pt").to(device)
        with torch.no_grad():
            out=model(**inputs)
        mask=inputs["attention_mask"].unsqueeze(-1).float()
        emb=((out.last_hidden_state*mask).sum(1)/mask.sum(1)).cpu().float()
        for i,k in enumerate(batch_keys):
            cache[k]=emb[i]
    del model,tok; gc.collect(); torch.cuda.empty_cache()
    torch.save(cache,path)
    print(f"saved E5 cache: {len(cache)} texts",flush=True)
    return cache

# ============ Phase 2+3: Collect activations + compute MS ============

def compute_ms_weighted(activations:np.ndarray, embeddings:torch.Tensor)->float:
    """MS score (Eq 9) using efficient O(N*d) trick.
    activations: (N,) float, embeddings: (N, d) float, both on CPU."""
    if len(activations)<2:return float("nan")
    acts=torch.from_numpy(activations).float()
    emb=F.normalize(embeddings.float(),dim=-1)
    eps=1e-8
    a_min,a_max=acts.min(),acts.max()
    a_tilde=(acts-a_min)/(a_max-a_min+eps)
    weighted_emb=(a_tilde.unsqueeze(-1)*emb).sum(dim=0)
    aTSa=(weighted_emb@weighted_emb).item()
    aTa=(a_tilde**2).sum().item()
    sum_a=a_tilde.sum().item()
    num=aTSa-aTa
    den=sum_a**2-aTa
    if den<eps:return float("nan")
    return num/den

def main():
    args=parse_args()
    fu=args.followup
    dev=torch.device(args.device if torch.cuda.is_available() else "cpu")
    one_dir=f"outputs/real_alpha_followup_{fu}/one_sae/final"
    two_dir=f"outputs/real_alpha_followup_{fu}/two_sae/final"
    out_base=Path(args.out)
    out_base.parent.mkdir(parents=True,exist_ok=True)

    train_pairs=json.load(open(Path(args.cache_dir)/"splits.json"))["train"]
    N=len(train_pairs)
    print(f"N={N}",flush=True)

    # ===== Single SAE: load cached C + Hungarian =====
    one_d=Path(one_dir)
    C1=np.load(one_d/"diagnostic_B_C_train.npy")
    rates1=np.load(one_d/"diagnostic_B_firing_rates.npz")
    ai1=np.where(rates1["rate_i"]>0.001)[0];at1=np.where(rates1["rate_t"]>0.001)[0]
    one_model=TopKSAE.from_pretrained(one_dir).to(dev).eval()
    W1n=one_model.W_dec.detach().cpu().float().numpy();W1n/=(np.linalg.norm(W1n,axis=1,keepdims=True)+1e-12)
    r1,c1=linear_sum_assignment(-C1[np.ix_(ai1,at1)])
    oi1=ai1[r1];oj1=at1[c1];cos1=(W1n[oi1]*W1n[oj1]).sum(1)
    Cm1=C1[np.ix_(ai1,at1)][r1,c1]
    self_mask=oi1==oj1;self_idx=np.where(self_mask)[0]
    c02_mask=Cm1>=args.c_thr;c02_idx=np.where(c02_mask)[0]
    print(f"Single: alive_i={len(ai1)}, alive_t={len(at1)}, self={len(self_idx)}, C>={args.c_thr}={len(c02_idx)}",flush=True)
    del C1,rates1; gc.collect()

    # ===== Two SAE: load cached C + Hungarian =====
    two_d=Path(two_dir)
    C2=np.load(two_d/"diagnostic_B_C_train.npy")
    rates2=np.load(two_d/"diagnostic_B_firing_rates.npz")
    ai2=np.where(rates2["rate_i"]>0.001)[0];at2=np.where(rates2["rate_t"]>0.001)[0]
    sd2=load_file(two_d/"model.safetensors")
    Wi=sd2["image_sae.W_dec"].float().numpy();Wt=sd2["text_sae.W_dec"].float().numpy()
    r2,c2=linear_sum_assignment(-C2[np.ix_(ai2,at2)])
    oi2=ai2[r2];oj2=at2[c2]
    Cm2=C2[np.ix_(ai2,at2)][r2,c2]
    two_c02=np.where(Cm2>=args.c_thr)[0]
    del C2,sd2,rates2; gc.collect()
    two_model=TwoSidedTopKSAE.from_pretrained(two_dir).to(dev).eval()
    print(f"Two: C>={args.c_thr}={len(two_c02)}",flush=True)

    # ===== Collect target latent activations (streaming, cached) =====
    single_targets_img=set(int(oi1[i]) for i in c02_idx)
    single_targets_txt=set(int(oj1[i]) for i in c02_idx)
    two_targets_img=set(int(oi2[i]) for i in two_c02)
    two_targets_txt=set(int(oj2[i]) for i in two_c02)

    acts_cache=out_base.parent/f"mono_acts_fu{fu}.pt"
    if acts_cache.exists():
        print(f"loading cached activations from {acts_cache}",flush=True)
        _ac=torch.load(acts_cache,map_location="cpu")
        s1_img_acts=_ac["s1_img"];s1_txt_acts=_ac["s1_txt"]
        t2_img_acts=_ac["t2_img"];t2_txt_acts=_ac["t2_txt"]
        del _ac
    else:
        ds=CachedClipPairsDataset(args.cache_dir,split="train",l2_normalize=True)
        N_=len(ds)
        img=torch.stack([ds._image_dict[int(p[0])] for p in ds.pairs])
        txt=torch.stack([ds._text_dict[f"{int(p[0])}_{int(p[1])}"] for p in ds.pairs])
        del ds; gc.collect()

        one_model=TopKSAE.from_pretrained(one_dir).to(dev).eval()
        two_model=TwoSidedTopKSAE.from_pretrained(two_dir).to(dev).eval()

        s1_img_acts={k:[] for k in single_targets_img}
        s1_txt_acts={k:[] for k in single_targets_txt}
        t2_img_acts={k:[] for k in two_targets_img}
        t2_txt_acts={k:[] for k in two_targets_txt}

        print("streaming activations...",flush=True)
        BS=1024
        with torch.no_grad():
            for s in range(0,N_,BS):
                e=min(s+BS,N_)
                hi=img[s:e].unsqueeze(1).to(dev);ht=txt[s:e].unsqueeze(1).to(dev)
                z1i=one_model(hidden_states=hi,return_dense_latents=True).dense_latents.squeeze(1).cpu().numpy()
                z1t=one_model(hidden_states=ht,return_dense_latents=True).dense_latents.squeeze(1).cpu().numpy()
                z2i=two_model.image_sae(hidden_states=hi,return_dense_latents=True).dense_latents.squeeze(1).cpu().numpy()
                z2t=two_model.text_sae(hidden_states=ht,return_dense_latents=True).dense_latents.squeeze(1).cpu().numpy()
                for b in range(z1i.shape[0]):
                    sid=s+b
                    for k in single_targets_img:
                        v=z1i[b,k]
                        if v>0:s1_img_acts[k].append((float(v),sid))
                    for k in single_targets_txt:
                        v=z1t[b,k]
                        if v>0:s1_txt_acts[k].append((float(v),sid))
                    for k in two_targets_img:
                        v=z2i[b,k]
                        if v>0:t2_img_acts[k].append((float(v),sid))
                    for k in two_targets_txt:
                        v=z2t[b,k]
                        if v>0:t2_txt_acts[k].append((float(v),sid))
                if (s//BS)%50==0:print(f"  act batch {s//BS}/{(N_+BS-1)//BS}",flush=True)
        del one_model,two_model,img,txt; gc.collect(); torch.cuda.empty_cache()
        torch.save({"s1_img":s1_img_acts,"s1_txt":s1_txt_acts,
                     "t2_img":t2_img_acts,"t2_txt":t2_txt_acts},acts_cache)
        print(f"saved activation cache to {acts_cache}",flush=True)

    # ===== Phase 1b: load embedding caches =====
    if args.emb_type=="clip":
        img_emb_cache,txt_emb_cache=load_clip_caches(args.cache_dir)
    else:
        img_emb_cache=build_dino_cache(args.cache_dir,dev)
        txt_emb_cache=build_e5_cache(args.cache_dir,dev)

    # ===== Phase 3: Compute MS per latent =====
    def ms_for_acts(acts_list, emb_cache, get_key):
        """Compute MS for a list of (activation, sample_idx)."""
        if len(acts_list)<2:return float("nan")
        activations=np.array([a for a,_ in acts_list])
        keys=[get_key(sid) for _,sid in acts_list]
        embs=[]
        valid_acts=[]
        for i,(a,key) in enumerate(zip(activations,keys)):
            if key in emb_cache:
                embs.append(emb_cache[key])
                valid_acts.append(a)
        if len(embs)<2:return float("nan")
        return compute_ms_weighted(np.array(valid_acts),torch.stack(embs))

    def img_key(sid):return int(train_pairs[sid][0])
    def txt_key(sid):return f"{train_pairs[sid][0]}_{train_pairs[sid][1]}"

    print("computing MS scores...",flush=True)
    results=[]
    for pi in c02_idx:
        li=int(oi1[pi]);lj=int(oj1[pi])
        is_self=bool(self_mask[pi])
        c_val=float(Cm1[pi]);cos_val=float(cos1[pi])
        ms_img=ms_for_acts(s1_img_acts.get(li,[]),img_emb_cache,img_key)
        ms_txt=ms_for_acts(s1_txt_acts.get(lj,[]),txt_emb_cache,txt_key)

        # Find best matching two SAE pair (by Jaccard from earlier, or just use index)
        # Simple: match by largest C in two SAE for same rough concept — skip for now,
        # just record single side
        results.append(dict(
            type="single",collapsed=is_self,
            img_latent=li,txt_latent=lj,C=c_val,cos=cos_val,
            ms_img=ms_img,ms_txt=ms_txt,
            n_img=len(s1_img_acts.get(li,[])),n_txt=len(s1_txt_acts.get(lj,[]))))

    for pi in two_c02:
        li=int(oi2[pi]);lj=int(oj2[pi])
        c_val=float(Cm2[pi])
        ms_img=ms_for_acts(t2_img_acts.get(li,[]),img_emb_cache,img_key)
        ms_txt=ms_for_acts(t2_txt_acts.get(lj,[]),txt_emb_cache,txt_key)
        results.append(dict(
            type="two",collapsed=False,
            img_latent=li,txt_latent=lj,C=c_val,cos=0,
            ms_img=ms_img,ms_txt=ms_txt,
            n_img=len(t2_img_acts.get(li,[])),n_txt=len(t2_txt_acts.get(lj,[]))))

    # ===== Phase 4: Summary =====
    json.dump(results,open(str(out_base)+".json","w"),indent=2)
    print(f"saved {out_base}.json ({len(results)} entries)",flush=True)

    # Summary stats
    def stats(arr):
        arr=[x for x in arr if not np.isnan(x)]
        if not arr:return {"n":0}
        return {"n":len(arr),"mean":float(np.mean(arr)),"median":float(np.median(arr)),"std":float(np.std(arr))}

    s_col=[r for r in results if r["type"]=="single" and r["collapsed"]]
    s_noncol=[r for r in results if r["type"]=="single" and not r["collapsed"]]
    t_all=[r for r in results if r["type"]=="two"]

    print(f"\n{'group':>20} | {'n':>4} | {'MS_img mean':>11} | {'MS_img med':>10} | {'MS_txt mean':>11} | {'MS_txt med':>10}")
    print("-"*80)
    for label,group in [("single collapsed",s_col),("single non-col",s_noncol),("two SAE",t_all)]:
        si=stats([r["ms_img"] for r in group])
        st=stats([r["ms_txt"] for r in group])
        print(f"{label:>20} | {si.get('n',0):>4} | {si.get('mean',0):>11.4f} | {si.get('median',0):>10.4f} | {st.get('mean',0):>11.4f} | {st.get('median',0):>10.4f}")

    summary={"single_collapsed_img":stats([r["ms_img"] for r in s_col]),
             "single_collapsed_txt":stats([r["ms_txt"] for r in s_col]),
             "single_noncollapsed_img":stats([r["ms_img"] for r in s_noncol]),
             "single_noncollapsed_txt":stats([r["ms_txt"] for r in s_noncol]),
             "two_img":stats([r["ms_img"] for r in t_all]),
             "two_txt":stats([r["ms_txt"] for r in t_all])}
    json.dump(summary,open(str(out_base)+"_summary.json","w"),indent=2)
    print(f"saved {out_base}_summary.json",flush=True)

if __name__=="__main__":
    main()
