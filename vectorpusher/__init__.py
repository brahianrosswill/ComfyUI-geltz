# vectorpusher

import torch
import comfy.model_management as model_management

EPS=1e-8

@torch.no_grad()
def _normalize(x, dim=-1, eps=EPS):
    return x/(torch.norm(x, dim=dim, keepdim=True)+eps)

@torch.no_grad()
def _get_embeddings(clip_model):
    return clip_model.transformer.text_model.embeddings.token_embedding.weight.to(device=model_management.get_torch_device())

@torch.no_grad()
def _sched(s):
    k=int(round(8+12*s))
    tau=6.0+6.0*s
    step_scale=0.06+0.14*s
    theta_max_deg=8.0+12.0*s
    return {"k":k,"tau":tau,"step_scale":step_scale,"theta_max_deg":theta_max_deg}

@torch.no_grad()
def winsorized_alpha(vals,tau,clip_c=1.5):
    z=tau*vals
    m=z.median()
    mad=(z-m).abs().median()+1e-8
    z=z.clamp(m-clip_c*mad,m+clip_c*mad)
    return torch.softmax(z,dim=0)

@torch.no_grad()
def batched_topk_neighbors(w0,Wn,k=16,tau=8.0):
    w0n=_normalize(w0,dim=0)
    sims=(Wn@w0n)
    sims[(sims!=sims)]=-1.0
    vals,idx=torch.topk(sims,k=k+1,largest=True)
    if torch.allclose(Wn[idx[0]],w0n,atol=1e-6):
        vals,idx=vals[1:],idx[1:]
    alpha=winsorized_alpha(vals,tau)
    return idx,alpha

@torch.no_grad()
def trust_region_step(w0,Wn,idx,alpha,theta_max_deg=12.0,step_scale=0.15):
    w0n=_normalize(w0,dim=0)
    w0_mag=float(torch.norm(w0))
    t=(Wn[idx]*alpha[:,None]).sum(dim=0)
    t=t-(t@w0n)*w0n
    nt=torch.norm(t)
    if nt<EPS:
        return w0
    t=t/(nt+EPS)
    w_star=w0+step_scale*w0_mag*t
    cosang=torch.dot(_normalize(w0,dim=0),_normalize(w_star,dim=0)).clamp(-1,1).item()
    ang=float(torch.arccos(torch.tensor(cosang)))*180.0/3.141592653589793
    if ang>theta_max_deg:
        scale=theta_max_deg/(ang+1e-8)
        w_star=w0+scale*(w_star-w0)
    return _normalize(w_star,dim=0)*w0_mag

@torch.no_grad()
def _entropy_conf(alpha):
    H=-(alpha*(alpha.clamp_min(1e-12)).log()).sum()
    denom=torch.log(torch.tensor(float(alpha.numel()),device=alpha.device))+1e-12
    conf=1.0-(H/denom)
    return float(conf.clamp(0.0,1.0))

@torch.no_grad()
def _att_scalar(attn):
    if isinstance(attn,(float,int)):
        a=float(attn)
    elif torch.is_tensor(attn):
        a=float(attn.float().mean().item())
    else:
        a=1.0
    if a!=a:
        a=1.0
    return max(0.0,min(1.0,a))

@torch.no_grad()
def _kl(a,b):
    return float((a*(a.add(EPS).log()-b.add(EPS).log())).sum())

@torch.no_grad()
def blended_kl_bounded_step(w0,Wn,sched,att,kappa=0.03):
    k0=sched["k"]
    ks=[max(4,int(round(0.5*k0))),k0,max(5,int(round(1.5*k0)))]
    taus=[sched["tau"]*0.9,sched["tau"],sched["tau"]*1.1]
    g=_att_scalar(att)
    deltas=None
    for k_i,tau_i in zip(ks,taus):
        idx_i,alpha_i=batched_topk_neighbors(w0,Wn,k=k_i,tau=tau_i)
        conf=_entropy_conf(alpha_i)
        step_scale=sched["step_scale"]*(0.5+0.5*g)*conf/len(ks)
        w_star_i=trust_region_step(w0,Wn,idx_i,alpha_i,theta_max_deg=sched["theta_max_deg"],step_scale=step_scale)
        d_i=w_star_i-w0
        deltas=d_i if deltas is None else deltas+d_i
    w_prop=w0+deltas
    idx0,alpha0=batched_topk_neighbors(w0,Wn,k=k0,tau=sched["tau"])
    for _ in range(5):
        wpn=_normalize(w_prop,dim=0)
        beta=torch.softmax(sched["tau"]*(Wn[idx0]@wpn),dim=0)
        if _kl(alpha0,beta)<=kappa:
            break
        deltas*=0.5
        w_prop=w0+deltas
    w0_mag=float(torch.norm(w0))
    return _normalize(w_prop,dim=0)*w0_mag

@torch.no_grad()
def vectorpusher_tokens_minimal(clip,text,sculpt_strength):
    tokens=clip.tokenize(text)
    ignored={49406,49407,0,2}
    for branch in tokens:
        clip_model=getattr(clip.cond_stage_model,f"clip_{branch}",None)
        W=_get_embeddings(clip_model)
        Wn=_normalize(W,dim=1)
        s=float(sculpt_strength)
        if branch.lower()=="g":
            s=min(1.0,s*1.3)
        sched=_sched(s)
        for x in range(len(tokens[branch])):
            for y in range(len(tokens[branch][x])):
                tok,attn=tokens[branch][x][y]
                if tok in ignored or s<=0.0:
                    continue
                new_vec=blended_kl_bounded_step(W[tok],Wn,sched,attn)
                tokens[branch][x][y]=(new_vec,attn)
    return tokens

class vectorpusher:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":{"clip":("CLIP",),"text":("STRING",{"multiline":True}),"sculpt_strength":("FLOAT",{"default":0.5,"min":0.0,"max":1.0,"step":0.01})}}
    RETURN_TYPES=("CONDITIONING","STRING")
    RETURN_NAMES=("Conditioning","Params")
    FUNCTION="exec"
    CATEGORY="conditioning"
    @torch.no_grad()
    def exec(self,clip,text,sculpt_strength):
        tokens=vectorpusher_tokens_minimal(clip,text,float(sculpt_strength))
        cond,pooled=clip.encode_from_tokens(tokens,return_pooled=True)
        conditioning=[[cond,{"pooled_output":pooled}]]
        params=f"vectorpusher: sculpt_strength={round(float(sculpt_strength),3)}"
        return conditioning,params

@torch.no_grad()
def add_to_first_if_shorter(conditioning1,conditioning2,x=0):
    min_dim=min(conditioning1[x][0].shape[1],conditioning2[x][0].shape[1])
    if conditioning2[x][0].shape[1]>conditioning1[x][0].shape[1]:
        conditioning2[x][0][:,:min_dim,]=conditioning1[x][0][:,:min_dim,]
        conditioning1=conditioning2
    return conditioning1

NODE_CLASS_MAPPINGS={"vectorpusher":vectorpusher}
NODE_DISPLAY_NAME_MAPPINGS={"vectorpusher":"vectorpusher"}
