# Sigma-Weighted Shuffle

import torch
import math

class SWS:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "intensity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }
    RETURN_TYPES=("MODEL",)
    RETURN_NAMES=("model",)
    FUNCTION="patch"
    CATEGORY="model_patches"
    def patch(self,model,intensity,seed):
        m=model.clone()
        smax=25.0
        smin=0.28
        alpha=4.0
        wnd_max=8
        log_smin=torch.log(torch.tensor(smin))
        log_smax=torch.log(torch.tensor(smax))
        log_range=log_smax-log_smin
        state={"calls":0,"cycle":64}
        proj_cache={}
        perm_cache={}
        def get_u_and_w(extra):
            sig=None
            idx=None
            cur=None
            tot=None
            if isinstance(extra,dict):
                if extra.get("sigma",None) is not None:
                    sig=extra["sigma"]
                elif extra.get("sigmas",None) is not None:
                    idx=extra.get("sigmas_index",None)
                    if idx is None: idx=extra.get("step",None)
                    if idx is None: idx=extra.get("t_index",None)
                    if idx is not None:
                        try: sig=float(extra["sigmas"][int(idx)])
                        except: sig=None
                if sig is None:
                    for k in ["sigmas_step","step","t_index","k_step","model_step","curr_iter","timestep"]:
                        if extra.get(k,None) is not None:
                            try: cur=float(extra[k]); break
                            except: pass
                    for k in ["sigmas_total","steps","total_steps","num_steps","max_steps"]:
                        if extra.get(k,None) is not None:
                            try: tot=float(extra[k]); break
                            except: pass
            if sig is not None:
                s=float(sig)
                s=max(min(s,smax),smin)
                u=float(((torch.log(torch.tensor(s)) - log_smin) / log_range).clamp(0,1).item())
            elif cur is not None and tot is not None and tot>0:
                u=max(0.0,min(1.0,cur/tot))
            else:
                state["calls"]+=1
                u=(state["calls"]%state["cycle"])/float(state["cycle"])
            w=float(torch.sigmoid(torch.tensor(alpha*(u-0.5))).item()/torch.sigmoid(torch.tensor(alpha*0.5)).item())
            return u,w
        def infer_hw(T,extra):
            H=None; W=None
            if isinstance(extra,dict):
                if extra.get("h",None) is not None and extra.get("w",None) is not None:
                    H=int(extra["h"]); W=int(extra["w"])
                elif extra.get("hw",None) is not None and isinstance(extra["hw"],(tuple,list)) and len(extra["hw"])==2:
                    H=int(extra["hw"][0]); W=int(extra["hw"][1])
                elif extra.get("spatial",None) is not None and isinstance(extra["spatial"],(tuple,list)) and len(extra["spatial"])==2:
                    H=int(extra["spatial"][0]); W=int(extra["spatial"][1])
            if H is None or W is None or H*W!=T:
                s=int(math.sqrt(T))
                if s*s==T:
                    H=s; W=s
                else:
                    W=int(round(math.sqrt(T)))
                    H=max(1,T//max(1,W))
                    if H*W<T: H=min(H+1,T)
                    if H*W!=T:
                        H=1; W=T
            return H,W
        def proj_mat(df,dt,dev):
            key=(df,dt,str(dev))
            P=proj_cache.get(key,None)
            if P is None:
                # Use user-provided seed instead of deterministic calculation
                torch.manual_seed(seed + df*1000003 + dt*9176)
                if dev.type == 'cuda':
                    torch.cuda.manual_seed_all(seed + df*1000003 + dt*9176)
                M=torch.randn(df,dt,device=dev,dtype=torch.float32)
                U, S, Vh = torch.linalg.svd(M, full_matrices=False)
                P = U[:, :dt] if df >= dt else Vh[:dt, :].T
                if P.shape != (df, dt):
                    P = torch.nn.functional.pad(P, (0, dt-P.shape[1])) if P.shape[1] < dt else P[:, :dt]
                proj_cache[key]=P
            return P
        def baseline_attn(q,k,sample_state=None,sample_max_q=384,sample_max_k=384):
            if q.dim()==4: q=q.reshape(q.shape[0]*q.shape[1], q.shape[2], q.shape[3])
            if k.dim()==4: k=k.reshape(k.shape[0]*k.shape[1], k.shape[2], k.shape[3])
            dq=q.shape[-1]; dk=k.shape[-1]
            if dq==dk:
                q2=q; k2=k
            elif dk>dq:
                P=proj_mat(dk,dq,k.device)
                k2=torch.matmul(k.float(),P).to(k.dtype); q2=q
            else:
                P=proj_mat(dq,dk,q.device)
                q2=torch.matmul(q.float(),P).to(q.dtype); k2=k
            B,Tq,D=q2.shape
            S=k2.shape[1]
            ss = {} if sample_state is None else dict(sample_state)
            if Tq>sample_max_q:
                if "idx_q" not in ss:
                    step=max(1,int(Tq//sample_max_q))
                    idx_q=torch.arange(0,Tq,step,device=q2.device)
                    if idx_q.numel()>sample_max_q: idx_q=idx_q[:sample_max_q]
                    ss["idx_q"]=idx_q
                q2=q2.index_select(1,ss["idx_q"])
            if S>sample_max_k:
                if "idx_k" not in ss:
                    step=max(1,int(S//sample_max_k))
                    idx_k=torch.arange(0,S,step,device=k2.device)
                    if idx_k.numel()>sample_max_k: idx_k=idx_k[:sample_max_k]
                    ss["idx_k"]=idx_k
                k2=k2.index_select(1,ss["idx_k"])
            d=q2.shape[-1]
            logits=torch.einsum('btd,bsd->bts', q2, k2)/math.sqrt(max(d,1))
            probs=logits.softmax(dim=-1)
            return logits,probs,ss
        def entropy_alpha_from_probs(probs,u,intensity):
            eps=1e-12
            A=probs.clamp_min(eps)
            H=-(A*(A.log())).sum(dim=-1)
            H_mean=H.mean(dim=(-1,-2),keepdim=True)
            H_min=H.amin(dim=(-1,-2),keepdim=True)
            H_max=H.amax(dim=(-1,-2),keepdim=True).clamp_min(H_min+1e-6)
            H_norm=((H_mean-H_min)/(H_max-H_min)).squeeze()
            H_norm=torch.nan_to_num(H_norm,nan=0.0).clamp(0,1)
            H_scalar=H_norm.mean()
            u_prime=((torch.tensor(u)-0.0)/(1.0-0.0)).clamp(0,1)
            w_u=(u_prime*(1.0-u_prime))*4.0
            a=float(intensity)*float(H_scalar)*float(w_u)
            return max(0.0,min(a,1.0))
        def kl_guard(P1,P0):
            eps=1e-8
            P0=P0.clamp_min(eps).float()
            P1=P1.clamp_min(eps).float()
            KL=(P1*(P1.log()-P0.log())).sum(dim=-1).mean(dim=(-1,-2))
            return KL.mean().item()
        def bsearch_alpha_k(q,k,A0,k_perm,a_max,kl_cap,sample_state,steps=7):
            lo=0.0; hi=float(max(0.0,min(1.0,a_max)))
            best=0.0
            for _ in range(steps):
                mid=0.5*(lo+hi)
                k_try=(1.0-mid)*k+mid*k_perm
                _,A1,ss=baseline_attn(q,k_try,sample_state)
                if kl_guard(A1,A0) > kl_cap:
                    hi=mid
                else:
                    best=mid
                    lo=mid
            return best
        def pick_window(H,W,u,wm):
            target=max(2,int(round(max(2.0, wm*max(0.1,1.0-u)))))
            candidates=[s for s in [8,7,6,5,4,3,2] if s<=wm and s<=target]
            for s in candidates:
                if H%s==0 and W%s==0:
                    return s
            for s in [4,3,2]:
                if s<=wm and H%s==0 and W%s==0:
                    return s
            return 1
        def window_perm_indices(H,W,s,device):
            key=(H,W,s,str(device))
            idx=perm_cache.get(key,None)
            if idx is not None:
                return idx
            I=torch.arange(H*W,device=device).reshape(H,W)
            nH=H//s; nW=W//s
            W0=I.reshape(nH,s,nW,s).permute(0,2,1,3).reshape(nH*nW,s*s)
            shift=max(1,(s*s)//4)
            p=(torch.arange(s*s,device=device)+shift)%(s*s)
            W1=W0.index_select(1,p)
            idx=W1.reshape(nH,nW,s,s).permute(0,2,1,3).reshape(H,W).reshape(-1)
            perm_cache[key]=idx
            return idx
        def apply_idx(X,idx):
            B,T,D=X.shape
            idx_b=idx.unsqueeze(0).expand(B,-1)
            idx_e=idx_b.unsqueeze(-1).expand(B,-1,D)
            return X.gather(dim=1,index=idx_e)
        def sws(q,k,v,extra):
            if q.dim()==4: q=q.reshape(q.shape[0]*q.shape[1], q.shape[2], q.shape[3])
            if k.dim()==4: k=k.reshape(k.shape[0]*k.shape[1], k.shape[2], k.shape[3])
            if v.dim()==4: v=v.reshape(v.shape[0]*v.shape[1], v.shape[2], v.shape[3])
            u,_=get_u_and_w(extra)
            tau = 1.0 + 1.0 * (1.0 - u)
            scale=(1.0/float(tau))**0.5
            q=q*scale
            k=k*scale
            _,A0,ss=baseline_attn(q,k,None)
            a=entropy_alpha_from_probs(A0,u,intensity)
            if a<=0.0:
                return q,k,v
            Bk,Tk,Dk=k.shape
            Bv,Tv,Dv=v.shape
            Hk,Wk=infer_hw(Tk,extra)
            Hv,Wv=infer_hw(Tv,extra)
            sk=pick_window(Hk,Wk,u,wnd_max)
            sv=pick_window(Hv,Wv,u,wnd_max)
            if sk>1:
                idx_k=window_perm_indices(Hk,Wk,sk,k.device)
                k_perm=apply_idx(k,idx_k)
            else:
                k_perm=k
            if sv>1:
                idx_v=window_perm_indices(Hv,Wv,sv,v.device)
                v_perm=apply_idx(v,idx_v)
            else:
                v_perm=v
            kl_cap=0.08*(1.0-u)+0.01
            a_k_max=0.25*a
            a_v_base=0.6*a
            u_stop=0.75
            if u>=u_stop:
                a_k=0.0
            else:
                a_k=bsearch_alpha_k(q,k,A0,k_perm,a_k_max,kl_cap,ss,steps=7)
            v_gate=0.5+0.5*(1.0-u)
            a_v=max(0.0,min(1.0,a_v_base*v_gate))
            k_final=(1.0-a_k)*k+a_k*k_perm
            v_final=(1.0-a_v)*v+a_v*v_perm
            return q,k_final,v_final
        m.set_model_attn2_patch(sws)
        return (m,)

NODE_CLASS_MAPPINGS={"SWS":SWS}
NODE_DISPLAY_NAME_MAPPINGS={"SWS":"Sigma-Weighted Shuffle"}