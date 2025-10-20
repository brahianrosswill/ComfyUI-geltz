# Sigma-Weighted Shuffle

import torch
import math

class SWS:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{"model":("MODEL",),"intensity":("FLOAT",{"default":0.25,"min":0.0,"max":1.0,"step":0.05})}}
    RETURN_TYPES=("MODEL",)
    RETURN_NAMES=("model",)
    FUNCTION="patch"
    CATEGORY="model_patches"
    def patch(self,model,intensity):
        m=model.clone()
        smax=25.0
        smin=0.28
        alpha=4.0
        wnd_max=8
        beta_mom=0.8
        log_smin=torch.log(torch.tensor(smin))
        log_smax=torch.log(torch.tensor(smax))
        log_range=log_smax-log_smin
        state={"calls":0,"cycle":64,"Pk_prev":None,"Pv_prev":None,"Pk_shape":None,"Pv_shape":None,"gate_ema":None}
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
                            try:
                                cur=float(extra[k]); break
                            except:
                                pass
                    for k in ["sigmas_total","steps","total_steps","num_steps","max_steps"]:
                        if extra.get(k,None) is not None:
                            try:
                                tot=float(extra[k]); break
                            except:
                                pass
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
        def baseline_attn(q,k):
            if q.dim()==4: q=q.reshape(q.shape[0]*q.shape[1], q.shape[2], q.shape[3])
            if k.dim()==4: k=k.reshape(k.shape[0]*k.shape[1], k.shape[2], k.shape[3])
            dq=q.shape[-1]; dk=k.shape[-1]
            def proj_mat(df,dt,dev):
                cache=getattr(baseline_attn,'_cache',None)
                if cache is None:
                    cache={}
                    setattr(baseline_attn,'_cache',cache)
                key=(df,dt,str(dev))
                P=cache.get(key,None)
                if P is None:
                    g=torch.Generator(device=dev)
                    g.manual_seed(df*1000003+dt*9176)
                    M=torch.randn(df,dt,device=dev,generator=g,dtype=torch.float32)
                    Q,_=torch.linalg.qr(M,mode='reduced')
                    if Q.shape[1]!=dt:
                        Q=torch.nn.functional.pad(Q,(0,dt-Q.shape[1]))
                    P=Q
                    cache[key]=P
                return P
            if dq==dk:
                q2=q; k2=k
            elif dk>dq:
                P=proj_mat(dk,dq,k.device)
                k2=torch.matmul(k.float(),P).to(k.dtype); q2=q
            else:
                P=proj_mat(dq,dk,q.device)
                q2=torch.matmul(q.float(),P).to(q.dtype); k2=k
            d=q2.shape[-1]
            logits=torch.einsum('btd,bsd->bts', q2, k2)/math.sqrt(max(d,1))
            probs=logits.softmax(dim=-1)
            return logits,probs
        def entropy_gate(probs,u,intensity):
            eps=1e-12
            A=probs.clamp_min(eps).float()
            S=A.shape[-1]
            H=-(A*(A.log())).sum(dim=-1)
            ESS=torch.exp(H).clamp_min(1.0)
            ess_norm=(ESS/float(S)).mean()
            u_t=torch.tensor(u).clamp(0,1)
            w_u=(u_t*(1.0-u_t))*4.0
            g=float(intensity)*float(ess_norm)*float(w_u)
            if state["gate_ema"] is None:
                state["gate_ema"]=g
            else:
                state["gate_ema"]=0.7*state["gate_ema"]+0.3*g
            return max(0.0,min(1.0,state["gate_ema"]))
        def coords_hw(T,H,W,device):
            h=torch.arange(H,device=device).unsqueeze(1).repeat(1,W).reshape(-1).float()
            w=torch.arange(W,device=device).unsqueeze(0).repeat(H,1).reshape(-1).float()
            return h,w
        def gaussian_sinkhorn(T,H,W,local_r,device,tau,iters,sharp,iden_bias):
            h,w=coords_hw(T,H,W,device)
            hi=h.unsqueeze(1).repeat(1,T)
            wi=w.unsqueeze(1).repeat(1,T)
            hj=h.unsqueeze(0).repeat(T,1)
            wj=w.unsqueeze(0).repeat(T,1)
            dist2=(hi-hj).pow(2)+(wi-wj).pow(2)
            r=max(1.0,float(local_r))
            s2=(r*r)/2.0
            K=torch.exp(-(dist2)/(max(s2,1e-6)))
            nb_mask=((hi-hj).abs()+(wi-wj).abs()<=local_r).to(K.dtype)
            K=K*nb_mask
            if iden_bias>0.0:
                K=K+iden_bias*torch.eye(T,device=device,dtype=K.dtype)
            K=(K/tau).softmax(dim=-1)
            for _ in range(iters):
                K=K/(K.sum(dim=-1,keepdim=True).clamp_min(1e-8))
                K=K/(K.sum(dim=0,keepdim=True).clamp_min(1e-8))
            if sharp>1.0:
                K=K.pow(sharp)
                for _ in range(2):
                    K=K/(K.sum(dim=-1,keepdim=True).clamp_min(1e-8))
                    K=K/(K.sum(dim=0,keepdim=True).clamp_min(1e-8))
            return K
        def rebalance(P,steps=2):
            for _ in range(steps):
                P=P/(P.sum(dim=-1,keepdim=True).clamp_min(1e-8))
                P=P/(P.sum(dim=0,keepdim=True).clamp_min(1e-8))
            return P
        def apply_P(X,P):
            B,T,D = X.shape
            if P.device != X.device:
                P = P.to(device=X.device)
            if P.dtype != X.dtype:
                P = P.to(dtype=X.dtype)
            Pb = P.unsqueeze(0).expand(B, -1, -1)
            if X.dtype in (torch.float16, torch.bfloat16):
                out = torch.matmul(Pb.to(torch.float32), X.to(torch.float32)).to(X.dtype)
            else:
                out = torch.matmul(Pb, X)
            return out
        def kl_guard(P1,P0):
            eps=1e-8
            P0=P0.clamp_min(eps).float()
            P1=P1.clamp_min(eps).float()
            KL=(P1*(P1.log()-P0.log())).sum(dim=-1).mean(dim=(-1,-2))
            return KL.mean().item()
        def bsearch_alpha_k(q,k,A0,k_perm,a_max,kl_cap,steps=7):
            lo=0.0; hi=float(max(0.0,min(1.0,a_max)))
            best=0.0
            for _ in range(steps):
                mid=0.5*(lo+hi)
                k_try=(1.0-mid)*k+mid*k_perm
                _,A1=baseline_attn(q,k_try)
                if kl_guard(A1,A0) > kl_cap:
                    hi=mid
                else:
                    best=mid
                    lo=mid
            return best
        def sws(q,k,v,extra):
            if q.dim()==4: q=q.reshape(q.shape[0]*q.shape[1], q.shape[2], q.shape[3])
            if k.dim()==4: k=k.reshape(k.shape[0]*k.shape[1], k.shape[2], k.shape[3])
            if v.dim()==4: v=v.reshape(v.shape[0]*v.shape[1], v.shape[2], v.shape[3])
            u,_=get_u_and_w(extra)
            tau = 1.0 + 1.0 * (1.0 - u)
            scale=(1.0/float(tau))**0.5
            q=q*scale
            k=k*scale
            _,A0=baseline_attn(q,k)
            a=entropy_gate(A0,u,intensity)
            if a<=0.0:
                return q,k,v
            Bk,Tk,Dk=k.shape
            Bv,Tv,Dv=v.shape
            Hk,Wk=infer_hw(Tk,extra)
            Hv,Wv=infer_hw(Tv,extra)
            r_base=max(1,int(wnd_max*(1.0-u)))
            rk=max(1,int(min(Hk,Wk, r_base)//2))
            rv=max(1,int(min(Hv,Wv, r_base)//2))
            sharp=1.0+2.0*(1.0-u)
            id_bias=0.25*(1.0-u)
            Pk=gaussian_sinkhorn(Tk,Hk,Wk,rk,device=k.device,tau=0.3,iters=3,sharp=sharp,iden_bias=id_bias)
            Pv=gaussian_sinkhorn(Tv,Hv,Wv,rv,device=v.device,tau=0.3,iters=3,sharp=sharp,iden_bias=id_bias)
            if state["Pk_prev"] is not None and state["Pk_shape"]==(Tk,Hk,Wk):
                Pk=rebalance(beta_mom*state["Pk_prev"]+(1.0-beta_mom)*Pk,steps=2)
            if state["Pv_prev"] is not None and state["Pv_shape"]==(Tv,Hv,Wv):
                Pv=rebalance(beta_mom*state["Pv_prev"]+(1.0-beta_mom)*Pv,steps=2)
            state["Pk_prev"]=Pk.detach()
            state["Pv_prev"]=Pv.detach()
            state["Pk_shape"]=(Tk,Hk,Wk)
            state["Pv_shape"]=(Tv,Hv,Wv)
            k_perm=apply_P(k,Pk)
            v_perm=apply_P(v,Pv)
            kl_cap=0.08*(1.0-u)+0.01
            a_k_max=0.3*a
            a_v_base=0.6*a
            u_stop=0.75
            if u>=u_stop:
                a_k=0.0
            else:
                a_k=bsearch_alpha_k(q,k,A0,k_perm,a_k_max,kl_cap,steps=7)
            v_gate=0.5+0.5*(1.0-u)
            a_v=max(0.0,min(1.0,a_v_base*v_gate))
            k_final=(1.0-a_k)*k+a_k*k_perm
            v_final=(1.0-a_v)*v+a_v*v_perm
            return q,k_final,v_final
        m.set_model_attn2_patch(sws)
        return (m,)

NODE_CLASS_MAPPINGS={"SWS":SWS}
NODE_DISPLAY_NAME_MAPPINGS={"SWS":"Sigma-Weighted Shuffle"}

