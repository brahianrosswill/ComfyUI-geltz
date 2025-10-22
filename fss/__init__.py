# Fast Sigma Shuffle

import torch
import math

class FSS:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{"model":("MODEL",),"intensity":("FLOAT",{"default":0.35,"min":0.0,"max":1.0,"step":0.05})}}
    RETURN_TYPES=("MODEL",)
    RETURN_NAMES=("model",)
    FUNCTION="patch"
    CATEGORY="model_patches"

    def patch(self,model,intensity):
        m=model.clone()
        smax=25.0
        smin=0.28
        wnd_max=6

        log_smin=torch.log(torch.tensor(smin))
        log_smax=torch.log(torch.tensor(smax))
        log_range=log_smax-log_smin
        state={"calls":0,"cycle":64}

        def get_u(extra):
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
            return u

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

        def tri(x):
            t=torch.tensor(x)
            return float((t*(1.0-t))*4.0)

        def step_index(extra):
            if isinstance(extra,dict):
                for k in ["sigmas_index","t_index","step","k_step","model_step","curr_iter","timestep"]:
                    v=extra.get(k,None)
                    if v is not None:
                        try: return int(v)
                        except: pass
            state["calls"]+=1
            return state["calls"]

        def make_shifts(H,W,r,idx):
            if r<=0: return [(0,0)]
            a=(idx*73)% (2*r+1); b=(idx*127)% (2*r+1); c=(idx*211)% (2*r+1)
            dx1=a-r; dy1=b-r
            dx2=0;   dy2=c-r
            dx3=c-r; dy3=0
            return [(int(dx1),int(dy1)),(int(dx2),int(dy2)),(int(dx3),int(dy3))]

        def roll_hw(X,H,W,dx,dy):
            B,T,D=X.shape
            Y=X.reshape(B,H,W,D)
            if dx!=0: Y=torch.roll(Y,shifts=int(dx),dims=1)
            if dy!=0: Y=torch.roll(Y,shifts=int(dy),dims=2)
            return Y.reshape(B,T,D)

        def mix_multi(X,H,W,shifts,weights):
            acc=torch.zeros_like(X)
            for (dx,dy),w in zip(shifts,weights):
                if abs(w)>0:
                    acc=acc + w*roll_hw(X,H,W,dx,dy)
            return acc

        def cosine_guard(X,Y):
            x=X.to(dtype=torch.float32)
            y=Y.to(dtype=torch.float32)
            x=x/(x.norm(dim=-1,keepdim=True).clamp_min(1e-8))
            y=y/(y.norm(dim=-1,keepdim=True).clamp_min(1e-8))
            return float((x*y).sum(dim=-1).mean().item())

        def rms(x):
            return x.pow(2).mean(dim=-1,keepdim=True).sqrt()

        def fss(q,k,v,extra):
            if intensity<=1e-6:
                return q,k,v
            if q.dim()==4: q=q.reshape(q.shape[0]*q.shape[1], q.shape[2], q.shape[3])
            if k.dim()==4: k=k.reshape(k.shape[0]*k.shape[1], k.shape[2], k.shape[3])
            if v.dim()==4: v=v.reshape(v.shape[0]*v.shape[1], v.shape[2], v.shape[3])

            u=get_u(extra)
            tau=1.0 + 1.0*(1.0-u)
            scale=(1.0/float(tau))**0.5
            q=q*scale
            k=k*scale

            Bk,Tk,Dk=k.shape
            Hk,Wk=infer_hw(Tk,extra)
            r_k=max(1,int(min(Hk,Wk, max(1,int(wnd_max*(1.0-u))))//2))
            idx=step_index(extra)

            shifts0=make_shifts(Hk,Wk,r_k,idx)
            shifts1=make_shifts(Hk,Wk,r_k,idx+97)
            shifts2=make_shifts(Hk,Wk,r_k,idx+193)
            k_p0=mix_multi(k,Hk,Wk,shifts0,[0.6,0.25,0.15])
            k_p1=mix_multi(k,Hk,Wk,shifts1,[0.6,0.25,0.15])
            k_p2=mix_multi(k,Hk,Wk,shifts2,[0.6,0.25,0.15])
            k_perm=(0.5*k_p0 + 0.3*k_p1 + 0.2*k_p2)

            a_base=float(intensity)*tri(u)
            c_gate=max(0.0,min(1.0,(cosine_guard(k,k_perm)-0.7)/0.3))
            a_k=0.15*a_base*c_gate

            rk=rms(k).clamp_min(1e-6)
            delta_k=k_perm-k
            rd=rms(delta_k).clamp_min(1e-6)

            gamma=0.35 + (0.85-0.35)*(1.0-u)
            scale_d=torch.clamp((gamma*rk)/rd,max=1.0)
            k_new=k + a_k*(delta_k*scale_d)

            n0=k.norm(dim=-1,keepdim=True).clamp_min(1e-8)
            n1=k_new.norm(dim=-1,keepdim=True).clamp_min(1e-8)
            k_new=k_new*(n0/n1)

            d=k_new-k
            rd2=rms(d).clamp_min(1e-6)
            cap=(0.25 + 0.35*(1.0-u))*rk
            g=torch.clamp(cap/rd2,max=1.0)
            k_final=k + d*g

            v_perm=(0.5*mix_multi(v,Hk,Wk,shifts0,[0.6,0.25,0.15]) + 0.3*mix_multi(v,Hk,Wk,shifts1,[0.6,0.25,0.15]) + 0.2*mix_multi(v,Hk,Wk,shifts2,[0.6,0.25,0.15]))
            delta_v=v_perm-v
            rv=rms(v).clamp_min(1e-6)
            rvd=rms(delta_v).clamp_min(1e-6)
            a_v=0.08*a_base*c_gate
            sv=torch.clamp((gamma*rv)/rvd,max=1.0)
            v_new=v + a_v*(delta_v*sv)
            nv0=v.norm(dim=-1,keepdim=True).clamp_min(1e-8)
            nv1=v_new.norm(dim=-1,keepdim=True).clamp_min(1e-8)
            v_new=v_new*(nv0/nv1)
            dv=v_new-v
            rdv=rms(dv).clamp_min(1e-6)
            capv=(0.2 + 0.3*(1.0-u))*rv
            gv=torch.clamp(capv/rdv,max=1.0)
            v_final=v + dv*gv

            return q,k_final,v_final

        m.set_model_attn2_patch(fss)
        return (m,)

NODE_CLASS_MAPPINGS={"FSS":FSS}
NODE_DISPLAY_NAME_MAPPINGS={"FSS":"Fast Sigma Shuffle"}
