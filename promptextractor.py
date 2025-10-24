import argparse, json, os, re
from PIL import Image

def oneline(s): return " ".join((s or "").split()).lower()

def parse_a1111_parameters(s):
    lines=s.strip().split("\n")
    d={"prompt":oneline(lines[0]),"negative_prompt":"","settings":{}}
    if len(lines)>1 and lines[1].startswith("Negative prompt:"):
        d["negative_prompt"]=oneline(lines[1].split(":",1)[1]); st="\n".join(lines[2:])
    else:
        st="\n".join(lines[1:])
    for k,v in re.findall(r"([^,]+?):\s*([^,]+)(?:,|$)",st):
        d["settings"][k.strip().lower()]=oneline(v)
    return d

def parse_comfyui_metadata(s):
    try: wf=json.loads(s)
    except Exception: return {"prompt":"","negative_prompt":"","settings":{}}
    ks=mdl=lat=None
    for _,nd in wf.items():
        t=nd.get("class_type")
        if t=="KSampler": ks=nd
        elif t=="CheckpointLoaderSimple": mdl=nd
        elif t=="EmptyLatentImage": lat=nd
    if not ks: return {"prompt":"","negative_prompt":"","settings":{}}
    inp=ks.get("inputs",{})
    st={"steps":inp.get("steps"),"cfg scale":inp.get("cfg"),"sampler":inp.get("sampler_name"),"scheduler":inp.get("scheduler"),"seed":inp.get("seed")}
    if mdl: st["model"]=mdl["inputs"].get("ckpt_name","unknown")
    if lat:
        w,h=lat["inputs"].get("width"),lat["inputs"].get("height")
        if w and h: st["size"]=f"{w}x{h}"
    def txt(ref):
        if ref and isinstance(ref,list) and ref:
            nd=wf.get(str(ref[0]),{})
            return oneline(nd.get("inputs",{}).get("text",""))
        return ""
    return {"prompt":txt(inp.get("positive")),"negative_prompt":txt(inp.get("negative")),"settings":{k.lower():oneline(str(v)) for k,v in st.items() if v is not None}}

def parse_universal_metadata(meta):
    r={"prompt":"","negative_prompt":"","settings":{},"source":"unknown"}
    if "parameters" in meta:
        try: p=parse_a1111_parameters(meta["parameters"]); p["source"]="automatic1111"; return p
        except Exception: pass
    if "prompt" in meta:
        try: p=parse_comfyui_metadata(meta["prompt"]); p["source"]="comfyui"; return p
        except Exception: pass
    for k,v in meta.items():
        if isinstance(v,str) and "Negative prompt:" in v:
            try: p=parse_a1111_parameters(v); p["source"]=f"generic ({k.lower()})"; return p
            except Exception: pass
    if meta: r["settings"]={k.lower():oneline(str(v)) for k,v in meta.items() if k.lower() not in ["software","dpi"]}
    return r

def format_output_unified(d, image_path):
    b=os.path.basename(image_path).lower()
    lines=[]
    lines.append(f"metadata for: {b}")
    lines.append(f"[source detected: {oneline(d.get('source','unknown'))}]")
    pos=oneline(d.get("prompt","")) or "(not found)"
    neg=oneline(d.get("negative_prompt","")) or "(not found)"
    lines.append(f"positive: {pos}")
    lines.append(f"negative: {neg}")
    lines.append("generation settings:")
    if d.get("settings"):
        for k,v in d["settings"].items():
            if v is not None: lines.append(f"- {k.lower()}: {oneline(str(v))}")
    else:
        lines.append("(not found)")
    return "\n".join(lines)

def extract_metadata_from_png(p):
    try:
        with Image.open(p) as img: return img.info
    except Exception: return {}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("input_file"); ap.add_argument("-o","--output")
    a=ap.parse_args()
    if not os.path.exists(a.input_file): print(f"error: input file not found at '{a.input_file}'"); return
    base,_=os.path.splitext(a.input_file); out=a.output or f"{base}_metadata.txt"
    raw=extract_metadata_from_png(a.input_file)
    if not raw:
        with open(out,"w",encoding="utf-8") as f: f.write("no special metadata found in this image.")
        print(f"empty file created at: {out}"); return
    d=parse_universal_metadata(raw)
    txt=format_output_unified(d,a.input_file)
    try:
        with open(out,"w",encoding="utf-8") as f: f.write(txt)
        print(f"success! metadata saved to: {out}")
    except Exception as e:
        print(f"error: could not write to output file '{out}'. details: {str(e).lower()}")

if __name__=="__main__": main()
