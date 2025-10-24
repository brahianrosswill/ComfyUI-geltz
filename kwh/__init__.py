import torch
import torch.nn.functional as F

class KuwaharaBlur:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{"image":("IMAGE",),"radius":("INT",{"default":2,"min":1,"max":10,"step":1,"display":"slider"}),"strength":("FLOAT",{"default":0.3,"min":0.0,"max":1.0,"step":0.05,"display":"slider"})}}
    RETURN_TYPES=("IMAGE",)
    FUNCTION="apply_kuwahara"
    CATEGORY="image/filters"

    def apply_kuwahara(self,image,radius,strength):
        img=image.permute(0,3,1,2)
        f=self.kuwahara(img,radius).permute(0,2,3,1)
        return (image*(1-strength)+f*strength,)

    def kuwahara(self,img,r):
        B,C,H,W=img.shape; k=r+1
        kernel=torch.ones(1,1,k,k,device=img.device)/(k*k)
        p=F.pad(img,(r,r,r,r),mode="reflect")
        quads=[p[:,:,:H+r,:W+r],p[:,:,:H+r,r:W+2*r],p[:,:,r:H+2*r,:W+r],p[:,:,r:H+2*r,r:W+2*r]]
        cat=torch.cat(quads,0)
        mean=F.conv2d(cat,kernel.expand(C,1,-1,-1),groups=C)
        var=F.conv2d(cat*cat,kernel.expand(C,1,-1,-1),groups=C)-mean*mean
        mean=mean.view(4,B,C,H,W); var=var.sum(1,keepdim=True).view(4,B,1,H,W)
        idx=var.argmin(0)
        mask=torch.nn.functional.one_hot(idx.squeeze(1),4).permute(3,0,1,2).unsqueeze(2).to(mean.dtype)
        return (mean*mask).sum(0)

NODE_CLASS_MAPPINGS={"KuwaharaBlur":KuwaharaBlur}
NODE_DISPLAY_NAME_MAPPINGS={"KuwaharaBlur":"Kuwahara Blur"}
