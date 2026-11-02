"""RGB-only fruit crop, a small NumPy network and calibrated probabilities."""
import hashlib,json
from pathlib import Path
import cv2,numpy as np
from scipy.optimize import minimize_scalar

def require(ok,reason):
    if not ok:raise ValueError(reason)
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def config_digest(c):return hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()
def fruit_crop(rgb,k,camera_xyz,optical_rotation,handle_xyz,c):
    require(rgb.dtype==np.uint8 and rgb.ndim==3 and rgb.shape[2]==3,'invalid_rgb')
    center=np.asarray(handle_xyz)+c['fruit_offset_from_handle_m'];half=np.asarray(c['fruit_crop_half_extent_m'])
    corners=center+np.asarray([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])*half
    rays=(np.asarray(optical_rotation).T@(corners-np.asarray(camera_xyz)).T).T
    require(np.isfinite(rays).all() and np.min(rays[:,2])>.01,'crop_behind_camera')
    uv=(np.asarray(k)@rays.T).T;uv=uv[:,:2]/uv[:,2,None]
    lo=np.floor(uv.min(axis=0)).astype(int);hi=np.ceil(uv.max(axis=0)).astype(int)
    require(lo[0]>=0 and lo[1]>=0 and hi[0]<rgb.shape[1] and hi[1]<rgb.shape[0],'fruit_crop_clipped')
    require(np.min(hi-lo)>=6,'fruit_crop_too_small')
    return rgb[lo[1]:hi[1]+1,lo[0]:hi[0]+1].copy(),[int(lo[0]),int(lo[1]),int(hi[0]+1),int(hi[1]+1)]
def features(crop,c):
    require(crop.size>0 and crop.dtype==np.uint8,'invalid_crop')
    # Spatial RGB values enter a learned network; there is no class-color rule.
    return (cv2.resize(crop,(c['input_size'],c['input_size']),interpolation=cv2.INTER_AREA).astype(np.float32).reshape(-1)/255.-.5)
def softmax(logits,temperature=1.):
    require(np.isfinite(logits).all() and np.isfinite(temperature) and temperature>0,'invalid_logits_or_temperature')
    z=np.asarray(logits,dtype=np.float64)/temperature;z=z-z.max(axis=-1,keepdims=True);a=np.exp(z);return a/a.sum(axis=-1,keepdims=True)
def nll(logits,labels,temperature=1.):
    p=softmax(logits,temperature);return float(-np.log(np.maximum(p[np.arange(len(labels)),labels],1e-12)).mean())
def fit_temperature(logits,labels,bounds):
    result=minimize_scalar(lambda t:nll(logits,labels,t),bounds=bounds,method='bounded')
    require(result.success,'temperature_fit_failed');t=float(result.x)
    return t if nll(logits,labels,t)<=nll(logits,labels,1.) else 1.
def metrics(probabilities,labels,bins):
    p=np.asarray(probabilities);labels=np.asarray(labels);confidence=p.max(axis=1);pred=p.argmax(axis=1);correct=pred==labels
    onehot=np.eye(p.shape[1])[labels];rows=[];ece=0.
    for i in range(bins):
        selected=(confidence>=i/bins)&(confidence<=(i+1)/bins if i==bins-1 else confidence<(i+1)/bins)
        count=int(selected.sum());acc=float(correct[selected].mean()) if count else None;conf=float(confidence[selected].mean()) if count else None
        if count:ece+=count/len(labels)*abs(acc-conf)
        rows.append(dict(lower=i/bins,upper=(i+1)/bins,count=count,accuracy=acc,confidence=conf))
    return dict(samples=len(labels),accuracy=float(correct.mean()),nll=float(-np.log(np.maximum(p[np.arange(len(labels)),labels],1e-12)).mean()),
        brier=float(np.square(p-onehot).sum(axis=1).mean()),ece=float(ece),ece_bins=bins,reliability=rows)
class Network:
    def __init__(self,w1,b1,w2,b2):self.w1=w1;self.b1=b1;self.w2=w2;self.b2=b2
    def logits(self,x):return np.maximum(np.asarray(x)@self.w1+self.b1,0.)@self.w2+self.b2
    @classmethod
    def initial(cls,c):
        rng=np.random.default_rng(c['train_seed']);n=3*c['input_size']**2;h=c['hidden_units'];k=len(c['class_ids'])
        return cls((rng.normal(size=(n,h))*np.sqrt(2/n)).astype(np.float32),np.zeros(h,np.float32),(rng.normal(size=(h,k))*np.sqrt(2/h)).astype(np.float32),np.zeros(k,np.float32))
    def state(self):return [self.w1,self.b1,self.w2,self.b2]
    def train(self,x,y,validation,c):
        rng=np.random.default_rng(c['train_seed']);moment=[np.zeros_like(v) for v in self.state()];variance=[v.copy() for v in moment];step=0;best=float('inf');selected=None;history=[]
        vx,vy=validation
        for epoch in range(c['epochs']):
            order=rng.permutation(len(y))
            for start in range(0,len(y),c['batch_size']):
                idx=order[start:start+c['batch_size']];xx=x[idx];yy=y[idx];z=xx@self.w1+self.b1;a=np.maximum(z,0.);p=softmax(a@self.w2+self.b2).astype(np.float32)
                p[np.arange(len(yy)),yy]-=1.;p/=len(yy);hidden=(p@self.w2.T)*(z>0)
                gradients=[xx.T@hidden+c['weight_decay']*self.w1,hidden.sum(axis=0),a.T@p+c['weight_decay']*self.w2,p.sum(axis=0)];step+=1
                for value,g,m,v in zip(self.state(),gradients,moment,variance):
                    m[:]=.9*m+.1*g;v[:]=.999*v+.001*g*g;value[:]-=c['learning_rate']*(m/(1-.9**step))/(np.sqrt(v/(1-.999**step))+1e-8)
            loss=nll(self.logits(vx),vy)
            if loss<best:best=loss;selected=[v.copy() for v in self.state()];best_epoch=epoch
            if epoch%20==0:history.append(dict(epoch=epoch,validation_nll=loss))
        for v,chosen in zip(self.state(),selected):v[:]=chosen
        return dict(selected_epoch=best_epoch,selected_validation_nll=best,history=history,selection_partition='validation',test_used_for_selection=False)
class Classifier:
    def __init__(self,folder,c):
        folder=Path(folder);self.c=c;self.meta=json.loads((folder/'model.json').read_text());require(self.meta['config_sha256']==config_digest(c),'model_config_mismatch')
        require(digest(folder/'weights.npz')==self.meta['weights_sha256'],'weights_hash_mismatch')
        with np.load(folder/'weights.npz',allow_pickle=False) as a:self.net=Network(*[a[n] for n in ('w1','b1','w2','b2')])
        self.temperature=self.meta['temperature'];require(np.isfinite(self.temperature) and self.temperature>0,'invalid_temperature');self.sha=digest(folder/'weights.npz')
        require(self.meta['class_ids']==c['class_ids'],'class_order_mismatch')
        require(self.net.w1.shape==(3*c['input_size']**2,c['hidden_units']) and self.net.w2.shape==(c['hidden_units'],len(c['class_ids'])) and self.net.b1.shape==(c['hidden_units'],) and self.net.b2.shape==(len(c['class_ids']),),'model_shape_mismatch')
        require(all(np.isfinite(v).all() for v in self.net.state()),'nonfinite_model')
    def infer(self,crop):
        p=softmax(self.net.logits(features(crop,self.c)[None,:]),self.temperature)[0]
        confidence=float(p.max());return dict(probabilities=p.tolist(),confidence=confidence,entropy=float(-(p*np.log(np.maximum(p,1e-12))).sum()),accepted=confidence>=self.c['acceptance_confidence'])
