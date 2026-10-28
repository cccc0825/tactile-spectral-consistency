"""Source-only response contrast calibration of tactile spatial spectra."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'; os.environ['OMP_NUM_THREADS']='2'
import numpy as np
from scipy.special import logsumexp
from scipy.fft import dct

def integral(psd,frequency,edges):
    p=np.asarray(psd,dtype=float); f=np.asarray(frequency); e=np.asarray(edges)
    if e.ndim==1:e=np.broadcast_to(e,(len(p),len(e)))
    assert e.min()>=f[0] and e.max()<=f[-1]
    cum=np.c_[np.zeros(len(p)),np.cumsum((p[:,:-1]+p[:,1:])*.5*np.diff(f),axis=1)]
    ix=np.minimum(np.searchsorted(f,e,side='right')-1,len(f)-2);dx=e-f[ix]
    pv=np.take_along_axis(p,ix,1);slope=(np.take_along_axis(p,ix+1,1)-pv)/(f[ix+1]-f[ix])
    return np.take_along_axis(cum,ix,1)+pv*dx+.5*slope*dx**2

def log_masses(p,f,v,lo=40,hi=800,ref=40,beta=1,bins=128):
    edge=np.geomspace(lo,hi,bins+1)[None,:]*(np.asarray(v)[:,None]/ref)**beta
    mass=np.maximum(np.diff(integral(p,f,edge),axis=1),0)
    assert np.all(mass.sum(1)>0)
    return np.log(np.maximum(mass/mass.sum(1,keepdims=True),1e-12))

def mel(p,f,high=1200,bins=128):
    hz=lambda m:700*(10**(m/2595)-1)
    ml=lambda x:2595*np.log10(1+x/700)
    e=hz(np.linspace(ml(20),ml(high),bins+2));w=[]
    for i in range(bins):w.append(np.maximum(0,np.minimum((f-e[i])/(e[i+1]-e[i]),(e[i+2]-f)/(e[i+2]-e[i+1]))))
    w=np.array(w);assert np.all(w.sum(1)>0)
    a=p@w.T*(f[1]-f[0]);return np.log(np.maximum(a/a.sum(1,keepdims=True),1e-12))

class ResponseCalibration:
    def __init__(self,lo=40,hi=800,ref=40,support=(20,1200),lam=.01,wrong_order=False):
        self.lo=lo;self.hi=hi;self.ref=ref;self.support=support;self.lam=lam;self.wrong_order=wrong_order
    def basis(self,v):
        edges=np.geomspace(self.lo,self.hi,129);centers=np.sqrt(edges[:-1]*edges[1:])
        f=np.asarray(v)[:,None]/self.ref*centers
        z=np.log(f/self.support[0])/np.log(self.support[1]/self.support[0])
        return np.cos(np.pi*z[:,:,None]*np.arange(1,13))
    def fit(self,z,v,y):
        speeds=np.unique(v);classes=np.unique(y);assert len(speeds)>=2
        means=np.stack([np.mean([z[(v==s)&(y==c)].mean(0) for c in classes],axis=0) for s in speeds])
        assert np.isfinite(means).all(), 'Every source class must appear at every source speed'
        b=self.basis(speeds[::-1] if self.wrong_order else speeds)
        a=b-b.mean(0,keepdims=True);a-=a.mean(1,keepdims=True)
        delta=means-means.mean(0,keepdims=True);delta-=delta.mean(1,keepdims=True)
        mat=a.reshape(-1,12);t=delta.ravel();gram=mat.T@mat
        penalty=np.arange(1,13,dtype=float)**4;penalty/=penalty.mean()
        reg=self.lam*np.trace(gram)/12*np.diag(penalty)
        self.theta=np.linalg.solve(gram+reg,mat.T@t)
        resid=t-mat@self.theta
        self.fit_diagnostics={'source_speeds':speeds.tolist(),'source_classes':len(classes),'design_rank':int(np.linalg.matrix_rank(mat)),'contrast_variance_explained':float(1-np.sum(resid**2)/np.sum(t**2)),'theta':self.theta.tolist(),'source_n':len(z)}
        return self
    def transform(self,z,v):
        a=z-np.einsum('nbk,k->nb',self.basis(v),self.theta)
        return a-logsumexp(a,axis=1,keepdims=True)

def features_for_split(p,f,v,y,train,settings,method):
    kw={k:settings[k] for k in ['lo','hi','ref']}
    info={}
    if method=='temporal':x=log_masses(p,f,v,beta=0,**kw)
    elif method=='spatial':x=log_masses(p,f,v,**kw)
    elif method in ('mel','mfcc'):
        x=mel(p,f,settings['mel_hi'])
        if method=='mfcc':x=dct(x,type=2,norm='ortho',axis=1)[:,1:21]
    elif method=='pooled_whitening':
        # Source temporal geometric mean. It is not a measured transfer function.
        floor=np.maximum(p[train].sum(1,keepdims=True)*1e-12,1e-30)
        h=np.mean(np.log(np.maximum(p[train],floor)),axis=0)
        corrected=p*np.exp(-h)[None,:]
        x=log_masses(corrected,f,v,**kw);info={'pooled_log_psd':h.tolist()}
    else:
        z=log_masses(p,f,v,**kw)
        strength={'corrected':.01,'corrected_weak':.001,'corrected_strong':.1,'wrong_order':.01}[method]
        model=ResponseCalibration(**kw,support=settings['support'],lam=strength,wrong_order=method=='wrong_order').fit(z[train],v[train],y[train])
        x=model.transform(z,v);info=model.fit_diagnostics
    assert np.isfinite(x).all()
    return x,info

def algebra_test():
    """Synthetic additive model checks only. Never a real-data performance result."""
    rng=np.random.default_rng(100);v=np.tile([30,40,50],10);y=np.repeat(np.arange(10),3)
    mod=ResponseCalibration(lam=1e-10);theta=rng.normal(size=12);u=rng.normal(size=(10,128))
    z=u[y]+np.einsum('nbk,k->nb',mod.basis(v),theta)+rng.normal(size=(30,1))
    mod.fit(z,v,y);out=mod.transform(z,v)
    error=max(np.max(np.abs(out[y==c]-out[y==c][0])) for c in range(10))
    assert error<1e-5,error
    # A transform of held-out observations cannot alter the fitted coefficients.
    before=mod.theta.copy();mod.transform(z+1000,np.full(30,60));assert np.array_equal(before,mod.theta)
    return {'synthetic_test':True,'maximum_within_class_error':float(error),'pass':True}

if __name__=='__main__':
    import json
    print(json.dumps(algebra_test(),indent=2))
