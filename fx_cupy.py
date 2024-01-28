import numpy as np
import cupy as cp
import time

def reshape_cast(dat,nn,nblock):
    nsamp=nn*nblock
    nant=dat.shape[0]
    out=cp.empty([nant,nblock,nn],dtype='float32')
    out[:]=cp.reshape(dat[:,:nsamp],[nant,nblock,nn])
    return out

def get_win(n):
    x=cp.linspace(-np.pi,np.pi,n,dtype='float32')
    y=cp.sinc(x)
    return y

def reshape_pfb(dat,win,ntap):
    out=cp.zeros(1,dtype='float32')
    nb=dat.shape[1]
    nn=dat.shape[-1]
    for i in range(ntap):
        out=out+dat[:,i:nb+i-ntap,:]*win[i*nn:(i+1)*nn]
    return out

nchan=2**14
fsamp=2e8
dt=0.1

nn=2*nchan
nblock=int((dt*fsamp)/nn)
nsamp=nblock*nn
ntap=4
win=get_win(nn*ntap)

nfeed=8
try:
    assert(hdat.shape[0]==nfeed)
    assert(hdat.shape[1]==nsamp)
except:
    hdat=(3*np.random.randn(nfeed,nsamp)).astype('int8')

tot=0
do_pfb=True

for iter in range(10):
    t1=time.time()
    ddat=cp.asarray(hdat)
    dblocks=reshape_cast(ddat,nn,nblock)
    if do_pfb:
        dblocks=reshape_pfb(dblocks,win,ntap)
    dft=cp.fft.rfft(dblocks,axis=-1)
    tmp=cp.transpose(dft,(2,0,1))
    prod=tmp@(cp.conj(cp.transpose(tmp,(0,2,1))))
    tot=tot+cp.asnumpy(prod)
    t2=time.time()
    print('iter time ',t2-t1)
