import numpy as np
import cupy as cp
from cupyx.profiler import benchmark
import cupyx
import time
import ctypes

mylib=ctypes.cdll.LoadLibrary("libfx_cupy.so")
conv_cols_gpu=mylib.conv_cols
conv_cols_gpu.argtypes=(ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int)
cherk_batched_gpu=mylib.cherk_batched
cherk_batched_gpu.argtypes=(ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_int)


def conv_cols(din,win):
    n=din.shape[1]
    m=din.shape[0]
    k=win.shape[0]

    itemsize=0
    if din.dtype=='int8':
        itemsize=1
    if din.dtype=='int16':
        itemsize=2
    if din.dtype=='int32':
        itemsize=4
    if din.dtype=='float32':
        itemsize=-4
    assert(not(itemsize==0))
    out=cp.empty([m,n],dtype='float32')
    conv_cols_gpu(din.data.ptr,out.data.ptr,win.data.ptr,n,m,k,itemsize)
    return out

def cherk_batched(mat):
    out=cp.empty([mat.shape[0],mat.shape[1],mat.shape[1]],dtype='complex64')
    cherk_batched_gpu(mat.data.ptr,out.data.ptr,mat.shape[0],mat.shape[1],mat.shape[2])
    return out
    


def cast_pfb(dat,win,ntap,nchan):
    """Apply the window function/adding part of a PFB while including a cast
    to float32.  Since I was too lazy to handle the logic for multiple antennas,
    the final ntap-1 blocks will be garbage.  Rather than zero them here, I plan
    to ignore those blocks when doing the correlation."""
    nblock=dat.shape[1]//(2*nchan)
    nant=dat.shape[0]
    dd=cp.reshape(dat,[nant*nblock,2*nchan])
    ww=cp.reshape(win,[ntap,2*nchan])
    return cp.reshape(conv_cols(dd,ww),[nant,nblock,2*nchan])
    

def pfb_xcorr_block(dat,win,ntap,nchan):
    dat_pfb=cast_pfb(dat,win,ntap,nchan)
    dft=cp.fft.rfft(dat_pfb,axis=-1)

    tmp=cp.transpose(dft,(2,0,1))
    #print('tmp shape is ',tmp.shape)
    tmp[:,:,-ntap+1:]=0
    #prod=tmp@(cp.conj(cp.transpose(tmp,(0,2,1))))
    prod=cherk_batched(tmp)
    return prod
    
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
    hdat=(3*np.random.randn(nfeed,nsamp)).astype('int16')
    tmp=cupyx.empty_like_pinned(hdat)
    tmp[:]=hdat
    hdat=tmp
    
tot=0
do_pfb=True



for iter in range(10):
    cp.cuda.runtime.deviceSynchronize()
    t1=time.time()
    ddat=cp.asarray(hdat)
    if False:
        dblocks=reshape_cast(ddat,nn,nblock)
        if do_pfb:
            dblocks_org=dblocks
            dblocks=reshape_pfb(dblocks_org,win,ntap)
        dft=cp.fft.rfft(dblocks,axis=-1)
        tmp=cp.transpose(dft,(2,0,1))
        prod=tmp@(cp.conj(cp.transpose(tmp,(0,2,1))))
        tot=tot+cp.asnumpy(prod)
    else:
        tot=tot+pfb_xcorr_block(ddat,win,ntap,nchan)
    cp.cuda.runtime.deviceSynchronize()
    t2=time.time()
    print('iter time ',t2-t1)

print('transfer: ',benchmark(cp.asarray,(hdat,),n_repeat=100))
print('reshape_cast: ',benchmark(reshape_cast,(ddat,nn,nblock),n_repeat=100))
ddat=cp.asarray(hdat)
print('new pfb: ',benchmark(pfb_xcorr_block,(ddat,win,ntap,nchan),n_repeat=100))
print('cherk_batched: ',benchmark(cherk_batched,(tmp,),n_repeat=100))

niter=100

cp.cuda.runtime.deviceSynchronize()
t1=time.time()
for i in range(niter):
    dblocks=reshape_pfb(dblocks_org,win,ntap)
cp.cuda.runtime.deviceSynchronize()
t2=time.time()
print('pfb: ',(t2-t1)/niter*1e3,' msec')

cp.cuda.runtime.deviceSynchronize()
t1=time.time()
for i in range(niter):
    dft=cp.fft.rfft(dblocks,axis=-1)
cp.cuda.runtime.deviceSynchronize()
t2=time.time()
print('fft: ',(t2-t1)/niter*1e3,' msec')

cp.cuda.runtime.deviceSynchronize()
t1=time.time()
for i in range(niter):
    tmp=cp.transpose(dft,(2,0,1))
cp.cuda.runtime.deviceSynchronize()
t2=time.time()
print('transpose: ',(t2-t1)/niter*1e3,' msec')

cp.cuda.runtime.deviceSynchronize()
t1=time.time()
for i in range(niter):
    prod=tmp@(cp.conj(cp.transpose(tmp,(0,2,1))))
cp.cuda.runtime.deviceSynchronize()
t2=time.time()
print('xcorr: ',(t2-t1)/niter*1e3,' msec')


tot=0
cp.cuda.runtime.deviceSynchronize()
t1=time.time()
for i in range(niter):
    tot=tot+cp.asnumpy(prod)
cp.cuda.runtime.deviceSynchronize()
t2=time.time()
print('copy to host: ',(t2-t1)/niter*1e3,' msec')




#print('pfb: ',benchmark(reshape_pfb,(dblocks_org,win,ntap)))
#print('dft: ',benchmark(cp.fft.rfft,(dblocks,None,-1)))
      
