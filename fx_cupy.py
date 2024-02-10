import cupy as cp
import ctypes

mylib=ctypes.cdll.LoadLibrary("libfx_cupy.so")
conv_cols_gpu=mylib.conv_cols
conv_cols_gpu.argtypes=(ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int)
conv_cols_complex_gpu=mylib.conv_cols_complex
conv_cols_complex_gpu.argtypes=(ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int)
cherk_batched_gpu=mylib.cherk_batched
cherk_batched_gpu.argtypes=(ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_int)


def conv_cols(din,win):
    n=din.shape[1]
    m=din.shape[0]
    k=win.shape[0]
    if din.dtype=='complex64':
        itemsize=-8
        out=cp.empty([m,n],dtype='complex64')
        conv_cols_complex_gpu(din.data.ptr,out.data.ptr,win.data.ptr,n,m,k,itemsize)
        return out
    if din.dtype=='complex128':
        itemsize=-16
        out=cp.empty([m,n],dtype='complex64')
        conv_cols_complex_gpu(din.data.ptr,out.data.ptr,win.data.ptr,n,m,k,itemsize)
        return out

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
    

def cast_pfb(dat,win,ntap,nchan,iq=False):
    """Apply the window function/adding part of a PFB while including a cast
    to float32.  Since I was too lazy to handle the logic for multiple antennas,
    the final ntap-1 blocks will be garbage.  Rather than zero them here, I plan
    to ignore those blocks when doing the correlation."""
    if iq:
        nn=nchan
    else:
        nn=2*nchan
    nblock=dat.shape[1]//(nn)
    nant=dat.shape[0]
    dd=cp.reshape(dat,[nant*nblock,nn])
    ww=cp.reshape(win,[ntap,nn])
    return cp.reshape(conv_cols(dd,ww),[nant,nblock,nn])
    

def pfb_xcorr_block(dat,win,ntap,nchan,full=False):
    dat_pfb=cast_pfb(dat,win,ntap,nchan)
    dft=cp.fft.rfft(dat_pfb,axis=-1)

    tmp=cp.transpose(dft,(2,0,1))
    #print('tmp shape is ',tmp.shape)
    tmp[:,:,-ntap+1:]=0
    #prod=tmp@(cp.conj(cp.transpose(tmp,(0,2,1))))
    prod=cherk_batched(tmp)
    if full:
        return prod,tmp
    else:
        return prod
    
def reshape_cast(dat,nn,nblock):
    nsamp=nn*nblock
    nant=dat.shape[0]
    out=cp.empty([nant,nblock,nn],dtype='float32')
    out[:]=cp.reshape(dat[:,:nsamp],[nant,nblock,nn])
    return out

def get_win(nchan,ntap):
    winft=cp.zeros(nchan*ntap//2+1,dtype='complex64')
    if ntap%2==0:
        winft[:ntap//2]=1
        winft[ntap//2]=0.5
    else:
        winft[:(ntap+1)//2]=1
    win=cp.fft.irfft(winft)
    win=win/win.max()
    return cp.fft.ifftshift(win)

def get_win_sinc(n): #prolly buggy...
    x=cp.linspace(-cp.pi,cp.pi,n,dtype='float32')
    y=cp.sinc(x)
    return y

def reshape_pfb(dat,win,ntap):
    out=cp.zeros(1,dtype='float32')
    nb=dat.shape[1]
    nn=dat.shape[-1]
    for i in range(ntap):
        out=out+dat[:,i:nb+i-ntap,:]*win[i*nn:(i+1)*nn]
    return out

def make_autos(dat,nchan,ntap,win=None,iq=False):
    if win is None:
        win=get_win(nchan,ntap)
    crud=cast_pfb(dat,win,ntap,nchan,iq=iq)
    if iq:
        crudft=cp.fft.fft(crud,axis=-1)
    else:
        crudft=cp.fft.rfft(crud,axis=-1)
    #this is inefficient, but probably doesn't matter in most cases.
    #it would not be too hard to write a kernel that sums, but I am lazy
    spec=cp.sum(cp.abs(crudft[:,:-ntap+1,:])**2,axis=1)
    if iq:
        return(cp.fft.fftshift(spec,axes=-1))
    else:
        return spec
