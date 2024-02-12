import numpy as np
import cupy as cp
import fx_cupy
from matplotlib import pyplot as plt
from cupyx.profiler import benchmark

samps=np.load('samples_IQ_2p4MHz.npy')
nchan=1024
ntap=4

#test out the auto spectrum code
#for array sizes likely to be in play, the time is going
#to be completely dominated by transferring from host to GPU
#you'll probably want to send samples in as low precision as possible.
#as of now, IQ code will handle floats and doubles, but easy to have it
#handle ints of various sizes as well.  In the meantime, if you send
#as complex short ints, you can cast on the GPU to complex easily
#e.g. dsamps_complex=dsamps[:,::2].astype('float32')+1J*dsamps[:,1::2].astype('float32')
dsamps=cp.asarray(np.reshape(samps,[1,len(samps)]))
dspec,dft=fx_cupy.make_autos(dsamps,nchan,ntap,iq=True,full=True)
spec=np.squeeze(cp.asnumpy(dspec))

np.save('spec.npy',spec)
plt.clf()
plt.plot(spec)
plt.savefig('iqspec_'+repr(nchan)+'_chan_'+repr(ntap)+'_tap.png')

#test out the cross-correlation code.  The autos should be the diagonals
#if you had multiple data streams, you would want dsamps to have dimension [nstream,nsamp]
win=fx_cupy.get_win(nchan,ntap)
dspec2=fx_cupy.pfb_xcorr_block(dsamps,win,ntap,nchan,iq=True)
#with IQ samples, LO frequency naturally comes out at k=0, followed
#by positive frequencies, then negative frequencies.  If you want
#usual frequency ordering, put in an fftshift
spec2=np.fft.fftshift(np.squeeze(cp.asnumpy(dspec2[:,0,0])))  
np.save('spec2.npy',np.real(spec2))

print('fractional error on channel 0 autos is ',np.std(spec2-spec)/np.std(spec2))
print('\ntimings:')
print('transfer: ',benchmark(cp.asarray,(np.reshape(samps,[1,len(samps)]),),n_repeat=100))
print('xcorr: ',benchmark(fx_cupy.pfb_xcorr_block,(dsamps,win,ntap,nchan,False,True),n_repeat=100))

nant=8
samps2=np.empty([8,len(samps)],dtype=samps.dtype)
for i in range(nant):
    samps2[i,:]=samps
dsamps2=cp.asarray(samps2)
print('\nwith ',nant,' streams: ')
print('transfer: ',benchmark(cp.asarray,(samps2,),n_repeat=100))
print('xcorr: ',benchmark(fx_cupy.pfb_xcorr_block,(dsamps2,win,ntap,nchan,False,True),n_repeat=100))
