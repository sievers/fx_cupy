import numpy as np
import cupy as cp
import fx_cupy
from matplotlib import pyplot as plt

samps=np.load('samples_IQ_2p4MHz.npy')
nchan=1024
ntap=4

dsamps=cp.asarray(np.reshape(samps,[1,len(samps)]))
dspec=fx_cupy.make_autos(dsamps,nchan,ntap,iq=True)
spec=np.squeeze(cp.asnumpy(dspec))

np.save('spec.npy',spec)
plt.clf()
plt.plot(spec)
plt.savefig('iqspec_'+repr(nchan)+'_chan_'+repr(ntap)+'_tap.png')
