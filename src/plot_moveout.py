import pyasdf
import obspy
import numpy as np 
import matplotlib.pyplot as plt
from obspy.signal.filter import bandpass

'''
display the moveout (2D matrix) of the cross-correlation functions stacked for all time chuncks.

PARAMETERS:
 ---------------------
sfile: cross-correlation functions outputed by S2
dtype: datatype either 'Allstack0pws' or 'Allstack0linear'
freqmin: min frequency to be filtered
freqmax: max frequency to be filtered
ccomp:   cross component
dist_inc: distance bins to stack over
disp_lag: lag times for displaying
savefig: set True to save the figures (in pdf format)
sdir: diresied directory to save the figure (if not provided, save to default dir)

'''

# basic parameters for plotting
sfile = "/Users/chengxin/Documents/ANU/DAS/melb_DAS2022/CCF/2022_01_17_00_01_01.00T2022_01_17_00_02_01.00.h5"
freqmin = 1
freqmax = 10
source = 'C0100'
nsta = 3000
savefig = True
disp_lag = 30

# extract common variables
with pyasdf.ASDFDataSet(sfile,mode='r') as ds:
    try:
        slist = ds.auxiliary_data.list()
        rlist = ds.auxiliary_data[source].list()
        dt    = ds.auxiliary_data[source][rlist[0]].parameters['dt']
        maxlag= ds.auxiliary_data[source][rlist[0]].parameters['maxlag']
        data = ds.auxiliary_data[source][rlist[0]].data[:]
    except Exception:
        raise IOError("cannot load %s for parameters"%sfile)


# lags for display   
if not disp_lag:
    disp_lag=maxlag
if disp_lag>maxlag:
    raise ValueError('lag excceds maxlag!')

t = np.linspace(-maxlag,maxlag,len(data))
indx = np.where(np.abs(t)<=disp_lag)[0]

# make space for the cc matrix
nwin = len(rlist)
if nsta<=nwin:
    nwin = nsta
data = np.zeros(shape=(nwin,len(indx)),dtype=np.float32)
dist = np.zeros(nwin,dtype=np.float32)
ngood= np.zeros(nwin,dtype=np.int16)    

# load cc and parameter matrix
with pyasdf.ASDFDataSet(sfile,mode='r') as ds:
    for ii in range(nwin):
        path = rlist[ii]
        try:
            # load data to variables
            dist[ii] = ds.auxiliary_data[source][path].parameters['dist']
            tdata    = ds.auxiliary_data[source][path].data[:]
            tdata    = tdata[indx]
        except Exception:
            print("continue! cannot read data for %s "%path)
            continue
        
        # filter the data
        data[ii] = bandpass(tdata,freqmin,freqmax,int(1/dt),corners=4, zerophase=True)

# normalize the matri
for ii in range(data.shape[0]):
    print(ii,np.max(np.abs(data[ii])))
    data[ii] /= np.max(np.abs(data[ii]))

# plotting figures
fig,ax = plt.subplots()
ax.matshow(data,cmap='seismic',extent=[-disp_lag,disp_lag,0,data.shape[0]],aspect='auto')
ax.set_title('%5.3f-%5.2f Hz'%(freqmin,freqmax))
ax.set_xlabel('time [s]')
ax.set_ylabel('distance [m]')
ax.xaxis.set_ticks_position('bottom')

# save figure or show
if savefig:
    outfname = 'source_'+source+'_moveout_'+str(freqmin)+'_'+str(freqmax)+'Hz.pdf'
    fig.savefig(outfname, format='pdf', dpi=400)
    plt.close()
else:
    plt.show()
