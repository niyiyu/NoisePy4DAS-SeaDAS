import os
import obspy
import numpy as np 
import matplotlib.pyplot as plt

'''
a script gathers some functions to visualize the DAS data and outputs

by Chengxin Jiang @ANU(Apr2022)
'''

def plot_raw_waveform(data,prepro_para,ttag):
    '''
    plot the matrix/waveform of the filtered 1min DAS data

    PARAMETERS:
    -----------
    data:
    prepro_para:
    '''
    # load parameters
    nsta,npts = data.shape
    sps = prepro_para['samp_freq']
    taxis = np.linspace(0,(npts-1)/sps,sps)
    freqmin,freqmax = prepro_para['freqmin'],prepro_para['freqmax']

    # do normalization
    for ii in range(nsta):
        data[ii] /= np.max(np.abs(data[ii]))

    # check output folder
    if not os.path.isdir(prepro_para['FIGDIR']):
        os.mkdir(prepro_para['FIGDIR'])
    outfname = prepro_para['FIGDIR']+'/'+ \
               obspy.UTCDateTime(ttag).strftime('%Y_%m_%d_%H_%M_%S.%f')[:-4]+'_raw.pdf'

    # plotting figures
    fig,ax = plt.subplots()
    ax.matshow(data,cmap='seismic',extent=[taxis[0],taxis[-1],nsta,0],aspect='auto')
    ax.set_title('%5.3f-%5.2f Hz'%(freqmin,freqmax))
    ax.set_xlabel('time [s]')
    ax.set_ylabel('channel index')
    ax.xaxis.set_ticks_position('bottom')

    # save figures
    fig.savefig(outfname, format='pdf', dpi=400)
    plt.close()  


def plot_moveout(corr,prepro_para,ssta,ttag):
    '''
    show the moveout of the resulted CCFs
    '''
    # load parameters
    maxlag = prepro_para['maxlag']
    freqmin,freqmax = prepro_para['freqmin'],prepro_para['freqmax']

    # do normalization
    for ii in range(corr.shape[0]):
        corr[ii] /= np.max(np.abs(corr[ii]))    

    # check output dir
    if not os.path.isdir(prepro_para['FIGDIR']):
        os.mkdir(prepro_para['FIGDIR'])
    outfname = prepro_para['FIGDIR'] + '/' +\
               obspy.UTCDateTime(ttag).strftime('%Y_%m_%d_%H_%M_%S.%f')[:-4]+ '_C'+\
               str(ssta)+'_moveout_'+str(freqmin)+'_'+str(freqmax)+'Hz.pdf'

    # plotting figures
    fig,ax = plt.subplots()
    ax.matshow(corr,cmap='seismic',extent=[-maxlag,maxlag,0,corr.shape[0]],aspect='auto')
    ax.set_title('%5.3f-%5.2f Hz'%(freqmin,freqmax))
    ax.set_xlabel('time [s]')
    ax.set_ylabel('channel index')
    ax.xaxis.set_ticks_position('bottom')

    # save figure
    fig.savefig(outfname, format='pdf', dpi=400)
    plt.close()