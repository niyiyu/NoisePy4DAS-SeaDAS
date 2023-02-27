import os
import glob
import obspy
import h5py
import pyasdf
import scipy
import numpy as np
from numba import jit
import plotting_module
from scipy.signal import hilbert
from scipy.fftpack import next_fast_len
from obspy.signal.filter import bandpass
from obspy.core.util.base import _get_function_from_entry_point


####################################################
############## CORE FUNCTIONS ######################
####################################################

def update_das_para(para):
    '''
    update DAS parameters by loading some sample data

    PARAMETERS:
    -----------------------
    para: dict containing all important parameters 
    '''
    allfiles_path = os.path.join(para['DATADIR'],'*.'+para['input_fmt'])                    # make sure all sac/mseed files can be found through this format
    allfiles = sorted(glob.glob(allfiles_path))

    with h5py.File(allfiles[0],"r") as f:
        tdata = f['DAS'][:]
        npts,nsta = tdata.shape

    para.update({'npts':npts,
                 'nsta':nsta,
                })    
    return allfiles,para

def output_para_check_memory_S1(prepro_para,metadata):
    '''
    '''
    # make directory
    if not os.path.isdir(prepro_para['DATADIR']):
        os.mkdir(prepro_para['DATADIR'])
    if not os.path.isdir(prepro_para['CCFDIR']):
        os.mkdir(prepro_para['CCFDIR'])

    # keep the parameter info in a file
    fout = open(metadata,'w')
    fout.write(str(prepro_para))
    fout.close()

    # rough estimation on memory needs needed in S1 (assume float32 dtype)
    memory_size = prepro_para['nsta']*prepro_para['npts']*4/1024**3   

    return memory_size 


def preprocess_raw_make_stat(tdata,prepro_para,ttag):
    '''
    this function pre-processes the raw data stream by:
        1) check samping rate and gaps in the data;
        2) remove sigularity, trend and mean of each trace
        3) filter and correct the time if integer time are between sampling points
        4) trim data to a day-long sequence and interpolate it to ensure starting at 00:00:00.000
    (used in S0A & S0B)

    PARAMETERS:
    -----------------------
    data: 2D data matrix
    prepro_para: dict containing fft parameters, such as frequency bands and selection for instrument response removal etc.
    tvec: time vector of the data

    RETURNS:
    -----------------------
    ntr: obspy stream object of cleaned, merged and filtered noise data
    '''
    
    # load paramters from fft dict
    freqmin    = prepro_para['freqmin']
    freqmax    = prepro_para['freqmax']
    samp_freq  = prepro_para['samp_freq']
    npts_chunk = prepro_para['npts']
    nchan      = prepro_para['nsta']
    sps        = prepro_para['sps']

    # check the consistency of 2D matrix for each time series
    npts,nsta = tdata.shape
    if npts!=npts_chunk or nsta!=nchan:
        raise ValueError('data size not consistent between provided input files')
    tdata = tdata.T

    # parameters for butterworth filter
    f1 = 0.9*freqmin
    f2 = freqmin
    if 1.1*freqmax > 0.45*samp_freq:
        f3 = 0.4*samp_freq
        f4 = 0.45*samp_freq
    else:
        f3 = freqmax
        f4= 1.1*freqmax
    pre_filt  = [f1,f2,f3,f4]

    # remove nan/inf, mean and trend of each trace 
    tttindx = np.where(np.isnan(tdata))
    if len(tttindx) >0:
        tdata[tttindx]=0
    tttindx = np.where(np.isinf(tdata))
    if len(tttindx) >0:
        tdata[tttindx]=0

    # 2D array processing
    tdata = demean(tdata)
    tdata = detrend(tdata)
    tdata = taper(tdata)

    # merge, taper and filter the data
    for ii in range(tdata.shape[0]):
        tdata[ii] = np.float32(bandpass(tdata[ii],
                                pre_filt[0],
                                pre_filt[-1],
                                df=sps,
                                corners=4,
                                zerophase=True))

    # make downsampling if needed
    if abs(samp_freq-sps) > 1E-4:
        decimation_factor = int(np.round(sps/samp_freq))
        if decimation_factor > 10:
            raise ValueError('more than one downsampling step is required')
        print('check')
        
        # need to test some parameters of the decimate function
        tdata = scipy.signal.decimate(tdata,
                                      decimation_factor,
                                      n=2,
                                      ftype='iir',
                                      axis=1,
                                      zero_phase=True)  

    # statistic to detect segments that may be associated with earthquakes
    trace_madS = np.zeros(nsta,dtype=np.float32)
    trace_stdS = np.zeros(nsta,dtype=np.float32)	        
    for ii in range(nsta):
        all_madS = mad(tdata[ii])           # median absolute deviation over all noise window
        all_stdS = np.std(tdata[ii])        # standard deviation over all noise window   
        trace_madS[ii] = (np.max(np.abs(tdata[ii]))/all_madS)
        trace_stdS[ii] = (np.max(np.abs(tdata[ii]))/all_stdS)
    
    # plot the matrix form of the filtered waveforms
    if prepro_para['plot_raw']:
        plotting_module.plot_raw_waveform(tdata,prepro_para,ttag)

    return trace_stdS,tdata


def noise_processing(dataS,fft_para):
    '''
    this function performs time domain and frequency domain normalization if needed. in real case, we prefer use include
    the normalization in the cross-correaltion steps by selecting coherency or decon (Prieto et al, 2008, 2009; Denolle et al, 2013)
    
    PARMAETERS:
    ------------------------
    fft_para: dictionary containing all useful variables used for fft and cc
    dataS: 2D matrix of all segmented noise data
    
    RETURNS:
    ------------------------
    source_white: 2D matrix of data spectra
    '''

    # load parameters first
    time_norm   = fft_para['time_norm']
    freq_norm   = fft_para['freq_norm']
    smooth_N    = fft_para['smooth_N']
    N = dataS.shape[0]

    #------to normalize in time or not------
    if time_norm != 'no':

        if time_norm == 'one_bit': 	# sign normalization
            white = np.sign(dataS)
        elif time_norm == 'rma': # running mean: normalization over smoothed absolute average
            white = np.zeros(shape=dataS.shape,dtype=dataS.dtype)
            for kkk in range(N):
                white[kkk,:] = dataS[kkk,:]/moving_ave(np.abs(dataS[kkk,:]),smooth_N)

    else:	# don't normalize
        white = dataS

    #-----to whiten or not------
    if freq_norm != 'no':
        source_white = whiten(white,fft_para)	# whiten and return FFT
    else:
        Nfft = int(next_fast_len(int(dataS.shape[1])))
        source_white = scipy.fftpack.fft(white, Nfft, axis=1) # return FFT

    return source_white


def source_index_filenames(prepro_para,ttag):
    '''
    '''
    # make source and receiver chan index
    if prepro_para['source_type']=='all':
        sindx = np.arange(0,prepro_para['nsta'])
    elif prepro_para['source_type']=='selected':
        sindx = prepro_para['source_chan']
    elif prepro_para['source_type']=='regular':
        sindx = prepro_para['nsta']

    outname = obspy.UTCDateTime(ttag).strftime('%Y_%m_%d_%H_%M_%S.%f')[:-4]+'T'+ \
          (obspy.UTCDateTime(ttag)+60).strftime('%Y_%m_%d_%H_%M_%S.%f')[:-4]

    tmpfile = os.path.join(prepro_para['CCFDIR'],outname+'.tmp')
    cc_h5 = os.path.join(prepro_para['CCFDIR'],outname+'.h5')

    return sindx,cc_h5,tmpfile


def make_receiver_array(para,source_indx):
    '''
    '''

    # check the receiver type
    if para['receiver_type']=="all":
        indx1 = 0
        indx2 = para['nsta']
    elif para['receiver_type']=="adjacent":
        indx1 = source_indx-para['receiver_chan']
        if indx1<0:
            indx1 = 0 
        indx2 = source_indx+para['receiver_chan']
        if indx2>para['nsta']:
            indx2 = para['nsta']
    
    return indx1,indx2


def smooth_source_spect(fft1,cc_para):
    '''
    this function smoothes amplitude spectrum of the 2D spectral matrix. (used in S1)
    PARAMETERS:
    ---------------------
    cc_para: dictionary containing useful cc parameters
    fft1:    source spectrum matrix

    RETURNS:
    ---------------------
    sfft1: complex numpy array with normalized spectrum
    '''
    cc_method = cc_para['cc_method']
    smoothspect_N = cc_para['smoothspect_N']

    if cc_method == 'deconv':

        #-----normalize single-station cc to z component-----
        temp = moving_ave(np.abs(fft1),smoothspect_N)
        try:
            sfft1 = np.conj(fft1)/temp**2
        except Exception:
            raise ValueError('smoothed spectrum has zero values')

    elif cc_method == 'coherency':
        temp = moving_ave(np.abs(fft1),smoothspect_N)
        try:
            sfft1 = np.conj(fft1)/temp
        except Exception:
            raise ValueError('smoothed spectrum has zero values')

    elif cc_method == 'xcorr':
        sfft1 = np.conj(fft1)

    else:
        raise ValueError('no correction correlation method is selected at L59')

    return sfft1


def correlate(fft1_smoothed_abs,fft2,D,Nfft,ssta,ttag):
    '''
    this function does the cross-correlation in freq domain and has the option to keep sub-stacks of
    the cross-correlation if needed. it takes advantage of the linear relationship of ifft, so that
    stacking is performed in spectrum domain first to reduce the total number of ifft. (used in S1)
    PARAMETERS:
    ---------------------
    fft1_smoothed_abs: smoothed power spectral density of the FFT for the source station
    fft2: raw FFT spectrum of the receiver station
    D: dictionary containing following parameters:
        maxlag:  maximum lags to keep in the cross correlation
        dt:      sampling rate (in s)
        nwin:    number of segments in the 2D matrix
        method:  cross-correlation methods selected by the user
        freqmin: minimum frequency (Hz)
        freqmax: maximum frequency (Hz)
    Nfft:    number of frequency points for ifft
    ssta: name of the virtual source.
    
    RETURNS:
    ---------------------
    s_corr: 1D or 2D matrix of the averaged or sub-stacks of cross-correlation functions in time domain
    t_corr: timestamp for each sub-stack or averaged function
    n_corr: number of included segments for each sub-stack or averaged function

    MODIFICATIONS:
    ---------------------
    output the linear stack of each time chunk even when substack is selected (by Chengxin @Aug2020)
    '''
    #----load paramters----
    sps     = D['samp_freq']
    dt      = 1/sps
    maxlag  = D['maxlag']
    method  = D['cc_method']
    smoothspect_N = D['smoothspect_N']

    nwin  = fft2.shape[0]
    Nfft2 = fft2.shape[1]

    #------convert all 2D arrays into 1D to speed up--------
    corr = np.zeros(nwin*Nfft2,dtype=np.complex64)
    fft1 = np.ones(shape=(nwin,1))*fft1_smoothed_abs.reshape(1,fft1_smoothed_abs.size)  # duplicate fft1_smoothed_abs for nwin rows
    corr = fft1.reshape(fft1.size,)*fft2.reshape(fft2.size,)

    if method == "coherency":
        temp = moving_ave(np.abs(fft2.reshape(fft2.size,)),smoothspect_N)
        corr /= temp
    corr  = corr.reshape(nwin,Nfft2)

    # loop through each cross correlation
    s_corr = np.zeros(shape=(nwin,Nfft),dtype=np.float32)   # stacked correlation
    ampmax = np.zeros(nwin,dtype=np.float32)
    crap   = np.zeros(Nfft,dtype=np.complex64)
    for i in range(nwin):
        crap[:Nfft2] = corr[i,:]
        crap[:Nfft2] = crap[:Nfft2]-np.mean(crap[:Nfft2])   # remove the mean in freq domain (spike at t=0)
        crap[-(Nfft2)+1:] = np.flip(np.conj(crap[1:(Nfft2)]),axis=0)
        crap[0]=complex(0,0)
        s_corr[i,:] = np.real(np.fft.ifftshift(scipy.fftpack.ifft(crap, Nfft, axis=0)))

    # remove abnormal trace: to be improved
    '''
    ampmax = np.max(s_corr,axis=1)
    tindx  = np.where( (ampmax<20*np.median(ampmax)) & (ampmax>0))[0]
    s_corr = s_corr[tindx,:]
    '''
    tindx = []

    # #####################################
    # t = np.arange(-Nfft2+1, Nfft2)*dt
    # the t vector is defined incorrectly in the previous version, as the starting time window 
    # should start from -Nfft2*dt rather than -Nfft2+1. This causes the cross-correlation to shift
    # 1 sample point to the positive axis, which is particularly problematic for long-period studies. 
    # this bug can not be found without Dr. Xingli Fan's help! Thank you Xingli! 
    ########################################
    t = np.arange(-Nfft2,Nfft2)*dt
    ind = np.where(np.abs(t) <= maxlag)[0]
    if s_corr.ndim==1:
        s_corr = s_corr[ind]
    elif s_corr.ndim==2:
        s_corr = s_corr[:,ind]

    # plotting figures or not
    if D['plot_ccfs']:
        plotting_module.plot_moveout(s_corr,D,ssta,ttag)

    return s_corr,tindx


def save_ccfs(cc_h5,prepro_para,sta_indx,corr):
    '''
    '''
    with pyasdf.ASDFDataSet(cc_h5,mpi=False) as ccf_ds:
        # use the channel number as a way to figure out distance
        param = {'sspace':prepro_para['sspace'],
                 'nchan':corr.shape[0],
                 'npts':corr.shape[1],
                 'dt': 1/prepro_para['samp_freq'],
                 'maxlag':prepro_para['maxlag'],
                 'freqmin':prepro_para['freqmin'],
                 'freqmax':prepro_para['freqmax']}
    
        # source-receiver pair
        data_type = 'C'+format(sta_indx,'04d')
        path = 'ZZ'
        ccf_ds.add_auxiliary_data(data=corr, 
                                  data_type=data_type, 
                                  path=path, 
                                  parameters=param)


def output_para_check_memory_S2(stack_para):
    '''
    '''
    # save fft metadata for future reference
    stack_metadata  = os.path.join(stack_para['STACKDIR'],'stack_data.txt') 

    # save metadata 
    fout = open(stack_metadata,'w')
    fout.write(str(stack_para))
    fout.close()

    # crude estimation on memory needs (assume float32)
    num_chunck = stack_para['num_chunk']
    npts_chunk = stack_para['npts_chunk']
    memory_size = num_chunck*npts_chunk*stack_para['nsta']*4/1024**3

    return memory_size
        

def stacking(cc_array,stack_para,stack_h5,tpara):
    '''
    this function stacks the cross correlation data according to the user-defined substack_len parameter

    PARAMETERS:
    ----------------------
    cc_array: 2D numpy float32 matrix containing all segmented cross-correlation data
    cc_time:  1D numpy array of timestamps for each segment of cc_array
    cc_ngood: 1D numpy int16 matrix showing the number of segments for each sub-stack and/or full stack
    stack_para: a dict containing all stacking parameters

    RETURNS:
    ----------------------
    cc_array, cc_ngood, cc_time: same to the input parameters but with abnormal cross-correaltions removed
    allstacks1: 1D matrix of stacked cross-correlation functions over all the segments
    nstacks:    number of overall segments for the final stacks
    '''
    # load useful parameters from dict
    samp_freq = stack_para['samp_freq']
    smethod   = stack_para['stack_method']
    comp      = stack_para['comp']
    nsta,npts_chunk = stack_para['nsta'],stack_para['npts_chunk']
    stack_method = stack_para['stack_method']

    # remove abnormal data
    ampmax = np.max(cc_array,axis=1)
    tindx  = np.where( (ampmax<20*np.median(ampmax)) & (ampmax>0))[0]
    if not len(tindx):
        allstacks1=[];allstacks2=[];allstacks3=[]
        nstacks=0;cc_array=[]
        return cc_array,allstacks1,allstacks2,allstacks3,nstacks
    else:

        # remove ones with bad amplitude
        cc_array = cc_array[tindx,:]

        # do stacking
        allstacks1 = np.zeros(npts_chunk*nsta,dtype=np.float32)
        allstacks2 = np.zeros(npts_chunk*nsta,dtype=np.float32)
        allstacks3 = np.zeros(npts_chunk*nsta,dtype=np.float32)

        if smethod == 'linear':
            allstacks1 = np.mean(cc_array,axis=0)
        elif smethod == 'pws':
            allstacks1 = pws(cc_array,samp_freq)
        elif smethod == 'robust':
            allstacks1,w,nstep = robust_stack(cc_array,0.001)

    allstacks1 = allstacks1.reshape(nsta,npts_chunk)

    if not len(allstacks1):
        return cc_array,allstacks1,allstacks2,allstacks3,nstacks

    # write stacked data into ASDF file
    with pyasdf.ASDFDataSet(stack_h5,mpi=False) as ds:
        if stack_method != 'all':
            data_type = 'Allstack_'+stack_method
            allstacks1 = allstacks1.reshape(nsta,npts_chunk)
            ds.add_auxiliary_data(data=allstacks1, 
                                    data_type=data_type, 
                                    path=comp, 
                                    parameters=tpara)
        else:
            ds.add_auxiliary_data(data=allstacks1, 
                                    data_type='Allstack_linear', 
                                    path=comp, 
                                    parameters=tpara)
            ds.add_auxiliary_data(data=allstacks2, 
                                    data_type='Allstack_pws', 
                                    path=comp, 
                                    parameters=tpara)
            ds.add_auxiliary_data(data=allstacks3, 
                                    data_type='Allstack_robust', 
                                    path=comp, 
                                    parameters=tpara)

    # keep a track of all sub-stacked data from S1
    if stack_para['keep_substack']:
        for ii in range(cc_array.shape[0]):
            with pyasdf.ASDFDataSet(stack_h5,mpi=False) as ds:
                data_type = 'T'+str(int(allstacks3[ii]))
                ds.add_auxiliary_data(data=cc_array[ii], 
                                        data_type=data_type, 
                                        path=comp, 
                                        parameters=tpara)        



####################################################
############## UTILITY FUNCTIONS ###################
####################################################

def detrend(data):
    '''
    this function removes the signal trend based on QR decomposion
    NOTE: QR is a lot faster than the least square inversion used by
    scipy (also in obspy).
    PARAMETERS:
    ---------------------
    data: input data matrix
    RETURNS:
    ---------------------
    data: data matrix with trend removed
    '''
    #ndata = np.zeros(shape=data.shape,dtype=data.dtype)
    if data.ndim == 1:
        npts = data.shape[0]
        X = np.ones((npts,2))
        X[:,0] = np.arange(0,npts)/npts
        Q,R = np.linalg.qr(X)
        rq  = np.dot(np.linalg.inv(R),Q.transpose())
        coeff = np.dot(rq,data)
        data = data-np.dot(X,coeff)
    elif data.ndim == 2:
        npts = data.shape[1]
        X = np.ones((npts,2))
        X[:,0] = np.arange(0,npts)/npts
        Q,R = np.linalg.qr(X)
        rq = np.dot(np.linalg.inv(R),Q.transpose())
        for ii in range(data.shape[0]):
            coeff = np.dot(rq,data[ii])
            data[ii] = data[ii] - np.dot(X,coeff)
    return data

def demean(data):
    '''
    this function remove the mean of the signal
    PARAMETERS:
    ---------------------
    data: input data matrix
    RETURNS:
    ---------------------
    data: data matrix with mean removed
    '''
    #ndata = np.zeros(shape=data.shape,dtype=data.dtype)
    if data.ndim == 1:
        data = data-np.mean(data)
    elif data.ndim == 2:
        for ii in range(data.shape[0]):
            data[ii] = data[ii]-np.mean(data[ii])
    return data

def taper(data):
    '''
    this function applies a cosine taper using obspy functions
    PARAMETERS:
    ---------------------
    data: input data matrix
    RETURNS:
    ---------------------
    data: data matrix with taper applied
    '''
    #ndata = np.zeros(shape=data.shape,dtype=data.dtype)
    if data.ndim == 1:
        npts = data.shape[0]
        # window length
        if npts*0.05>20:
            wlen = 20
        else:
            wlen = npts*0.05
        # taper values
        func = _get_function_from_entry_point('taper', 'hann')
        if 2*wlen == npts:
            taper_sides = func(2*wlen)
        else:
            taper_sides = func(2*wlen+1)
        # taper window
        win  = np.hstack((taper_sides[:wlen], np.ones(npts-2*wlen),taper_sides[len(taper_sides) - wlen:]))
        data *= win
    elif data.ndim == 2:
        npts = data.shape[1]
        # window length
        if npts*0.05>20:
            wlen = 20
        else:
            wlen = npts*0.05
        # taper values
        func = _get_function_from_entry_point('taper', 'hann')
        if 2*wlen == npts:
            taper_sides = func(2*wlen)
        else:
            taper_sides = func(2*wlen + 1)
        # taper window
        win  = np.hstack((taper_sides[:wlen], np.ones(npts-2*wlen),taper_sides[len(taper_sides) - wlen:]))
        for ii in range(data.shape[0]):
            data[ii] *= win
    return data

def mad(arr):
    """
    Median Absolute Deviation: MAD = median(|Xi- median(X)|)
    PARAMETERS:
    -------------------
    arr: numpy.ndarray, seismic trace data array
    RETURNS:
    data: Median Absolute Deviation of data
    """
    if not np.ma.is_masked(arr):
        med = np.median(arr)
        data = np.median(np.abs(arr - med))
    else:
        med = np.ma.median(arr)
        data = np.ma.median(np.ma.abs(arr-med))
    return data

@jit(nopython = True)
def moving_ave(A,N):
    '''
    this Numba compiled function does running smooth average for an array.
    PARAMETERS:
    ---------------------
    A: 1-D array of data to be smoothed
    N: integer, it defines the half window length to smooth

    RETURNS:
    ---------------------
    B: 1-D array with smoothed data
    '''
    A = np.concatenate((A[:N],A,A[-N:]),axis=0)
    B = np.zeros(A.shape,A.dtype)

    tmp=0.
    for pos in range(N,A.size-N):
        # do summing only once
        if pos==N:
            for i in range(-N,N+1):
                tmp+=A[pos+i]
        else:
            tmp=tmp-A[pos-N-1]+A[pos+N]
        B[pos]=tmp/(2*N+1)
        if B[pos]==0:
            B[pos]=1
    return B[N:-N]


def whiten(data, fft_para):
    '''
    This function takes 1-dimensional timeseries array, transforms to frequency domain using fft,
    whitens the amplitude of the spectrum in frequency domain between *freqmin* and *freqmax*
    and returns the whitened fft.
    PARAMETERS:
    ----------------------
    data: numpy.ndarray contains the 1D time series to whiten
    fft_para: dict containing all fft_cc parameters such as
        dt: The sampling space of the `data`
        freqmin: The lower frequency bound
        freqmax: The upper frequency bound
        smooth_N: integer, it defines the half window length to smooth
        freq_norm: whitening method between 'one-bit' and 'RMA'
    RETURNS:
    ----------------------
    FFTRawSign: numpy.ndarray contains the FFT of the whitened input trace between the frequency bounds
    '''

    # load parameters
    samp_freq = fft_para['samp_freq']
    delta   = 1/samp_freq
    freqmin = fft_para['freqmin']
    freqmax = fft_para['freqmax']
    smooth_N  = fft_para['smoothspect_N']
    freq_norm = fft_para['freq_norm']

    # Speed up FFT by padding to optimal size for FFTPACK
    if data.ndim == 1:
        axis = 0
    elif data.ndim == 2:
        axis = 1

    Nfft = int(next_fast_len(int(data.shape[axis])))

    Napod = 100
    Nfft = int(Nfft)
    freqVec = scipy.fftpack.fftfreq(Nfft, d=delta)[:Nfft // 2]
    J = np.where((freqVec >= freqmin) & (freqVec <= freqmax))[0]
    low = J[0] - Napod
    if low <= 0:
        low = 1

    left = J[0]
    right = J[-1]
    high = J[-1] + Napod
    if high > Nfft/2:
        high = int(Nfft//2)

    FFTRawSign = scipy.fftpack.fft(data, Nfft,axis=axis)
    # Left tapering:
    if axis == 1:
        FFTRawSign[:,0:low] *= 0
        FFTRawSign[:,low:left] = np.cos(
            np.linspace(np.pi / 2., np.pi, left - low)) ** 2 * np.exp(
            1j * np.angle(FFTRawSign[:,low:left]))
        # Pass band:
        if freq_norm == 'phase_only':
            FFTRawSign[:,left:right] = np.exp(1j * np.angle(FFTRawSign[:,left:right]))
        elif freq_norm == 'rma':
            for ii in range(data.shape[0]):
                tave = moving_ave(np.abs(FFTRawSign[ii,left:right]),smooth_N)
                FFTRawSign[ii,left:right] = FFTRawSign[ii,left:right]/tave
        # Right tapering:
        FFTRawSign[:,right:high] = np.cos(
            np.linspace(0., np.pi / 2., high - right)) ** 2 * np.exp(
            1j * np.angle(FFTRawSign[:,right:high]))
        FFTRawSign[:,high:Nfft//2] *= 0

        # Hermitian symmetry (because the input is real)
        FFTRawSign[:,-(Nfft//2)+1:] = np.flip(np.conj(FFTRawSign[:,1:(Nfft//2)]),axis=axis)
    else:
        FFTRawSign[0:low] *= 0
        FFTRawSign[low:left] = np.cos(
            np.linspace(np.pi / 2., np.pi, left - low)) ** 2 * np.exp(
            1j * np.angle(FFTRawSign[low:left]))
        # Pass band:
        if freq_norm == 'phase_only':
            FFTRawSign[left:right] = np.exp(1j * np.angle(FFTRawSign[left:right]))
        elif freq_norm == 'rma':
            tave = moving_ave(np.abs(FFTRawSign[left:right]),smooth_N)
            FFTRawSign[left:right] = FFTRawSign[left:right]/tave
        # Right tapering:
        FFTRawSign[right:high] = np.cos(
            np.linspace(0., np.pi / 2., high - right)) ** 2 * np.exp(
            1j * np.angle(FFTRawSign[right:high]))
        FFTRawSign[high:Nfft//2] *= 0

        # Hermitian symmetry (because the input is real)
        FFTRawSign[-(Nfft//2)+1:] = FFTRawSign[1:(Nfft//2)].conjugate()[::-1]

    return FFTRawSign

def pws(arr,sampling_rate,power=2,pws_timegate=5.):
    '''
    Performs phase-weighted stack on array of time series. Modified on the noise function by Tim Climents.
    Follows methods of Schimmel and Paulssen, 1997.
    If s(t) is time series data (seismogram, or cross-correlation),
    S(t) = s(t) + i*H(s(t)), where H(s(t)) is Hilbert transform of s(t)
    S(t) = s(t) + i*H(s(t)) = A(t)*exp(i*phi(t)), where
    A(t) is envelope of s(t) and phi(t) is phase of s(t)
    Phase-weighted stack, g(t), is then:
    g(t) = 1/N sum j = 1:N s_j(t) * | 1/N sum k = 1:N exp[i * phi_k(t)]|^v
    where N is number of traces used, v is sharpness of phase-weighted stack

    PARAMETERS:
    ---------------------
    arr: N length array of time series data (numpy.ndarray)
    sampling_rate: sampling rate of time series arr (int)
    power: exponent for phase stack (int)
    pws_timegate: number of seconds to smooth phase stack (float)

    RETURNS:
    ---------------------
    weighted: Phase weighted stack of time series data (numpy.ndarray)
    '''

    if arr.ndim == 1:
        return arr
    N,M = arr.shape
    analytic = hilbert(arr,axis=1, N=next_fast_len(M))[:,:M]
    phase = np.angle(analytic)
    phase_stack = np.mean(np.exp(1j*phase),axis=0)
    phase_stack = np.abs(phase_stack)**(power)

    # smoothing
    #timegate_samples = int(pws_timegate * sampling_rate)
    #phase_stack = moving_ave(phase_stack,timegate_samples)
    weighted = np.multiply(arr,phase_stack)
    return np.mean(weighted,axis=0)


def robust_stack(cc_array,epsilon):
    """
    this is a robust stacking algorithm described in Palvis and Vernon 2010

    PARAMETERS:
    ----------------------
    cc_array: numpy.ndarray contains the 2D cross correlation matrix
    epsilon: residual threhold to quit the iteration
    RETURNS:
    ----------------------
    newstack: numpy vector contains the stacked cross correlation

    Written by Marine Denolle
    """
    res  = 9E9  # residuals
    w = np.ones(cc_array.shape[0])
    nstep=0
    newstack = np.median(cc_array,axis=0)
    while res > epsilon:
        stack = newstack
        for i in range(cc_array.shape[0]):
            crap = np.multiply(stack,cc_array[i,:].T)
            crap_dot = np.sum(crap)
            di_norm = np.linalg.norm(cc_array[i,:])
            ri = cc_array[i,:] -  crap_dot*stack
            ri_norm = np.linalg.norm(ri)
            w[i]  = np.abs(crap_dot) /di_norm/ri_norm#/len(cc_array[:,1])
        # print(w)
        w =w /np.sum(w)
        newstack =np.sum( (w*cc_array.T).T,axis=0)#/len(cc_array[:,1])
        res = np.linalg.norm(newstack-stack,ord=1)/np.linalg.norm(newstack)/len(cc_array[:,1])
        nstep +=1
        if nstep>10:
            return newstack, w, nstep
    return newstack, w, nstep
