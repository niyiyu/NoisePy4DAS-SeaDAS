import sys
import os
import h5py
import time
import DAS_module
import numpy as np
from mpi4py import MPI

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
os.system('export HDF5_USE_FILE=FALSE')

'''
this script merges and simplies the S1-S2 NoisePy routine
for DAS data processing specifically

by Chengxin Jiang @ANU (Feb/2022)
Please consider cite the NoisePy paper (DOI:10.1785/0220190364) if you find the scripts useful for your research

TODO:
1) add options to cut off the channel pairs that are out of interests: start with one virtual source
2) deal with traces of large earthquakes -> maybe filter out during stacking?
'''

#######################################################
################PARAMETER SECTION######################
#######################################################
tt0=time.time()

# data/file paths
rootpath  = '/Users/chengxin/Documents/ANU/DAS/melb_DAS2022'            # absolute path for your project

# assemble parameters for data pre-processing
prepro_para = {'DATADIR':os.path.join(rootpath,'raw_data'),             # dir where raw data is stored
               'CCFDIR':os.path.join(rootpath,'CCF'),                   # new dir to output CCFs data 
               'flag': True,                                            # print intermediate variables and computing time 
               'MAX_MEM':4.0,                                           # maximum memory allowed per core in GB 
               # DAS parameters
               'gaug_len':10,                                           # gauge length of the array 
               'sspace':4,                                              # sensor spacing for inter-station distance (assuming linear array)
               'input_fmt':'h5',                                        # input file format options of 'h5' or 'tdmc'
               'source_type':'selected',                                # choose between "selected" (specific channels) and "all" (all channels)
               'source_chan':list(np.arange(1100,2100,100)),                 # array of the source channel if source_type = 'selected'
               'receiver_type':'adjacent',                              # choose among "adjacent" (adjacent channels), "range" (certain channel range) and "all" (all channels)
               'receiver_chan':500,                                     # if "adjacent" is chosen, the receiver channel ranges between source-receiver_chan and source+receiver_chan
               # processing parameters
               'freqmin':0.05,                                          # min frequency to filter
               'freqmax':20,                                            # max frequency to filter
               'sps':125,                                               # current sampling rate
               'samp_freq':125,                                         # targeted sampling rate
               'freq_norm':'rma',                                       # 'no' for no whitening, or 'rma' for running-mean average, 'phase_only' for sign-bit normalization in freq domain.
               'time_norm':'no',                                        # 'no' for no normalization, or 'rma', 'one_bit' for normalization in time domain
               'cc_method':'xcorr',                                     # 'xcorr' for pure cross correlation, 'deconv' for deconvolution; FOR "COHERENCY" PLEASE set freq_norm to "rma", time_norm to "no" and cc_method to "xcorr"
               'smooth_N':200,                                          # moving window length for time domain normalization if selected (points)
               'smoothspect_N':200,                                     # moving window length to smooth spectrum amplitude (points)
               'maxlag':20,                                             # lags of cross-correlation to save (sec)
               'max_over_std':10,                                       # threahold to remove window of bad signals: set it to 10*9 if prefer not to remove them
               # plotting parameters
               'plot_raw':True,                                         # plot the raw waveform as 2D matrix
               'plot_ccfs':False,                                       # show the resulted ccfs for each virtual source
               'FIGDIR':os.path.join(rootpath,'figures')}               # directory for saving figures

metadata = os.path.join(prepro_para['CCFDIR'],'prepro_fft_info.txt')
# load one sample file to get some basic parameters
allfiles,prepro_para = DAS_module.update_das_para(prepro_para) 

##########################################################
#################PROCESSING SECTION#######################
##########################################################

#---------MPI-----------
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
#-----------------------

if rank == 0:
 
    if len(allfiles)==0:
        raise ValueError('cannot find any %s files within %s'%(prepro_para['input_fmt'],prepro_para['DATADIR']))               
    splits = len(allfiles)

    # write parameter into file and have rough memory estimation
    memory_size = DAS_module.output_para_check_memory_S1(prepro_para,metadata)

    if memory_size > prepro_para['MAX_MEM']:
        raise ValueError('Require %5.3fG memory but only %5.3fG provided)! \
                          Reduce inc_hours to avoid this issue!' % (memory_size,prepro_para['MAX_MEM']))
else:
    splits,allfiles = [None for _ in range(2)]

# broadcast the variables
splits     = comm.bcast(splits,root=0)
allfiles = comm.bcast(allfiles,root=0)

# MPI: loop through each time-chunk
for ick in range(rank,splits,size):
    t0=time.time()
    nchan = prepro_para['nsta']

    with h5py.File(allfiles[ick],'r') as f:
        tdata = f['DAS'][:]
        temp  = allfiles[ick].split('/')[-1].split('_')
        ttag  = str(temp[3]+temp[4].split('.')[0])
        print(ttag)

        # perform pre-processing
        trace_stdS,dataS = DAS_module.preprocess_raw_make_stat(tdata,prepro_para,ttag)
        t1 = time.time()
        if prepro_para['flag']:
            print('pre-processing & making stat takes %6.2fs'%(t1-t0))

        # do normalization if needed
        white_spect = DAS_module.noise_processing(dataS,prepro_para)
        Nfft = white_spect.shape[1];Nfft2 = Nfft//2

        # load fft data in memory for cross-correlations
        spec_data = white_spect[:,:Nfft2]
        del dataS,white_spect
    
    # make source chan index and name output files
    sindx,cc_h5,tmpfile = DAS_module.source_index_filenames(prepro_para,ttag)

    ftmp = open(tmpfile,'w')
    t2 = time.time()
    if prepro_para['flag']:
        print('it takes %6.2fs before getting into the cross correlation'%(t2-t1))

    # loop through each source
    for iiS in sindx:
        if prepro_para['flag']:
            print('working on %dth channel'%iiS)

        # define receiver array based on options
        indx1,indx2 = DAS_module.make_receiver_array(prepro_para,iiS)   

        # smooth the source spectrum
        sfft1 = DAS_module.smooth_source_spect(spec_data[iiS],prepro_para)
        corr,tindx = DAS_module.correlate(sfft1,spec_data[indx1:indx2],prepro_para,Nfft,iiS,ttag)

        # save cross-correlation as ASDF files
        DAS_module.save_ccfs(cc_h5,prepro_para,iiS,corr)
        ftmp.write(str(iiS)+'\n')
    
    t3=time.time()
    print('it takes '+str(t3-t2)+' s to cross correlate one chunk of data')

tt1=time.time()
print('step0B takes '+str(tt1-tt0)+' s')

comm.barrier()
if rank == 0:
    prepro_para.update({'nsta':corr.shape[0], \
                        'npts':corr.shape[1]})
    # save metadata 
    fout = open(metadata,'w')
    fout.write(str(prepro_para))
    fout.close() 

    sys.exit()
