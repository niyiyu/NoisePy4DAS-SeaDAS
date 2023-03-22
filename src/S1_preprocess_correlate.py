import sys
sys.path.append("/tmp/DASStore")
# sys.path.append("/home/niyiyu/notebooks/DASStore")

import os
import time
import numpy as np
import DAS_module
from mpi4py import MPI
from datetime import datetime, timedelta
from dasstore.zarr import Client
import tiledb

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")


os.system('export HDF5_USE_FILE=FALSE')

'''
this script merges and simplies the S1-S2 NoisePy routine
for DAS data processing specifically

by Chengxin Jiang @ANU (Feb/2022)

TODO:
0) interpolate the data into regular time range if each h5 has different length;
1) add options to cut off the channel pairs that are out of interests
2) deal with traces of large earthquakes -> maybe filter out during stacking?
'''

#######################################################
################PARAMETER SECTION######################
#######################################################
tt0=time.time()

# pnwstore1_endpoint = "pnwstore1.ess.washington.edu:9000"

ccf_bucket = "seadas-december-2022-ccf"
bucket = "seadas-december-2022"
endpoint = "s3.us-west-2.amazonaws.com"

config = tiledb.Config()
config["vfs.s3.scheme"] = "https"
config["vfs.s3.region"] = "us-west-2"
config["vfs.s3.endpoint_override"] = ""
config["vfs.s3.use_virtual_addressing"] = "true"
ctx = tiledb.Ctx(config)

secure = True
role_assigned = True

# useful parameters for preprocessing the data
input_fmt   = 'zarr'                                                      # input file format between 'sac' and 'mseed' 
sps         = 100                                                         # current sampling rate
samp_freq   = 50                                                          # targeted sampling rate
freqmin     = 1                                                           # pre filtering frequency bandwidth
freqmax     = 20                                                          # note this cannot exceed Nquist freq
flag        = False                                                       # print intermediate variables and computing time
gaug_len    = 2                                                           # gauge length of the array for inter-station distance (assuming linear array)

# useful parameters for cross correlating the data
freq_norm   = 'phase_only'                                              # 'no' for no whitening, or 'rma' for running-mean average, 'phase_only' for sign-bit normalization in freq domain.
time_norm   = 'one_bit'                                                 # 'no' for no normalization, or 'rma', 'one_bit' for normalization in time domain
cc_method   = 'xcorr'                                                   # 'xcorr' for pure cross correlation, 'deconv' for deconvolution; FOR "COHERENCY" PLEASE set freq_norm to "rma", time_norm to "no" and cc_method to "xcorr"
smooth_N    = 100                                                       # moving window length for time domain normalization if selected (points)
smoothspect_N  = 100                                                    # moving window length to smooth spectrum amplitude (points)
maxlag      = 10                                                        # lags of cross-correlation to save (sec)

cc_len      = 60                                                             # cross-correlation time window length
step        = 60                                                             # striding
npts_chunk  = sps*cc_len

# criteria for data selection
max_over_std = 10                                                       # threahold to remove window of bad signals: set it to 10*9 if prefer not to remove them

# maximum memory allowed per core in GB
MAX_MEM   = 4.0        

# channel list
cha_list = np.array(range(500, 1100))  
nsta = len(cha_list)
# n_pair = int(nsta * (nsta-1)/2)                                       # if only cross-correlate
n_pair = int((nsta+1)*nsta/2)                                           # if with auto-correlation

# worker info
ARRAY_SIZE = int(os.environ['JOB_ARRAY_SIZE'])
ARRAY_INDEX = int(os.environ['AWS_BATCH_JOB_ARRAY_INDEX'])
# ARRAY_SIZE = 1
# ARRAY_INDEX = 0

#---------MPI-----------
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
#-----------------------

# rank = 0
# size = 1

# prepare client and time range
client = Client(bucket, endpoint, secure = secure, role_assigned = role_assigned)
if rank == 0:
    acq_t0 = datetime.fromisoformat(client.meta['acquisition.acquisition_start_time'])
    acq_t1 = datetime.fromisoformat(client.meta['acquisition.acquisition_end_time'])
    n_hour = ((acq_t1 - acq_t0).days + 1) * 24
    # n_days = (t1 - t0 + timedelta(days=1)).days
    # minute_list = [t0 + timedelta(days = i) for i in range(n_days)]
else:
    # date_list = []
    acq_t0 = 0
    n_hour = 0

# split jobs by rank
# date_list   = comm.bcast(date_list,root=0)
n_hour = comm.bcast(n_hour,root=0)
acq_t0 = datetime.fromtimestamp(comm.bcast(acq_t0.timestamp(),root=0))
array_split = np.array_split(np.arange(n_hour), ARRAY_SIZE)[ARRAY_INDEX] 
rank_split = np.array_split(array_split, size)[rank]

##################################################
# assemble parameters for data pre-processing
prepro_para = {'CCF_BUCKET':ccf_bucket,
               'input_fmt':input_fmt,
               'freqmin':freqmin,
               'freqmax':freqmax,
               'sps':sps,
               'npts_chunk':cc_len*sps,
               'nsta':nsta,
               'cha_list':cha_list,
               'samp_freq':samp_freq,
               'freq_norm':freq_norm,
               'time_norm':time_norm,
               'cc_method':cc_method,
               'smooth_N':smooth_N,
               'smoothspect_N':smoothspect_N,
               'maxlag':maxlag,
               'max_over_std':max_over_std,
               'MAX_MEM':MAX_MEM}
# metadata = os.path.join(CCFDIR,'prepro_fft_info.txt') 

##########################################################
#################PROCESSING SECTION#######################
##########################################################

if rank == 0:
    # output parameter info
    # fout = open(metadata,'w')
    # fout.write(str(prepro_para));fout.close()

    # rough estimation on memory needs needed in S1 (assume float32 dtype)
    memory_size = nsta*npts_chunk*4/1024**3
    if memory_size > MAX_MEM:
        raise ValueError('Require %5.3fG memory but only %5.3fG provided)! Reduce inc_hours to avoid this issue!' % (memory_size,MAX_MEM))

# MPI: loop through each time-chunk
for i in rank_split:
    t0=time.time()

    corr_full = np.zeros([1001, n_pair])
    stack_full = np.zeros([1, n_pair], dtype = int)
    starthour = acq_t0 + timedelta(hours = int(i))

    # for each hour
    for ii in range(60):
        starttime = starthour + timedelta(minutes = int(ii))
        endtime   = starthour + timedelta(minutes = int(ii+1))
        tdata = client.get_data(cha_list, starttime, endtime).T
        print(starttime)

        # check if data has NaN
        if not np.isnan(tdata).any():

            # perform pre-processing
            trace_stdS,dataS = DAS_module.preprocess_raw_make_stat(tdata,prepro_para)
            t1 = time.time()
            if flag:
                print('pre-processing & make stat takes %6.2fs'%(t1-t0))

            # do normalization if needed
            white_spect = DAS_module.noise_processing(dataS,prepro_para)
            Nfft = white_spect.shape[1];Nfft2 = Nfft//2

            # load fft data in memory for cross-correlations
            data = white_spect[:,:Nfft2]
            del dataS,white_spect
            
            # remove channel of potential local traffic noise
            ind = np.where((trace_stdS<prepro_para['max_over_std']) &
                            (trace_stdS>0) &
                            (np.isnan(trace_stdS)==0))[0]
            if not len(ind):
                raise ValueError('the max_over_std criteria is too high which results in no data')
            sta = cha_list[ind]
            white_spect = data[ind]

            # do cross-correlation now
            temp = starttime.strftime('%Y_%m_%d_%H_%M_%S.%f')[:-4]+'T'+ endtime.strftime('%Y_%m_%d_%H_%M_%S.%f')[:-4]
            t2 = time.time()
            if flag:
                print('it takes %6.2fs before getting into the cross correlation'%(t2-t1))

                
            for iiS in range(len(sta)):             # range(len(sta)) if only cross-correlate
                if flag:
                    print('working on source %s'%sta[iiS])

                # smooth the source spectrum
                sfft1 = DAS_module.smooth_source_spect(white_spect[iiS],prepro_para)
                corr,tindx = DAS_module.correlate(sfft1, white_spect[iiS:],prepro_para,Nfft)

                # update the receiver list
                tsta = sta[iiS:]
                receiver_lst = tsta[tindx]

                iS = int((500 + 1200 - sta[iiS] + 1) * (sta[iiS] - 500)/2)

                # stacking one minute
                corr_full[:, iS + receiver_lst - sta[iiS]] += corr.T
                stack_full[:, iS + receiver_lst - sta[iiS]] += 1

        t3=time.time()
        print('it takes '+str(t3-t2)+' s to cross correlate one chunk of data')
    
    with tiledb.open(f"s3://{ccf_bucket}/", 'w', ctx = ctx) as A:
        A[:, :, i] = corr_full / stack_full

tt1=time.time()
print('step0B takes '+str(tt1-tt0)+' s')

# comm.barrier()
# if rank == 0:
#     sys.exit()