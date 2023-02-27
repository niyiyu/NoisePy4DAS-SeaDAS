import sys
import time
import pyasdf
import os, glob
import numpy as np
import DAS_module

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

'''
Stacking script of NoisePy_4DAS to:
    1) load cross-correlation data for sub-stacking (if needed) and all-time average;
    2) stack data with either linear or phase weighted stacking (pws) methods (or both);
    3) save outputs in ASDF or SAC format depend on user's choice (for latter option, find the script of write_sac
       in the folder of application_modules;

Authors: Chengxin Jiang (chengxin_jiang@fas.harvard.edu)
         Marine Denolle (mdenolle@fas.harvard.edu)

NOTE: 
    This version stacks the 2D matrix in default, which is expected to consume large memory. This is
    the fundamental reason why no MPI is applied. Seek the MPI version for stacking by channel-pair.
'''

tt0=time.time()

########################################
#########PARAMETER SECTION##############
########################################

# absolute path parameters
rootpath  = '/Users/chengxin/Documents/ANU/DAS/melb_DAS2022'        # root path for this data processing
CCFDIR    = os.path.join(rootpath,'CCF')                            # dir where CC data is stored
STACKDIR  = os.path.join(rootpath,'STACK')                          # dir where stacked data is going to

# load fc_para parameters from Step1
ccfiles   = sorted(glob.glob(os.path.join(CCFDIR,'*.h5')))
fc_metadata = os.path.join(CCFDIR,'prepro_fft_info.txt')
fc_para     = eval(open(fc_metadata).read())

# make a dictionary to store all variables: also for later cc
stack_para={'rootpath':rootpath,
            'STACKDIR':STACKDIR,
            # parameters from S1
            'samp_freq':fc_para['samp_freq'],
            'maxlag':fc_para['maxlag'],
            'npts':fc_para['npts'],
            'nsta':fc_para['nsta'],
            'MAX_MEM':fc_para['MAX_MEM'],
            # parameters for stacking
            'num_chunk':len(ccfiles),                                           # number of cross correlation time chunk to be stacked
            'npts_chunk': int(2*fc_para['maxlag']*fc_para['samp_freq'])+1,      #
            'comp':'ZZ',                                                        # cross correlation path names
            'keep_substack':False,                                              # keep all sub-stacks in final ASDF file
            'flag':True,                                                        # output intermediate args for debugging
            'stack_method':'linear'}                                            # linear, pws, robust, nroot, selective, auto_covariance or all

#######################################
###########PROCESSING SECTION##########
#######################################

if not os.path.isdir(STACKDIR):
    os.mkdir(STACKDIR)

# write parameter into file and have rough memory estimation
memory_size = DAS_module.output_para_check_memory_S2(stack_para)
if memory_size > stack_para['MAX_MEM']:
    raise ValueError('Require %5.3fG memory but only %5.3fG provided)! \
                        Cannot load cc data all once!' % (memory_size,stack_para['MAX_MEM']))

if stack_para['flag']:
    print('Good on memory (need %5.2f G and %s G provided)!' % (memory_size,stack_para['MAX_MEM']))

source_chan = fc_para['source_chan']


# loop through each virtual source
for ista in range (len(source_chan)):
    t0=time.time()
    print(stack_para['num_chunk'])

    if stack_para['flag']:
        print('working on %dth source'%(source_chan[ista]))
    
    # naming output files
    ssta = 'C'+format(source_chan[ista],'04d')
    toutfn = os.path.join(STACKDIR,ssta+'.tmp')   
    if os.path.isfile(toutfn):
        continue 
    stack_h5 = os.path.join(STACKDIR,ssta+'.h5')       

    # allocate array to store fft data/info
    cc_array = np.zeros((stack_para['num_chunk'],stack_para['nsta']*stack_para['npts_chunk']),dtype=np.float32)

    # loop through all time-chuncks
    for ii,ifile in enumerate(ccfiles):

        # load the data from daily compilation
        ds=pyasdf.ASDFDataSet(ifile,mpi=False,mode="r")
        try:
            tpara = ds.auxiliary_data[ssta][stack_para['comp']].parameters 
            tdata = ds.auxiliary_data[ssta][stack_para['comp']].data[:]
        except Exception: 
            if stack_para['flag']:
                print('continue! no pair of %s in %s'%(ssta,ifile))
            continue
                  
        # read data and parameter matrix
        cc_array[ii] = tdata.reshape(tdata.size,)

    t1=time.time()
    if stack_para['flag']:
        print('loading CCF data takes %6.2fs'%(t1-t0))

    # stacking 
    DAS_module.stacking(cc_array,stack_para,stack_h5,tpara)
    del cc_array    
    
    t2 = time.time()
    if stack_para['flag']:
        print('takes %6.2fs to stack one component with %s stacking method' %(t2-t1,stack_para['stack_method']))

    # write file stamps 
    ftmp = open(toutfn,'w');ftmp.write('done');ftmp.close()

tt1 = time.time()
print('it takes %6.2fs to process step 2 in total' % (tt1-tt0))
