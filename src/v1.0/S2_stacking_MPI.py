import sys
import time
import pyasdf
import os, glob
import numpy as np
import DAS_module
from mpi4py import MPI

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

'''
Stacking script of NoisePy to:
    1) load cross-correlation data for sub-stacking (if needed) and all-time average;
    2) stack data with either linear or phase weighted stacking (pws) methods (or both);
    3) save outputs in ASDF or SAC format depend on user's choice (for latter option, find the script of write_sac
       in the folder of application_modules;
    4) rotate from a E-N-Z to R-T-Z system if needed.

Authors: Chengxin Jiang (chengxin_jiang@fas.harvard.edu)
         Marine Denolle (mdenolle@fas.harvard.edu)

NOTE: 
    0. MOST occasions you just need to change parameters followed with detailed explanations to run the script. 
    1. assuming 3 components are E-N-Z 
    2. auto-correlation is not kept in the stacking due to the fact that it has only 6 cross-component.
    this tends to mess up the orders of matrix that stores the CCFs data
'''

tt0=time.time()

########################################
#########PARAMETER SECTION##############
########################################

# absolute path parameters
rootpath  = '/Users/chengxin/Documents/ANU/DAS/melb_DAS2022'        # root path for this data processing
CCFDIR    = os.path.join(rootpath,'CCF')                            # dir where CC data is stored
STACKDIR  = os.path.join(rootpath,'STACK')                          # dir where stacked data is going to

# define new stacking para
keep_substack= False                                                # keep all sub-stacks in final ASDF file
flag         = True                                                 # output intermediate args for debugging
stack_method = 'linear'                                             # linear, pws, robust, nroot, selective, auto_covariance or all

# maximum memory allowed per core in GB
MAX_MEM = 4.0

##################################################
# we expect no parameters need to be changed below

# load fc_para parameters from Step1
ccfiles   = sorted(glob.glob(os.path.join(CCFDIR,'*.h5')))

fc_metadata = os.path.join(CCFDIR,'prepro_fft_info.txt')
fc_para     = eval(open(fc_metadata).read())
samp_freq   = fc_para['samp_freq']
maxlag      = fc_para['maxlag']
npts   = fc_para['npts']
comp = 'ZZ'

# make a dictionary to store all variables: also for later cc
stack_para={'samp_freq':samp_freq,
            'rootpath':rootpath,
            'STACKDIR':STACKDIR,
            'maxlag':maxlag,
            'keep_substack':keep_substack,
            'stack_method':stack_method}

# save fft metadata for future reference
stack_metadata  = os.path.join(STACKDIR,'stack_data.txt') 

#######################################
###########PROCESSING SECTION##########
#######################################

#--------MPI---------
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    if not os.path.isdir(STACKDIR):
        os.mkdir(STACKDIR)

    # save metadata 
    fout = open(stack_metadata,'w')
    fout.write(str(stack_para))
    fout.close()

    # 
    nsta = fc_para['nsta']
    # load sample file for source and receiver list
    splits  = nsta
    if splits==0:
        raise IOError('Abort! no available CCF data for stacking')

else:
    splits,nsta = [None for _ in range(2)]

# broadcast the variables
splits    = comm.bcast(splits,root=0)
nsta = comm.bcast(nsta,root=0)

# MPI loop: loop through each user-defined time chunck
for ista in range (rank,splits,size):
    t0=time.time()
    rsta = 'C'+format(ista,'04d')

    # crude estimation on memory needs (assume float32)
    num_chunck = len(ccfiles)
    npts_chunk = int(2*maxlag*samp_freq)+1
    memory_size = num_chunck*npts_chunk*4/1024**3

    if memory_size > MAX_MEM:
        raise ValueError('Require %5.3fG memory but only %5.3fG provided)! \
                        Cannot load cc data all once!' % (memory_size,MAX_MEM))
    if flag:
        print('Good on memory (need %5.2f G and %s G provided)!' % (memory_size,MAX_MEM))
        
    # allocate array to store fft data/info
    cc_array = np.zeros((num_chunck,npts_chunk),dtype=np.float32)

    # loop through each source
    source_chan = fc_para['source_chan']
    for isour in source_chan:
        ssta = 'C'+format(isour,'04d')

        if flag:print('working on %dth receiver'%(ista))
        idir  = ssta
        tdir  = os.path.join(STACKDIR,idir)
        if not os.path.isdir(tdir):
            os.mkdir(tdir)

        # continue when file is done
        toutfn = os.path.join(tdir,ssta+"_"+rsta+".tmp")   
        if os.path.isfile(toutfn):
            continue        

        # loop through all time-chuncks
        iseg = 0
        for ifile in ccfiles:

            # load the data from daily compilation
            ds=pyasdf.ASDFDataSet(ifile,mpi=False,mode="r")
            try:
                tpara = ds.auxiliary_data[ssta][comp].parameters 
                tdata = ds.auxiliary_data[ssta][comp].data[ista]
            except Exception: 
                if flag:
                    print('continue! no pair of %s-%sin %s'%(ssta,rsta,ifile))
                continue
                    
            # read data and parameter matrix
            cc_array[iseg] = tdata
            iseg+=1

        t1=time.time()
        if flag:
            print('loading CCF data takes %6.2fs'%(t1-t0))

        # continue when there is no data or for auto-correlation
        if iseg <= 1: 
            continue
        outfn = ssta+"_"+rsta+".h5"         

        if stack_method =='all':
            bigstack1=np.zeros(shape=(9,npts_chunk*nsta),dtype=np.float32)
            bigstack2=np.zeros(shape=(9,npts_chunk*nsta),dtype=np.float32)

        t2=time.time()
        stack_h5 = os.path.join(STACKDIR,idir+'/'+outfn)
        # output stacked data
        cc_final,allstacks1,allstacks2,allstacks3 = DAS_module.stacking(cc_array,stack_para)
        if not len(allstacks1):
            continue

        # write stacked data into ASDF file
        with pyasdf.ASDFDataSet(stack_h5,mpi=False) as ds:
            if stack_method != 'all':
                data_type = 'Allstack_'+stack_method
                ds.add_auxiliary_data(data=allstacks1, 
                                        data_type=data_type, 
                                        path='ZZ', 
                                        parameters=tpara)
            else:
                ds.add_auxiliary_data(data=allstacks1, 
                                        data_type='Allstack_linear', 
                                        path='ZZ', 
                                        parameters=tpara)
                ds.add_auxiliary_data(data=allstacks2, 
                                        data_type='Allstack_pws', 
                                        path='ZZ', 
                                        parameters=tpara)
                ds.add_auxiliary_data(data=allstacks3, 
                                        data_type='Allstack_robust', 
                                        path='ZZ', 
                                        parameters=tpara)

        # keep a track of all sub-stacked data from S1
        if keep_substack:
            for ii in range(cc_final.shape[0]):
                with pyasdf.ASDFDataSet(stack_h5,mpi=False) as ds:
                    data_type = 'T'+str(int(allstacks3[ii]))
                    ds.add_auxiliary_data(data=cc_final[ii], 
                                            data_type=data_type, 
                                            path='ZZ', 
                                            parameters=tpara)            
        
        t3 = time.time()
        if flag:print('takes %6.2fs to stack one component with %s stacking method' %(t3-t1,stack_method))

        # write file stamps 
        ftmp = open(toutfn,'w');ftmp.write('done');ftmp.close()

tt1 = time.time()
print('it takes %6.2fs to process step 2 in total' % (tt1-tt0))
comm.barrier()

# merge all path_array and output
if rank == 0:
    sys.exit()

