import numpy as np
import os

import sys
sys.path.append('/users/jvergara/python_code')
#import matplotlib
#matplotlib.use('Agg')
import Jesuslib_eth as jle


os.chdir(jle.scratch)


os.chdir(jle.scratch+'PRECIP_BUG')

list_commits=['5dd6b264', 'c9856bd1' ,'a701a638' ,'8dd75c76' ,'320e20aa' ,'e8ae0f69' ,'86a45963' ,'411bb617' ,
              '427a212c' ,'b0532d63', 'd58b1727', 'e48a4699' ,'34fc5a14' ,'45d4031f', '4d434db7', 'c567547d' ,'978f5aa7']
list_commits=['34fc5a14' ,'45d4031f', '4d434db7', 'c567547d' ,'978f5aa7']

read_commits=[]
with open('/users/jvergara/commits.txt') as f:
    lines=f.readlines()
for line in lines:
    if line[:6]=='commit':
        read_commits.append(line[7:-1])
#%%

def compile_and_run(commit):
    os.chdir(jle.scratch+'PRECIP_BUG')
    print (commit,len(commit))
    a=os.system('git clone ../my_fork_second_compile %s'%commit)
#        if a:
#            raise NameError('fail when cloning %s'%commit)
    print('Cloning:',commit, a)
    a=os.chdir(commit)
    a=os.system('git checkout %s'%commit)
#        if a:
#            raise NameError('fail when Checking out %s'%commit)
    print('Checking out:',commit, a)
        
    a=os.chdir('cosmo')
        #compiling
        
    a=os.system('./test/jenkins/build.sh -z -t gpu -c cray')
#        if a:
#            raise NameError('fail when compiling %s'%commit)
    
    a=os.system('cp cosmo test/climate/crClim2km_DVL/bin/lm_f90')
#        if a:
#            raise NameError('fail when copying the compiled version %s'%commit)
    os.chdir('test/climate/crClim2km_DVL/')
    #    a=os.system("sed '/5_trajectories 6_climate_analysis 7_msd 8_front_tracking 9_tracking_coldfront850/d' run_daint.sh")
    a=os.system("sed -i 's/5_trajectories//g' run_daint.sh")
    a=os.system("sed -i 's/6_climate_analysis//g' run_daint.sh")
    a=os.system("sed -i 's/7_msd//g' run_daint.sh")
    a=os.system("sed -i 's/8_front_tracking//g' run_daint.sh")
    a=os.system("sed -i 's/8_tracking_cyclone_slp//g' run_daint.sh")
    a=os.system("sed -i 's/9_tracking_coldfront850//g' run_daint.sh")
    a=os.system("sed -i 's/nmaxwait=0/nmaxwait=3600/g' 3_lm2lm/run")
    a=os.system("sed -i 's/RELHUM_2M/TOT_PR/g' 4_lm_f/run")
    a=os.system("sed -i 's/02:30/00:45/g' 4_lm_f/run")
    a=os.system("sed -i 's/LM_NL_HSTOP=72/LM_NL_HSTOP=24/g' run_daint.sh")
    a=os.system("cp /project/c14/install/daint/int2lm/int2lm_cray bin/int2lm")
    a=os.system('./run_daint.sh')
    print (a)


import time
import multiprocessing
processes=15
print ('Number of commits', len(read_commits))
list_of_chunks=np.array_split(read_commits,len(read_commits)/processes+1)
start=time.time()
for chunk in list_of_chunks:
    jobs=[]
    for commit in chunk:
        p = multiprocessing.Process(target=compile_and_run, args=(commit,))
        print (commit,p)
        jobs.append(p)
        p.start()
    for job in jobs:
        job.join()
#for commit in read_commits:
    
    
'''
    os.chdir(jle.scratch+'PRECIP_BUG')

    try:
        print (commit,len(commit))
        a=os.system('git clone ../my_fork_second_compile %s'%commit)
#        if a:
#            raise NameError('fail when cloning %s'%commit)
        print('Cloning:',commit, a)
        a=os.chdir(commit)
        a=os.system('git checkout %s'%commit)
#        if a:
#            raise NameError('fail when Checking out %s'%commit)
        print('Checking out:',commit, a)
        
        a=os.chdir('cosmo')
        #compiling
        
        a=os.system('./test/jenkins/build.sh -z -t gpu -c cray')
#        if a:
#            raise NameError('fail when compiling %s'%commit)
    
        a=os.system('cp cosmo test/climate/crClim2km_DVL/bin/lm_f90')
#        if a:
#            raise NameError('fail when copying the compiled version %s'%commit)
    
        os.chdir('test/climate/crClim2km_DVL/')
    #    a=os.system("sed '/5_trajectories 6_climate_analysis 7_msd 8_front_tracking 9_tracking_coldfront850/d' run_daint.sh")
        a=os.system("sed -i 's/5_trajectories//g' run_daint.sh")
        a=os.system("sed -i 's/6_climate_analysis//g' run_daint.sh")
        a=os.system("sed -i 's/7_msd//g' run_daint.sh")
        a=os.system("sed -i 's/8_front_tracking//g' run_daint.sh")
        a=os.system("sed -i 's/8_tracking_cyclone_slp//g' run_daint.sh")
        a=os.system("sed -i 's/9_tracking_coldfront850//g' run_daint.sh")
        a=os.system("sed -i 's/nmaxwait=0/nmaxwait=3600/g' 3_lm2lm/run")
        a=os.system("sed -i 's/RELHUM_2M/TOT_PR/g' 4_lm_f/run")
        a=os.system("sed -i 's/02:30/00:45/g' 4_lm_f/run")
        a=os.system("sed -i 's/LM_NL_HSTOP=72/LM_NL_HSTOP=24/g' run_daint.sh")
        
        a=os.system("cp /project/c14/install/daint/int2lm/int2lm_cray bin/int2lm")
        
        a=os.system('./run_daint.sh')
        print (a)
        
    
        os.chdir(jle.scratch+'PRECIP_BUG')
    except:
        print('Commit:', commit, '  Failed')

'''
