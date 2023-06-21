import numpy as np
import h5py
import time
import os
import shutil
import sys
import subprocess

def generate_mvn(means, stds, corrs, nsamp):
    '''
    function for generating correlated multivariate normal random variables as numpy array
    :param means: 1d array of means for N variables
    :param stds: 1d array of standard deviations for N variables
    :param corrs: NxN correlation matrix
    :param size: Number of samples to generate
    :return: (size x N) array of sampled variables
    '''

    covs = np.matmul(np.matmul(np.diag(stds), corrs),
                    np.diag(stds))
    return np.random.multivariate_normal(means, covs, size=nsamp)


### loop over different experiment sizes
for nsamp, nparam in [(30,100), (30000,10), (10,100), (10000,10), (10000000,1), (1000,10), (1000000,1), (100,10), (100000,1)][::-1]:
    data_dir = f'nsamp{nsamp}_nparam{nparam}/'
    try:
        shutil.rmtree(data_dir)
    except:
        pass
    os.mkdir(data_dir)
    os.mkdir(data_dir + 'data_csv/')
    os.mkdir(data_dir + 'data_hdf5/')
    print(f'nsamp: {nsamp}, nparam: {nparam}\n')
    sys.stdout.flush()

    ### First store as csv write
    ### Loop over alternative means, stds, & corrs for 2-variable system.
    np.random.seed(101)
    mean_samps = np.random.uniform(-1, 1, size=(nparam,2))
    std_samps = np.random.uniform(0, 1, size=(nparam,2))
    corr_samps = np.random.uniform(0, 1, size=nparam)

    t0 = time.perf_counter()
    for i in range(nparam):
        means = mean_samps[i,:]
        for j in range(nparam):
            stds = std_samps[j,:]
            for k in range(nparam):
                corr = corr_samps[k]
                corrs = np.array([[1, corr], [corr, 1]])
                ### generate sample for this combination of parameters
                samps = generate_mvn(means, stds, corrs, nsamp)
                ### store sample as csv
                np.savetxt(f'{data_dir}/data_csv/means{i}_stds{j}_corr{k}.csv', samps, delimiter=',',
                           header=f'means: {means}\nstds: {stds}\ncorr: {corr}')

    ### print run time
    dt = time.perf_counter() - t0
    print(f'CSV write: {dt:0.4f} seconds')
    sys.stdout.flush()

    ### now loop back over and get read each file to compare actual to sampled params for each set
    t0 = time.perf_counter()
    params_actual, params_samp = {}, {}
    for i in range(nparam):
        params_actual[f'means{i}'], params_samp[f'means{i}'] = {}, {}
        for j in range(nparam):
            params_actual[f'means{i}'][f'stds{j}'], params_samp[f'means{i}'][f'stds{j}'] = {}, {}
            for k in range(nparam):
                params_actual[f'means{i}'][f'stds{j}'][f'corr{k}'], params_samp[f'means{i}'][f'stds{j}'][f'corr{k}'] = {}, {}

                ### first get actual params from file
                with open(f'{data_dir}/data_csv/means{i}_stds{j}_corr{k}.csv', 'r') as f:
                    paramstrings = [next(f) for _ in range(3)]
                    m1 = float(paramstrings[0].split('[')[1].lstrip().split(' ')[0])
                    m2 = float(paramstrings[0].split(']')[0].rstrip().split(' ')[-1])
                    s1 = float(paramstrings[1].split('[')[1].lstrip().split(' ')[0])
                    s2 = float(paramstrings[1].split(']')[0].rstrip().split(' ')[-1])
                    c = float(paramstrings[2].split('\n')[0].split(' ')[-1])
                    params_actual[f'means{i}'][f'stds{j}'][f'corr{k}']['means'] = np.array([m1, m2])
                    params_actual[f'means{i}'][f'stds{j}'][f'corr{k}']['stds'] = np.array([s1, s2])
                    params_actual[f'means{i}'][f'stds{j}'][f'corr{k}']['corr'] = c
                ### now read sample from csv and calculate actual params
                samps = np.genfromtxt(f'{data_dir}/data_csv/means{i}_stds{j}_corr{k}.csv', delimiter=',')
                params_samp[f'means{i}'][f'stds{j}'][f'corr{k}']['means'] = samps.mean(axis=0)
                params_samp[f'means{i}'][f'stds{j}'][f'corr{k}']['means'] = samps.std(axis=0)
                params_samp[f'means{i}'][f'stds{j}'][f'corr{k}']['means'] = np.corrcoef(samps[:,0], samps[:,1])[0,1]

    ### print run time
    dt = time.perf_counter() - t0
    print(f'CSV read: {dt:0.4f} seconds')
    sys.stdout.flush()


    ### print directory size and remove
    subprocess.run(['du', '-h', f'{data_dir}/data_csv/'])
    print('')
    shutil.rmtree(f'{data_dir}/data_csv/')





    ### now repeat with hdf5 write
    ### Loop over alternative means, stds, & corrs for 2-variable system.
    np.random.seed(101)
    mean_samps = np.random.uniform(-1, 1, size=(nparam,2))
    std_samps = np.random.uniform(0, 1, size=(nparam,2))
    corr_samps = np.random.uniform(0, 1, size=nparam)

    t0 = time.perf_counter()
    with h5py.File(f'{data_dir}/data_hdf5/data.hdf5', 'w') as f:
        for i in range(nparam):
            means = mean_samps[i,:]
            for j in range(nparam):
                stds = std_samps[j,:]
                for k in range(nparam):
                    corr = corr_samps[k]
                    corrs = np.array([[1, corr], [corr, 1]])
                    ### generate sample for this combination of parameters
                    samps = generate_mvn(means, stds, corrs, nsamp)
                    ### store sample as hdf5 dataset
                    f[f'means{i}/stds{j}/corr{k}'] = samps
                    f[f'means{i}/stds{j}/corr{k}'].attrs['corr'] = [corr]
                f[f'means{i}/stds{j}'].attrs['stds'] = [stds[0], stds[1]]
            f[f'means{i}'].attrs['means'] = [means[0], means[1]]

    ### print run time
    dt = time.perf_counter() - t0
    print(f'HDF5 write: {dt:0.4f} seconds')
    sys.stdout.flush()



    ### now loop back over and get read hdf5 to compare actual to sampled params for each set
    t0 = time.perf_counter()
    params_actual, params_samp = {}, {}

    with h5py.File(f'{data_dir}/data_hdf5/data.hdf5', 'r') as f:
        for i in range(nparam):
            params_actual[f'means{i}'], params_samp[f'means{i}'] = {}, {}
            for j in range(nparam):
                params_actual[f'means{i}'][f'stds{j}'], params_samp[f'means{i}'][f'stds{j}'] = {}, {}
                for k in range(nparam):
                    params_actual[f'means{i}'][f'stds{j}'][f'corr{k}'], params_samp[f'means{i}'][f'stds{j}'][f'corr{k}'] = {}, {}

                    params_actual[f'means{i}'][f'stds{j}'][f'corr{k}']['means'] = f[f'means{i}'].attrs['means']
                    params_actual[f'means{i}'][f'stds{j}'][f'corr{k}']['stds'] = f[f'means{i}/stds{j}'].attrs['stds']
                    params_actual[f'means{i}'][f'stds{j}'][f'corr{k}']['corr'] = f[f'means{i}/stds{j}/corr{k}'].attrs['corr']
                    params_samp[f'means{i}'][f'stds{j}'][f'corr{k}']['means'] = f[f'means{i}/stds{j}/corr{k}'][...].mean(axis=0)
                    params_samp[f'means{i}'][f'stds{j}'][f'corr{k}']['means'] = f[f'means{i}/stds{j}/corr{k}'][...].std(axis=0)
                    params_samp[f'means{i}'][f'stds{j}'][f'corr{k}']['means'] = np.corrcoef(f[f'means{i}/stds{j}/corr{k}'][:, 0], f[f'means{i}/stds{j}/corr{k}'][:, 1])[0, 1]

    ### print run time
    dt = time.perf_counter() - t0
    print(f'HDF5 read: {dt:0.4f} seconds')
    sys.stdout.flush()


    ### print directory size and remove
    subprocess.run(['du', '-h', f'{data_dir}/data_hdf5/'])
    print('')
    shutil.rmtree(f'{data_dir}/data_hdf5/')

    shutil.rmtree(data_dir)






