import h5py
import numpy as np

### Create a new HDF5 file with a write connection
with h5py.File('example.hdf5', 'w') as f_write:
    ### create a new 2x2 short int array called mammals, within animals group
    ds1 = f_write.create_dataset('animals/mammals', (2,2), dtype='i8')
    ### set values for mammals array
    ds1[...] = [[1,2],[3,4]]
    ### create/set metadata attribute tied to mammals array
    ds1.attrs['names'] = ['cat', 'dog']
    ### here is a different way to create and set a new dataset directly with data
    f_write['animals/reptiles'] = [[1.2, 2.3], [3.4, 4.5], [5.6, 6.7]]
    ### attribute tied directly to group rather than dataset
    f_write['animals'].attrs['data collection'] = 'made up' 
    ### here is a new group and dataset, this time storing character string data.
    f_write['vehicles/cars'] = ['prius', 'focus']

### Reopen the HDF5 file in read mode to verify our data was stored correctly
with h5py.File('example.hdf5', 'r') as f_read:
    ### assign mammals array to variable ds1
    ds1 = f_read['animals/mammals']
    ### read stored array
    print(ds1[...])
    ### read attribute
    print(ds1.attrs['names'])
    ### you can also read array data directly without naming a variable first
    print(f_read['vehicles/cars'][...])
