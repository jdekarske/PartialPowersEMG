import h5py

f = h5py.File('data/2/training task_20190527-194920/windowedEMG.hdf5', 'r')
print(f.keys())