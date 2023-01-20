import h5py 
import matplotlib.pyplot as plt 

import argparse
parser = argparse.ArgumentParser(description="Density estimation for dense hydrogen")
parser.add_argument("--rdf_file", default="../data/ds_n_64_dim_3_d_4_h1_64_h2_16_lr_0.001/epoch_001700.h5",help="")
args = parser.parse_args()


rdf = h5py.File(args.rdf_file, 'r')
rdf_data = rdf['data']

plt.plot(rdf_data[0], rdf_data[1], linestyle='-', c='blue', label='data')

nsteps = len(rdf.keys()) -1
for i in range(0, nsteps, 10):
    rdf_model = rdf['mcstep_%g'%i]
    plt.plot(rdf_model[0], rdf_model[1], linestyle='-', c='red', label='model%g'%i)

plt.legend()
plt.show()
