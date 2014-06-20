import glob
import numpy as np
import sys
import scipy.io as sio

# Este programa guarda, en una matriz de matlab, las labels para todas las samples 
# que hay en los ficheros que corresponden a un pattern.
# La matriz de salida tiene tantas filas como samples y tantas columnas como posible atributos
# ej: python all_labels.py 'test_labels_*.npy' '/home/jmansanet/test_labels.mat'

proto = sys.argv[1]
outputfile = sys.argv[2]
filenames = []
filenames.append(sorted(glob.glob(proto)))
numfiles = len(filenames[0])
assert numfiles > 0, 'num files = %d' % (numfiles)

total_samples = 0
data = np.load(filenames[0][0])
curr_size = data.shape[0]
total_samples = total_samples + curr_size

for i in range(1, numfiles):
	curr = np.load(filenames[0][i])
	curr_size = curr.shape[0]
	total_samples = total_samples + curr_size
	print 'File %s with %d samples' % (filenames[0][i], curr_size)
	data = np.append(data,curr,axis=0)

print '----------------------------'
print 'Total samples = %d' % (total_samples)
sio.savemat(outputfile,mdict={'data': data})