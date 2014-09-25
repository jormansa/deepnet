import numpy as np
import scipy.io as sio
import os
import sys
import glob

# convert .npy to .mat, or viceversa
# THE KEY MUST HAVE THE SAME NAME AS THE FILE

to_convert = sys.argv[1]
filenames = []
filenames.append(sorted(glob.glob(to_convert)))
numfiles = len(filenames[0])
assert numfiles > 0, 'no files to convert'
for i in range(0, numfiles):
	fileName, fileExtension = os.path.splitext(filenames[0][i])
	fileName = os.path.basename(fileName)
	if fileExtension == '.mat':
		mat_contents = sio.loadmat(filenames[0][i])
		data = mat_contents[fileName]
		np.save(fileName + '.npy', data)
		print 'Converted %s to %s.npy' % (filenames[0][i],fileName)
	else:
		data = np.load(filenames[0][i])
		sio.savemat(fileName + '.mat',mdict={fileName: data},do_compression=True)
		print 'Converted %s to %s.mat' % (filenames[0][i],fileName)
