import numpy as np
import scipy.io as sio
import os
import sys

# convert .npy to .mat, or viceversa
# THE KEY MUST HAVE THE SAME NAME AS THE FILE

to_convert = sys.argv[1]
fileName, fileExtension = os.path.splitext(to_convert)
if fileExtension == '.mat':
	mat_contents = sio.loadmat(to_convert)
	data = mat_contents[fileName]
	np.save(fileName + '.npy', data)
	print 'Converted %s to %s.npy' % (to_convert,fileName)
else:
	data = np.load(to_convert)
	sio.savemat(fileName + '.mat',mdict={fileName: data})
	print 'Converted %s to %s.mat' % (to_convert,fileName)
