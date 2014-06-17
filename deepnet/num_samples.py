import glob
import numpy as np
import sys

proto = sys.argv[1]
filenames = []
filenames.append(sorted(glob.glob(proto)))
numfiles = len(filenames[0])
assert numfiles > 0, 'num files = %d' % (numfiles)

total_samples = 0
for i in range(0, numfiles):
	curr = np.load(filenames[0][i])
	curr_size = curr.shape[0]
	total_samples = total_samples + curr_size
	print 'File %s with %d samples' % (filenames[0][i], curr_size)

print '----------------------------'
print 'Total samples = %d' % (total_samples)
