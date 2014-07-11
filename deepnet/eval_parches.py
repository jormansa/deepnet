
import numpy as np
from trainer import *
import scipy.io as sio

# le pasamos el file_pattern de labels, el file_parttern de ids, el numpy de representacion de salida
# ej: python eval_parches.py "/home/jmansanet/pruebas/data/2/test_labels*.npy" "/home/jmansanet/pruebas/data/2/test_id*.npy" test/output_layer-00001-of-00001.npy

file_pattern_labels = sys.argv[1]
file_pattern_ids    = sys.argv[2]

# get guess labels
out_repr = np.load(sys.argv[3])
labels_pred = np.argmax(out_repr,1);

# load true labels
#test_op_file = sys.argv[1]
#op = ReadOperation(test_op_file)
#data_proto_file = os.path.join(op.data_proto_prefix, op.data_proto)
#dataset_proto = util.ReadData(data_proto_file)
#data_proto = next(d for d in dataset_proto.data if d.name == "test_labels")
filenames = []
filenames.append(sorted(glob.glob(file_pattern_labels)))
numfiles = len(filenames[0])
assert numfiles > 0, 'num files labels = %d' % (numfiles)
labels_true = np.zeros(labels_pred.shape[0])
collect_pos = 0
for i in range(0, numfiles):
	curr = np.load(filenames[0][i])
	curr_size = curr.shape[0]
	labels_true[collect_pos:collect_pos + curr_size] = curr.T
	collect_pos += curr_size

# check if labels start with 1 instead of with 0
if np.min(labels_true) == 1:
	labels_pred = np.add(labels_pred,1)

sio.savemat('labels_pred.mat',mdict={'labels_pred': labels_pred})
sio.savemat('labels_true.mat',mdict={'labels_true': labels_true})

acc = sum(labels_true == labels_pred)/float(labels_true.shape[0])
print 'Test acc : %.5f ' % (acc)
#ind_pos = labels_true == 1
#tpr = sum(labels_true[ind_pos] == labels_pred[ind_pos])/float(sum(ind_pos))
#print 'TPR : %.5f ' % (tpr)
#ind_neg = labels_true == 0
#tnr = sum(labels_true[ind_neg] == labels_pred[ind_neg])/float(sum(ind_neg))
#print 'TNR : %.5f ' % (tnr)

# load ids
filenames = []
filenames.append(sorted(glob.glob(file_pattern_ids)))
assert numfiles > 0, 'num files ids = %d' % (numfiles)
numfiles = len(filenames[0])
readids = np.zeros(labels_pred.shape[0])
collect_pos = 0

for i in range(0, numfiles):
	curr = np.load(filenames[0][i])
	curr_size = curr.shape[0]
	readids[collect_pos:collect_pos + curr_size] = curr.T
	collect_pos += curr_size

readids = readids.astype(int)
sio.savemat('readids.mat',mdict={'readids': readids})

