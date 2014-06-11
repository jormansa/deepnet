
import scipy.io as sio
from neuralnet import *
from fastdropoutnet import *
from dbm import *
from dbn import *
from sparse_coder import *
from choose_matrix_library import *
import numpy as np
from time import sleep
from matplotlib import pyplot as plt

def LockGPU(max_retries=10):
  for retry_count in range(max_retries):
    board = gpu_lock.obtain_lock_id()
    if board != -1:
      break
    sleep(1)
  if board == -1:
    print 'No GPU board available.'
    sys.exit(1)
  else:
    cm.cuda_set_device(board)
    cm.cublas_init()
  return board

# TODO: repasar el código !!
# para MAC
#cm.cuda_set_device(0)
#cm.cublas_init()

if use_gpu == 'yes':
    board = LockGPU()

# cargamos el modelo
modelname = 'model_trainfull_test_classifier_BEST'
evalname = '../eval.pbtxt'
best_model = NeuralNet(modelname, None, evalname)
#best_model = NeuralNet(sys.argv[1], sys.argv[2], sys.argv[2])
best_model.LoadModelOnGPU()
best_model.PrintNetwork()
best_model.SetUpData()
num_steps = best_model.validation_data_handler.num_batches
best_model.validation_stop_steps = num_steps


# sacamos los targets y las predictions
step = 0
stats = []
#stopcondition = best_model.TestStopCondition
stopcondition = best_model.ValidationStopCondition
stop = stopcondition(step)
#datagetter = best_model.GetTestBatch
datagetter = best_model.GetValidationBatch
#stats_list = best_model.net.test_stats
#stats_list = best_model.net.validation_stats
#num_batches = best_model.test_data_handler.num_batches
num_batches = best_model.validation_data_handler.num_batches

output_layer = best_model.output_datalayer[0]
collect_pos = 0
batchsize = best_model.e_op.batchsize
#numdims = output_layer.state.shape[0]
# CUIDADO !!!!!
numdims = 2
predictions = np.zeros((batchsize * num_batches, numdims))
targets = np.zeros(predictions.shape)

while not stop:
	datagetter()
	losses = best_model.EvaluateOneBatch()
	predictions[collect_pos:collect_pos + batchsize] = output_layer.state.asarray().T
	targets[collect_pos:collect_pos + batchsize] = output_layer.data.asarray().T
	collect_pos += batchsize
	if stats:
		for loss, acc in zip(losses, stats):
			Accumulate(acc, loss)
	else:
		stats = losses
	step += 1
	stop = stopcondition(step)

predictions = predictions[:collect_pos]
targets = targets[:collect_pos]
#sio.savemat('preds.mat',mdict={'predictions': predictions})

# Mostramos la informacion de acierto
labels_pred = np.argmax(predictions,1);
labels_target = targets[:,0]
acc = sum(targets[:,0] == labels_pred)/float(targets.shape[0])
print 'Test acc : %.5f ' % (acc)
ind_pos = labels_target == 1
tpr = sum(labels_target[ind_pos] == labels_pred[ind_pos])/float(sum(ind_pos))
print 'TPR : %.5f ' % (tpr)
ind_neg = labels_target == 0
tnr = sum(labels_target[ind_neg] == labels_pred[ind_neg])/float(sum(ind_neg))
print 'TNR : %.5f ' % (tnr)

# Mostramos la informacion de los que han fallado
ind_errors_test = np.where(targets[:,0] != labels_pred)
#test_data = best_model.test_data_handler.GetCPUBatches()
test_data = best_model.validation_data_handler.GetCPUBatches()
for i in range(0, ind_errors_test[0].shape[0]):
	error1 = test_data[0][ind_errors_test[0][i]]
	plt.imshow(np.transpose(error1.reshape(21,21)), cmap=plt.cm.gray)
	print 'Image : %d truth %d guess %d ' % (ind_errors_test[0][i], int(labels_target[ind_errors_test[0][i]]), int(labels_pred[ind_errors_test[0][i]]))
	raw_input("Press Enter to continue...")

# free gpu
if use_gpu == 'yes':
	cm.cublas_shutdown()

