from neuralnet import *
from fastdropoutnet import *
from dbm import *
from dbn import *
from sparse_coder import *
from choose_matrix_library import *
import numpy as np
from time import sleep

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

def FreeGPU(board):
  cm.cublas_shutdown()
  #gpu_lock.free_lock(board)

def LoadExperiment(model_file, train_op_file, eval_op_file):
  model = util.ReadModel(model_file)
  train_op = util.ReadOperation(train_op_file)
  eval_op = util.ReadOperation(eval_op_file)
  return model, train_op, eval_op

def CreateDeepnet(model, train_op, eval_op):
  if model.model_type == deepnet_pb2.Model.FEED_FORWARD_NET:
    return NeuralNet(model, train_op, eval_op)
  elif model.model_type == deepnet_pb2.Model.DBM:
    return DBM(model, train_op, eval_op)
  elif model.model_type == deepnet_pb2.Model.DBN:
    return DBN(model, train_op, eval_op)
  elif model.model_type == deepnet_pb2.Model.SPARSE_CODER:
    return SparseCoder(model, train_op, eval_op)
  elif model.model_type == deepnet_pb2.Model.FAST_DROPOUT_NET:
    return FastDropoutNet(model, train_op, eval_op)
  else:
    raise Exception('Model not implemented.')

def main():
  if use_gpu == 'yes':
    board = LockGPU()

  # train on validation: stop criterion is number of steps
  if len(sys.argv) == 4:
    model, train_op, eval_op = LoadExperiment(sys.argv[1], sys.argv[2], sys.argv[3])
    model = CreateDeepnet(model, train_op, eval_op)
    # set max_tcee to 0
    max_tcee = 0
    print 'Stop criterion: number of steps'
    model.max_tcee = max_tcee
    model.Train()
    if use_gpu == 'yes':
      FreeGPU(board)
  # train with train_all: stop criterion is max train cross entropy error
  # The 5th param is the best network trained on validation  
  elif len(sys.argv) == 5:
    best_model = NeuralNet(sys.argv[4])
    correct = best_model.net.best_valid_stat.correct_preds
    count_valid = best_model.net.best_valid_stat.count
    print 'Best valid error on loaded network: %.6f ' % (correct/count_valid)
    tot_cr_err = best_model.net.train_stat_es.cross_entropy
    count_train = best_model.net.train_stat_es.count
    max_tcee = tot_cr_err/count_train
    print 'Stop criterion: max train cross entropy error =  : %.6f ' % (max_tcee)

    model, train_op, eval_op = LoadExperiment(sys.argv[1], sys.argv[2], sys.argv[3])
    model = CreateDeepnet(model, train_op, eval_op)
    model.max_tcee = max_tcee
    model.Train()

    # check if max train cross entropy criterion is not reached: instead -> network trained for all epochs
    t_op = util.ReadOperation("models/model_train_op_LAST")
    trained_steps = t_op.current_step
    train_max_steps = model.train_stop_steps
    tot_cr_err = model.net.train_stats[-1].cross_entropy
    count_train = model.net.train_stats[-1].count
    if trained_steps == train_max_steps:
      print 'WARNING: current tcee %.6f > target tcee %.6f ' % (tot_cr_err/count_train,model.max_tcee)
    else:
      print 'OK: current tcee %.6f < target tcee %.6f at epoch %d' % (tot_cr_err/count_train,model.max_tcee,trained_steps)

    if use_gpu == 'yes':
      FreeGPU(board)
  # evaluate network: not to train
  elif len(sys.argv) == 3:
    best_model = NeuralNet(sys.argv[1], sys.argv[2], sys.argv[2])
    best_model.SetUpTrainer()
    best_model.Evaluate(False, False)
    if use_gpu == 'yes':
      FreeGPU(board)
  else:
    raise Exception('Not correct number of input params')  


if __name__ == '__main__':
  main()
