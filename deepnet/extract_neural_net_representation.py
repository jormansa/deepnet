"""Push the data through a network and get representations at each layer."""
from neuralnet import *
from trainer import *
import sys
import scipy.io as sio
import glob
import numpy as np
import os

def ExtractRepresentations(model_file, train_op_file, layernames,
                           base_output_dir, memory = '100M', skip_outputs=True,
                           datasets=['test'], gpu_mem='2G', main_mem='30G',numtimes=1, no_translations=False):
  if isinstance(model_file, str):
    model = util.ReadModel(model_file)
  else:
    model = model_file
  if isinstance(train_op_file, str):
    op = ReadOperation(train_op_file)
  else:
    op = train_op_file
  if not os.path.isdir(base_output_dir):
    os.makedirs(base_output_dir)
  op.randomize = False
  op.get_last_piece = True
  net = CreateDeepnet(model, op, op)
  net.LoadModelOnGPU()
  net.SetUpData(skip_outputs=skip_outputs)

  # set translations
  if no_translations:
    print 'WARNING: evaluating WITHOUT translations'
    if net.train_data_handler is not None:
      for i in range(0, net.train_data_handler.gpu_cache.num_data):
        net.train_data_handler.gpu_cache.translate[i] = False

    if net.validation_data_handler is not None:
      for i in range(0, net.validation_data_handler.gpu_cache.num_data):
        net.validation_data_handler.gpu_cache.translate[i] = False

    if net.test_data_handler is not None:
      for i in range(0, net.test_data_handler.gpu_cache.num_data):
        net.test_data_handler.gpu_cache.translate[i] = False
  else:
    print 'WARNING: evaluating WITH translations'

  data_pb = deepnet_pb2.Dataset()
  data_pb.name = model.name
  data_pb.gpu_memory = gpu_mem
  data_pb.main_memory =  main_mem
  output_proto_file = os.path.join(base_output_dir, 'data.pbtxt')
  for dataset in datasets:
    output_dir = os.path.join(base_output_dir, dataset)
    if not os.path.isdir(output_dir):
      os.makedirs(output_dir)
    print 'Writing to %s' % output_dir
    size = net.WriteRepresentationToDisk(
	layernames, output_dir, memory=memory, dataset=dataset, numtimes=numtimes)
# Write protocol buffer.
    for i, lname in enumerate(layernames):
      if not size or size[i] == 0:
        continue
      layer = net.GetLayerByName(lname)
      data = data_pb.data.add()
      data.name = '%s_%s' % (lname, dataset)
      data.file_pattern = os.path.join(output_dir, '*-of-*.npy')
      data.size = size[i]
      data.dimensions.append(layer.state.shape[0])
  with open(output_proto_file, 'w') as f:
    text_format.PrintMessage(data_pb, f)

def Usage():
  print 'python %s <model_file> <train_op_file> <output_dir> <numtimes> <no_translations> <set2eval> <layer name1> [layer name2 [..]]' % sys.argv[0]

def main():
  if len(sys.argv) < 8:
    Usage()
    sys.exit(0)
  if use_gpu == 'yes':  
    board = LockGPU()
  model_file = sys.argv[1]
  model = util.ReadModel(model_file)
  random.seed(model.seed)
  train_op_file = sys.argv[2]
  output_dir = sys.argv[3]
  numtimes = int(sys.argv[4])
  no_translations = bool(int(sys.argv[5]))
  set2eval = sys.argv[6]
  layernames = sys.argv[7:]
  ExtractRepresentations(model_file, train_op_file, layernames, output_dir,
                         #memory='1G', datasets=['train', 'validation', 'test'])
                         memory='1G', datasets=[set2eval], 
                         numtimes=numtimes, no_translations=no_translations)

  # Save outputs to mat
  filenames = []
  patt = output_dir + '/' + set2eval + '/*.npy'
  filenames.append(sorted(glob.glob(patt)))
  numfiles = len(filenames[0])
  assert numfiles > 0, 'no ouput files'
  for i in range(0, numfiles):
    outmat = np.load(filenames[0][i])
    sio.savemat(os.path.splitext(filenames[0][i])[0]+'.mat',mdict={'outmat': outmat})


  if use_gpu == 'yes':
    FreeGPU(board)


if __name__ == '__main__':
  main()
