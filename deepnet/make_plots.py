from deepnet import deepnet_pb2
import matplotlib.pyplot as plt
import glob, sys, gzip, numpy as np

def preds(metrics_list):
  y = []
  for metric in metrics_list:
    count = metric.count
    y.append( 100*(1- metric.correct_preds/metric.count))
  return y

def error(metrics_list):
  y = []
  for metric in metrics_list:
    count = metric.count
    y.append(metric.error/metric.count)
  return y

def get_plot(v, skip, label):
  y = v[skip:]
  x = np.arange(skip, len(v))
  return plt.plot(x, y, label=label)

def Usage():
  print 'python %s <model_file> <stat_to_plot>' % sys.argv[0]

if __name__ == '__main__':
  plt.ion()
  proto = sys.argv[1]
  stat_to_plot = sys.argv[2]
  proto = glob.glob(proto + "*")[-1]
  print proto
  skip = 0
  if len(sys.argv) > 3:
    skip = int(sys.argv[3])
  model_pb = deepnet_pb2.Model()
  f = gzip.open(proto, 'rb')
  model_pb.ParseFromString(f.read())
  f.close()
  if stat_to_plot == 'pred':
    train = preds(model_pb.train_stats)
    valid = preds(model_pb.validation_stats)
    test = preds(model_pb.test_stats)
  elif stat_to_plot == 'error':
    train = error(model_pb.train_stats)
    valid = error(model_pb.validation_stats)
    test = error(model_pb.test_stats)
  else:
    print 'No available stat %s to plot' % (stat_to_plot)
    sys.exit(0)

  x = np.arange(len(train))
  plt.figure(1)
  p1 = get_plot(train, skip, 'train')
  p2 = get_plot(valid, skip, 'valid')
  p3 = get_plot(test, skip, 'test')
  plt.legend()
  plt.xlabel('Iterations')
  plt.ylabel('Error %')
  plt.grid(True)
  #plt.axis([0, 50, 30, 42])
  plt.draw()
  raw_input('Press any key')
