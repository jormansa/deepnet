#import visualize
from neuralnet import *
from trainer import *

# display features of neural network
def display_wsorted(w, s, r, c, fig, vmax=None, vmin=None, dataset='mnist',
                    title='weights_sorted'):
    if dataset == 'norb':
        numvis = 4096
    else:
        numvis = w.shape[0]
    numhid = w.shape[1]
    sc = s
    sr = numvis / s
    padding = numhid - r * c
    if isinstance(w, np.ndarray):
        w = w.T[:, :sr * sc]
    else:
        w = w.asarray().T[:, :sr * sc]

    vh = w.reshape(sr * numhid, sc)
    pvh = np.zeros((sr, sc, r, c))
    pvh2 = np.zeros((sr * r, sc * c))
    norm_list = []
    for i in range(r):
        for j in range(c):
            pvh[:, :, i, j] = vh[(i * c + j) * sr: (i * c + j + 1) * sr, :]
            norm = (pvh[:, :, i, j] ** 2).sum()
            norm_list.append((norm, i, j))
    norm_list.sort(reverse=True)
    index = 0

    sc_old = sc
    sr_old = sr
    sc = sr_old
    sr = sc_old

    pvh2 = pvh2.T
    pvh = np.transpose(pvh, (1, 0, 2, 3))
    for norm, i, j in norm_list:
        ii = index / c
        jj = index % c
        pvh2[ii * sr:(ii + 1) * sr, jj * sc:(jj + 1) * sc] = pvh[:, :, i, j]
        index += 1

    plt.figure(fig)
    plt.clf()

    #plt.suptitle(title)
    # vmax = 0.5
    # vmin = -0.5
    plt.imshow(pvh2, cmap=plt.cm.gray, interpolation='nearest', vmax=vmax, vmin=vmin)
    scale = 1
    xmax = sc * c
    ymax = sr * r
    color = 'k'
    for x in range(1,c):
        plt.axvline(x=x*sc/scale,ymin=0,ymax=ymax/scale, color = color, linewidth=2)

    for y in range(1,r):
        plt.axhline(y=y*sr/scale, xmin=0,xmax=xmax/scale, color = color, linewidth=2)

    plt.axis('off')
    plt.draw()
    raw_input('Press any key')

    return pvh

# display features convolutional neural network
def display_convw2(w, s, r, c, fig, title='conv_filters'):
  """w: num_filters X sizeX**2 * num_colors."""
  num_f, num_d = w.shape
  #assert s**2 * 3 == num_d
  assert s**2 * 1 == num_d
  #pvh = np.zeros((s*r, s*c, 3))
  pvh = np.zeros((s*r, s*c, 1))
  for i in range(r):
    for j in range(c):
      #pvh[i*s:(i+1)*s, j*s:(j+1)*s, :] = w[i*c + j, :].reshape(3, s, s).T
      pvh[i*s:(i+1)*s, j*s:(j+1)*s, :] = w[i*c + j, :].reshape(1, s, s).T
  mx = pvh.max()
  mn = pvh.min()
  pvh = 255*(pvh - mn) / (mx-mn)
  pvh = pvh.astype('uint8')
  plt.figure(fig)
  plt.suptitle(title)
  plt.imshow(pvh, interpolation="nearest")
  scale = 1
  xmax = s * c
  ymax = s * r
  color = 'k'
  for x in range(0, c):
    plt.axvline(x=x*s/scale, ymin=0, ymax=ymax/scale, color=color)
  for y in range(0, r):
    plt.axhline(y=y*s/scale, xmin=0, xmax=xmax/scale, color=color)
  plt.draw()
  return pvh

def display_convw(w, s, r, c, fig, vmax=None, vmin=None, dataset='mnist', title='conv_filters'):

  """
  w2 = np.zeros(w.shape)
  d = w.shape[1]/3
  print w.shape
  for i in range(w.shape[0]):
    for j in range(w.shape[1]/3):
      w2[i, j] = w[i, 3*j]
      w2[i, j + d] = w[i, 3*j+1]
      w2[i, j + 2*d] = w[i, 3*j+2]
  w = w2
  """

  numhid = w.shape[0]
  size_x = s
  size_y = s    # For now.
  num_channels = w.shape[1] / (size_x*size_y)
  #assert num_channels == 3
  assert w.shape[1] % size_x*size_y == 0
  if isinstance(w, np.ndarray):
    vh = w.reshape(size_x*numhid*num_channels, size_y)
  else:
    vh = w.asarray().reshape(size_x*numhid*num_channels, size_y)
  pvh = np.zeros((size_x*r, size_y*c, num_channels))
  for i in range(r):
    for j in range(c):
      for ch in range(num_channels):
        pvh[i*size_x:(i+1)*size_x, j*size_y:(j+1)*size_y, ch] = \
            vh[(num_channels*(i*c+j)+ch)*size_x:(num_channels*(i*c+j)+ch+1)*size_x,:]

  # pvh /= np.std(pvh)
  plt.figure(fig)
  plt.clf()
  plt.title(title)
  #plt.imshow(pvh, vmax=vmax, vmin=vmin)
  plt.imshow(pvh[:,:,0],cmap=plt.cm.gray, vmax=vmax, vmin=vmin)
  scale = 1
  xmax = size_x*c
  ymax = size_y*r
  color = 'k'
  for x in range(0, c):
    plt.axvline(x=x*size_x/scale, ymin=0,ymax=ymax/scale, color = color)
  for y in range(0, r):
    plt.axhline(y=y*size_y/scale, xmin=0,xmax=xmax/scale, color = color)
  plt.draw()

  return pvh

# input params
model_file      = sys.argv[1]
train_op_file   = sys.argv[2]
rows            = int(sys.argv[3])
cols            = int(sys.argv[4])
vmin            = float(sys.argv[5])
vmax            = float(sys.argv[6])
filename        = sys.argv[7]

# load model
#model = util.ReadModel('/Volumes/hdd/Users/jordimansanet/Dropbox/series/pruebas_deepnet/deepnet/examples/rbm_gender/pretrained/1layer/bin/1/512/models_classifier/model_trainfull_test_classifier_LAST')
#op = ReadOperation('/Volumes/hdd/Users/jordimansanet/Dropbox/series/pruebas_deepnet/deepnet/examples/rbm_gender/pretrained/1layer/bin/1/512/eval.pbtxt')
model = util.ReadModel(model_file)
op = ReadOperation(train_op_file)
op.randomize = False
op.get_last_piece = True
net = CreateDeepnet(model, op, op)
net.LoadModelOnGPU()
net.PrintNetwork()

# visualize
net.net.hyperparams.enable_display = True
edge = net.edge[0]
#pvh = display_wsorted(edge.params['weight'].asarray(),
#                          edge.proto.receptive_field_width,
#                          rows,
#                          cols,
#                          edge.fig,
#                          vmax=vmax, 
#                          vmin=vmin,
#                          title=edge.name)
#plt.savefig(filename, bbox_inches='tight')

pvh = display_convw(edge.params['weight'].asarray(),
                            edge.proto.receptive_field_width,
                            rows,
                            cols,
                            edge.fig,
                            title='')
#                            title=edge.name)

#pvh = display_convw2(edge.params['weight'].asarray(),
#                            edge.proto.receptive_field_width,
#                            rows,
#                            cols,
#                            edge.fig,
#                            title=edge.name)
plt.axis('off')
plt.savefig('foo.png', bbox_inches='tight')
raw_input('Press any key')