import os
from options.test_options import TestOptions
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import save_images
from itertools import islice
from util import html
import numpy as np

# helper function
def get_specify_z(opt,speci_index):
    z_samples = np.zeros((opt.n_samples + 1, opt.nz))
    for i in range (np.shape(z_samples)[0]):
        z_samples[i][speci_index]=float(i)/10
    z_samples[i][speci_index]=2
    return z_samples

def get_specify_z2(opt,idx1,idx2):
    z_samples = np.zeros((opt.n_samples + 1, opt.nz))
    for i in range (np.shape(z_samples)[0]):
        z_samples[i][idx1] = float(i)/10
        z_samples[i][idx2] = 0.5
        #z_samples[i][idx3] = np.random.randn(1)
    z_samples[i][idx1] = 2
    z_samples[i][idx2] = 2
    #z_samples[i][idx3] = 2
    return z_samples

#option
opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads=1
opt.batchSize = 1   # test code only supports batchSize=1
opt.serial_batches = True  # no shuffle

# create dataset
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
model.eval()
print('Loading model %s' % opt.model)

# create website
web_dir = os.path.join(opt.results_dir, opt.phase +
                       '_sync' if opt.sync else opt.phase)
webpage = html.HTML(web_dir, 'Training = %s, Phase = %s, G = %s, E = %s' % (
    opt.name, opt.phase, opt.G_path, opt.E_path))

# sample random z
if opt.sync:
    z_samples = get_specify_z2(opt,0,1)

# test stage
for i, data in enumerate(islice(dataset, opt.how_many)):
    model.set_input(data)
    print('process input image %3.3d/%3.3d' % (i, opt.how_many))
    if not opt.sync:
        z_samples = get_specify_z2(opt,3,4)
    for nn in range(opt.n_samples + 1):
        encode_B = nn == 0 and not opt.no_encode
        _, real_A, fake_B, real_B, _ = model.test_simple(
            z_samples[nn], encode_real_B=encode_B)
        if nn == 0:
            all_images = [real_A, real_B, fake_B]
            all_names = ['input', 'ground truth', 'encoded']
            #from skimage import io
            #io.imshow(real_A)
        else:
            all_images.append(fake_B)
            all_names.append('random sample%2.2d' % nn)

    img_path = 'input image%3.3i' % i
    save_images(webpage, all_images, all_names, img_path, None,
                width=opt.fineSize, aspect_ratio=opt.aspect_ratio)

webpage.save()