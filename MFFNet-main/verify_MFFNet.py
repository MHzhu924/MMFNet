import torch
import torch.nn.functional as F
import numpy as np
import pdb, os, argparse
from scipy import misc
import time
import imageio
from model.MFFNet_models import MFFNet
from data import test_dataset

torch.cuda.set_device(0)
parser = argparse.ArgumentParser()
parser.add_argument('--equi_input_width', type=int, default=512, help='training dataset width size')
parser.add_argument('--equi_input_height', type=int, default=1024, help='training dataset height size')
opt = parser.parse_args()

dataset_path = './dataset/'

model = MFFNet()

model.load_state_dict(torch.load('./models/MFFNet_360-SOD.pth.29'))

model.cuda()
model.eval()


test_datasets = ['360-SOD']
# test_datasets = ['360-SSOD']

for dataset in test_datasets:
    save_path = './results/MFFNet/' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root = dataset_path + dataset + '/360-SOD-te/image/'
    print(dataset)
    gt_root = dataset_path + dataset + '/360-SOD-te/groundtruth/'
    test_loader = test_dataset(image_root, gt_root, inputsize=(opt.equi_input_width, opt.equi_input_height))
    time_sum = 0
    for i in range(test_loader.size):
        image,gt,name= test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        with torch.no_grad():
            time_start = time.time()
            res, s2, s3, s4,  s1_sig, s2_sig, s3_sig, s4_sig = model(image)
            time_end = time.time()

        time_sum = time_sum+(time_end-time_start)
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        res = (res *255).astype(np.uint8)  # 转换为unit8 类型
        imageio.imwrite(save_path+name, res)
        if i == test_loader.size-1:
            print('Running time {:.5f}'.format(time_sum/test_loader.size))
            print('Average speed: {:.4f} fps'.format(test_loader.size/time_sum))
            