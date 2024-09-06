import torch
import torchvision
import torch.optim

from model.mitnet import MITNet
import numpy as np
import torch.nn as nn
from PIL import Image
import glob
import time,os,cv2

def att(channal):
    cv2.normalize(channal, channal, 0, 255, cv2.NORM_MINMAX)
    M = np.ones(channal.shape, np.uint8) * 255
    img_new = cv2.subtract(M, channal)
    return img_new
def define_model():
    model= MITNet(in_chn=3, wf=20, depth=4).cuda()
    return model
def process(img):
    #img = np.array(img)
    b, g, r = cv2.split(img)
    b = att(b)
    g = att(g)
    r = att(r)
    new_image = cv2.merge([r,g,b])
    return new_image

def dehaze_image(image_hazy_path):
    # --------------------------------------------------
    img_hazy = np.array(Image.open(image_hazy_path).convert('RGB'))
    downsample = 16

    if img_hazy.shape[0] > 4000:
        index = int(img_hazy.shape[1]/4),int(img_hazy.shape[0]/4)
        img_hazy = cv2.resize(img_hazy, index)
    if img_hazy.shape[0] > 2000:
        index = int(img_hazy.shape[1]/2),int(img_hazy.shape[0]/2)
        img_hazy = cv2.resize(img_hazy, index)
        #denosing = cv2.resize(denosing, index)

    size = img_hazy.shape[:2]
    height = size[0]
    width = size[1]
    # h = size[0] - np.mod(size[0], self.downsample)
    # w = size[1] - np.mod(size[1], self.downsample)
    # haze_img = haze_img[0:h, 0:w, :]

    pad_height = (downsample - height % downsample) % downsample
    pad_width = (downsample - width % downsample) % downsample
    img_hazy = cv2.copyMakeBorder(img_hazy, 0, pad_height, 0, pad_width, cv2.BORDER_CONSTANT)

    # if img_hazy.shape[0] % 16 != 0 or img_hazy.shape[1] % 16 != 0:
    #     i = img_hazy.shape[0]
    #     j = img_hazy.shape[1]
    #     if img_hazy.shape[0] % 16 != 0:
    #         while i % 16 != 0:
    #             i-=1
    #     if img_hazy.shape[1]% 16 != 0:
    #         while j % 16 != 0:
    #             j -= 1
    #     img_hazy = cv2.resize(img_hazy, (j,i))
    #-----------------------------------------------------

    img_hazy = (np.asarray(img_hazy) / 255.0)

    img_hazy = torch.from_numpy(img_hazy).float().permute(2, 0, 1).cuda().unsqueeze(0)

    with torch.no_grad():
        clean_image= dehaze_net(img_hazy)[3]

        clean_image = clean_image[:, :, :size[0], :size[1]]

    index = image_hazy_path.split('\\')[-1]

    torchvision.utils.save_image(clean_image, "./model_data/%s" %(index))
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    pth_path = "weight/model_best_isaid.pt"
    dehaze_net = define_model().cuda()
    #dehaze_net = nn.DataParallel(dehaze_net).cuda()
    dehaze_net.load_state_dict(torch.load(pth_path))




    hazy_path = r"./figures/*"
    hazy_list = glob.glob(hazy_path)####
    print('image num:',len(hazy_list))
    dataname =hazy_path.split('\\')[-1].split('/')[0]


    for Id in range(len(hazy_list)):
        dehaze_image(hazy_list[Id])
        print(hazy_list[Id], "done!")
