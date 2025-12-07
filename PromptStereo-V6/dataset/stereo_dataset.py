import os
import torch
import numpy as np
import imageio.v3 as imageio
from glob import glob
from util.reader import *
from util.augmentor import Augmentor

class StereoDataset(torch.utils.data.Dataset):
    def __init__(self, sparse=False, aug_params=None, reader=None, mask=None):
        self.augmentor = None
        self.reader = reader
        self.mask = mask
        self.image_list = []
        self.disp_list = []

        if aug_params:
            self.augmentor = Augmentor(sparse, aug_params)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        left = imageio.imread(self.image_list[index][0]).astype(np.float32)[..., :3]
        right = imageio.imread(self.image_list[index][1]).astype(np.float32)[..., :3]
        disp, valid = self.reader(self.disp_list[index], self.mask)

        disp_right, valid_right = self.reader(self.disp_list[index].replace('left', 'right'), self.mask)

        if self.augmentor:
            left, right, disp, valid, _, _ = self.augmentor(left=left, right=right, disp=disp, valid=valid, disp_right=disp_right, valid_right=valid_right)

        left = torch.from_numpy(left).permute(2, 0, 1)
        right = torch.from_numpy(right).permute(2, 0, 1)
        disp = torch.from_numpy(disp)[None]
        valid = torch.from_numpy(valid)[None]

        return left, right, disp, valid

class SceneFlow(StereoDataset):
    def __init__(self, aug_params=None, root='/data/StereoData/sceneflow', dstype='frames_finalpass', things_test=False, mask='all'):
        super(SceneFlow, self).__init__(sparse=False, aug_params=aug_params, reader=sceneflow_disp_reader, mask=mask)
        assert os.path.exists(root)
        self.root = root
        self.dstype = dstype

        if things_test:
            self._add_things('TEST')
        else:
            self._add_things('TRAIN')
            self._add_monkaa('TRAIN')
            self._add_driving('TRAIN')

    def _add_things(self, split='TRAIN'):
        left_list = sorted(glob(os.path.join(self.root, self.dstype, split, '*/*/left/*.png')))
        right_list = [img.replace('left', 'right') for img in left_list]
        disp_list = [img.replace(self.dstype, 'disparity').replace('.png', '.pfm') for img in left_list]

        assert len(left_list) == len(right_list) == len(disp_list)

        for idx, (left, right, disp) in enumerate(zip(left_list, right_list, disp_list)):
            self.image_list += [[left, right]]
            self.disp_list += [disp]

    def _add_monkaa(self, split='TRAIN'):
        left_list = sorted(glob(os.path.join(self.root, self.dstype, split, '*/left/*.png')))
        right_list = [img.replace('left', 'right') for img in left_list]
        disp_list = [img.replace(self.dstype, 'disparity').replace('.png', '.pfm') for img in left_list]

        assert len(left_list) == len(right_list) == len(disp_list)

        for idx, (left, right, disp) in enumerate(zip(left_list, right_list, disp_list)):
            self.image_list += [[left, right]]
            self.disp_list += [disp]

    def _add_driving(self, split='TRAIN'):
        left_list = sorted(glob(os.path.join(self.root, self.dstype, split, '*/*/*/left/*.png')))
        right_list = [img.replace('left', 'right') for img in left_list]
        disp_list = [img.replace(self.dstype, 'disparity').replace('.png', '.pfm') for img in left_list]

        assert len(left_list) == len(right_list) == len(disp_list)

        for idx, (left, right, disp) in enumerate(zip(left_list, right_list, disp_list)):
            self.image_list += [[left, right]]
            self.disp_list += [disp]

class KITTI(StereoDataset):
    def __init__(self, aug_params=None, root='data/StereoData/kitti', year='2015', split='training', mask='all'):
        super(KITTI, self).__init__(sparse=True, aug_params=aug_params, reader=kitti_disp_reader, mask=mask)
        assert os.path.exists(root)

        if year == '2012':
            left_list = sorted(glob(os.path.join(root, year, split, 'colored_0/*_10.png')))
            right_list = sorted(glob(os.path.join(root, year, split, 'colored_1/*_10.png')))
            disp_list = sorted(glob(os.path.join(root, year, split, 'disp_occ/*_10.png')))

        if year == '2015':
            left_list = sorted(glob(os.path.join(root, year, split, 'image_2/*_10.png')))
            right_list = sorted(glob(os.path.join(root, year, split, 'image_3/*_10.png')))
            disp_list = sorted(glob(os.path.join(root, year, split, 'disp_occ_0/*_10.png')))

        assert len(left_list) == len(right_list) == len(disp_list)

        for _, (left, right, disp) in enumerate(zip(left_list, right_list, disp_list)):
            self.image_list += [[left, right]]
            self.disp_list += [disp]

class Middlebury(StereoDataset):
    def __init__(self, aug_params=None, root='/data/StereoData/middlebury', year='MiddEval3', split='training', resolution='H', mask='noc'):
        super(Middlebury, self).__init__(sparse=True, aug_params=aug_params, reader=middlebury_disp_reader, mask=mask)
        assert os.path.exists(root)

        if year == 'MiddEval3':
            left_list = sorted(glob(os.path.join(root, year, split + resolution, '*/im0.png')))
            right_list = sorted(glob(os.path.join(root, year, split + resolution, '*/im1.png')))
            disp_list = sorted(glob(os.path.join(root, year, split + resolution, '*/disp0GT.pfm')))
        elif year == '2021':
            left_list = sorted(glob(os.path.join(root, year, 'data', '*/im0.png')))
            right_list = sorted(glob(os.path.join(root, year, 'data', '*/im1.png')))
            disp_list = sorted(glob(os.path.join(root, year, 'data', '*/disp0.pfm')))    

        assert len(left_list) == len(right_list) == len(disp_list)

        for _, (left, right, disp) in enumerate(zip(left_list, right_list, disp_list)):
            self.image_list += [[left, right]]
            self.disp_list += [disp]

class ETH3D(StereoDataset):
    def __init__(self, aug_params=None, root='/data/StereoData/eth3d', split='training', mask='noc'):
        super(ETH3D, self).__init__(sparse=True, aug_params=aug_params, reader=eth3d_disp_reader, mask=mask)
        assert os.path.exists(root)

        left_list = sorted(glob(os.path.join(root, f'two_view_{split}', '*/im0.png')))
        right_list = sorted(glob(os.path.join(root, f'two_view_{split}', '*/im1.png')))
        disp_list = sorted(glob(os.path.join(root, 'two_view_training_gt', '*/disp0GT.pfm')))

        assert len(left_list) == len(right_list) == len(disp_list)

        for _, (left, right, disp) in enumerate(zip(left_list, right_list, disp_list)):
            self.image_list += [[left, right]]
            self.disp_list += [disp]