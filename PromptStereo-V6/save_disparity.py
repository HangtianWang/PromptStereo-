import os
import hydra
import torch
import numpy as np
import imageio.v3 as imageio
from glob import glob
from tqdm import tqdm
from hydra.utils import instantiate
from matplotlib import pyplot as plt
from accelerate import load_checkpoint_and_dispatch
from accelerate.logging import get_logger
from model import fetch_model
from util.padder import InputPadder

@hydra.main(version_base=None, config_path='config', config_name='save_disparity')
def main(cfg):
    logger = get_logger(__name__)
    accelerator = instantiate(cfg.accelerator)

    model = fetch_model(cfg, logger)
    model = load_checkpoint_and_dispatch(model, cfg.checkpoint)
    logger.info(f'Loading checkpoint from {cfg.checkpoint}.')

    model = accelerator.prepare_model(model)
    model.eval()

    left_list = sorted(glob(cfg.left_list))
    right_list = sorted(glob(cfg.right_list))
    left_list = left_list[accelerator.process_index::accelerator.num_processes]
    right_list = right_list[accelerator.process_index::accelerator.num_processes]

    for left_file, right_file in tqdm(zip(left_list, right_list), total=len(left_list), dynamic_ncols=True, disable=not accelerator.is_main_process):
        left = imageio.imread(left_file).astype(np.float32)[..., :3]
        right = imageio.imread(right_file).astype(np.float32)[..., :3]
        left = torch.from_numpy(left).permute(2, 0, 1)[None].cuda()
        right = torch.from_numpy(right).permute(2, 0, 1)[None].cuda()

        disp_gt = imageio.imread(left_file.replace('image_2', 'disp_occ_0')).astype(np.float32) / 256
        valid = disp_gt > 0

        padder = InputPadder(left.shape, divis_by=32)
        left, right = padder.pad(left, right)

        with torch.no_grad():
            disp_pred = model(left, right, iters=cfg.model.valid_iters, test_mode=True)
            disp_pred = padder.unpad(disp_pred) 

        out = (torch.abs(disp_pred.squeeze() - torch.from_numpy(disp_gt).cuda()) > 3).float() * 255
        out[~valid] = 0

        os.makedirs(cfg.disp_dir, exist_ok=True)
        # plt.imsave(f'{cfg.disp_dir}/{left_file.split('/')[cfg.base_index].split('.')[0]}.png', disp_pred.squeeze().cpu().numpy(), cmap='jet')
        plt.imsave(f'{cfg.disp_dir}/{left_file.split('/')[cfg.base_index].split('.')[0]}.png', out.squeeze().cpu().numpy(), cmap='grey')

    accelerator.end_training()

if __name__ == '__main__':
    main()