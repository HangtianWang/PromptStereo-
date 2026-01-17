import torch
import cv2
import hydra
import numpy as np
import matplotlib.pyplot as plt
import argparse
from omegaconf import OmegaConf
from model.promptstereo.promptstereoV0 import PromptStereoV0
from safetensors.torch import load_file
from util.padder import InputPadder

def load_image(path, size=None):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    
    # 1. 必须保留：BGR 转 RGB (因为 dataset 用的是 imageio，默认 RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if size:
        img = cv2.resize(img, size)
    
    # 2. 关键修改：不要除以 255.0，保持 [0, 255] 范围
    img = img.astype(np.float32) 
    
    # [H, W, 3] -> [1, 3, H, W]
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).cuda()
    return img

def dict_to_namespace(d):
    if isinstance(d, dict):
        ns = argparse.Namespace()
        for k, v in d.items():
            setattr(ns, k, dict_to_namespace(v))
        return ns
    elif isinstance(d, list):
        return [dict_to_namespace(i) for i in d]
    else:
        return d

def main():
    # 1. 简易加载配置
    print("Loading Config...")
    config_path = 'config/model/promptstereo.yaml'
    full_cfg = OmegaConf.load(config_path)
    cfg_omega = full_cfg.instance.cfg

    # 2. 实例化 Config 中定义的 DepthAnythingV2 子模块
    print("Instantiating Pretrained Module...")
    # 注意：hydra.utils.instantiate 需要 OmegaConf 对象或字典
    vit_instance = hydra.utils.instantiate(cfg_omega.pretrained_model.instance)
    
    # --- 关键修改开始 ---
    # 将 OmegaConf 配置转为普通字典，再转为 Namespace
    # 这样既支持 cfg.xx.xx 的访问方式，也可以存入 vit_instance 对象
    cfg_dict = OmegaConf.to_container(cfg_omega, resolve=True)
    cfg = dict_to_namespace(cfg_dict)
    
    # 现在可以放心地把模型对象塞进去了
    cfg.pretrained_model.instance = vit_instance
    # --- 关键修改结束 ---

   # 3. 初始化并加载模型
    print("Initializing PromptStereoV0...")
    model = PromptStereoV0(cfg).cuda().eval()
    
    ckpt_path = '/data/wht/checkpoints/promptstereo/exp0/final/model.safetensors' 
    
    print(f"Loading weights from {ckpt_path}...")
    state_dict = load_file(ckpt_path)
    model.load_state_dict(state_dict, strict=True)
    
    # 4. 准备输入数据 (请确保输入尺寸是 32 的倍数
    left = load_image('demo_imgs/octogons/im0.png')
    right = load_image('demo_imgs/octogons/im1.png')
    padder = InputPadder(left.shape, divis_by=32)
    left, right = padder.pad(left, right)

    print("Running Inference...")
    with torch.no_grad():
        out = model(left, right, test_mode=True)

    # 5. 获取并保存可视化结果
    if hasattr(model, 'debug_fused_disp'):
        fused_disp = model.debug_fused_disp
        stereo_disp = model.stereo_disp
        mono_depth = model.mono_depth
        
        # 1. 关键步骤：去掉 Padding，恢复原始尺寸
        fused_disp = padder.unpad(fused_disp)
        out = padder.unpad(out)
        stereo_disp = padder.unpad(stereo_disp)
        mono_depth = padder.unpad(mono_depth)
        
        # 2. 转为 numpy [H, W]
        disp_vis = fused_disp.squeeze().cpu().numpy()
        out_vis = out.squeeze().cpu().numpy()
        stereo_vis = stereo_disp.squeeze().cpu().numpy()
        mono_vis = mono_depth.squeeze().cpu().numpy()
        
        # 3. 归一化到 0-255 (为了可视化)
        d_min = disp_vis.min()
        d_max = disp_vis.max()
        if d_max - d_min > 0:
            disp_norm = (disp_vis - d_min) / (d_max - d_min)
        else:
            disp_norm = np.zeros_like(disp_vis)

        out_min = out_vis.min()
        out_max = out_vis.max()
        if out_max - out_min > 0:
            out_norm = (out_vis - out_min) / (out_max - out_min)
        else:
            out_norm = np.zeros_like(out_vis)

        s_min = stereo_vis.min()
        s_max = stereo_vis.max()
        if s_max - s_min > 0:
            stereo_norm = (stereo_vis - s_min) / (s_max - s_min)
        else:
            stereo_norm = np.zeros_like(stereo_vis)

        m_min = mono_vis.min()
        m_max = mono_vis.max()
        if m_max - m_min > 0:
            mono_norm = (mono_vis - m_min) / (m_max - m_min)
        else:
            mono_norm = np.zeros_like(mono_vis)
            
        disp_uint8 = (disp_norm * 255).astype(np.uint8)
        out_uint8 = (out_norm * 255).astype(np.uint8)
        stereo_uint8 = (stereo_norm * 255).astype(np.uint8)
        mono_uint8 = (mono_norm * 255).astype(np.uint8)
        
        # 4. 应用伪彩色 (INFERNO 配色方案接近 Magma，且对比度高)
        disp_color = cv2.applyColorMap(disp_uint8, cv2.COLORMAP_INFERNO)
        out_color = cv2.applyColorMap(out_uint8, cv2.COLORMAP_INFERNO)
        stereo_color = cv2.applyColorMap(stereo_uint8, cv2.COLORMAP_INFERNO)
        mono_color = cv2.applyColorMap(mono_uint8, cv2.COLORMAP_INFERNO)
        
        # 5. 直接保存图片，不含白边
        save_path0 = 'demo_imgs/vis_fusion_imgs/exp0_fusuion.png'
        save_path1 = 'demo_imgs/vis_fusion_imgs/exp0_out.png'
        save_path2 = 'demo_imgs/vis_fusion_imgs/exp0_stereo.png'
        save_path3 = 'demo_imgs/vis_fusion_imgs/exp0_mono.png'
        cv2.imwrite(save_path0, disp_color)
        cv2.imwrite(save_path1, out_color)
        cv2.imwrite(save_path2, stereo_color)
        cv2.imwrite(save_path3, mono_color)
        print(f"Visualization saved to: {save_path0} (Shape: {disp_color.shape})")
        print(f"Visualization saved to: {save_path1} (Shape: {out_color.shape})")
        print(f"Visualization saved to: {save_path2} (Shape: {stereo_color.shape})")
        print(f"Visualization saved to: {save_path3} (Shape: {mono_color.shape})")

    else:
        print("Error: 'debug_fused_disp' not found using model instance.")

if __name__ == '__main__':
    main()