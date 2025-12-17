import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import copy

# 假设你的项目结构如下，请根据实际情况调整 import
from .promptstereoVx import PromptStereoVx  # 你的原始模型文件
from util.util import * # 你的工具函数

# ==========================================
# 1. 核心修正模块 (复用之前的 Code Review 结果)
# ==========================================

def compute_scale_shift_safe(monocular_val, target_val, mask=None):
    """Batch-safe Scale/Shift 计算"""
    B, C, H, W = monocular_val.shape
    device = monocular_val.device
    mono_flat = monocular_val.view(B, -1)
    target_flat = target_val.view(B, -1)
    
    if mask is None:
        mask = (target_flat > 0) & (mono_flat > 1e-3)
    else:
        mask = mask.view(B, -1)

    scale_list, shift_list = [], []
    for i in range(B):
        m_valid = mono_flat[i][mask[i]]
        t_valid = target_flat[i][mask[i]]
        if len(m_valid) < 10:
            scale_list.append(torch.tensor(1.0, device=device))
            shift_list.append(torch.tensor(0.0, device=device))
            continue
        ones = torch.ones_like(m_valid)
        X = torch.stack([m_valid, ones], dim=1)
        y = t_valid
        A = torch.matmul(X.t(), X) + 1e-6 * torch.eye(2, device=device)
        b = torch.matmul(X.t(), y)
        try:
            params = torch.linalg.solve(A, b)
            scale_list.append(params[0])
            shift_list.append(params[1])
        except RuntimeError:
            scale_list.append(torch.tensor(1.0, device=device))
            shift_list.append(torch.tensor(0.0, device=device))

    scale = torch.stack(scale_list).view(B, 1, 1, 1).detach()
    shift = torch.stack(shift_list).view(B, 1, 1, 1).detach()
    return scale, shift

class DisparityAtt_Debug(nn.Module):
    """用于验证的 DisparityAtt，支持 Logits 加法逻辑"""
    def __init__(self, maxdisp, sigma_val=2.0, amp_val=5.0): # amp_val 设大一点方便可视化
        super(DisparityAtt_Debug, self).__init__()
        self.D = maxdisp 
        self.register_buffer('disp_grid', torch.arange(self.D, dtype=torch.float32).view(1, 1, self.D, 1, 1))
        self.sigma = nn.Parameter(torch.tensor(sigma_val))
        self.amplitude = nn.Parameter(torch.tensor(amp_val))

    def forward(self, cv, pred_disp):
        # pred_disp 已经是下采样后的尺度
        norm_disp = pred_disp.unsqueeze(2)
        dist = self.disp_grid - norm_disp
        sigma = torch.clamp(torch.abs(self.sigma), min=0.5)
        gaussian_weight = torch.exp(- (dist ** 2) / (2 * sigma ** 2))
        # Logits Additive: CV (Logits) + Gaussian
        output = cv - self.amplitude * gaussian_weight
        return output, gaussian_weight # 返回 weight 用于可视化

# ==========================================
# 2. 验证模型包装器
# ==========================================

class VerificationWrapper(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        # 共享权重
        self.cfg = original_model.cfg
        self.fnet = original_model.fnet
        self.stem = original_model.stem
        self.desc = original_model.desc
        self.hourglass = original_model.hourglass
        self.classifier = original_model.classifier
        
        # 实例化我们的 Debug 版 Attention 模块
        # 注意：这里我们手动替换掉原模型里的 disp_att，或者新建一个
        self.disp_att = DisparityAtt_Debug(self.cfg.gwc_max_disp // (2 ** self.cfg.n_downsample))
        
        # 冻结所有权重，只做推断
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, left, right):
        B, _, H, W = left.shape
        stem_left, left = normalize_image(left)
        stem_right, right = normalize_image(right)

        # 1. 特征提取
        feat_mono, feat_stereo, depth = self.fnet(torch.cat((left, right), dim=0))
        feat_left = [stereo[:B] for stereo in feat_stereo]
        feat_right = [stereo[B:] for stereo in feat_stereo]
        
        stem_list = []
        for i, block in enumerate(self.stem):
            if i >= 1:
                stem_list.append(block(stem_list[-1]))
            else:
                stem_list.append(block(torch.cat((stem_left, stem_right), dim=0)))
        
        match_left = self.desc(torch.cat((feat_left[0], stem_list[1][:B]), dim=1))
        match_right = self.desc(torch.cat((feat_right[0], stem_list[1][B:]), dim=1))

        # 2. 构建 Cost Volume
        down_factor = 2 ** self.cfg.n_downsample
        max_disp_down = self.cfg.gwc_max_disp // down_factor
        
        gwc_volume = build_gwc_volume(match_left, match_right, max_disp_down, self.cfg.gwc_group)
        concat_volume = build_concat_volume(match_left, match_right, max_disp_down)
        comb_volume = torch.cat([gwc_volume, concat_volume], dim=1)
        
        # 3. 聚合得到 Logits (Before Attention)
        # geometry_volume 通常是 Logits
        geometry_volume_before = self.hourglass(comb_volume, feat_left)
        
        # 4. 计算 Initial Disparity (双目)
        # classifier 输出的是 1 channel 的 logits
        logits_before = self.classifier(geometry_volume_before).squeeze(1) # [B, D, H/4, W/4]
        prob_before = F.softmax(logits_before, dim=1)
        init_disp = disparity_regression(prob_before, max_disp_down)

        # 5. 单目对齐
        # depth: [2B, 1, H/4, W/4], 取左图 depth[:B]
        mono_depth = depth[:B].clone().detach()
        # 简单处理：假设 mono 输出的是 inverse depth，需要转为 disparity 空间
        # 如果你的模型输出就是 depth，则 mono_disp = 1 / depth
        # 这里假设输出是 Metric Depth，所以我们转视差: Disp = const / Depth
        # 为简单起见，直接把 mono_depth 当做需要被线性变换的量
        mono_val = 1.0 / (mono_depth + 1e-5) # 转为视差形态
        
        scale, shift = compute_scale_shift_safe(mono_val, init_disp.detach())
        aligned_mono_disp = scale * mono_val + shift

        # 6. 应用 Attention
        # 传入 Logits (geometry_volume_before)
        geometry_volume_after, gaussian_map = self.disp_att(geometry_volume_before, aligned_mono_disp)
        
        logits_after = self.classifier(geometry_volume_after).squeeze(1)
        prob_after = F.softmax(logits_after, dim=1)
        
        return {
            "prob_before": prob_before,  # [B, D, H/4, W/4]
            "prob_after": prob_after,    # [B, D, H/4, W/4]
            "init_disp": init_disp,      # [B, 1, H/4, W/4]
            "aligned_mono": aligned_mono_disp, # [B, 1, H/4, W/4]
            "gaussian_map": gaussian_map # [B, 1, D, H/4, W/4]
        }

# ==========================================
# 3. 可视化执行脚本
# ==========================================

def visualize_at_pixel(ret_dict, batch_idx, y, x, save_path="vis_debug.png"):
    """
    在指定像素坐标 (y, x) 绘制 Probability 分布对比图
    注意：y, x 是在下采样分辨率下的坐标 (H/4, W/4)
    """
    prob_before = ret_dict["prob_before"][batch_idx, :, y, x].cpu().numpy()
    prob_after = ret_dict["prob_after"][batch_idx, :, y, x].cpu().numpy()
    
    # 提取 Scalar 值
    d_mono = ret_dict["aligned_mono"][batch_idx, 0, y, x].item()
    d_init = ret_dict["init_disp"][batch_idx, 0, y, x].item()
    
    # Gaussian 分布 (Logits层面的加成)
    # 把它取出来做个归一化方便展示形态
    gaussian_curve = ret_dict["gaussian_map"][batch_idx, 0, :, y, x].cpu().numpy()
    
    D = len(prob_before)
    x_axis = np.arange(D)
    
    plt.figure(figsize=(12, 6))
    
    # 绘制概率分布
    plt.plot(x_axis, prob_before, 'b--', label='Prob Before (Stereo Only)', alpha=0.6, linewidth=2)
    plt.plot(x_axis, prob_after, 'r-', label='Prob After (Fused)', linewidth=2.5)
    
    # 绘制指示线
    plt.axvline(x=d_mono, color='g', linestyle=':', linewidth=2, label=f'Aligned Mono: {d_mono:.2f}')
    plt.axvline(x=d_init, color='b', linestyle=':', alpha=0.5, label=f'Init Stereo: {d_init:.2f}')
    
    # 绘制 Gaussian 形态 (用副坐标轴或缩放)
    # 为了图表清晰，我们把 gaussian 缩放到 prob 的高度范围示意一下
    scale_factor = prob_before.max() 
    plt.plot(x_axis, gaussian_curve * scale_factor, 'g-', alpha=0.3, label='Gaussian Attention Shape (Scaled)')

    plt.title(f"Cost Volume Probability Distribution @ ({y},{x})")
    plt.xlabel("Disparity Index")
    plt.ylabel("Probability")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Visualization saved to {save_path}")
    plt.show()

# ==========================================
# 4. Main 入口 (模拟运行)
# ==========================================
if __name__ == '__main__':
    # 这里模拟配置和模型加载，你需要替换成真实的加载逻辑
    import hydra
    from omegaconf import OmegaConf
    import os
    import types
    from accelerate import load_checkpoint_and_dispatch

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config_path = os.path.join(project_root, 'config', 'model', 'promptstereo.yaml')
    
    print(f"Loading config from: {config_path}")
    full_cfg = OmegaConf.load(config_path)

    OmegaConf.resolve(full_cfg)
    cfg_omegaconf = full_cfg.instance.cfg

    def dict_to_namespace(d):
        x = types.SimpleNamespace()
        for k, v in d.items():
            if isinstance(v, dict):
                setattr(x, k, dict_to_namespace(v))
            else:
                setattr(x, k, v)
        return x

    cfg_dict = OmegaConf.to_container(cfg_omegaconf, resolve=True)
    cfg = dict_to_namespace(cfg_dict)

    print(f"Instantiating pretrained model: {cfg.pretrained_model.name}")
    instance_config = cfg_omegaconf.pretrained_model.instance
    model_instance = hydra.utils.instantiate(instance_config)
    cfg.pretrained_model.instance = model_instance

    print("Initializing PromptStereo...")
    model = PromptStereoVx(cfg).cuda().eval()
    model = load_checkpoint_and_dispatch(model, "/data/wht/checkpoints/promptstereo/exp1/final/model.safetensors")
    model = model.cuda().eval()
    # --- 2. 包装模型 ---
    print("Wrapping model with VerificationWrapper...")
    vis_model = VerificationWrapper(model).cuda().eval()
    
    # --- 3. 准备数据 ---
    H, W = 448, 896
    left_img = torch.randn(1, 3, H, W).cuda() # 替换为真实图片读取
    right_img = torch.randn(1, 3, H, W).cuda()
    
    print("Running forward pass...")
    with torch.no_grad():
        ret = vis_model(left_img, right_img)
        
    # --- 4. 选择像素进行可视化 ---
    # 选择图像中心附近的一个点 (注意分辨率是 H/4, W/4)
    y_center = (H // 4) // 2
    x_center = (W // 4) // 2
    
    # 或者手动指定一个你感兴趣的区域（比如弱纹理区）
    # y_center, x_center = 50, 100 
    
    print(f"Visualizing pixel at ({y_center}, {x_center})...")
    visualize_at_pixel(ret, 0, y_center, x_center, "vis_debug_center.png")
    
    # 建议：多找几个点看看
    # 1. 找一个 Init Disp 和 Mono Disp 差距很大的点（可能是遮挡或误匹配）
    diff_map = torch.abs(ret['init_disp'] - ret['aligned_mono'])
    y_max, x_max = torch.unravel_index(torch.argmax(diff_map), diff_map.shape[2:])
    print(f"Visualizing max difference pixel at ({y_max}, {x_max})...")
    visualize_at_pixel(ret, 0, y_max, x_max, "vis_debug_conflict.png")