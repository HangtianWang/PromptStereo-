import torch
import torchvision
import torch.nn.functional as F

def freeze_module(module):
    module.eval()
    for p in module.parameters():
      p.requires_grad = False

def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()

    return module

def normalize_image(image):
    transform = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    scale_image = F.interpolate(image, scale_factor=14 / 16, mode='bilinear', align_corners=True)
    image = transform(image / 255).contiguous()
    scale_image = transform(scale_image / 255).contiguous()

    return image, scale_image

def groupwise_correlation(left, right, group):
    B, C, H, W = left.shape
    channel = C // group
    cost = (left * right).view([B, group, channel, H, W]).mean(dim=2)

    return cost

def build_gwc_volume(left, right, max_disp, group):
    B, C, H, W = left.shape
    volume = left.new_zeros([B, group, max_disp, H, W])

    for i in range(max_disp):
        if i > 0:
            volume[:, :, i, :, i:] = groupwise_correlation(left[:, :, :, i:], right[:, :, :, :-i], group)
        else:
            volume[:, :, i, :, :] = groupwise_correlation(left, right, group)

    return volume

def disparity_regression(x, max_disp):
    disp_value = torch.arange(0, max_disp, dtype=x.dtype, device=x.device)
    disp_value = disp_value.view(1, max_disp, 1, 1)

    return torch.sum(x * disp_value, 1, keepdim=True)

def corr_sampler(corr, coord):
    # 两个输入 [B*H*W, G, 1, D/n] [B*H*W, 1, 2*r+1, 1]
    _, _, _, W = corr.shape
    x_grid = 2 * coord / (W - 1) - 1
    y_grid = torch.zeros_like(x_grid)
    grid = torch.cat([x_grid, y_grid], dim=-1)

    corr = F.grid_sample(corr, grid, align_corners=True)

    return corr

def fmap_sampler(fmap, disp):
    if isinstance(fmap, list):
        warped_fmap = []
        for i in range(len(fmap)):
            scale = disp.shape[-1] / fmap[i].shape[-1]
            if scale > 1:
                disp = F.interpolate(disp, fmap[i].shape[-2:], mode='bilinear', align_corners=True) / scale

            B, C, H, W = fmap[i].shape
            x0 = torch.arange(W).to(disp.device).view(1, 1, 1, W).repeat(B, 1, H, 1)
            y0 = torch.arange(H).to(disp.device).view(1, 1, H, 1).repeat(B, 1, 1, W)
            grid = (torch.cat((x0, y0), dim=1) + torch.cat((-disp, torch.zeros_like(disp)), dim=1)).permute(0, 2, 3, 1)
            grid[..., 0] = 2 * grid[..., 0].clone() / (W - 1) - 1
            grid[..., 1] = 2 * grid[..., 1].clone() / (H - 1) - 1

            warped_fmap.append(F.grid_sample(fmap[i], grid, align_corners=True))
    else:
        scale = disp.shape[-1] / fmap.shape[-1]

        B, C, H, W = fmap.shape
        x0 = torch.arange(W).to(disp.device).view(1, 1, 1, W).repeat(B, 1, H, 1)
        y0 = torch.arange(H).to(disp.device).view(1, 1, H, 1).repeat(B, 1, 1, W)
        grid = (torch.cat((x0, y0), dim=1) + torch.cat((-disp, torch.zeros_like(disp)), dim=1)).permute(0, 2, 3, 1)
        grid[..., 0] = 2 * grid[..., 0].clone() / (W - 1) - 1
        grid[..., 1] = 2 * grid[..., 1].clone() / (H - 1) - 1

        warped_fmap = F.grid_sample(fmap, grid, align_corners=True)

    return warped_fmap

def context_upsample(disp, weight, factor):
    B, C, H, W = disp.shape
    unfold_disp = F.unfold(disp, 3, 1, 1).view(B, -1, H, W)
    unfold_disp = F.interpolate(unfold_disp, (H * factor, W * factor), mode='nearest').view(B, 9, H * factor, W * factor)
    disp = torch.sum(unfold_disp * weight, dim=1).unsqueeze(1)

    return disp

def compute_scale_and_shift(disp):
    flat_disp = disp.flatten(2)
    # 计算中位数
    shift = torch.nanquantile(flat_disp, 0.5, dim=2)
    scale = torch.abs(flat_disp - shift[..., None]).nanmean(dim=2)
    shift[shift.isnan()] = 0
    scale[scale.isnan()] = 1

    return scale, shift

def normalize_disparity(disp):
    dtype = disp.dtype
    scale, shift = compute_scale_and_shift(disp.float())
    norm_disp = (disp - shift[..., None, None]) / scale[..., None, None]

    return norm_disp.to(dtype), scale.to(dtype), shift.to(dtype)

def compute_scale_shift(monocular_depth, gt_depth, mask=None):
    """
    计算 monocular depth 和 ground truth depth 之间的 scale 和 shift.
    
    参数:
    monocular_depth (torch.Tensor): 单目深度图，形状为 (H, W) 或 (N, H, W)
    gt_depth (torch.Tensor): ground truth 深度图，形状为 (H, W) 或 (N, H, W)
    mask (torch.Tensor, optional): 有效区域的掩码，形状为 (H, W) 或 (N, H, W)
    
    返回:
    scale (float): 计算得到的 scale
    shift (float): 计算得到的 shift
    """
    
    flattened_depth_maps = monocular_depth.clone().view(-1).contiguous()
    sorted_depth_maps, _ = torch.sort(flattened_depth_maps)
    percentile_10_index = int(0.2 * len(sorted_depth_maps))
    threshold_10_percent = sorted_depth_maps[percentile_10_index]

    if mask is None:
        mask = (gt_depth > 0) & (monocular_depth > 1e-2) & (monocular_depth > threshold_10_percent)
    
    monocular_depth_flat = monocular_depth[mask]
    gt_depth_flat = gt_depth[mask]
    
    X = torch.stack([monocular_depth_flat, torch.ones_like(monocular_depth_flat)], dim=1)
    y = gt_depth_flat
    
    # 使用最小二乘法计算 [scale, shift]
    A = torch.matmul(X.t(), X) + 1e-6 * torch.eye(2, device=X.device)
    b = torch.matmul(X.t(), y)
    A = A.to(torch.float32)
    b = b.to(torch.float32)
    
    params = torch.linalg.solve(A, b)
    
    scale, shift = params[0].item(), params[1].item()
    
    return scale, shift