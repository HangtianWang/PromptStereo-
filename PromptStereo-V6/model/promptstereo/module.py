import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import math
from torch.distributions import Beta

class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, deconv=False, is_3d=False, norm='instance', relu='leaky', **kwargs):
        super(BasicConv, self).__init__()
        self.use_norm = norm
        self.use_relu = relu

        if is_3d:
            if deconv:
                self.conv = nn.ConvTranspose3d(in_channel, out_channel, bias=False, **kwargs)
            else:
                self.conv = nn.Conv3d(in_channel, out_channel, bias=False, **kwargs)
            if norm == 'batch':
                self.norm = nn.BatchNorm3d(out_channel)
            elif norm == 'instance':
                self.norm = nn.InstanceNorm3d(out_channel)
        else:
            if deconv:
                self.conv = nn.ConvTranspose2d(in_channel, out_channel, bias=False, **kwargs)
            else:
                self.conv = nn.Conv2d(in_channel, out_channel, bias=False, **kwargs)
            if norm == 'batch':
                self.norm = nn.BatchNorm2d(out_channel)
            elif norm == 'instance':
                self.norm = nn.InstanceNorm2d(out_channel)

        if relu == 'leaky':
            self.relu = nn.LeakyReLU(inplace=True)
        elif relu == 'relu':
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        
        if self.use_norm:
            x = self.norm(x)
        if self.use_relu:
            x = self.relu(x)

        return x

class Conv2x(nn.Module):
    def __init__(self, in_channel, out_channel, deconv=False, is_3d=False, norm='instance', relu='leaky'):
        super(Conv2x, self).__init__()

        if deconv and is_3d:
            kernel = (4, 4, 4)
        elif deconv:
            kernel = 4
        else:
            kernel = 3

        self.conv1 = BasicConv(in_channel, out_channel, deconv, is_3d, norm, relu, kernel_size=kernel, stride=2, padding=1)
        self.conv2 = BasicConv(out_channel * 2, out_channel * 2, False, is_3d, norm, relu, kernel_size=3, stride=1, padding=1)

    def forward(self, x, y):
        x = self.conv1(x)
        x = torch.cat((x, y), dim=1)
        x = self.conv2(x)

        return x

class FeatureAtt(nn.Module):
    def __init__(self, cv_channel, feat_channel):
        super(FeatureAtt, self).__init__()
        self.feat_att = nn.Sequential(
            BasicConv(feat_channel, feat_channel // 2, kernel_size=1),
            nn.Conv2d(feat_channel // 2, cv_channel, 1)
        )

    def forward(self, cv, feat):
        feat_att = self.feat_att(feat).unsqueeze(2)
        cv = torch.sigmoid(feat_att) * cv

        return cv
    
# baseline的HourGlass
class HourGlass0(nn.Module):
    def __init__(self, cfg):
        super(HourGlass0, self).__init__()
        cv_channel = cfg.gwc_group

        self.corr_stem = BasicConv(cfg.gwc_group, cfg.gwc_group, is_3d=True, kernel_size=3, padding=1)

        self.conv1 = nn.Sequential(
            BasicConv(cv_channel, cv_channel * 2, is_3d=True, kernel_size=3, stride=2, padding=1),
            BasicConv(cv_channel * 2, cv_channel * 2, is_3d=True, kernel_size=3, padding=1)
        )
        self.conv2 = nn.Sequential(
            BasicConv(cv_channel * 2, cv_channel * 4, is_3d=True, kernel_size=3, stride=2, padding=1),
            BasicConv(cv_channel * 4, cv_channel * 4, is_3d=True, kernel_size=3, padding=1)
        )
        self.conv3 = nn.Sequential(
            BasicConv(cv_channel * 4, cv_channel * 6, is_3d=True, kernel_size=3, stride=2, padding=1),
            BasicConv(cv_channel * 6, cv_channel * 6, is_3d=True, kernel_size=3, padding=1)
        )

        self.conv3_up = BasicConv(cv_channel * 6, cv_channel * 4, deconv=True, is_3d=True, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1))
        self.conv2_up = BasicConv(cv_channel * 4, cv_channel * 2, deconv=True, is_3d=True, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1))
        self.conv1_up = BasicConv(cv_channel * 2, cv_channel, deconv=True, is_3d=True, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1))

        self.agg_0 = nn.Sequential(
            BasicConv(cv_channel * 8, cv_channel * 4, is_3d=True, kernel_size=1),
            BasicConv(cv_channel * 4, cv_channel * 4, is_3d=True, kernel_size=3, padding=1),
            BasicConv(cv_channel * 4, cv_channel * 4, is_3d=True, kernel_size=3, padding=1)
        )
        self.agg_1 = nn.Sequential(
            BasicConv(cv_channel * 4, cv_channel * 2, is_3d=True, kernel_size=1),
            BasicConv(cv_channel * 2, cv_channel * 2, is_3d=True, kernel_size=3, padding=1),
            BasicConv(cv_channel * 2, cv_channel * 2, is_3d=True, kernel_size=3, padding=1)
        )

        self.feature_att_4 = FeatureAtt(cv_channel, cfg.feat_dim[0])
        self.feature_att_8 = FeatureAtt(cv_channel * 2, cfg.feat_dim[1])
        self.feature_att_16 = FeatureAtt(cv_channel * 4, cfg.feat_dim[2])
        self.feature_att_32 = FeatureAtt(cv_channel * 6, cfg.feat_dim[3])
        self.feature_att_up_16 = FeatureAtt(cv_channel * 4, cfg.feat_dim[2])
        self.feature_att_up_8 = FeatureAtt(cv_channel * 2, cfg.feat_dim[1])

    def forward(self, x, feat):
        x = self.corr_stem(x)
        x = self.feature_att_4(x, feat[0])

        conv1 = self.conv1(x)
        conv1 = self.feature_att_8(conv1, feat[1])

        conv2 = self.conv2(conv1)
        conv2 = self.feature_att_16(conv2, feat[2])

        conv3 = self.conv3(conv2)
        conv3 = self.feature_att_32(conv3, feat[3])

        conv3_up = self.conv3_up(conv3)
        conv2 = torch.cat((conv3_up, conv2), dim=1)
        conv2 = self.agg_0(conv2)
        conv2 = self.feature_att_up_16(conv2, feat[2])

        conv2_up = self.conv2_up(conv2)
        conv1 = torch.cat((conv2_up, conv1), dim=1)
        conv1 = self.agg_1(conv1)
        conv1 = self.feature_att_up_8(conv1, feat[1])

        conv = self.conv1_up(conv1)

        return conv

# 针对volume=corr_volume+cat_volume重新设计的HourGlass
class HourGlass(nn.Module):
    def __init__(self, cfg):
        super(HourGlass, self).__init__()
        in_channel = cfg.gwc_group+ 2 * cfg.feat_dim[0]
        cv_channel = cfg.gwc_group

        # 修改了corr_stem，对于com_volume先做融合，再做特征提取
        self.corr_stem = nn.Sequential(
            BasicConv(in_channel, cv_channel, is_3d=True, kernel_size=1, padding=0),
            BasicConv(cv_channel, cv_channel, is_3d=True, kernel_size=3, padding=1),
        )

        self.conv1 = nn.Sequential(
            BasicConv(cv_channel, cv_channel * 2, is_3d=True, kernel_size=3, stride=2, padding=1),
            BasicConv(cv_channel * 2, cv_channel * 2, is_3d=True, kernel_size=3, padding=1)
        )
        self.conv2 = nn.Sequential(
            BasicConv(cv_channel * 2, cv_channel * 4, is_3d=True, kernel_size=3, stride=2, padding=1),
            BasicConv(cv_channel * 4, cv_channel * 4, is_3d=True, kernel_size=3, padding=1)
        )
        self.conv3 = nn.Sequential(
            BasicConv(cv_channel * 4, cv_channel * 6, is_3d=True, kernel_size=3, stride=2, padding=1),
            BasicConv(cv_channel * 6, cv_channel * 6, is_3d=True, kernel_size=3, padding=1)
        )

        self.conv3_up = BasicConv(cv_channel * 6, cv_channel * 4, deconv=True, is_3d=True, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1))
        self.conv2_up = BasicConv(cv_channel * 4, cv_channel * 2, deconv=True, is_3d=True, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1))
        self.conv1_up = BasicConv(cv_channel * 2, cv_channel, deconv=True, is_3d=True, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1))

        self.agg_0 = nn.Sequential(
            BasicConv(cv_channel * 8, cv_channel * 4, is_3d=True, kernel_size=1),
            BasicConv(cv_channel * 4, cv_channel * 4, is_3d=True, kernel_size=3, padding=1),
            BasicConv(cv_channel * 4, cv_channel * 4, is_3d=True, kernel_size=3, padding=1)
        )
        self.agg_1 = nn.Sequential(
            BasicConv(cv_channel * 4, cv_channel * 2, is_3d=True, kernel_size=1),
            BasicConv(cv_channel * 2, cv_channel * 2, is_3d=True, kernel_size=3, padding=1),
            BasicConv(cv_channel * 2, cv_channel * 2, is_3d=True, kernel_size=3, padding=1)
        )

        self.feature_att_4 = FeatureAtt(cv_channel, cfg.feat_dim[0])
        self.feature_att_8 = FeatureAtt(cv_channel * 2, cfg.feat_dim[1])
        self.feature_att_16 = FeatureAtt(cv_channel * 4, cfg.feat_dim[2])
        self.feature_att_32 = FeatureAtt(cv_channel * 6, cfg.feat_dim[3])
        self.feature_att_up_16 = FeatureAtt(cv_channel * 4, cfg.feat_dim[2])
        self.feature_att_up_8 = FeatureAtt(cv_channel * 2, cfg.feat_dim[1])

    def forward(self, x, feat):
        x = self.corr_stem(x)
        x = self.feature_att_4(x, feat[0])

        conv1 = self.conv1(x)
        conv1 = self.feature_att_8(conv1, feat[1])

        conv2 = self.conv2(conv1)
        conv2 = self.feature_att_16(conv2, feat[2])

        conv3 = self.conv3(conv2)
        conv3 = self.feature_att_32(conv3, feat[3])

        conv3_up = self.conv3_up(conv3)
        conv2 = torch.cat((conv3_up, conv2), dim=1)
        conv2 = self.agg_0(conv2)
        conv2 = self.feature_att_up_16(conv2, feat[2])

        conv2_up = self.conv2_up(conv2)
        conv1 = torch.cat((conv2_up, conv1), dim=1)
        conv1 = self.agg_1(conv1)
        conv1 = self.feature_att_up_8(conv1, feat[1])

        conv = self.conv1_up(conv1)

        return conv

    
class DisparityAtt(nn.Module):
    def __init__(self, maxdisp, learnable_sigma=True):
        super(DisparityAtt, self).__init__()
        self.D = maxdisp 
        self.register_buffer('disp_grid', torch.arange(self.D, dtype=torch.float32).view(1, 1, self.D, 1, 1))

        if learnable_sigma:
            self.sigma = nn.Parameter(torch.tensor(2.0)) 
        else:
            self.sigma = 2.0
        
        self.amplitude = nn.Parameter(torch.tensor(0.5))

    def forward(self, cv, pred_disp):
        """
        cv: [B, G, D, H, W] Cost Volume (Logits)
        pred_disp: [B, 1, H, W] Aligned Monocular Disparity
        """
        #[B, 1, 1, H, W]
        norm_disp = pred_disp.unsqueeze(2)
        
        #[B, 1, D, H, W]
        dist = self.disp_grid - norm_disp

        sigma = torch.clamp(torch.abs(self.sigma), min=0.5)
        
        #高斯分布 [B, 1, D, H, W]
        gaussian_weight = torch.exp(- (dist ** 2) / (2 * sigma ** 2))
        output = cv + self.amplitude * gaussian_weight * cv
        # print("cv.max:",cv.max(),"amplitude * gaussian_weight", (self.amplitude * gaussian_weight).max(), "amplitude", self.amplitude)
        
        return output
    

class SobelEdge(nn.Module):
    def __init__(self):
        super(SobelEdge, self).__init__()
        
        self.kernel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
        self.kernel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)

    def forward(self, img):
        """
        img: [B, 3, H, W]
        """
        img = F.interpolate(img, scale_factor=0.25, mode='bilinear', align_corners=True)
        B, C, H, W = img.shape
        if C == 3:
            gray = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
        else:
            gray = img
            
        kx = self.kernel_x.to(img.device).type(img.dtype)
        ky = self.kernel_y.to(img.device).type(img.dtype)

        gx = F.conv2d(gray, kx, padding=1)
        gy = F.conv2d(gray, ky, padding=1)

        edge = torch.sqrt(gx**2 + gy**2 + 1e-6)
        edge = edge / (edge.max().detach() + 1e-6)
        
        return edge # [B, 1, H/4, W/4]
    
class LBPEncoder(nn.Module):
    def __init__(self, offsets_str='(-1,-1), (1,1), (1,-1), (-1,1)'):
        super().__init__()
        # 1. 解析字符串获取偏移量列表
        matches = re.findall(r'\((-?\d+),\s*(-?\d+)\)', offsets_str)
        if not matches:
            raise ValueError(f"Invalid offsets format: {offsets_str}")
        offsets = [(int(y), int(x)) for y, x in matches]
        
        # 2. 根据最大偏移量确定卷积核大小
        num_neighbors = len(offsets)
        max_offset = max(max(abs(y), abs(x)) for y, x in offsets)
        kernel_size = 2 * max_offset + 1
        center = max_offset # 中心点坐标
        
        # 3. 构建深度卷积层 (不更新权重)
        self.lbp_conv = nn.Conv2d(
            in_channels=1, 
            out_channels=num_neighbors,
            kernel_size=kernel_size, 
            padding=max_offset,
            padding_mode="replicate", 
            bias=False
        )
        
        # 4. 手动设置卷积权重: Neighbor - Center
        weight = torch.zeros((num_neighbors, 1, kernel_size, kernel_size))
        for i, (dy, dx) in enumerate(offsets):
            # 将中心设为 -1，邻域设为 1，实现差分计算
            weight[i, 0, center + dy, center + dx] = 1.0
            weight[i, 0, center, center] = -1.0
            
        self.lbp_conv.weight = nn.Parameter(weight, requires_grad=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x: [Batch, 1, H, W] Grayscale image
        Returns:
            out: [Batch, num_neighbors, H, W] LBP features in range (0, 1)
        """
        return self.sigmoid(self.lbp_conv(x))
    

class BMResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(BMResidualBlock, self).__init__()
  
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.BatchNorm2d(planes)
        
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.Sequential()

        if stride == 1 and in_planes == planes:
            self.downsample = None
        
        else:    
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)


    def forward(self, x):
        y = x
        y = self.conv1(y)
        y = self.norm1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.norm2(y)
        y = self.relu(y)

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)
    

class BetaModulator(nn.Module):
    def __init__(self, lbp_dim=4, hidden_dim=None, modulation_ratio=1.0, norm_fn='batch'):
        super(BetaModulator, self).__init__()
        self.modulation_ratio = modulation_ratio
        
        # 确定中间层维度，默认维持 lbp_dim
        if hidden_dim is None:
            hidden_dim = lbp_dim
            
        # 1. 编码器 (Encoder): 提取 LBP 特征的深层表示
        # 输入维度: lbp_dim * 2 (因为拼接了 disp 和 depth 的 LBP 特征)
        self.conv1 = nn.Sequential(
            nn.Conv2d(lbp_dim * 2, hidden_dim * 2, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim * 2, hidden_dim * 2, kernel_size=3, padding=1, bias=True),
        )
        
        # 2. 上下文聚合 (Context Aggregation): U-Net 类似的下采样和上采样
        down_dim = 64 if hidden_dim * 2 < 64 else 128
        self.down = nn.Sequential(
            BMResidualBlock(hidden_dim * 2, down_dim, norm_fn, stride=2),
            BMResidualBlock(down_dim, 128, norm_fn, stride=1)
        )
        self.up = nn.ConvTranspose2d(128, hidden_dim * 2, kernel_size=2, stride=2)
        
        # 3. 参数预测头 (Head): 预测 Beta 分布的 alpha 和 beta 参数
        # 输入: concat(Encoder特征, 上采样特征)
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_dim * 4, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.Softplus(),
            nn.Conv2d(hidden_dim, 2, kernel_size=1, padding=0, bias=False),
            nn.Softplus(), # 保证输出非负
        )

    def forward(self, lbp_disp, lbp_depth):
        """
        输入:
             lbp_disp: [B, lbp_dim, H, W] 视差图的 LBP 特征
             lbp_depth: [B, lbp_dim, H, W] 深度图(先验)的 LBP 特征
        输出:
             modulation: [B, 1, H, W] 融合系数 (0~1)
        """
        # 特征提取
        x1 = self.conv1(torch.cat([lbp_disp, lbp_depth], dim=1))
        x2 = self.up(self.down(x1))
        
        # 预测 alpha 和 beta 参数 (加 1 是为了保证分布的形状是有意义的，避免极端情况)
        beta_paras = self.conv2(torch.cat([x1, x2], dim=1)) + 1 

        # 构建 Beta 分布
        alpha, beta = torch.split(beta_paras, 1, dim=1)
        distribution = Beta(alpha, beta)

        # 采样或取均值
        if self.training:
            # 训练时使用重参数化技巧进行采样，增加随机性以提高鲁棒性
            modulation = distribution.rsample()
        else:
            # 推理时使用均值，保证确定性
            modulation = distribution.mean
        
        return modulation
    

import torch
import torch.nn as nn

class MonoParamEstimator(nn.Module):
    def __init__(self, in_channels=2, use_global_pool=True):
        """
        Args:
            in_channels (int): 输入通道数 (通常是 disp + depth = 2)
            use_global_pool (bool): 是否开启全局池化以输出全局参数 a, b
        """
        super(MonoParamEstimator, self).__init__()
        self.use_global_pool = use_global_pool

        # 逐级降低分辨率，提取抽象特征
        self.enc1 = self._conv_block(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = self._conv_block(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        
        self.bottleneck = self._conv_block(64, 128)

        # 恢复空间分辨率，并通过 concat 融合浅层细节
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = self._conv_block(128 + 64, 64)
        
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = self._conv_block(64 + 32, 32)

        # 输出 2 通道: Channel 0 -> Scale (a), Channel 1 -> Shift (b)
        self.head = nn.Conv2d(32, 2, kernel_size=1)
        
        # 池化，将 HxW 压缩为 1x1，实现全局参数估计
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def _conv_block(self, in_ch, out_ch):
        """基础卷积块: Conv-BN-ReLU x2"""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, disp, depth):
        """
        Args:
            disp: [B, 1, H, W] 视差图
            depth: [B, 1, H, W] 单目深度
        Returns:
            a: [B, 1, 1, 1] Scale 参数
            b: [B, 1, 1, 1] Shift 参数
        """
        # 拼接输入
        x = torch.cat([disp, depth], dim=1)
        
        # Forward U-Net
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b_feat = self.bottleneck(self.pool2(e2))
        
        d2 = self.dec2(torch.cat([self.up2(b_feat), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        
        # 预测密集参数图 [B, 2, H, W]
        out = self.head(d1)
        
        # 全局池化，获取全局参数 [B, 2, 1, 1]
        if self.use_global_pool:
            out = self.global_pool(out)
            
        # 拆分为 a 和 b
        a, b = torch.split(out, 1, dim=1)
        
        return a, b

class WindowAttentionWithMap(nn.Module):
    def __init__(self, dim=64, window_size=8, num_heads=4):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def window_partition(self, x):
        """
        Args:
            x: (B, H, W, C)
        Returns:
            windows: (num_windows*B, window_size*window_size, C)
        """
        B, H, W, C = x.shape
        # 将特征图重塑为窗口布局
        x = x.view(B, H // self.window_size, self.window_size, W // self.window_size, self.window_size, C)
        # 维度置换: [B, H/M, W/M, M, M, C] -> [B, H/M, W/M, M*M, C]
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.window_size * self.window_size, C)
        return windows

    def window_reverse(self, windows, H, W):
        """
        Args:
            windows: (num_windows*B, window_size*window_size, C)
        Returns:
            x: (B, H, W, C)
        """
        B = int(windows.shape[0] / (H * W / self.window_size / self.window_size))
        x = windows.view(B, H // self.window_size, W // self.window_size, self.window_size, self.window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

    def forward(self, x):
        # x shape: [B, C, H, W] -> [1, 64, 112, 224]
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1) # [B, H, W, C]

        # 切分窗口
        x_windows = self.window_partition(x)  # [B*N_win, 64, 64]
        
        # 计算 Q, K, V
        B_, N, C_ = x_windows.shape
        qkv = self.qkv(x_windows).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B*N_win, Heads, M^2, Head_Dim]

        # 计算 Attention Score
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1)) # [B*N_win, Heads, M^2, M^2]
        
        # 获取 Attention Map 并保存
        attn_map = attn.softmax(dim=-1) 

        # 特征加权
        x_windows_new = (attn_map @ v).transpose(1, 2).reshape(B_, N, C_)
        x_windows_new = self.proj(x_windows_new)

        # 还原回大图
        x_out = self.window_reverse(x_windows_new, H, W) # [B, H, W, C]
        x_out = x_out.permute(0, 3, 1, 2) # [B, C, H, W]

        return x_out, attn_map
    

class AttentionGuidedCostRefiner(nn.Module):
    def __init__(self, window_size=8):
        super().__init__()
        self.window_size = window_size

    def window_partition(self, x):
        # 复用同样的切分逻辑
        B, H, W, C = x.shape
        x = x.view(B, H // self.window_size, self.window_size, W // self.window_size, self.window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.window_size * self.window_size, C)
        return windows

    def window_reverse(self, windows, H, W):
        B = int(windows.shape[0] / (H * W / self.window_size / self.window_size))
        x = windows.view(B, H // self.window_size, W // self.window_size, self.window_size, self.window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

    def forward(self, cost_volume, attn_map):
        """
        cost_volume: [B, G, D, H, W]
        attn_map: [B*N_win, Heads, M^2, M^2] (From Step 1)
        """
        B, G, D, H, W = cost_volume.shape
        
        # 变形: 将 G 和 D 合并为通道
        # 现在的形状: [B, G*D, H, W]
        cv_reshaped = cost_volume.view(B, G * D, H, W).permute(0, 2, 3, 1) # [B, H, W, G*D]

        # 切分窗口
        # Shape: [B*N_win, M^2, G*D]
        cv_windows = self.window_partition(cv_reshaped)

        # 处理 Attention Map 特征图可能有多个 Head，但代价体不需要 Head 的概念,对 Heads 维度取平均，得到通用的空间亲和矩阵。
        # Input: [B*N_win, Heads, M^2, M^2] -> Output: [B*N_win, M^2, M^2]
        attn_map_avg = attn_map.mean(dim=1)

        # 矩阵乘法
        # [B*N_win, M^2, M^2] @ [B*N_win, M^2, G*D] -> [B*N_win, M^2, G*D]
        # 含义：每个像素的代价 = 它周围窗口内像素代价的加权和
        cv_refined_windows = torch.matmul(attn_map_avg, cv_windows)

        # 还原形状
        cv_refined = self.window_reverse(cv_refined_windows, H, W) # [B, H, W, G*D]
        cv_refined = cv_refined.permute(0, 3, 1, 2).view(B, G, D, H, W)
        
        # 残差连接
        return cost_volume + cv_refined
    

class PositionEncodingSine1D(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        # 预计算频率项
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.register_buffer('div_term', div_term)

    def forward(self, w, device):
        pe = torch.zeros(w, self.d_model, device=device)
        position = torch.arange(0, w, dtype=torch.float, device=device).unsqueeze(1)
        pe[:, 0::2] = torch.sin(position * self.div_term)
        pe[:, 1::2] = torch.cos(position * self.div_term)
        return pe # [W, C]

class EpipolarAttentionBlock(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        # 1. 自注意力 (基于官方 TransformerEncoderLayer)
        self.self_attn = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=d_model*4, 
            batch_first=True, norm_first=True
        )
        
        # 2. 交叉注意力机制
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        
        # 前馈网络 (包含 Pre-Norm)
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4), 
            nn.GELU(), 
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, left, right):
        # --- A. Self Attention 阶段 ---
        # 左图和右图虽然独立进行 Self-Attention，但我们把它们在 Batch 维度拼起来一次算完，极致提速。
        concat = torch.cat([left, right], dim=0) # [2*BH, W, C]
        concat = self.self_attn(concat)
        left_self, right_self = concat.chunk(2, dim=0) # 拆开还原 [BH, W, C]
        
        # --- B. Cross Attention 阶段 ---
        # 1. L 查 R: 左图去匹配右图
        left_cross, _ = self.cross_attn(
            self.norm_q(left_self), self.norm_kv(right_self), self.norm_kv(right_self), need_weights=False
        )
        left = left_self + left_cross
        left = left + self.ffn(left)
        
        # 2. R 查 L: 右图反向匹配左图 (对称设计，让特征充分交流)
        right_cross, _ = self.cross_attn(
            self.norm_q(right_self), self.norm_kv(left_self), self.norm_kv(left_self), need_weights=False
        )
        right = right_self + right_cross
        right = right + self.ffn(right)
        
        return left, right

class EpipolarStereoTransformer(nn.Module):
    def __init__(self, d_model=128, nhead=8, num_layers=4):
        super().__init__()
        self.d_model = d_model
        
        self.pos_encoder = PositionEncodingSine1D(d_model)
        self.layers = nn.ModuleList([
            EpipolarAttentionBlock(d_model, nhead) for _ in range(num_layers)
        ])
        
        # 将最终特征投影一次，用于计算相关性内积
        self.match_proj = nn.Linear(d_model, d_model)

    def forward(self, left_feat, right_feat):
        """
        left_feat, right_feat: [B, C, H, W]  (通常是 1/4 尺寸)
        """
        B, C, H, W = left_feat.shape
        assert C == self.d_model, f"输入通道必须与 d_model 等价 (当前: {C} vs {self.d_model})"
        
        # 形变转换为极线序列: [B, C, H, W] -> [B*H, W, C]
        left = left_feat.permute(0, 2, 3, 1).reshape(B * H, W, C)
        right = right_feat.permute(0, 2, 3, 1).reshape(B * H, W, C)
        
        # 注入位置编码
        pos = self.pos_encoder(W, left.device).unsqueeze(0) # [1, W, C]
        left = left + pos
        right = right + pos
        
        # Transformer 堆叠计算
        for layer in self.layers:
            left, right = layer(left, right)
            
        # 使用线性层将最终特征转换为 Q 和 K
        q = self.match_proj(left)  # [BH, W, C]
        k = self.match_proj(right) # [BH, W, C]
        
        # 矩阵乘法得到稠密的 Cost Matrix [BH, W_L, W_R]
        attn_scores = torch.bmm(q, k.transpose(1, 2)) / (self.d_model ** 0.5)
        
        # 几何约束 (Masking): 立体匹配中视差 >= 0，即 x_L >= x_R
        # 我们屏蔽掉所有 x_R > x_L 的区域 (矩阵的上三角部分设为 -inf)
        mask = torch.triu(torch.ones(W, W, device=left.device), diagonal=1).bool()
        attn_scores.masked_fill_(mask, float('-inf'))
        
        # 最优传输近似 (为了极简，我们这里用标准的 Softmax 代替 Sinkhorn)
        # 求得对于每一个左图像素在右图上的匹配概率分布
        attn_probs = F.softmax(attn_scores, dim=-1) # [BH, W_L, W_R]
        
        # 计算视差期望值： d = x_L - x_R 
        i_idx = torch.arange(W, device=left.device).view(1, W, 1) # x_L
        j_idx = torch.arange(W, device=left.device).view(1, 1, W) # x_R
        disp_grid = (i_idx - j_idx).float() # [1, W, W] 物理视差矩阵
        
        # 概率分布和视差值求点积
        disp_1d = torch.sum(attn_probs * disp_grid, dim=-1) # [BH, W_L]
        
        # 还原回图像尺寸
        init_disp = disp_1d.view(B, 1, H, W)
        
        return init_disp