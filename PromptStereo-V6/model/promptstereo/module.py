import torch
import torch.nn as nn
import torch.nn.functional as F
import re
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
        
        self.amplitude = nn.Parameter(torch.tensor(0.01))

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
        output = cv + self.amplitude * gaussian_weight
        
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