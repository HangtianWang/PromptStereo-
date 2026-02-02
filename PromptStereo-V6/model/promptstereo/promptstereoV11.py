# 1.代价体采用concat_volume+corr_volume
# 2.增加从代价体中回归的熵值、左右图的差异不取平均、去除梯度图用于优化融合置信度
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import load_checkpoint_and_dispatch
from .corr import CombinedGeometryEncodingVolume
from .update import MultiPromptUpdateBlock
from .extractor import FeatureExtractor
from .module import *
from util.util import *

class PromptStereoV3(nn.Module):
    def __init__(self, cfg):
        super(PromptStereoV3, self).__init__()
        self.cfg = cfg

        vit = cfg.pretrained_model.instance
        vit = load_checkpoint_and_dispatch(vit, cfg.pretrained_model.checkpoint, strict=True)

        self.fnet = FeatureExtractor(cfg, vit.state_dict())
        self.update_block = MultiPromptUpdateBlock(cfg, vit.depth_head.state_dict())

        del vit

        self.hourglass = HourGlass(cfg)
        self.classifier = nn.Conv3d(cfg.gwc_group, 1, 3, 1, 1, bias=False)

        self.stem = nn.ModuleList([
            nn.Sequential(
                BasicConv(in_channel, out_channel, kernel_size=3, stride=2, padding=1),
                BasicConv(out_channel, out_channel, relu='relu', kernel_size=3, stride=1, padding=1)
            ) for in_channel, out_channel in zip([3] + cfg.stem_dim[:-1], cfg.stem_dim)
        ])

        self.spx_4 = nn.Sequential(
            BasicConv(cfg.feat_dim[0], cfg.stem_dim[1], kernel_size=3, stride=1, padding=1),
            BasicConv(cfg.stem_dim[1], cfg.stem_dim[1], relu='relu', kernel_size=3, stride=1, padding=1)
        )
        self.spx_2 = Conv2x(cfg.stem_dim[1], cfg.stem_dim[0], deconv=True)
        self.spx = nn.ConvTranspose2d(2 * cfg.stem_dim[0], 9, kernel_size=4, stride=2, padding=1)

        self.desc = nn.Sequential(
            BasicConv(cfg.feat_dim[0] + cfg.stem_dim[1], cfg.feat_dim[0] + cfg.stem_dim[1], kernel_size=3, stride=1, padding=1),
            nn.Conv2d(cfg.feat_dim[0] + cfg.stem_dim[1], cfg.feat_dim[0], kernel_size=1, stride=1, padding=0)
        )

        self.cnet = nn.ModuleList([
            nn.Sequential(
                BasicConv(feat_dim + stem_dim, feat_dim + stem_dim, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(feat_dim + stem_dim, cfg.pretrained_model.features, 1, 1, 0)
            ) for feat_dim, stem_dim in zip(cfg.feat_dim, cfg.stem_dim[1:])
        ])
        
        self.hnet = nn.ModuleList([
            nn.Sequential(
                BasicConv(cfg.pretrained_model.features * 2, cfg.pretrained_model.features * 2, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(cfg.pretrained_model.features * 2, cfg.pretrained_model.features, 1, 1, 0)
            ) for _ in range(len(cfg.pretrained_model.out_channels))
        ])

        self.conf = nn.Sequential(
            BasicConv(cfg.pretrained_model.features * 3, cfg.pretrained_model.features * 3, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(cfg.pretrained_model.features * 3, 1, 1, 1, 0),
            nn.Sigmoid()
        )
        self.conf2 = nn.Sequential(
            nn.Conv2d(2, 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )

        self.disp_att = DisparityAtt(self.cfg.gwc_max_disp // (2 ** self.cfg.n_downsample))
        self.sobel = SobelEdge()

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
            if isinstance(m, nn.SyncBatchNorm):
                m.eval()

    def upsample_disp(self, disp, mask):
        B, C, H, W = disp.shape
        factor = 2 ** self.cfg.n_downsample
        mask = mask.view(B, 1, 9, factor, factor, H, W)
        mask = torch.softmax(mask, dim=2)

        disp = F.unfold(factor * disp, [3, 3], padding=1)
        disp = disp.view(B, C, 9, 1, 1, H, W)
        disp = torch.sum(mask * disp, dim=2)
        disp = disp.permute(0, 1, 4, 2, 5, 3).contiguous()
        
        return disp.view(B, C, factor * H, factor * W)

    def forward(self, left, right, iters=16, test_mode=False):
        B, _, H, W = left.shape
        # stem是原图尺寸[B,C,H,W]，后一个经过放缩14/16，尺寸为[B,C,Hs,Ws]
        stem_left, left = normalize_image(left)
        stem_right, right = normalize_image(right)
        # 假设cfg.feat_dim为 [D0, D1, D2, D3]
        # 单目深度图 [B, 1, H/4, W/4]
        feat_mono, feat_stereo, depth = self.fnet(torch.cat((left, right), dim=0))
        # 单目特征图只取左侧
        # 单目特征组 [B,D3, H/4, W/4] [B, D3, H/8, W/8] [B, D3, H/16, W/16] [B, D3, H/32, W/32]
        ctx_mono = feat_mono[:B]
        # 拆分双目左右特征图
        # 左右形状都为 [B, D0, H/4, W/4] [B, D1, H/8, W/8] [B, D2, H/16, W/16] [B, D3, H/32, W/32]
        feat_left = [stereo[:B] for stereo in feat_stereo]
        feat_right = [stereo[B:] for stereo in feat_stereo]
        # 可学习的特特征图提取分支,假设stem_dim为[S1,S2,S3,S4,S5]
        # [2B,S1,H/2,W/2], [2B,S2,H/4,W/4], [2B,S3,H/8,W/8], [2B,S4,H/16,W/16], [2B,S5,H/32,W/32]
        stem_list = []
        for i, block in enumerate(self.stem):
            if i >= 1:
                stem_list.append(block(stem_list[-1]))
            else:
                stem_list.append(block(torch.cat((stem_left, stem_right), dim=0)))
        # 融合预训练分支和可学习分支提取的特征，四分之一分辨率，输出为[B, D0, H/4, W/4]
        match_left = self.desc(torch.cat((feat_left[0], stem_list[1][:B]), dim=1))
        match_right = self.desc(torch.cat((feat_right[0], stem_list[1][B:]), dim=1))
        # 构建代价体，代价体聚合，初始视差回归
        # [B,G,maxdisp//4,H/4,W/4]
        gwc_volume = build_gwc_volume(match_left, match_right, self.cfg.gwc_max_disp // (2 ** self.cfg.n_downsample), self.cfg.gwc_group)
        # [B,2*D0,maxdisp//4,H/4,W/4]
        concat_volume = build_concat_volume(match_left, match_right, self.cfg.gwc_max_disp // (2 ** self.cfg.n_downsample))
        # [B,2*D0+G,maxdisp//4,H/4,W/4]
        comb_volume = torch.cat([gwc_volume, concat_volume], dim=1)
        # [B,G,maxdisp//4,H/4,W/4]
        geometry_encoding_volume = self.hourglass(comb_volume, feat_left)
        # [B,maxdisp//4,H/4,W/4]
        prob = F.softmax(self.classifier(geometry_encoding_volume).squeeze(1), dim=1)
        # [B,1,H/4,W/4]
        init_disp = disparity_regression(prob, self.cfg.gwc_max_disp // (2 ** self.cfg.n_downsample))

        del gwc_volume, concat_volume, comb_volume, prob

        # 指导初始视差上采样的准备
        if not test_mode:
            # [B, S1, H/4, W/4]
            xspx = self.spx_4(match_left)
            # [B, 2*S0, H/2, W/2]
            xspx = self.spx_2(xspx, stem_list[0][:B])
            # [B, 9, H/4, W/4]
            spx_pred = self.spx(xspx)
            # [B, 9, H/4, W/4]
            spx_pred = F.softmax(spx_pred, 1)
        
        # Motion Prompt要求的输入之一
        # 输入形状为[B, D0, H/4, W/4] [B, D0, H/4, W/4] [B,G,maxdisp//4,H/4,W/4] 2 4
        corr_block = CombinedGeometryEncodingVolume(match_left, match_right, geometry_encoding_volume, self.cfg.corr_level, self.cfg.corr_radius)
        # 融合预训练视觉大模型和可学习CNN提取的双目特征
        ctx_stereo = [block(torch.cat((x, y), dim=1)) for block, x, y in zip(self.cnet, feat_stereo, stem_list[1:])]
        ctx_left = [stereo[:B] for stereo in ctx_stereo]
        ctx_right = [stereo[B:] for stereo in ctx_stereo]
        warped_ctx_right = fmap_sampler(ctx_right, init_disp)
        net = [block(torch.cat((x, y), dim=1)) for block, x, y in zip(self.hnet, ctx_left, warped_ctx_right)]

        # 单双目视差融合成迭代起点的初始视差+Structure Prompt的要求输入
        norm_depth, _, _ = normalize_disparity(depth)
        _, scale, shift = normalize_disparity(init_disp)
        aligned_depth = norm_depth * scale[..., None, None] + shift[..., None, None]
        # scale, shift = compute_scale_shift(depth.clone().squeeze(1).to(torch.float32), init_disp.clone().squeeze(1).to(torch.float32))
        # aligned_depth = scale * depth + shift

        # 利用对齐后的单目视差图沿视差维度增强代价体，获得新的初始视差
        # geometry_encoding_volume = self.disp_att(geometry_encoding_volume, aligned_depth)
        # Init disp from geometry encoding volume [B,maxdisp//4,H/4,W/4]
        # prob = F.softmax(self.classifier(geometry_encoding_volume).squeeze(1), dim=1)
        # [B,1,H/4,W/4]
        # init_disp = disparity_regression(prob, self.cfg.gwc_max_disp // (2 ** self.cfg.n_downsample))

        # del prob
        # 获得融合置信度conf
        # 从代价体计算熵值[B, 1, H, W], 反应某一个像素的视差可信度
        entropy = calculate_disparity_entropy(geometry_encoding_volume)
        # 计算左图和右图warp之后的左图的差异，显式反应遮挡关系
        # error_map = get_occlusion_proxy(ctx_left[0], warped_ctx_right[0])
        # 计算梯度，反应边界位置，单目边界较模糊应该信赖锐利的双目边界
        # grad_map = self.sobel(stem_left)
        # depth [B,1,H/4,W/4], init_disp [B,1,H/4,W/4]
        conf = self.conf(torch.cat((ctx_left[0], warped_ctx_right[0], ctx_left[0] - warped_ctx_right[0]), dim=1))
        # 融合熵和置信度1得到最终置信度
        conf = self.conf2(torch.cat((conf, entropy), dim=1))
        disp = conf * init_disp + (1 - conf) * aligned_depth

        disp_pred = []
        for itr in range(iters):
            disp = disp.detach()
            # corr形状为[B,C_out,H/4,W/4]，C_out=(2r+1)*(G+1)*level
            corr = corr_block(disp)
            net, delta_disp, mask = self.update_block(net, corr, disp, ctx_mono, norm_depth)
            disp = disp + delta_disp

            if test_mode and itr < iters - 1:
                continue

            up_disp = self.upsample_disp(disp, mask)
            disp_pred.append(up_disp)
        
        if test_mode:
            return up_disp

        factor = 2 ** self.cfg.n_downsample
        init_disp = context_upsample(init_disp * 4, spx_pred, factor)

        return init_disp, disp_pred
    

if __name__ == '__main__':
    import os
    import hydra
    from omegaconf import OmegaConf
    import types

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
    model = PromptStereoV3(cfg).cuda().eval()

    H, W = 448, 896
    left = torch.randn(1, 3, H, W).cuda()
    right = torch.randn(1, 3, H, W).cuda()

    print(f"Running forward pass with input shape: {left.shape}")
    with torch.no_grad():
        init_disp, disp_preds = model(left, right, iters=2, test_mode=False)

    print(f"Init Disp Shape: {init_disp.shape}")
    print(f"Final Disp Shape: {disp_preds[-1].shape}")