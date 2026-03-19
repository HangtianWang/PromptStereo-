# 参考sttrTransformer，无迭代过程，注意力组计算完毕后直接回归得到视差
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import load_checkpoint_and_dispatch
from .corr import CombinedGeometryEncodingVolume
from .update import MultiPromptUpdateBlock
from .extractor import FeatureExtractor
from .module import *
from util.util import *

class PromptStereoV15(nn.Module):
    def __init__(self, cfg):
        super(PromptStereoV15, self).__init__()
        self.cfg = cfg

        vit = cfg.pretrained_model.instance
        vit = load_checkpoint_and_dispatch(vit, cfg.pretrained_model.checkpoint, strict=True)

        self.fnet = FeatureExtractor(cfg, vit.state_dict())
        # self.update_block = MultiPromptUpdateBlock(cfg, vit.depth_head.state_dict())

        del vit

        # self.hourglass = HourGlass0(cfg)
        # self.classifier = nn.Conv3d(cfg.gwc_group, 1, 3, 1, 1, bias=False)

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

        # self.cnet = nn.ModuleList([
        #     nn.Sequential(
        #         BasicConv(feat_dim + stem_dim, feat_dim + stem_dim, kernel_size=3, stride=1, padding=1),
        #         nn.Conv2d(feat_dim + stem_dim, cfg.pretrained_model.features, 1, 1, 0)
        #     ) for feat_dim, stem_dim in zip(cfg.feat_dim, cfg.stem_dim[1:])
        # ])
        
        # self.hnet = nn.ModuleList([
        #     nn.Sequential(
        #         BasicConv(cfg.pretrained_model.features * 2, cfg.pretrained_model.features * 2, kernel_size=3, stride=1, padding=1),
        #         nn.Conv2d(cfg.pretrained_model.features * 2, cfg.pretrained_model.features, 1, 1, 0)
        #     ) for _ in range(len(cfg.pretrained_model.out_channels))
        # ])

        # self.conf = nn.Sequential(
        #     BasicConv(cfg.pretrained_model.features * 2, cfg.pretrained_model.features * 2, kernel_size=3, stride=1, padding=1),
        #     nn.Conv2d(cfg.pretrained_model.features * 2, 1, 1, 1, 0),
        #     nn.Sigmoid()
        # )

        self.sttr_transformer = EpipolarStereoTransformer(d_model=self.cfg.feat_dim[0], nhead=8, num_layers=4)

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
        stem_left, left = normalize_image(left)
        stem_right, right = normalize_image(right)

        feat_mono, feat_stereo, depth = self.fnet(torch.cat((left, right), dim=0))
        ctx_mono = feat_mono[:B]
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

        # match_left shape: [B, d_model, H/4, W/4]
        # [B,1,H/4,W/4]
        init_disp = self.sttr_transformer(match_left, match_right)
    
        # 准备上采样权重
        spx_pred = None
        if not test_mode or True: # 测试时也需要上采样还原分辨率
            xspx = self.spx_4(match_left)
            xspx = self.spx_2(xspx, stem_list[0][:B])
            spx_pred = F.softmax(self.spx(xspx), 1)
        
        # 上采样
        factor = 2 ** self.cfg.n_downsample
        up_disp = context_upsample(init_disp * factor, spx_pred, factor)

        if test_mode:
            return up_disp
        else:
            init_disp_full = F.interpolate(
                init_disp * factor,  # 值域放大 4 倍
                size=(H, W),     # 尺寸放大到全尺寸
                mode='bilinear', 
                align_corners=True
            )
            
            return init_disp_full, [up_disp] 
    

if __name__ == '__main__':
    import os
    import hydra
    from omegaconf import OmegaConf
    import types
    from safetensors.torch import load_file

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
    model = PromptStereoV15(cfg).cuda().eval()

    # ckpt_path = "/data/wht/checkpoints/promptstereo/exp2/model.safetensors" 
    # from safetensors.torch import load_file
    # state_dict = load_file(ckpt_path)
    # model.load_state_dict(state_dict, strict=False)

    H, W = 448, 896
    left = torch.randn(1, 3, H, W).cuda()
    right = torch.randn(1, 3, H, W).cuda()

    print(f"Running forward pass with input shape: {left.shape}")
    with torch.no_grad():
        init_disp, disp_preds = model(left, right, iters=2, test_mode=False)

    print(f"Init Disp Shape: {init_disp.shape}")
    print(f"Final Disp Shape: {disp_preds[-1].shape}")