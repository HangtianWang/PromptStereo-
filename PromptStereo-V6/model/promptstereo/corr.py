import torch
import torch.nn.functional as F
from util.util import corr_sampler

class CombinedGeometryEncodingVolume:
    def __init__(self, left, right, gwc_volume, level=2, radius=4):
        # 有几层
        self.level = level
        # 搜索范围
        self.radius = radius
        self.gev_pyramid = []
        self.apc_pyramid = []

        B, G, D, H, W = gwc_volume.shape
        # [B*H*W,G,1,D]
        gwc_volume = gwc_volume.permute(0, 3, 4, 1, 2).contiguous().view(B * H * W, G, 1, D)

        self.gev_pyramid.append(gwc_volume)
        # 构建多level金字塔 [B*H*W, G, 1, D/n]
        for i in range(level - 1):
            gwc_volume = F.avg_pool2d(gwc_volume, [1, 2], [1, 2])
            self.gev_pyramid.append(gwc_volume)
        # apc形状为[B,H,W,1,W]
        apc_volume = CombinedGeometryEncodingVolume.corr(left, right)
        B, H, W, _, W = apc_volume.shape
        # [B*H*W,1,1,W]
        apc_volume = apc_volume.view(B * H * W, 1, 1, W)

        self.apc_pyramid.append(apc_volume)
        # 构建多level金字塔 [B*H*W, 1, 1, W/n]
        for i in range(level - 1):
            apc_volume = F.avg_pool2d(apc_volume, [1, 2], [1, 2])
            self.apc_pyramid.append(apc_volume)

    @staticmethod
    def corr(left, right):
        B, C, H, W = left.shape
        # 全对代价体，[B,H,W,W]
        corr = torch.einsum('aijk,aijh->ajkh', left, right)
        corr = corr.contiguous().view(B, H, W, 1, W)

        return corr

    def __call__(self, disp):
        r = self.radius
        B, _, H, W = disp.shape
        disp = disp.view(B * H * W, 1, 1, 1)
        x0 = torch.arange(W).to(disp.device).view(1, 1, W, 1).repeat(B, H, 1, 1).contiguous().view(B * H * W, 1, 1, 1)

        pyramid = []
        for i in range(self.level):
            gwc_volume = self.gev_pyramid[i]
            dx = torch.linspace(-r, r, 2 * r + 1, device=disp.device)
            dx = dx.view(1, 1, 2 * r + 1, 1)
            # [B*H*W, 1, 2*r+1, 1]
            coord = dx + disp / (2 ** i)
            # [B*H*W, G, 1, 2*r+1]
            gwc_volume = corr_sampler(gwc_volume, coord)
            # [B,(2*r+1)*G,H,W]
            gwc_volume = gwc_volume.view(B, H, W, -1).permute(0, 3, 1, 2)

            apc_volume = self.apc_pyramid[i]
            # [B*H*W, 1, 2*r+1, 1]
            coord = dx + (x0 - disp) / (2 ** i)
            # [B*H*W, 1, 1, 2*r+1]
            apc_volume = corr_sampler(apc_volume, coord)
            # [B,2*r+1,H,W]
            apc_volume = apc_volume.view(B, H, W, -1).permute(0, 3, 1, 2)

            pyramid.append(gwc_volume)
            pyramid.append(apc_volume)

        return torch.cat(pyramid, dim=1)
    