import math

import torch
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch import nn

from .kan import KAN, KANLinear
from .EUGA import EUGA




class KANLayer(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
        shift_size=5,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features

        grid_size = 5
        spline_order = 3
        scale_noise = 0.1
        scale_base = 1.0
        scale_spline = 1.0
        base_activation = torch.nn.SiLU
        grid_eps = 0.02
        grid_range = [-1, 1]

        self.fc1 = KANLinear(
            in_features,
            hidden_features,
            grid_size=grid_size,
            spline_order=spline_order,
            scale_noise=scale_noise,
            scale_base=scale_base,
            scale_spline=scale_spline,
            base_activation=base_activation,
            grid_eps=grid_eps,
            grid_range=grid_range,
        )
        self.fc2 = KANLinear(
            hidden_features,
            out_features,
            grid_size=grid_size,
            spline_order=spline_order,
            scale_noise=scale_noise,
            scale_base=scale_base,
            scale_spline=scale_spline,
            base_activation=base_activation,
            grid_eps=grid_eps,
            grid_range=grid_range,
        )
        self.fc3 = KANLinear(
            hidden_features,
            out_features,
            grid_size=grid_size,
            spline_order=spline_order,
            scale_noise=scale_noise,
            scale_base=scale_base,
            scale_spline=scale_spline,
            base_activation=base_activation,
            grid_eps=grid_eps,
            grid_range=grid_range,
        )

        self.dwconv_1 = DW_bn_relu(hidden_features)
        self.dwconv_2 = DW_bn_relu(hidden_features)
        self.dwconv_3 = DW_bn_relu(hidden_features)

        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        # pdb.set_trace()
        B, N, C = x.shape

        x = self.fc1(x.reshape(B * N, C))
        x = x.reshape(B, N, C).contiguous()
        x = self.dwconv_1(x, H, W)
        x = self.fc2(x.reshape(B * N, C))
        x = x.reshape(B, N, C).contiguous()
        x = self.dwconv_2(x, H, W)
        x = self.fc3(x.reshape(B * N, C))
        x = x.reshape(B, N, C).contiguous()
        x = self.dwconv_3(x, H, W)

        return x


class KANBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        sr_ratio=1,
    ):
        super().__init__()

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.layer = KANLayer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.layer(self.norm2(x), H, W))

        return x

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class DW_bn_relu(nn.Module):
    def __init__(self, dim=768):
        super(DW_bn_relu, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.bn = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU()

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class OverlapPatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2),
        )
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class ConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvLayer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, input):
        return self.conv(input)


class D_ConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(D_ConvLayer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, input):
        return self.conv(input)

class Gate_control_fusion(nn.Module):
    def __init__(self, channels, m=-0.80):
        super(Gate_control_fusion, self).__init__()
        self.channels = channels
        self.w = nn.Parameter(torch.FloatTensor([m]),requires_grad=True)
        self.mix_block = nn.Sigmoid()
    def forward(self, fea1, fea2):
        W = self.mix_block(self.w)
        out = fea1 * W.expand_as(fea1) + fea2 * (1 - W.expand_as(fea2))
        return out

class MDB(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(MDB, self).__init__()
        self.up = nn.Conv2d(in_channels=in_channels,
                            out_channels=hidden_channels,
                            kernel_size=1)
        self.x1_MSC = nn.ConvTranspose2d(in_channels=hidden_channels,out_channels=hidden_channels,kernel_size=3,stride=2,padding=1,output_padding=1)
        self.x1_pool = nn.MaxPool2d(2)
        self.x1_norm = nn.BatchNorm2d(hidden_channels)

        self.x2_MSC1 = nn.ConvTranspose2d(in_channels=hidden_channels,out_channels=hidden_channels,kernel_size=3,stride=2,padding=1,output_padding=1)
        self.x2_pool1 = nn.MaxPool2d(2)
        self.x2_MSC2 = nn.ConvTranspose2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3,stride=2, padding=1, output_padding=1)
        self.x2_pool2 = nn.MaxPool2d(2)
        self.x2_norm = nn.BatchNorm2d(hidden_channels)

        self.x3_Conv1 = nn.Conv2d(in_channels=hidden_channels,out_channels=hidden_channels,kernel_size=3,dilation=2,padding=2)
        self.x3_norm = nn.BatchNorm2d(hidden_channels)

        self.x4_Conv1 = nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, dilation=2,
                                  padding=2)
        self.x4_norm1 = nn.BatchNorm2d(hidden_channels)
        self.x4_Conv2 = nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, dilation=3,
                                  padding=3)
        self.x4_norm2 = nn.BatchNorm2d(hidden_channels)


        self.final = nn.Conv2d(in_channels=hidden_channels,
                               out_channels=out_channels,
                               kernel_size=1)
        self.pool = nn.MaxPool2d(2)
        self.Gate_control_fusion = Gate_control_fusion(hidden_channels)

    def forward(self, x):
        x = self.up(x)
        x = self.pool(x)

        x1 = self.x1_MSC(x)
        x1 = self.x1_pool(x1)
        x1 = self.x1_norm(x1)

        x2 = self.x2_MSC1(x)
        x2 = self.x2_pool1(x2)
        x2 = self.x2_MSC2(x2)
        x2 = self.x2_pool2(x2)
        x2 = self.x2_norm(x2)

        x3 = self.x3_Conv1(x)
        x3 = self.x3_norm(x3)

        x4 = self.x4_Conv1(x)
        x4 = self.x4_norm1(x4)
        x4 = self.x4_Conv2(x4)
        x4 = self.x4_norm2(x4)
        feature = self.Gate_control_fusion(x1+x2,x3+x4)

        out = feature
        out = self.final(out)
        return out

class Evid_U_KAN(nn.Module):
    def __init__(
        self,
        num_classes,
        input_channels=3,
        img_size=224,
        embed_dims=[256, 320, 512],
        num_heads=[1, 2, 4, 8],
        mlp_ratios=[4, 4, 4, 4],
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        depths=[1, 1, 1],
        sr_ratios=[8, 4, 2, 1],
        **kwargs,
    ):
        super().__init__()

        kan_input_dim = embed_dims[0]
        self.num_classes = num_classes

        self.encoder1 = ConvLayer(input_channels, kan_input_dim // 4)
        self.encoder2 = ConvLayer(kan_input_dim // 4, kan_input_dim // 2)
        self.encoder3 = ConvLayer(kan_input_dim // 2, kan_input_dim)

        self.norm3 = norm_layer(embed_dims[1])
        self.norm4 = norm_layer(embed_dims[2])

        self.dnorm3 = norm_layer(embed_dims[1])
        self.dnorm4 = norm_layer(embed_dims[0])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.block1 = nn.ModuleList(
            [
                KANBlock(
                    dim=embed_dims[1],
                    num_heads=num_heads[0],
                    mlp_ratio=1,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[0],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[0],
                )
            ]
        )

        self.block2 = nn.ModuleList(
            [
                KANBlock(
                    dim=embed_dims[2],
                    num_heads=num_heads[0],
                    mlp_ratio=1,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[1],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[0],
                )
            ]
        )

        self.dblock1 = nn.ModuleList(
            [
                KANBlock(
                    dim=embed_dims[1],
                    num_heads=num_heads[0],
                    mlp_ratio=1,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[0],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[0],
                )
            ]
        )

        self.dblock2 = nn.ModuleList(
            [
                KANBlock(
                    dim=embed_dims[0],
                    num_heads=num_heads[0],
                    mlp_ratio=1,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[1],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[0],
                )
            ]
        )

        self.patch_embed3 = OverlapPatchEmbed(
            img_size=img_size // 4,
            patch_size=3,
            stride=2,
            in_chans=embed_dims[0],
            embed_dim=embed_dims[1],
        )
        self.patch_embed4 = OverlapPatchEmbed(
            img_size=img_size // 8,
            patch_size=3,
            stride=2,
            in_chans=embed_dims[1],
            embed_dim=embed_dims[2],
        )

        self.decoder1 = D_ConvLayer(embed_dims[2], embed_dims[1])
        self.decoder2 = D_ConvLayer(embed_dims[1], embed_dims[0])
        self.decoder3 = D_ConvLayer(embed_dims[0], embed_dims[0] // 2)
        self.decoder4 = D_ConvLayer(embed_dims[0] // 2, embed_dims[0] // 4)
        self.decoder5 = D_ConvLayer(embed_dims[0] // 4, embed_dims[0] // 4)

        self.final = nn.Conv2d(embed_dims[0] // 4, num_classes, kernel_size=1)
        self.soft = nn.Softmax(dim=1)
        self.MDB1 = MDB(in_channels=input_channels,hidden_channels=kan_input_dim // 8,out_channels=kan_input_dim // 4)
        self.MDB2 = MDB(in_channels=kan_input_dim // 4,hidden_channels=kan_input_dim // 4,out_channels=kan_input_dim // 2)
        self.MDB3 = MDB(in_channels=kan_input_dim // 2,hidden_channels=kan_input_dim // 2,out_channels=kan_input_dim)
        self.u_dict = {}
        self.u_fea = torch.Tensor([])
        self.EUGA = EUGA(kan_input_dim // 4)
        self.U_encoder1 = ConvLayer(1, kan_input_dim // 4)
        self.U_encoder2 = ConvLayer(kan_input_dim // 4, kan_input_dim // 2)
        self.U_encoder3 = ConvLayer(kan_input_dim // 2, kan_input_dim)

    def clear(self):
        self.u_dict.clear()

    def forward(self, x):
        inputs = x
        B, C, H, W = x.shape

        self.u_fea = torch.Tensor([])
        for i in range(inputs.size(0)):
            k = inputs[i].unsqueeze(0)
            k = tuple(k.reshape(-1).tolist())
            if k in self.u_dict:
                uncertainty = torch.Tensor(self.u_dict[k]).reshape([1, -1, H, W])
            else:
                uncertainty = torch.ones([1, 1, H, W])
            self.u_fea = torch.cat((self.u_fea, uncertainty), dim=0)
        self.u_fea = self.u_fea.cuda()
        ## Stage 1
        b1 = self.MDB1(x)
        out = F.relu(F.max_pool2d(self.encoder1(x), 2, 2))
        t1 = out
        t1 = self.EUGA(t1, self.u_fea)
        out = torch.add(out,b1)

        ### Stage 2
        b2 = self.MDB2(b1)
        out = F.relu(F.max_pool2d(self.encoder2(out), 2, 2))
        t2 = out
        out = torch.add(out,b2)

        ### Stage 3
        b3 = self.MDB3(b2)
        out = F.relu(F.max_pool2d(self.encoder3(out), 2, 2))
        t3 = out
        out = torch.add(out,b3)

        ### Tokenized KAN Stage
        ### Stage 4
        out, H, W = self.patch_embed3(out)
        for i, blk in enumerate(self.block1):
            out = blk(out, H, W)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t4 = out

        ### Bottleneck
        out, H, W = self.patch_embed4(out)
        for i, blk in enumerate(self.block2):
            out = blk(out, H, W)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        ### Stage 4
        out = F.relu(
            F.interpolate(self.decoder1(out), scale_factor=(2, 2), mode="bilinear")
        )

        out = torch.add(out, t4)
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.dblock1):
            out = blk(out, H, W)

        ### Stage 3
        out = self.dnorm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = F.relu(
            F.interpolate(self.decoder2(out), scale_factor=(2, 2), mode="bilinear")
        )
        out = torch.add(out, t3)
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.dblock2):
            out = blk(out, H, W)

        out = self.dnorm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        out = F.relu(
            F.interpolate(self.decoder3(out), scale_factor=(2, 2), mode="bilinear")
        )
        out = torch.add(out, t2)
        out = F.relu(
            F.interpolate(self.decoder4(out), scale_factor=(2, 2), mode="bilinear")
        )
        out = torch.add(out, t1)
        out = F.relu(
            F.interpolate(self.decoder5(out), scale_factor=(2, 2), mode="bilinear")
        )
        out = self.final(out)
        final_out = torch.relu(out)
        final_out = torch.exp(-final_out) + final_out - 1
        for i in range(inputs.size(0)):
            k = inputs[i].unsqueeze(0)
            k = tuple(k.reshape(-1).tolist())
            out = final_out[i].unsqueeze(0)
            alpha = out + 1
            S = torch.sum(alpha, dim=1)
            u = 2 / S
            v = tuple(u.reshape(-1).tolist())
            if k in self.u_dict:
                self.u_dict[k] = v
            else:
                self.u_dict[k] = v

        return final_out
