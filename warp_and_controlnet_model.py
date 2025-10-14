# warp_net.py
import torch, os
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, Any, Union, List
from diffusers.models.controlnet_flux import FluxControlNetModel

class DeformableWarp(nn.Module):
    def __init__(self, in_channels=3, hidden=32):
        super().__init__()
        # Small CNN to predict flow field (dx, dy) in pixels
        self.flow_net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 2, 3, padding=1)  # dx, dy
        )
        self.half()

    @staticmethod
    def _make_base_grid(B, H, W, device, dtype):
        # normalized base grid in [-1,1]
        ys, xs = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device, dtype=dtype),
            torch.linspace(-1, 1, W, device=device, dtype=dtype),
            indexing="ij"
        )
        grid = torch.stack((xs, ys), dim=-1)         # (H,W,2)
        grid = grid.unsqueeze(0).expand(B, H, W, 2)  # (B,H,W,2)
        return grid
    def forward(self, x, src_img):
        """
        x: features to predict flow from (could be src_img or concatenated inputs), (B,C,H,W)
        src_img: image to warp, (B,Cs,H,W)
        returns: warped image, flow (in pixels)
        """
        B, _, H, W = src_img.shape
        flow_px = self.flow_net(x)  # (B,2,H,W), pixel units (dx, dy)
        # convert pixel flow -> normalized flow for grid_sample
        # grid_sample expects normalized coords; +dx_px corresponds to +2*dx/(W-1)
        flow_norm = torch.empty_like(flow_px)
        flow_norm[:, 0, :, :] = 2.0 * flow_px[:, 0, :, :] / max(W - 1, 1)  # x
        flow_norm[:, 1, :, :] = 2.0 * flow_px[:, 1, :, :] / max(H - 1, 1)  # y
        base_grid = self._make_base_grid(B, H, W, src_img.device, src_img.dtype)  # (B,H,W,2)
        grid = base_grid + flow_norm.permute(0, 2, 3, 1)  # (B,H,W,2)
        warped = F.grid_sample(
            src_img, grid, mode="bilinear",
            padding_mode="border", align_corners=True
        )
        return warped, flow_px

class WarpAndFluxControlNetModel(FluxControlNetModel):
    def __init__(
        self,
        patch_size: int = 1,
        in_channels: int = 64,
        num_layers: int = 19,
        num_single_layers: int = 38,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 4096,
        pooled_projection_dim: int = 768,
        guidance_embeds: bool = False,
        axes_dims_rope: List[int] = [16, 56, 56],
        num_mode: int = None,
        conditioning_embedding_channels: int = None,
        ):
            
            # 按照 diffusers 的初始化流程走父类构造
            super().__init__(
                patch_size = patch_size,
                in_channels = in_channels,
                num_layers = num_layers,
                num_single_layers = num_single_layers,
                attention_head_dim = attention_head_dim,
                num_attention_heads = num_attention_heads,
                joint_attention_dim = joint_attention_dim,
                pooled_projection_dim = pooled_projection_dim,
                guidance_embeds = guidance_embeds,
                axes_dims_rope = axes_dims_rope,
                num_mode = num_mode,
                conditioning_embedding_channels = conditioning_embedding_channels,
            )
            # warper 在 from_unet 里挂载；这里占位，确保属性存在
            self.warper: Optional[nn.Module] = DeformableWarp(in_channels=48, hidden=64)
    
    @staticmethod
    def static_method(xxx, yyy):
        WarpAndFluxControlNetModel

    @classmethod
    def class_method(xxx, yyy):
        pass

    @classmethod
    def from_unet(cls, unet: nn.Module, **kwargs) -> "WarpAndFluxControlNetModel":
        # 1) 先构建原生 ControlNetModel
        base = super().from_unet(unet, **kwargs)

        # # 2) 将实例“升级”为子类对象（保留所有已初始化权重/缓冲区）
        # base.__class__ = cls
        # 3) 挂载 warper：输入通道与 conditioning_channels 对齐
        cond_ch = getattr(base.config, "conditioning_channels", kwargs.get("conditioning_channels", 48))
        base.warper = DeformableWarp(in_channels=cond_ch, hidden=64)

        return base

    # # —— 关键：覆盖 forward，签名与父类保持一致（SDXL 管线会按这个接口调用）——
    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        controlnet_cond: torch.Tensor,
        conditioning_scale: float = 1.0,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guess_mode: bool = False,
        return_dict: bool = True,
        **kwargs,
    ):
        
        # 安全防御：确保 warper 已挂载
        if self.warper is None:
            # 若通过 from_pretrained 等其它构造路径来到这里，可在此处按配置临时创建
            cond_ch = getattr(self.config, "conditioning_channels", 48)
            self.warper = DeformableWarp(in_channels=cond_ch, hidden=64)

        # —— 核心：对 controlnet_cond 做自我形变（x=cond, src=cond）——
        warped, _ = self.warper(controlnet_cond, controlnet_cond)

        # —— 调用父类逻辑（使用调用形式以保留 hooks/AMP 等）——
        return FluxControlNetModel.forward(
            self,
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=warped,
            conditioning_scale=conditioning_scale,
            class_labels=class_labels,
            timestep_cond=timestep_cond,
            attention_mask=attention_mask,
            added_cond_kwargs=added_cond_kwargs,
            cross_attention_kwargs=cross_attention_kwargs,
            guess_mode=guess_mode,
            return_dict=return_dict,
            **kwargs,
        )
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        # 调用基类的 from_pretrained 方法加载 ControlNetModel
        model = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        
        # 加载 warper 权重
        warper_path = os.path.join(pretrained_model_name_or_path, "warper.pt")
        if os.path.exists(warper_path):
            model.warper = DeformableWarp(in_channels=model.config.conditioning_channels, hidden=64)
            model.warper.load_state_dict(torch.load(warper_path, map_location=torch.device('cuda')))
            print(f"Warper weights loaded from {warper_path}")
        else:
            print("Warper weights not found. Initializing new warper.")
            model.warper = DeformableWarp(in_channels=model.config.conditioning_channels, hidden=64)
        
        return model

    def save_pretrained(self, save_directory: str, *args, **kwargs):
        # 调用基类的 save_pretrained 方法保存 ControlNetModel
        super().save_pretrained(save_directory, *args, **kwargs)
        
        # 保存 warper 权重
        warper_save_path = os.path.join(save_directory, "warper.pt")
        torch.save(self.warper.state_dict(), warper_save_path)
        print(f"Warper weights saved to {warper_save_path}")
