"""
参考图编码器 - 用于A1+阶段提取参考图特征

输入: 参考图 R [B, 3, H, W]
输出: 特征序列 [B, M, C]，其中M=h*w（下采样后的token数）
"""
import torch
import torch.nn as nn


class ReferenceEncoder(nn.Module):
    """
    轻量级参考图编码器
    
    架构: 4层CNN下采样（256→16），输出256维特征
    用于提取参考图的颜色语义特征，通过Cross-Attention注入UNet
    """
    
    def __init__(self, in_channels=3, feature_dim=256):
        super().__init__()
        self.feature_dim = feature_dim
        
        self.encoder = nn.Sequential(
            # 256 -> 128
            nn.Conv2d(in_channels, 64, 4, 2, 1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            
            # 128 -> 64
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.GroupNorm(16, 128),
            nn.SiLU(),
            
            # 64 -> 32
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.GroupNorm(32, 256),
            nn.SiLU(),
            
            # 32 -> 16
            nn.Conv2d(256, feature_dim, 4, 2, 1),
            nn.GroupNorm(32, feature_dim),
            nn.SiLU(),
        )
        
        # 可学习的"无条件"嵌入 (用于Classifier-Free Guidance训练)
        # 在特征空间中学习一个语义中性的点，表示"无参考图"状态
        # 形状: [1, num_tokens, feature_dim]，其中num_tokens = 16*16 = 256
        self.null_embed = nn.Parameter(torch.randn(1, 256, feature_dim) * 0.01)
        
    def forward(self, ref_image):
        """
        Args:
            ref_image: [B, 3, H, W] 参考图（H=W=256）
        Returns:
            context: [B, M, feature_dim] 特征序列（M=16*16=256）
        """
        features = self.encoder(ref_image)  # [B, feature_dim, 16, 16]
        B, C, H, W = features.shape
        # 展平为序列格式，符合Cross-Attention输入要求
        context = features.view(B, C, H * W).permute(0, 2, 1)  # [B, M, C]
        return context
    
    def get_null_context(self, batch_size):
        """
        返回用于无条件生成的null embedding
        
        Args:
            batch_size: 批次大小
        Returns:
            null_context: [B, M, feature_dim] 无条件嵌入
        """
        return self.null_embed.expand(batch_size, -1, -1)
