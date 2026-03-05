import torch
import torch.nn as nn
from typing import Optional, Tuple, Union


class Permutation(nn.Module):
    """Base permutation operating along a sequence dimension."""

    def __init__(self, seq_length: int) -> None:
        super().__init__()
        self.seq_length = seq_length

    def forward(self, x: torch.Tensor, dim: int = 1, inverse: bool = False) -> torch.Tensor:  # pragma: no cover - abstract
        raise NotImplementedError


class PermutationIdentity(Permutation):
    def forward(self, x: torch.Tensor, dim: int = 1, inverse: bool = False) -> torch.Tensor:
        return x


class PermutationFlip(Permutation):
    def forward(self, x: torch.Tensor, dim: int = 1, inverse: bool = False) -> torch.Tensor:
        return x.flip(dims=[dim])


class _AttentionBlock1D(nn.Module):
    """Simple Transformer-style block for 1D sequences."""

    def __init__(self, channels: int, num_heads: int, expansion: int = 4) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels * expansion),
            nn.GELU(),
            nn.Linear(channels * expansion, channels),
        )

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=attn_mask)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class MetaBlock(nn.Module):
    """TarFlow-style autoregressive block for 1D sequences.

    This is adapted to work on (B, T, C) sequences instead of image patches.
    """
    attn_mask: torch.Tensor
    
    def __init__(
        self,
        in_channels: int,
        channels: int,
        num_tokens: int,
        permutation: Permutation,
        num_layers: int = 1,
        head_dim: int = 64,
        expansion: int = 4,
        nvp: bool = True,
        num_classes: int = 0,
    ) -> None:
        super().__init__()
        self.proj_in = nn.Linear(in_channels, channels)
        self.pos_embed = nn.Parameter(torch.randn(num_tokens, channels) * 1e-2)
        self.class_embed = (
            nn.Parameter(torch.randn(num_classes, 1, channels) * 1e-2) if num_classes > 0 else None
        )

        num_heads = max(1, channels // head_dim)
        self.blocks = nn.ModuleList(
            [_AttentionBlock1D(channels, num_heads, expansion) for _ in range(num_layers)]
        )

        self.nvp = nvp
        out_dim = in_channels * 2 if nvp else in_channels
        self.proj_out = nn.Linear(channels, out_dim)
        # start near identity
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

        self.permutation = permutation
        # causal mask: forbid attending to future positions
        mask = torch.triu(torch.ones(num_tokens, num_tokens), diagonal=1)
        # MultiheadAttention expects float mask added to logits
        self.register_buffer("attn_mask", mask.masked_fill(mask == 1, float("-inf")))

    def forward(self, 
                traj_points, 
                bev_feature, 
                bev_spatial_shape, 
                agents_query, 
                ego_query, 
                status_encoding,
                global_img):
        
        x = self.permutation(x) #第一层好像是不变的，见论文2.2. Block Autoregressive Flows 中permutations的部分
        pos_embed = self.permutation(self.pos_embed, dim=0) # 为啥每一层要反向？
        x_in = x
        x = self.proj_in(x) + pos_embed
        if self.class_embed is not None:
            if y is not None:
                if (y < 0).any():
                    m = (y < 0).float().view(-1, 1, 1) #小于0的地方为1，其它为0
                    class_embed = (1 - m) * self.class_embed[y] + m * self.class_embed.mean(dim=0) #处理drop label的情况,这里对应Guidance
                else:
                    class_embed = self.class_embed[y]
                x = x + class_embed
            else:
                x = x + self.class_embed.mean(dim=0)
        # self.attn_mask2=torch.tril(self.attn_mask, diagonal=-1)
        # o_x = x.clone()
        # for block in self.attn_blocks:
        #     o_x = block(o_x, self.attn_mask2) 
        # o_x = self.proj_out(o_x)  #感觉可以这样改，但是忽略了残差连接，所以还是有问题哈
        for block in self.attn_blocks:
            x = block(x, self.attn_mask) 
        x = self.proj_out(x) #这里理一下思路，这里输出xtorch.Size([4, 1024, 384])
        x = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1) 
        # 上面这里再剖析一下，这里作用和attention_mask应该是不一样得。但是后面chunk了，移位感觉没用吧
        if self.nvp:
            xa, xb = x.chunk(2, dim=-1) #输入就是192维度的，输出变成384维度的，然后一分为二变回原来的通道1
        else:
            xb = x
            xa = torch.zeros_like(x)

        scale = (-xa.float()).exp().type(xa.dtype) #为啥取复数，函数怎么来的？
        return self.permutation((x_in - xb) * scale, inverse=True), -xa.mean(dim=[1, 2])
    
    def reverse_step(
        self,
        x: torch.Tensor,
        pos_embed: torch.Tensor,
        i: int,
        y: Union[torch.Tensor, None] = None,
        attn_temp: float = 1.0,
        which_cache: str = 'cond',
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_in = x[:, i : i + 1]  # get i-th patch but keep the sequence dimension （<i的逆向操作,i=0时进去的只有i=1）
        x = self.proj_in(x_in) + pos_embed[i : i + 1] #x是上一步生成的结果
        if self.class_embed is not None:
            if y is not None:
                x = x + self.class_embed[y]
            else:
                x = x + self.class_embed.mean(dim=0)

        for block in self.attn_blocks:
            x = block(x, attn_temp=attn_temp, which_cache=which_cache)  # here we use kv caching, so no attn_mask
        x = self.proj_out(x)

        if self.nvp:
            xa, xb = x.chunk(2, dim=-1)
        else:
            xb = x
            xa = torch.zeros_like(x)
        return xa, xb

    def set_sample_mode(self, flag: bool = True):
        for m in self.modules():
            if isinstance(m, Attention):
                m.sample = flag
                m.k_cache = {'cond': [], 'uncond': []}
                m.v_cache = {'cond': [], 'uncond': []}

    def reverse(
        self,
        x: torch.Tensor,
        y: Union[torch.Tensor, None] = None,
        guidance: float = 0,
        guide_what: str = 'ab',
        attn_temp: float = 1.0,
        annealed_guidance: bool = False,
    ) -> torch.Tensor:
        x = self.permutation(x)
        pos_embed = self.permutation(self.pos_embed, dim=0)
        self.set_sample_mode(True)
        T = x.size(1) 
        for i in range(x.size(1) - 1): #自回归采样，只能一个个计算每个batch内的(因为要用到每一步新生成的x),(只能预测i-1个，第0个轮流坐庄)
            za, zb = self.reverse_step(x, pos_embed, i, y, which_cache='cond')
            if guidance > 0 and guide_what: #guidance机制，对应论文中2.6. Guidance
                za_u, zb_u = self.reverse_step(x, pos_embed, i, None, attn_temp=attn_temp, which_cache='uncond')
                if annealed_guidance:
                    g = (i + 1) / (T - 1) * guidance
                else:
                    g = guidance
                if 'a' in guide_what:
                    za = za + g * (za - za_u)
                if 'b' in guide_what:
                    zb = zb + g * (zb - zb_u)

            scale = za[:, 0].float().exp().type(za.dtype)  # get rid of the sequence dimension
            x[:, i + 1] = x[:, i + 1] * scale + zb[:, 0] #为x[:, i + 1]赋值(应为这次用的是x<(i+1)生成的)
        self.set_sample_mode(False)
        return self.permutation(x, inverse=True)


class SequenceFlowModel(nn.Module):
    """TarFlow-style normalizing flow over 1D sequences.

    This mirrors the original Model/MetaBlock structure but operates on
    sequences x of shape (B, T, C) without patchify/unpatchify.
    """

    VAR_LR: float = 0.1
    var: torch.Tensor

    def __init__(
        self,
        in_channels: int,
        seq_len: int,
        channels: int,
        num_blocks: int,
        layers_per_block: int,
        nvp: bool = True,
        num_classes: int = 0,
        use_mean_det: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.seq_len = seq_len

        permutations = [PermutationIdentity(seq_len), PermutationFlip(seq_len)]

        blocks = []
        for i in range(num_blocks):
            blocks.append(
                TarDriveMetaBlock(
                    in_channels=in_channels,
                    channels=channels,
                    num_tokens=seq_len,
                    permutation=permutations[i % 2],
                    num_layers=layers_per_block,
                    head_dim=64,
                    expansion=4,
                    nvp=nvp,
                    num_classes=num_classes,
                    use_mean_det=use_mean_det,
                )
            )
        self.blocks = nn.ModuleList(blocks)

        # prior variance buffer (kept for parity with TarFlow, though not
        # strictly needed for training when using standard normal)
        self.register_buffer("var", torch.ones(seq_len, in_channels))

    def forward(self, 
                traj_feature,
                noisy_traj_points, 
                ego_query, 
                agents_query, 
                bev_feature, 
                bev_spatial_shape, 
                status_encoding, 
                global_img=None,
                best_mode=None):
        """Forward pass through all blocks.

        Args:
            x: (B, T, C) input sequence.
            y: optional class labels for conditioning.
        Returns:
            z: final latent sequence (B, T, C).
            outputs: list of intermediate sequences after each block.
            logdets: summed log-determinant per sample (B,).
        """
        # 统一 dtype 到 flow 参数的 dtype，避免 AMP 下 Half/Double 冲突
        dtype = self.blocks[0].proj_in.weight.dtype

        if traj_feature is not None:
            traj_feature = traj_feature.to(dtype)
        noisy_traj_points = noisy_traj_points.to(dtype)
        if bev_feature is not None:
            bev_feature = bev_feature.to(dtype)
        if agents_query is not None:
            agents_query = agents_query.to(dtype)
        if ego_query is not None:
            ego_query = ego_query.to(dtype)
        if status_encoding is not None:
            status_encoding = status_encoding.to(dtype)
        if global_img is not None:
            global_img = global_img.to(dtype)

        # noisy_traj_points: (B, M, T, C_in)
        poses_reg_list = []  # 每个 block 输出的 (B, M, T, C_in)
        poses_cls_list = []  # 每个 block 的 (B, M)
        
        traj_points = noisy_traj_points

        B = noisy_traj_points.size(0)
        M = noisy_traj_points.size(1)

        logdets = torch.zeros([B, M], device=noisy_traj_points.device, dtype=noisy_traj_points.dtype)

        # z, outputs, logdets, poses_cls_list
        for block in self.blocks:
            poses_reg, logdet, poses_cls = block(
                traj_feature,
                traj_points,
                bev_feature,
                bev_spatial_shape,
                agents_query,
                ego_query,
                status_encoding,
                global_img,
                best_mode,
            )
            logdets = logdets + logdet
            poses_reg_list.append(poses_reg)
            poses_cls_list.append(poses_cls)
            # 下一层以当前 block 输出作为输入（包含 x, y, heading）
            traj_points = poses_reg.clone().detach()

        # 返回最后一层的 z 以及中间结果，方便上层既算 flow loss 又算多模监督
        z = poses_reg_list[-1] # 注意这个z是归一化的z，用不了的
        return z, poses_reg_list, logdets, poses_cls_list

    def reverse(
        self,
        traj_feature: torch.Tensor,                # (B, M, D_traj) 轨迹特征
        z: torch.Tensor,                           # (B, M, T, C_in) latent, e.g. standard normal
        ego_query: torch.Tensor,
        bev_feature: torch.Tensor,
        bev_spatial_shape,
        agents_query: torch.Tensor,
        status_encoding: torch.Tensor,
        global_img: Optional[torch.Tensor] = None,
        best_mode: Optional[torch.Tensor] = None,
    ) -> Tuple[list[torch.Tensor], list[torch.Tensor]]:
        """TarFlow-style reverse: 从 latent z 生成轨迹序列.

        这里仿照 ml-tarflow/transformer_flow.py 中 Model.reverse 的结构：
        - 先用当前估计的 prior 方差 self.var 做缩放
        - 再按 block 逆序依次调用每个 block 的变换

        注意：这里使用 TarDriveMetaBlock 的 forward 作为生成步骤，
        因此是一个近似的采样过程，而不是严格的数学反变换。
        """

        dtype = self.blocks[0].proj_in.weight.dtype
        device = z.device

        # 对 latent 应用当前估计的方差缩放，形状 (T, C_in) -> (1,1,T,C_in)
        var = self.var.to(device=device, dtype=dtype)          # (T, C_in)
        z = z.to(device=device, dtype=dtype)
        z = z * var.sqrt().view(1, 1, self.seq_len, self.in_channels)

        # 条件也统一到同一 dtype
        if bev_feature is not None:
            bev_feature = bev_feature.to(device=device, dtype=dtype)
        if agents_query is not None:
            agents_query = agents_query.to(device=device, dtype=dtype)
        if ego_query is not None:
            ego_query = ego_query.to(device=device, dtype=dtype)
        if status_encoding is not None:
            status_encoding = status_encoding.to(device=device, dtype=dtype)
        if global_img is not None:
            global_img = global_img.to(device=device, dtype=dtype)

        poses_reg_list: list[torch.Tensor] = []
        poses_cls_list: list[torch.Tensor] = []

        x = z  # 当前序列 (B, M, T, C_in)

        # 按照 TarFlow 的做法，block 逆序迭代
        # 这里使用 TarDriveMetaBlock 的 forward 作为生成 step，
        # 不再累积 logdet（采样时用不到）。
        for block in reversed(self.blocks):
            x, _, poses_cls = block.reverse(
                traj_feature=traj_feature,          # 采样时不使用 traj_feature，可按需扩展
                traj_points=x,
                bev_feature=bev_feature,
                bev_spatial_shape=bev_spatial_shape,
                agents_query=agents_query,
                ego_query=ego_query,
                status_encoding=status_encoding,
                global_img=global_img,
            )
            poses_reg_list.append(x)
            poses_cls_list.append(poses_cls)

        return poses_reg_list, _,  poses_cls_list

    def update_prior(self, z: torch.Tensor):
        z2 = (z**2).mean(dim=0)
        self.var.lerp_(z2.detach(), weight=self.VAR_LR)
        # self.var = (1 - VAR_LR) * self.var + VAR_LR * z2_detached, 目前的VAR

    def get_loss(self, z: torch.Tensor, logdets: torch.Tensor):
        return 0.5 * z.pow(2).mean() - logdets.mean() #最小化这个函数即可,这个loss太抽象了


class TarDriveMetaBlock(nn.Module):
    """TarFlow-style MetaBlock 带 TarDrive 条件输入.

    输入:
      - traj_points:   (B, T, C_in)  序列 (例如 20×8 or 8 个轨迹点的 (x, y))
      - bev_feature:   BEV 特征图 (B, C_bev, H, W)
      - bev_spatial_shape: (H, W)  与 bev_feature 对应
      - agents_query:  (B, N_agents, D)
      - ego_query:     (B, 1, D)
      - status_encoding: (B, 1, D)
      - global_img:    (可选) 图像特征

    输出:
      - z:       (B, T, C_in)  仿射后的序列
      - logdet:  (B,)          每个样本的 log|det J|
      - poses_cls: (B, T)     序列上每个 token 的打分 (可按需要 reshape/汇聚)
    """
    def __init__(
        self,
        in_channels: int,
        channels: int,
        num_tokens: int,
        permutation: Permutation,
        num_layers: int = 1,
        head_dim: int = 64,
        expansion: int = 4,
        nvp: bool = True,
        num_classes: int = 0,
        use_mean_det: bool = False,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.channels = channels
        self.num_tokens = num_tokens
        self.permutation = permutation
        self.nvp = nvp
        self.use_mean_det = use_mean_det

        # 基础投影 + 位置编码
        self.proj_in = nn.Linear(in_channels, channels)
        self.pos_embed = nn.Parameter(torch.randn(num_tokens, channels) * 1e-2)

        # 自注意力 + MLP（因果 mask）
        num_heads = max(1, channels // head_dim)
        self.self_blocks = nn.ModuleList(
            [_AttentionBlock1D(channels, num_heads, expansion) for _ in range(num_layers)]
        )

        # 条件 cross-attention：BEV / agents / ego / status
        self.cross_bev = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.cross_agents = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.cross_ego = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.cross_status = nn.MultiheadAttention(channels, num_heads, batch_first=True)

        self.norm_bev = nn.LayerNorm(channels)
        self.norm_agents = nn.LayerNorm(channels)
        self.norm_ego = nn.LayerNorm(channels)
        self.norm_status = nn.LayerNorm(channels)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * expansion),
            nn.ReLU(),
            nn.Linear(channels * expansion, channels),
        )
        self.norm_ffn = nn.LayerNorm(channels)

        # NVP 仿射头
        out_dim = in_channels * 2 if nvp else in_channels
        self.proj_out = nn.Linear(channels, out_dim)
        self.proj_out.weight.data.fill_(0.0)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

        # 简单的分类头 (沿时间维度的打分)
        self.cls_head = nn.Linear(channels, 1)

        # 自回归 mask: 禁止关注未来位置
        mask = torch.triu(torch.ones(num_tokens, num_tokens), diagonal=1)
        self.register_buffer("attn_mask", mask.masked_fill(mask == 1, float("-inf")))

    def forward(
        self,
        traj_feature: torch.Tensor,  # (B, M, D_model)
        traj_points: torch.Tensor,   # (B, M, T, C_in)
        bev_feature: torch.Tensor,
        bev_spatial_shape,
        agents_query: torch.Tensor,
        ego_query: torch.Tensor,
        status_encoding: torch.Tensor,
        global_img: torch.Tensor = None,
        best_mode: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 展平多模维度, 在时间维度 T 上做自回归 flow
        B, M, T, C_in = traj_points.shape
        assert T == self.num_tokens and C_in == self.in_channels

        x = traj_points.view(B * M, T, C_in)  # (B*M, T, C_in)
        x = self.permutation(x, dim=1)
        x_in = x

        h = self.proj_in(x) + self.pos_embed[None, :, :]  # (B*M, T, C)

        # 1.5) 注入来自 decoder 的 per-mode traj_feature
        if traj_feature is not None:
            assert traj_feature.shape[0] == B and traj_feature.shape[1] == M
            traj_feat = traj_feature.view(B * M, 1, -1).expand(B * M, T, -1)
            assert traj_feat.size(-1) == self.channels
            h = h + traj_feat

        # 2) 自注意力 (带因果 mask)
        for blk in self.self_blocks:
            h = blk(h, attn_mask=self.attn_mask)

        # 3) 条件 cross-attention
        if bev_feature is not None:
            # bev_feature: (B, C_bev, H, W) -> (B*M, H*W, C_bev)
            B0, C_bev, H, W = bev_feature.shape
            assert B0 == B
            bev_tokens = bev_feature.flatten(2, 3).permute(0, 2, 1).contiguous()  # (B, H*W, C_bev)
            bev_tokens = bev_tokens.unsqueeze(1).expand(B, M, H * W, C_bev).reshape(B * M, H * W, C_bev)
            attn_out, _ = self.cross_bev(h, bev_tokens, bev_tokens)
            h = self.norm_bev(h + attn_out)

        if agents_query is not None:
            # agents_query: (B, N_agents, D) -> (B*M, N_agents, D)
            B0, N_agents, D = agents_query.shape
            assert B0 == B
            agents_tokens = agents_query.unsqueeze(1).expand(B, M, N_agents, D).reshape(B * M, N_agents, D)
            attn_out, _ = self.cross_agents(h, agents_tokens, agents_tokens)
            h = self.norm_agents(h + attn_out)

        if ego_query is not None:
            # ego_query: (B, 1, D) -> (B*M, 1, D)
            B0, _, D = ego_query.shape
            assert B0 == B
            ego_tokens = ego_query.unsqueeze(1).expand(B, M, 1, D).reshape(B * M, 1, D)
            attn_out, _ = self.cross_ego(h, ego_tokens, ego_tokens)
            h = self.norm_ego(h + attn_out)

        if status_encoding is not None:
            # status_encoding: (B, 1, D) -> (B*M, 1, D)
            B0, _, D = status_encoding.shape
            assert B0 == B
            status_tokens = status_encoding.unsqueeze(1).expand(B, M, 1, D).reshape(B * M, 1, D)
            attn_out, _ = self.cross_status(h, status_tokens, status_tokens)
            h = self.norm_status(h + attn_out)

        # 4) FFN
        h = self.norm_ffn(h + self.ffn(h))

        # 5) NVP 仿射输出 (自回归: t 位置只依赖 < t)
        out = self.proj_out(h)  # (B*M, T, 2*C_in)
        out = torch.cat([torch.zeros_like(out[:, :1]), out[:, :-1]], dim=1)

        if self.nvp:
            xa, xb = out.chunk(2, dim=-1)  # (B*M, T, C_in), (B*M, T, C_in)
        else:
            xb = out
            xa = torch.zeros_like(xb)

        scale = (-xa.float()).exp().type(xa.dtype)
        z = (x_in - xb) * scale  # (B*M, T, C_in)
        z = self.permutation(z, dim=1, inverse=True)

        # logdet: 单样本 log|det J|, 先在 (T, C_in) 上平均, 再在 M 上平均到 (B,),这个看要不要用
        logdet = -xa.mean(dim=[1, 2]) # 保留BM
        logdet = logdet.view(B, M)# (B,)

        # 每个 mode 的评分 (B, M)
        h_pool = h.mean(dim=1)                      # (B*M, C)
        poses_cls_flat = self.cls_head(h_pool).squeeze(-1)  # (B*M,)
        poses_cls = poses_cls_flat.view(B, M)

        z = z.view(B, M, T, C_in)
        return z, logdet, poses_cls
    
    def reverse(
        self,
        traj_feature: torch.Tensor,
        traj_points: torch.Tensor,
        bev_feature: torch.Tensor,
        bev_spatial_shape,
        agents_query: torch.Tensor,
        ego_query: torch.Tensor,
        status_encoding: torch.Tensor,
        global_img: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """TarFlow 风格的自回归反向采样。

        这里仿照 transformer_flow.py 中 MetaBlock.reverse 的结构：
        - 先在序列维度上做 permutation
        - 然后按 token 顺序逐步更新 x[:, i+1]
        - 最后再 inverse permutation 回到原始顺序

        注意：由于当前实现没有专门的 Attention KV cache，reverse_step
        在每个 i 上都会重新跑一遍 prefix 的 Transformer。T 比较小
        （轨迹长度通常只有几十），因此在测试阶段可以接受。
        """

        B, M, T, C_in = traj_points.shape
        assert T == self.num_tokens and C_in == self.in_channels

        # (B*M, T, C_in)，先做 permutation，对齐 forward 里的顺序
        x = traj_points.view(B * M, T, C_in)
        x = self.permutation(x, dim=1)

        # 位置编码同样做 permutation
        pos_embed = self.permutation(self.pos_embed, dim=0)  # (T, C)# 这个不确定

        BM = B * M

        # 展平 traj_feature 方便在时间维度上 broadcast
        traj_feat_flat = None
        if traj_feature is not None:
            assert traj_feature.shape[0] == B and traj_feature.shape[1] == M
            traj_feat_flat = traj_feature.view(BM, -1)  # (B*M, C)

        # 预先构造所有条件 token，避免在每一步重复计算
        bev_tokens = None
        if bev_feature is not None:
            B0, C_bev, H, W = bev_feature.shape
            assert B0 == B
            bev_tokens = bev_feature.flatten(2, 3).permute(0, 2, 1).contiguous()  # (B, H*W, C_bev)
            bev_tokens = bev_tokens.unsqueeze(1).expand(B, M, H * W, C_bev).reshape(BM, H * W, C_bev)

        agents_tokens = None
        if agents_query is not None:
            B0, N_agents, D = agents_query.shape
            assert B0 == B
            agents_tokens = agents_query.unsqueeze(1).expand(B, M, N_agents, D).reshape(BM, N_agents, D)

        ego_tokens = None
        if ego_query is not None:
            B0, _, D = ego_query.shape
            assert B0 == B
            ego_tokens = ego_query.unsqueeze(1).expand(B, M, 1, D).reshape(BM, 1, D)

        status_tokens = None
        if status_encoding is not None:
            B0, _, D = status_encoding.shape
            assert B0 == B
            status_tokens = status_encoding.unsqueeze(1).expand(B, M, 1, D).reshape(BM, 1, D)

        # 自回归地生成：逐步更新 x[:, i+1]
        for i in range(T - 1):
            za, zb = self.reverse_step(
                x=x,
                traj_feat_flat=traj_feat_flat,
                bev_tokens=bev_tokens,
                agents_tokens=agents_tokens,
                ego_tokens=ego_tokens,
                status_tokens=status_tokens,
                pos_embed=pos_embed,
                i=i,
            )

            # za / zb: (B*M, 1, C_in)
            scale = za[:, 0].float().exp().type(x.dtype)  # (B*M, C_in)
            x[:, i + 1] = x[:, i + 1] * scale + zb[:, 0]

        # 还原 permutation，并 reshape 回 (B, M, T, C_in)
        x = self.permutation(x, dim=1, inverse=True)
        traj_points_sample = x.view(B, M, T, C_in)

        # 采样阶段 logdet 不参与损失，直接置零即可
        logdet = torch.zeros(B, M, dtype=traj_points_sample.dtype, device=traj_points_sample.device)

        # 为了得到每个 mode 的分类分数，这里用当前采样的轨迹
        # 再跑一次轻量的 forward 头部（不求梯度）。
        with torch.no_grad():
            x_cls = traj_points_sample.view(B * M, T, C_in)
            x_cls = self.permutation(x_cls, dim=1)
            h = self.proj_in(x_cls) + self.pos_embed[None, :, :]

            if traj_feature is not None:
                traj_feat = traj_feature.view(BM, 1, -1).expand(BM, T, -1)
                h = h + traj_feat

            for blk in self.self_blocks:
                h = blk(h, attn_mask=self.attn_mask)

            if bev_tokens is not None:
                attn_out, _ = self.cross_bev(h, bev_tokens, bev_tokens)
                h = self.norm_bev(h + attn_out)

            if agents_tokens is not None:
                attn_out, _ = self.cross_agents(h, agents_tokens, agents_tokens)
                h = self.norm_agents(h + attn_out)

            if ego_tokens is not None:
                attn_out, _ = self.cross_ego(h, ego_tokens, ego_tokens)
                h = self.norm_ego(h + attn_out)

            if status_tokens is not None:
                attn_out, _ = self.cross_status(h, status_tokens, status_tokens)
                h = self.norm_status(h + attn_out)

            h = self.norm_ffn(h + self.ffn(h))
            h_pool = h.mean(dim=1)
            poses_cls_flat = self.cls_head(h_pool).squeeze(-1)
            poses_cls = poses_cls_flat.view(B, M)

        return traj_points_sample, logdet, poses_cls

    def reverse_step(
        self,
        x: torch.Tensor,
        traj_feat_flat: Optional[torch.Tensor],
        bev_tokens: Optional[torch.Tensor],
        agents_tokens: Optional[torch.Tensor],
        ego_tokens: Optional[torch.Tensor],
        status_tokens: Optional[torch.Tensor],
        pos_embed: torch.Tensor,
        i: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """单步自回归更新所需的 (za, zb)。

        x:         (B*M, T, C_in) 当前采样序列（已做 permutation）
        pos_embed: (T, C)        与 forward 对齐的 positional embedding
        i:         当前 step 索引，使用前 i+1 个 token 作为条件，
                   产生第 i+1 个 token 的仿射参数。
        """

        BM, T, C_in = x.shape

        # 只取前 i+1 个 token 作为上下文
        x_prefix = x[:, : i + 1, :]  # (B*M, i+1, C_in)

        h = self.proj_in(x_prefix) + pos_embed[None, : i + 1, :]

        if traj_feat_flat is not None:
            traj_cond = traj_feat_flat.view(BM, 1, -1).expand(BM, i + 1, -1)
            h = h + traj_cond

        # 自注意力，使用截断过的因果 mask
        attn_mask = self.attn_mask[: i + 1, : i + 1]
        for blk in self.self_blocks:
            h = blk(h, attn_mask=attn_mask)

        # 条件 cross-attention
        if bev_tokens is not None:
            attn_out, _ = self.cross_bev(h, bev_tokens, bev_tokens)
            h = self.norm_bev(h + attn_out)

        if agents_tokens is not None:
            attn_out, _ = self.cross_agents(h, agents_tokens, agents_tokens)
            h = self.norm_agents(h + attn_out)

        if ego_tokens is not None:
            attn_out, _ = self.cross_ego(h, ego_tokens, ego_tokens)
            h = self.norm_ego(h + attn_out)

        if status_tokens is not None:
            attn_out, _ = self.cross_status(h, status_tokens, status_tokens)
            h = self.norm_status(h + attn_out)

        h = self.norm_ffn(h + self.ffn(h))

        # 仿射头只取最后一个 token 的输出，对应 forward 里移位前的 out[i]
        out = self.proj_out(h)              # (B*M, i+1, 2*C_in or C_in)
        out_last = out[:, -1:, :]           # (B*M, 1, 2*C_in or C_in)

        if self.nvp:
            xa_raw, xb = out_last.chunk(2, dim=-1)
        else:
            xb = out_last
            xa_raw = torch.zeros_like(xb)

        # forward 中 scale = exp(-xa)，这里为了得到 x_{i+1} = z * exp(a_i) + b_i，
        # 令 za = -xa_raw，这样 scale = exp(za) 即可。
        za = -xa_raw
        zb = xb

        return za, zb

    # 麻烦了，不能用nn.transformers里现成的Attention模块了，得自己写个子类来加个sample模式标志位和缓存机制
    def set_sample_mode(self, flag: bool = True):
        for m in self.modules():
            if isinstance(m, _AttentionBlock1D):
                m.sample = flag
                m.k_cache = {'cond': [], 'uncond': []}
                m.v_cache = {'cond': [], 'uncond': []}


if __name__ == "__main__":
        """简单的 SequenceFlowModel 前向调试脚本。

        对齐你在 VSCode launch.json 里给的 TarDrive 训练配置：
            - batch_size = 2
            - ego_fut_mode = 20 (mode 数)
            - num_poses = 8  (每条轨迹点数)
            - d_model = 256

        这里只用随机张量构造一次 forward，主要检查：
            - dtype 对齐（不会再出现 Double / Half 冲突）
            - 各模块输入 / 输出 shape 是否一致
        """

        torch.manual_seed(0)

        # 配置超参数（与 TransfuserConfig 中 NF 相关部分保持一致的数量级即可）
        B = 2                # batch_size
        M = 20               # ego_fut_mode (mode 数)
        T = 8                # 轨迹长度 num_poses
        C_in = 3             # flow 维度: x, y, heading
        d_model = 256        # Transformer 通道

        # 构造一个简单的 SequenceFlowModel
        flow = SequenceFlowModel(
                in_channels=C_in,
                seq_len=T,
                channels=d_model,
                num_blocks=2,
                layers_per_block=2,
                nvp=True,
                num_classes=0,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        flow = flow.to(device)

        # 构造模拟输入，shape 仿照实际 TarDrive 训练：
        # traj_feature: (B, M, d_model)
        traj_feature = torch.randn(B, M, d_model, device=device, dtype=flow.blocks[0].proj_in.weight.dtype)

        # noisy_traj_points: (B, M, T, 3)
        noisy_traj_points = torch.randn(B, M, T, C_in, device=device, dtype=flow.blocks[0].proj_in.weight.dtype)

        # bev_feature: (B, 256, 64, 64)
        bev_feature = torch.randn(B, 256, 64, 64, device=device, dtype=flow.blocks[0].proj_in.weight.dtype)
        bev_spatial_shape = bev_feature.shape[2:]

        # agents_query: (B, 30, d_model)
        agents_query = torch.randn(B, 30, d_model, device=device, dtype=flow.blocks[0].proj_in.weight.dtype)

        # ego_query: (B, 1, d_model)
        ego_query = torch.randn(B, 1, d_model, device=device, dtype=flow.blocks[0].proj_in.weight.dtype)

        # status_encoding: (B, 1, d_model)
        status_encoding = torch.randn(B, 1, d_model, device=device, dtype=flow.blocks[0].proj_in.weight.dtype)

        # global_img: 这里随便给一个 (B, C, H, W)，暂时不用也可以传 None
        global_img = None

        with torch.no_grad():
                z, poses_reg_list, logdets, poses_cls_list = flow(
                        traj_feature,
                        noisy_traj_points,
                        ego_query,
                        agents_query,
                        bev_feature,
                        bev_spatial_shape,
                        status_encoding,
                        global_img,
                        best_mode=1,
                )

        print("[SequenceFlowModel Debug]")
        print(f"z.shape = {z.shape}")                       # 期望: (B, M, T, 3)
        print(f"len(poses_reg_list) = {len(poses_reg_list)}")
        print(f"poses_reg_list[-1].shape = {poses_reg_list[-1].shape}")
        print(f"logdets.shape = {logdets.shape}")           # 期望: (B,)
        print(f"poses_cls_list[-1].shape = {poses_cls_list[-1].shape}")  # 期望: (B, M)


        # 现在修改reverse函数,并且进行可逆性验证（z->x->z）以及生成结果的合理性检查。

        with torch.no_grad():
            traj_points_sample, logdet, poses_cls = flow.reverse(
                    traj_feature,
                    z,
                    ego_query,
                    bev_feature,
                    bev_spatial_shape,
                    agents_query,
                    status_encoding,
                    global_img,
            )
        print("[SequenceFlowModel Reverse Debug]")
        print(f"z.shape = {z.shape}")                       # 期望: (B, M, T, 3)
        print(f"len(poses_reg_list) = {len(poses_reg_list)}")
        print(f"poses_reg_list[-1].shape = {poses_reg_list[-1].shape}")
        print(f"poses_cls_list[-1].shape = {poses_cls_list[-1].shape}")  # 期望: (B, M)
        print(f"traj_points_sample.shape = {traj_points_sample.shape}")  # 期望: (B, M, T, 3)
        # 注意：traj_points_sample[-1]理论上应该和noisy_traj_points非常接近，因为模型是一个可逆的分布
        print(f"Whether the reverse is correct? {noisy_traj_points==traj_points_sample[-1]}")


    



