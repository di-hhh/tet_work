import torch
import torch.nn as nn
import math


class Embedding_v1(nn.Module):
    def __init__(self, in_features, out_features, is_vertex=True, num_freqs=6, hidden_dim_factor=4):
        super().__init__()
        self.is_vertex = is_vertex
        self.in_features = in_features  # 这里会接收真实的维度（比如 19）
        self.out_features = out_features
        self.num_freqs = num_freqs

        # 1. 频率映射矩阵
        # 形状: [in_features, num_freqs]
        self.B = nn.Parameter(
            torch.randn(in_features, num_freqs) * 2 * math.pi,
            requires_grad=False
        )

        # 计算频率映射后的维度: in_features * 2 * num_freqs
        freq_dim = in_features * 2 * num_freqs
        print(f"Frequency dimension: {freq_dim}")

        # 总输入维度 = 频率特征 + 原始预处理特征
        total_input_dim = freq_dim + in_features   # 拼接后的总维度
        print(f"Total input dimension: {total_input_dim}")

        hidden_dim = max(total_input_dim, out_features) * hidden_dim_factor
        print(f"Hidden dimension: {hidden_dim}")

        # Frequency dimension: 48   3*2*8
        # Total input dimension: 51 48+3
        # Hidden dimension: 256

        # Frequency dimension: 40  2*2*10
        # Total input dimension: 42 40+2
        # Hidden dimension: 256

        # RuntimeError: mat1 and mat2 shapes cannot be multiplied (4774x19 and 51x256)

        # 2. 定义 MLP 层
        self.mlp = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),

            nn.Linear(hidden_dim // 2, out_features),
            nn.BatchNorm1d(out_features),
            nn.GELU()
        )

        # 3. 修复初始化：使用 'relu' 作为 gain 的参考，因为 kaiming_normal 不支持 'gelu' 字符串
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # 修改点：nonlinearity 改为 'relu'，避免 ValueError
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        x shape: [N, in_features]
        注意：这里的 in_features 必须与 __init__ 传入的一致
        """
        # --- 步骤 A: 几何特定的预处理 ---
        x_processed = x.clone()

        eps = 1e-6
        if self.is_vertex:
            # if (x_processed[:, 0:1] < 0).any():
            #     print("⚠️ 警告：度（第0维）存在负值！")
            #     # 安全修复：将所有负值替换为极小正数（1e-6）
            #     x_processed[:, 0:1] = torch.clamp(x_processed[:, 0:1], min=0)
            #     print(f"✅ 已修复：{torch.sum(x_processed[:, 0:1] < 0).item()} 个负值被修正")
            # else:
            #     print("✅ 度（第0维）无负值，安全！")

            # 假设第0列是 Degree (非负整数)
            x_processed[:, 0:1] = torch.log1p(x_processed[:, 0:1].clamp(min=0))
            if torch.isnan(x_processed).any():
                print("草泥马1")

            # if (x_processed[:, 1:2] < 0).any():
            #     print("⚠️ 警告：尺寸场（第1维）存在负值！")
            #     # 安全修复：将所有负值替换为极小正数（1e-6）
            #     x_processed[:, 1:2] = torch.clamp(x_processed[:, 1:2], min=1e-6)
            #     print(f"✅ 已修复：{torch.sum(x_processed[:, 1:2] < 0).item()} 个负值被修正")
            # else:
            #     print("✅ 尺寸场（第1维）无负值，安全！")

            # 假设其余列是尺寸场 (正值)
            if x_processed.shape[1] > 1:
                x_processed[:, 1:2] = torch.log(x_processed[:, 1:2].abs() + eps)

            if torch.isnan(x_processed).any():
                print("草泥马2")

        else:  # Edge
            # 假设第0列是 Length (正值)
            if x_processed.shape[1] > 0:
                x_processed[:, 0:1] = torch.log(x_processed[:, 0:1] + eps)
            if torch.isnan(x_processed).any():
                print("草泥马3")

            # 假设第1列是 Curvature (有正负，如果有这一列)
            if x_processed.shape[1] > 1:
                x_processed[:, 1:2] = torch.tanh(x_processed[:, 1:2])
            if torch.isnan(x_processed).any():
                print("草泥马4")

        if torch.isnan(x_processed).any():
            raise RuntimeError("草泥马")

        # --- 步骤 B: 傅里叶频率映射 (Fourier Features) ---
        x_proj = x_processed.unsqueeze(-1) * self.B.unsqueeze(0)  # [N, in_features, num_freqs]
        x_sin = torch.sin(x_proj)  # [N, in_features, num_freqs]
        x_cos = torch.cos(x_proj)  # [N, in_features, num_freqs]
        x_freq = torch.cat([x_sin, x_cos], dim=-1)  # [N, in_features, 2*num_freqs]
        x_freq = x_freq.view(x.size(0), -1)  # [N, in_features * 2 * num_freqs]

        # --- 拼接特征---
        x_combined = torch.cat([x_freq, x_processed], dim=-1)  # [N, total_input_dim]
        # print(f"x_combined.shape:{x_combined.shape}")

        # --- 4. 再次检查 MLP 输入 ---
        if torch.isnan(x_combined).any():
             raise RuntimeError("嵌入层内部产生了 NaN！请检查预处理逻辑。")

        return self.mlp(x_combined)

