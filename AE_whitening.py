import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. データの準備 (相関のあるガウス分布)
# ---------------------------------------------------------
def generate_correlated_data(n_samples=2000):
    # 2次元の相関データを作成
    mean = [0, 0]
    cov = [[2.0, 1.5], [1.5, 1.5]] 
    data = np.random.multivariate_normal(mean, cov, n_samples)
    return torch.tensor(data, dtype=torch.float32)

# ---------------------------------------------------------
# 2. 白色化ペナルティ関数の定義
# ---------------------------------------------------------
def whitening_loss_func(h):
    """
    隠れ層の出力 h を白色化するための損失関数
    h: (batch_size, hidden_dim)
    """
    batch_size = h.size(0)
    # 平均を引いて中心化
    h_centered = h - h.mean(dim=0)
    # 共分散行列を計算
    cov_matrix = (h_centered.t() @ h_centered) / (batch_size - 1)
    
    # 対角成分を1に近づける (分散の正規化)
    diag_loss = torch.mean((torch.diag(cov_matrix) - 1.0) ** 2)
    # 非対角成分を0に近づける (無相関化)
    off_diag_mask = ~torch.eye(h.size(1), dtype=torch.bool)
    off_diag_loss = torch.mean(cov_matrix[off_diag_mask] ** 2)
    
    return diag_loss + off_diag_loss

# ---------------------------------------------------------
# 3. 線形Autoencoderモデル
# ---------------------------------------------------------
class WhiteningAE(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(WhiteningAE, self).__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=True)
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=True)

    def forward(self, x):
        h = self.encoder(x)
        x_hat = self.decoder(h)
        return x_hat, h

# ---------------------------------------------------------
# 4. 学習ループ
# ---------------------------------------------------------
data = generate_correlated_data()
model = WhiteningAE(input_dim=2, hidden_dim=2)
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 学習前の共分散行列を表示
data_centered = data - data.mean(dim=0)
initial_cov = (data_centered.t() @ data_centered) / (data.size(0) - 1)
print("学習前のデータの共分散行列:")
print(initial_cov.numpy())

# 学習ループ
for epoch in range(1001):
    model.train()
    optimizer.zero_grad()
    
    # 順伝播
    reconstructed, hidden = model(data)
    
    # 損失計算: 再構成誤差 + 白色化ペナルティ
    recon_loss = nn.MSELoss()(reconstructed, data)
    w_loss = whitening_loss_func(hidden)
    
    total_loss = recon_loss + 0.5 * w_loss # 重み係数は適宜調整
    
    total_loss.backward()
    optimizer.step()
    
    if epoch % 200 == 0:
        print(f"Epoch {epoch}: Total={total_loss.item():.4f}, Recon={recon_loss.item():.4f}, White={w_loss.item():.4f}")

# ---------------------------------------------------------
# 5. 結果の検証
# ---------------------------------------------------------
model.eval()
with torch.no_grad():
    _, h = model(data)
    # 学習後の隠れ層の共分散を確認
    h_centered = h - h.mean(dim=0)
    final_cov = (h_centered.t() @ h_centered) / (data.size(0) - 1)
    print("\n学習後の隠れ層の共分散行列 (単位行列に近いほど白色化されている):")
    print(final_cov.numpy())