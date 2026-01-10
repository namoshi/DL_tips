import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

# 1. ハイパーパラメータ
input_dim = 784
hidden_dim = 128  # ここで次元を大幅に削減（784 -> 128）
batch_size = 1024
epochs = 200
alpha = 1.0  # 白色化損失の重み
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. データ準備
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1) - 0.5) # 簡易的な中心化
])
train_loader = DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform), 
                          batch_size=batch_size, shuffle=True, drop_last=True)

# 3. 白色化損失関数（k次元の単位行列を目指す）
def get_whitening_loss(z):
    b, k = z.size()
    z_centered = z - z.mean(dim=0)
    cov = (z_centered.t() @ z_centered) / (b - 1)
    eye = torch.eye(k).to(device)
    return nn.MSELoss()(cov, eye)

# 4. モデル定義（線形次元削減 AE）
class ReducedWhiteningAE(nn.Module):
    def __init__(self, in_dim, h_dim):
        super().__init__()
        self.encoder = nn.Linear(in_dim, h_dim)
        self.decoder = nn.Linear(h_dim, in_dim)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z


class SimpleNet(nn.Module):
    def __init__(
            self, 
            input_dim=2048, 
            hid_dim=1024, 
            eps: float = 1e-3
        ):
        self.eps = eps

        # 学習可能な行列の初期化
        self.A_mat = nn.Parameter(torch.empty(input_dim, hid_dim))
        nn.init.xavier_uniform_(self.A_mat)

    # feat : (bs, input_dim)
    def forward(self, feat:torch.Tensor):
        # A による射影
        # (bs, input_dim) -> (bs, hid_dim)
        hidden = feat @ self.A_mat  

        # A の転置による射影
        # (bs, hid_dim) -> (bs, input_dim)
        out = hidden @ self.A_mat.T

        """
        self.A_mat は普通の行列と同様に演算が計算できる

        ex. 直交射影行列の計算
        W = self.A_mat  # (m, n), m > n
        WtW = W.t() @ W     # (n, n)

        I_mat = torch.eye(WtW.size(0), device=W.device)
        WtW_inv = torch.linalg.solve(WtW + eps * I_mat, I_mat) # (n, n)

        P = W @ WtW_inv @ W.t()     # (m, m)

        feat_A = feat @ P
        """

        return out


model = ReducedWhiteningAE(input_dim, hidden_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.002)
save_path = './trained_models/saved_ae_mnist.pth'

# 5. モデルが既に保存されていればロードして学習をスキップ
if os.path.exists(save_path):
    state = torch.load(save_path, map_location=device)
    model.load_state_dict(state)
    print(f"Loaded saved model from {save_path}")
    trained = True
else:
    trained = False

# 5. 学習
if not trained:
    model.train()
    for epoch in range(epochs):
        recon_loss = torch.tensor(0.0, device=device)
        white_loss = torch.tensor(0.0, device=device)
        for data, _ in train_loader:
            data = data.view(-1, 784).to(device)
            optimizer.zero_grad()
            
            recon, latent = model(data)
            
            recon_loss = nn.MSELoss()(recon, data)
            white_loss = get_whitening_loss(latent)
            
            # 合計損失: 再構成を優先しつつ白色化を混ぜる
            loss = recon_loss + alpha * white_loss
            
            loss.backward()
            optimizer.step()
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Recon Loss={recon_loss.item():.4f}, White Loss={white_loss.item():.4f}")

    # 学習完了後にモデルを保存
    torch.save(model.state_dict(), save_path)
    print(f"Saved trained model to {save_path}")

# 6. 結果の可視化
model.eval()
with torch.no_grad():
    sample, _ = next(iter(train_loader))
    sample = sample.view(-1, 784).to(device)
    recon, latent = model(sample)

    plt.figure(figsize=(10, 4))
    # オリジナル
    plt.subplot(1, 3, 1)
    plt.imshow(sample[0].cpu().view(28, 28), cmap='gray')
    plt.title("Original (784d)")
    
    # 再構成
    plt.subplot(1, 3, 3)
    plt.imshow(recon[0].cpu().view(28, 28), cmap='gray')
    plt.title("Reconstructed")
    plt.show()

    # 潜在変数の分散共分散行列の計算
    latent_centered = latent - latent.mean(dim=0)
    cov_matrix = (latent_centered.t() @ latent_centered) / (latent.size(0) - 1)
    print("潜在変数の分散共分散行列:")
    print(cov_matrix.cpu().numpy())