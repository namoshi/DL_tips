import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 再現性のためにシードを固定
torch.manual_seed(42)
np.random.seed(42)

# --- 1. 行列 A の定義 ---
# 例: 5次元空間内の3つのベクトル（行）を持つ行列
# ランクを意図的に下げるため、3行目は1行目と2行目の線形結合にします
row1 = [1.0, 2.0, 3.0, 4.0, 5.0]
row2 = [2.0, 3.0, 4.0, 5.0, 6.0]
row3 = [3.0, 5.0, 7.0, 9.0, 11.0] # row1 + row2

A_np = np.array([row1, row2, row3], dtype=np.float32)
m, n = A_np.shape # m=行数, n=列数(次元)

# ランクの計算（中間層の次元に使用）
rank = np.linalg.matrix_rank(A_np)
print(f"行列 A の形状: {A_np.shape}")
print(f"行列 A のランク: {rank}")
print("-" * 30)

# --- 2. SVDによる厳密な射影行列の計算 (正解データ) ---
# P = V_r * V_r^T (V_rは右特異ベクトル)
U, S, Vt = np.linalg.svd(A_np)
V_r = Vt[:rank, :].T # ランク分だけ取り出して転置 (n x r)
P_true = V_r @ V_r.T

# --- 3. 線形AutoEncoderの定義 ---
class LinearAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(LinearAutoEncoder, self).__init__()
        # バイアス(bias)をFalseにするのが重要です（線形部分空間は原点を通るため）
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=False)
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=False)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# モデルの初期化 (中間層の次元 = ランク)
model = LinearAutoEncoder(input_dim=n, hidden_dim=rank)

# --- 4. 学習ループ ---
# 行列 A の行ベクトル自体をトレーニングデータとします
data_tensor = torch.from_numpy(A_np)

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.0)

epochs = 50000
for epoch in range(epochs):
    # Forward
    outputs = model(data_tensor)
    loss = criterion(outputs, data_tensor)
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 1000 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.8f}')

print("-" * 30)

# --- 5. 結果の検証 ---
# 学習された重みの抽出
# PyTorchのLinear層は x @ W.T の形なので、
# 数式の P = W_dec @ W_enc に対応させるには、
# P_ae = decoder.weight @ encoder.weight となります。

W_enc = model.encoder.weight.detach().numpy() # shape (r, n)
W_dec = model.decoder.weight.detach().numpy() # shape (n, r)

# AEによって学習された射影行列
P_ae = W_dec @ W_enc

print("SVDで求めた真の射影行列 P:")
print(np.round(P_true, 3))
# print(P_true)
print("-" * 30)

print("AutoEncoderが学習した射影行列 P_ae:")
print(np.round(P_ae, 3))
# print(P_ae)
print("-" * 30)

# 誤差の確認 (フロベニウスノルム)
diff = np.linalg.norm(P_true - P_ae)
print(f"真の射影行列との誤差 (Frobenius Norm): {diff:.6f}")

# --- 6. 幾何学的な動作確認 ---
# 行空間に含まれないランダムなベクトル x を用意
x_random = np.random.randn(n)

# 真の射影
proj_true = P_true @ x_random

# AEによる射影 (行列演算で再現)
proj_ae = P_ae @ x_random

print("\nランダムベクトル x の射影結果の比較:")
print(f"真の射影: {np.round(proj_true, 3)}")
print(f"AEの射影: {np.round(proj_ae, 3)}")