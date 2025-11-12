import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import time

# ==============================================================================
# 1. Octave Convolution Layer (OctConv2D)
# ==============================================================================

class OctConv2D(nn.Module):
    """
    Octave Convolution Unit: Replaces standard Conv2D.
    Handles input factorization (alpha_in) and output factorization (alpha_out).
    alpha: ratio of channels dedicated to the low-frequency component.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, alpha_in=0.5, alpha_out=0.5, stride=1, padding=1, bias=False):
        super(OctConv2D, self).__init__()
        
        # アルファ値の制約
        assert 0 <= alpha_in <= 1
        assert 0 <= alpha_out <= 1
        
        self.alpha_in = alpha_in
        self.alpha_out = alpha_out
        
        # 入力チャネルの分割
        # OctConvの構造により、α_in=0, α_out=α や α_in=α, α_out=0 のケースが存在する [6, 7]
        self.C_in_H = int(in_channels * (1 - alpha_in))
        self.C_in_L = in_channels - self.C_in_H
        
        # 出力チャネルの分割
        self.C_out_H = int(out_channels * (1 - alpha_out))
        self.C_out_L = out_channels - self.C_out_H
        
        # OctConvのコアロジックを構成する4つの畳み込みパス [1, 8]

        # 1. Intra-frequency: H -> H (高周波の更新)
        if self.C_in_H > 0 and self.C_out_H > 0:
            self.W_H_to_H = nn.Conv2d(self.C_in_H, self.C_out_H, kernel_size, stride, padding, bias=bias)

        # 2. Inter-frequency: H -> L (ダウンサンプリングと情報交換)
        # pool(X^H, 2) -> f(pooled(X^H); W^H->L) [1]
        if self.C_in_H > 0 and self.C_out_L > 0:
            # 空間的な冗長性を減らすため、平均プーリングを使用する [1, 3, 4]
            self.pool = nn.AvgPool2d(kernel_size=2, stride=2) 
            self.W_H_to_L = nn.Conv2d(self.C_in_H, self.C_out_L, kernel_size, stride, padding, bias=bias)

        # 3. Inter-frequency: L -> H (アップサンプリングと情報交換)
        # upsample(f(X^L; W^L->H), 2) [1]
        if self.C_in_L > 0 and self.C_out_H > 0:
            self.W_L_to_H = nn.Conv2d(self.C_in_L, self.C_out_H, kernel_size, stride, padding, bias=bias)
        
        # 4. Intra-frequency: L -> L (低周波の更新)
        if self.C_in_L > 0 and self.C_out_L > 0:
            self.W_L_to_L = nn.Conv2d(self.C_in_L, self.C_out_L, kernel_size, stride, padding, bias=bias)
    
    def forward(self, x):
        # 入力xはテンソル（最初のレイヤー: alpha_in=0）またはタプル (X_H, X_L)
        if self.C_in_L == 0:
            X_H = x
            X_L = None
        else:
            X_H, X_L = x
        
        Y_H = 0.0
        Y_L = 0.0

        # High Frequency Output (Y_H = Y_H->H + Y_L->H)
        if self.C_out_H > 0:
            # 1. H -> H
            if self.C_in_H > 0:
                Y_H = Y_H + self.W_H_to_H(X_H)
            
            # 3. L -> H (Conv -> Upsample)
            if self.C_in_L > 0:
                Y_L_to_H_conv = self.W_L_to_H(X_L)
                # 最近傍補間によるアップサンプリング (scale_factor=2) [2, 5]
                Y_L_to_H_upsampled = F.interpolate(Y_L_to_H_conv, scale_factor=2, mode='nearest')
                Y_H = Y_H + Y_L_to_H_upsampled

        # Low Frequency Output (Y_L = Y_L->L + Y_H->L)
        if self.C_out_L > 0:
            # 4. L -> L
            if self.C_in_L > 0:
                Y_L = Y_L + self.W_L_to_L(X_L)
            
            # 2. H -> L (Pool -> Conv)
            if self.C_in_H > 0:
                X_H_pooled = self.pool(X_H)
                Y_H_to_L = self.W_H_to_L(X_H_pooled)
                Y_L = Y_L + Y_H_to_L

        # 出力の形式を決定
        if self.C_out_L == 0:
            # 最終レイヤー (alpha_out=0)
            return Y_H
        else:
            # 中間レイヤー (alpha_out>0)
            return Y_H, Y_L

# ==============================================================================
# 2. Network Definition (Simple CNN/OctConv comparison model)
# ==============================================================================

class OctConvBlock(nn.Module):
    """標準的な OctConv ブロック: OctConv -> BatchNorm -> ReLU"""
    def __init__(self, in_channels, out_channels, alpha_in, alpha_out, kernel_size=3, stride=1, padding=1):
        super(OctConvBlock, self).__init__()
        self.octconv = OctConv2D(in_channels, out_channels, kernel_size, 
                                 alpha_in=alpha_in, alpha_out=alpha_out, 
                                 stride=stride, padding=padding)
        # BatchNorm層の定義 (出力がタプルの場合、HとLで独立して行う必要がある)
        if alpha_out > 0 and alpha_out < 1:
            self.bn_H = nn.BatchNorm2d(self.octconv.C_out_H)
            self.bn_L = nn.BatchNorm2d(self.octconv.C_out_L)
        elif alpha_out == 1:
            self.bn_L = nn.BatchNorm2d(self.octconv.C_out_L)
        else: # alpha_out == 0
            self.bn_H = nn.BatchNorm2d(self.octconv.C_out_H)
        
        self.alpha_out = alpha_out

    def forward(self, x):
        y = self.octconv(x)
        
        if self.alpha_out > 0 and self.alpha_out < 1:
            Y_H, Y_L = y
            Y_H = F.relu(self.bn_H(Y_H))
            Y_L = F.relu(self.bn_L(Y_L))
            return Y_H, Y_L
        elif self.alpha_out == 1:
            Y_L = y
            Y_L = F.relu(self.bn_L(Y_L))
            return (None, Y_L) # HighはNoneとして扱う
        else: # alpha_out == 0
            Y_H = y
            Y_H = F.relu(self.bn_H(Y_H))
            return Y_H


class OctConvNet(nn.Module):
    def __init__(self, num_classes=10, alpha=0.25):
        super(OctConvNet, self).__init__()
        
        C1 = 64
        C2 = 128
        
        # 1. 最初のレイヤー: Vanilla Input -> Octave Output (alpha_in=0, alpha_out=alpha) [6]
        # CIFAR-10は3チャネル入力
        self.conv1 = OctConvBlock(in_channels=3, out_channels=C1, alpha_in=0, alpha_out=alpha)
        
        # 2. 中間レイヤー: Octave Input -> Octave Output (alpha_in=alpha, alpha_out=alpha) [9]
        self.conv2 = OctConvBlock(in_channels=C1, out_channels=C2, alpha_in=alpha, alpha_out=alpha)

        # 3. 最終レイヤー: Octave Input -> Vanilla Output (alpha_in=alpha, alpha_out=0) [7]
        # (ここでは分類のために高解像度に戻す)
        self.conv3 = OctConvBlock(in_channels=C2, out_channels=C2, alpha_in=alpha, alpha_out=0)
        
        # Global Average Pooling と Linear Layer (CIFAR-10のサイズに合わせて調整が必要)
        # CIFAR-10 (32x32) -> conv3 (32x32)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(C2, num_classes)

    def forward(self, x):
        # Stage 1: Output (H1, L1)
        x_H, x_L = self.conv1(x)
        
        # Stage 2: Input (H1, L1) -> Output (H2, L2)
        x_H, x_L = self.conv2((x_H, x_L))

        # Stage 3: Input (H2, L2) -> Output (H3, only High)
        x_H = self.conv3((x_H, x_L))

        # Classifier
        x = self.avgpool(x_H)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# Vanilla Conv Model (OctConvNetで alpha=0 を設定したのと理論的に同等)
class VanillaNet(nn.Module):
    def __init__(self, num_classes=10):
        super(VanillaNet, self).__init__()
        C1 = 64
        C2 = 128
        # Standard Conv2D layers
        self.conv1 = nn.Conv2d(3, C1, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(C1)
        self.conv2 = nn.Conv2d(C1, C2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(C2)
        self.conv3 = nn.Conv2d(C2, C2, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(C2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(C2, num_classes)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# ==============================================================================
# 3. Training and Evaluation Functions
# ==============================================================================

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch}: Loss = {loss.item():.4f}')

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
    return accuracy

def main():
    # ハイパーパラメータ
    EPOCHS = 20
    BATCH_SIZE = 128
    LEARNING_RATE = 0.01
    OCTCONV_ALPHA = 0.25 # OctConvの性能が最も良いとされる値に近い [10]

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # データローダーの準備 (CIFAR-10)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # ----------------------------------------
    # モデル 1: Vanilla Net (通常の Conv)
    # ----------------------------------------
    print("\nStarting VanillaNet training...")
    vanilla_model = VanillaNet().to(device)
    vanilla_optimizer = optim.Adam(vanilla_model.parameters(), lr=LEARNING_RATE)
    
    start_time_vanilla = time.time()
    for epoch in range(1, EPOCHS + 1):
        train(vanilla_model, device, train_loader, vanilla_optimizer, epoch)
    end_time_vanilla = time.time()
    
    accuracy_vanilla = test(vanilla_model, device, test_loader)

    # ----------------------------------------
    # モデル 2: OctConv Net (α=0.25)
    # ----------------------------------------
    print(f"\nStarting OctConvNet (alpha={OCTCONV_ALPHA}) training...")
    octconv_model = OctConvNet(alpha=OCTCONV_ALPHA).to(device)
    octconv_optimizer = optim.Adam(octconv_model.parameters(), lr=LEARNING_RATE)
    
    start_time_octconv = time.time()
    for epoch in range(1, EPOCHS + 1):
        train(octconv_model, device, train_loader, octconv_optimizer, epoch)
    end_time_octconv = time.time()
    
    accuracy_octconv = test(octconv_model, device, test_loader)

    # ----------------------------------------
    # 結果の比較
    # ----------------------------------------
    print("\n" + "="*40)
    print("CIFAR-10 比較結果")
    print(f"VanillaNet (Conv) - Accuracy: {accuracy_vanilla:.2f}%, Time: {end_time_vanilla - start_time_vanilla:.2f}s")
    print(f"OctConvNet (α={OCTCONV_ALPHA}) - Accuracy: {accuracy_octconv:.2f}%, Time: {end_time_octconv - start_time_octconv:.2f}s")
    print("="*40)

if __name__ == '__main__':
    main()
