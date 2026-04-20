# MiniCNN vs llm.c / tiny-cuda-nn / neural-network-cuda 完成度報告

## 先定位各自的賽道

| 專案 | 定位 | 目標讀者 |
|---|---|---|
| **llm.c** | GPT-2/3 production-grade LLM 訓練，7% 快於 PyTorch Nightly | 進階研究者、LLM 工程師 |
| **tiny-cuda-nn** | NVIDIA Labs 商用 fully-fused MLP，為 NeRF/神經場景設計 | 工業界 CUDA 工程師 |
| **neural-network-cuda** | 教學用最簡 CUDA 前饋網路，程式碼優先清晰 | 學習 GPU 程式的初學者 |
| **MiniCNN** | 雙 backend（PyTorch + 手寫 CUDA）CNN 框架，含純 NumPy autograd | 想學 framework 設計的中階開發者 |

MiniCNN 和 llm.c / tiny-cuda-nn 不在同一個賽道——硬要比是類比錯誤。真正的競品是 **neural-network-cuda + micrograd 的混合體**，但 MiniCNN 比兩者都更完整。

---

## CUDA kernel 層面對比

### 目前已有

| 功能 | 實作品質 |
|---|---|
| im2col + cuBLAS gemm（conv forward） | 成熟，有 cublas / handmade 雙路徑 |
| Conv backward（`conv_backward.cu` 221 行） | 完整，有 precol 優化版 |
| MaxPool forward/backward（4 個 kernel 變體） | 完整，有 NCHW 專用版 |
| SGD / momentum / fused conv update（weight_decay + clip） | 完整 |
| Layer Norm（`layer_norm.cu`，含 warp shuffle reduce） | kernel 已寫，但**未接入主訓練路徑** |
| LeakyReLU forward / backward | 完整 |
| Loss（cross-entropy + accuracy fused） | 完整 |
| Layout convert（NCHW ↔ CNHW） | 有 |
| cuBLAS context 管理 | 有 |

### 目前缺少（對比三個 repo）

| 缺口 | 對應哪個 repo 有 | 補難度 |
|---|---|---|
| **Adam / AdamW optimizer kernel** | llm.c | ★★☆ 中 |
| **BatchNorm CUDA kernel** | 三者皆無，MiniCNN 也無 | ★★☆ 中 |
| **Sigmoid / Tanh CUDA kernel** | neural-network-cuda（ReLU only，但同類） | ★★ 低中 |
| **Residual / skip connection CUDA** | llm.c | ★★ 低中 |
| **Global gradient norm kernel**（L2 norm for grad clip） | llm.c | ★★ 低中 |
| **Layer Norm 接入訓練路徑** | llm.c | ★ 低（kernel 已有） |
| **FP16 / half-precision 訓練** | tiny-cuda-nn、llm.c | ★★★ 高 |
| **Shared memory tiling in conv** | tiny-cuda-nn（最深）| ★★★ 高 |
| **Dropout CUDA kernel**（Philox RNG） | llm.c | ★★★ 高 |
| **Gradient checkpointing** | llm.c | ★★★ 高 |
| **Attention / MHA kernel** | llm.c（via cuDNN） | ★★★★ 極高 |
| **Tensor Core / CUTLASS matmul** | tiny-cuda-nn | ★★★★ 極高 |
| **Fully-fused MLP kernel**（matmul+bias+activation 合一） | tiny-cuda-nn | ★★★★ 極高 |
| **多 GPU（NCCL / MPI）** | llm.c | ★★★★★ 極高 |

---

## Python / Framework 層面對比

### MiniCNN 有但三個對比專案都沒有

- **雙 backend 共用 YAML 切換**（`engine.backend: torch / cuda_legacy`）——這是 MiniCNN 真正獨特的設計
- **純 NumPy autograd engine**，比 neural-network-cuda 完整數倍
- **Function API**（micrograd 沒有，neural-network-cuda 沒有）
- **config-driven 架構**（YAML → model，三個 repo 都是 hardcode）
- **多資料集支援**（CIFAR-10 / MNIST / random）
- **compiler + runtime inference pipeline**（`InferencePipeline`）
- **完整測試套件**（102 tests）

### Python / Framework 層面的缺口

| 缺口 | 補難度 | 說明 |
|---|---|---|
| **Loss 多樣性**：MSE、BCE、label smoothing | ★ 低 | autograd path 只有 cross_entropy |
| **Sigmoid / Tanh CUDA kernel** | ★★ 低中 | CUDA path 只有 LeakyReLU，對稱性差 |
| **BatchNorm CUDA kernel** | ★★☆ 中 | layer_norm kernel 基礎設施可複用 |
| **Layer Norm 接入訓練路徑** | ★ 低 | kernel 已寫好，只缺 Python 側接線 |
| **Conv1D / Conv3D** | ★★★ 高 | 目前只有 Conv2D |
| **Embedding layer** | ★★ 中 | 進入 NLP 的第一步 |
| **真實 FLOP / throughput 報告** | ★★ 中 | `minicnn info` 只有環境資訊，無效能數字 |
| **混合精度（AMP）整合** | ★★★ 高 | torch path 可用 AMP 但未整合 |
| **ONNX export** | ★★★ 高 | 目前只有 npz / pt checkpoint |
| **Gradient checkpointing** | ★★★ 高 | llm.c 有，節省顯存必需 |

---

## 三個維度的完成度評分

### 維度一：CUDA kernel 技術深度

```
MiniCNN              ████████░░░░  65%
neural-network-cuda  ████░░░░░░░░  30%  (教學用，atomic-heavy，無優化)
llm.c                ████████████  95%  (fused + cuDNN + multi-GPU)
tiny-cuda-nn         ██████████░░  85%  (Tensor Core + fully fused MLP)
```

MiniCNN 在「自己手寫 kernel」這條路上已超越 neural-network-cuda，但 llm.c 和 tiny-cuda-nn 走的是「極致優化」路線，需要數個月到一年的 CUDA 專業才能追上。

### 維度二：Framework 完整度（可用性 / 可擴充）

```
MiniCNN              ████████████  90%
neural-network-cuda  ██░░░░░░░░░░  15%  (單一 hardcode 網路，不可擴充)
llm.c                ██████░░░░░░  55%  (one-file 風格，YAML 不存在)
tiny-cuda-nn         ████████░░░░  65%  (C++ API，Python 整合有代價)
```

這是 MiniCNN 真正領先的地方：YAML config、雙 backend、autograd + CUDA 並存的架構，三個對比專案都沒有。

### 維度三：教學價值（新人可學性）

```
MiniCNN              █████████░░░  75%  (有 notebook，但仍需 CUDA 背景)
neural-network-cuda  ████████████  95%  (極簡，一個下午能讀完)
llm.c                ████████░░░░  70%  (有 blog，但 codebase 較大)
tiny-cuda-nn         ████░░░░░░░░  30%  (模板地獄，新手難讀)
```

---

## 建議補強優先清單

### 高 CP 值（低難度、高用戶體驗提升）

1. **Adam CUDA kernel** — `optimizer.cu` 加約 30 行，消除 CUDA path 只有 SGD 的限制
2. **BatchNorm CUDA kernel** — `layer_norm.cu` 已有 warp reduce 基礎，BN 只是另一種 normalization 方式
3. **MSE / BCE loss**（autograd path）— 各 ~15 行，擴展到 regression 與二元分類任務
4. **Layer Norm 接入訓練路徑** — kernel 已寫好，只缺 Python 側 ctypes binding 與 YAML 接線

### 中難度、補完「手寫 CUDA CNN」的功能完整性

5. **Sigmoid / Tanh CUDA kernel** — CUDA path 目前只有 LeakyReLU，加入後與 autograd path 對稱
6. **Global gradient norm kernel** — llm.c 做法：先 reduce 出 L2 norm，再 scale；MiniCNN 目前是 element-wise clip，語義不同
7. **Residual / skip connection CUDA** — 只是一個 element-wise add kernel，能解鎖 ResNet 架構

### 高難度、需要 1–2 週專門工作

8. **Shared memory tiling in conv forward** — 目前 im2col 有全局記憶體瓶頸，tiling 可提速 2–4x
9. **FP16 支援（torch path 整合 AMP）** — Python 側相對好做，kernel 側需改所有 `float*` 為 `half*`
10. **真實 FLOP / throughput 報告** — `minicnn info` 加上理論 / 實際 FLOP 對比，讓用戶知道效能基準

### 目前不建議投入（投報比過低）

- **Tensor Core / CUTLASS matmul** — 需要深度 CUDA 架構知識；cuBLAS 路徑已提供等效能
- **Attention / MHA kernel** — 開新賽道，與 CNN 框架定位衝突
- **多 GPU / NCCL** — 基礎設施工程量大，不是此框架的學習重點

---

## 結論

MiniCNN 在 **framework 設計完整度**上已超越三個對比專案，**CUDA kernel 深度**介於 neural-network-cuda 和 llm.c 之間。

補完以下四個點，CUDA path 即可達到「功能完整」水準，無需追求 tiny-cuda-nn 的 Tensor Core 級別優化：

1. Adam CUDA kernel
2. BatchNorm CUDA kernel
3. Global gradient norm（真正的 L2 grad clip）
4. Residual skip connection kernel

參考資料：
- [karpathy/llm.c](https://github.com/karpathy/llm.c)
- [NVlabs/tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn)
- [BobMcDear/neural-network-cuda](https://github.com/BobMcDear/neural-network-cuda)
