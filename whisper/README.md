# Whisper — BM1684X 移植

Whisper 语音识别移植到 Sophon BM1684X，支持中英文。  
支持 **base**（6 层，512 维）与 **large-v3-turbo**（32 层 encoder / 4 层 decoder，1280 维）两个模型，精度 F32 / F16 / W4F16。  
C++ 推理运行时从 bmodel 自动读取维度，一套代码两个模型通用。

## 快速开始

### 环境依赖

- 开发机：Python 3.8+、PyTorch、openai-whisper、onnx、Docker
- `0_Toolkits/soc-sdk-sp4/`：Sophon SOC SDK（bmruntime 头文件 + so）
- `1_third_party/fftw/`：fftw3f aarch64 静态库（Mel 特征计算）

### Step 1：导出 ONNX

```bash
cd whisper/python

# base（默认）
python export_onnx.py

# large-v3-turbo（encoder FP32 ≈2.4GB，需 WSL 内存 ≥14GB）
python export_onnx.py --model large-v3-turbo --asset_dir ../models/BM1684X_turbo
```

产物在 `models/onnx/`，同时生成 `mel_*_filters.txt`、`positional_embedding.npy`、`vocab.txt`（部署时随 bmodel 一起上板）。

### Step 2：编译 bmodel

```bash
# base（在 TPU-MLIR Docker 内）
docker exec sophon-tpumlir bash /workspace/whisper/python/gen_bmodel.sh F16

# turbo F16 / W4F16
docker exec sophon-tpumlir bash /workspace/whisper/python/gen_bmodel_turbo.sh F16
docker exec sophon-tpumlir bash /workspace/whisper/python/gen_bmodel_turbo.sh W4F16
```

> ⚠️ **所有模型编译必须加 `--disable_layer_group`**（decoder 图结构复杂；turbo encoder >500MB 的大模型不加会在板卡 kernel panic 重启）。脚本已内置，勿删。

### Step 3：交叉编译 C++

```bash
# 构建交叉编译镜像（只需一次）
docker build -t sophon-cross-build docker/

bash whisper/cpp/build.sh
# 产物：whisper/cpp/build/whisper_bm1684
```

### Step 4：上板运行

```bash
scp whisper/cpp/build/whisper_bm1684 root@<board_ip>:/data/whisper/

# base
./whisper_bm1684 models/BM1684X       test.wav zh F16
# turbo F16
./whisper_bm1684 models/BM1684X_turbo test.wav zh F16   turbo
# turbo W4F16
./whisper_bm1684 models/BM1684X_turbo_w4f16 test.wav zh W4F16 turbo
# 参数: <model_dir> <audio.wav> [zh|en] [F32|F16|W4F16] [base|turbo]
```

---

## 性能（BM1684X 实测）

### base（~5.8s 音频，5 轮平均）

| 精度 | 总耗时 |
|------|--------|
| F32  | ~1.86s |
| F16  | ~1.01s |

### large-v3-turbo — 内存 / 精度对比（~5.6s 音频）

| 指标 | F16 | W4F16 | 变化 |
|------|-----|-------|------|
| encoder bmodel | 1.3G | **369M** | 省 72% |
| decoder bmodel | 460M | **222M** | 省 52% |
| 部署总大小 | 1.7G | **594M** | 省 65% |
| encoder device 峰值 | 1.66G | **750M** | 省 55% |
| decoder device 峰值 | 669M | **420M** | 省 37% |
| 中英文转录 | 基准 | **逐字一致** | **无损** |

### large-v3-turbo — RTF（52 条校准数据，中英文混合）

> 校准数据：ChatTTS 合成 20 条 + FLEURS 中文 15 条 + LibriSpeech 英文 15 条 + 测试音频 2 条，共 639s。  
> 剔除 decoder > 5s 的极短音频异常条目后统计。

| 精度 | 平均 Encoder | 平均 Decoder | **平均 RTF** | 中文 RTF | 英文 RTF | 有效条数 |
|------|:-----------:|:-----------:|:-----------:|:--------:|:--------:|:-------:|
| F16   | 1372ms | 1228ms | **0.353** | 0.281 | 0.419 | 50/52 |
| W4F16 | 1474ms | 897ms  | **0.346** | 0.264 | 0.403 | 44/52 |

> RTF < 1 即实时，约 **2.8×~4× 实时**。  
> W4F16 decoder 快 27%，encoder 略慢，端到端 RTF 与 F16 基本持平。  
> **turbo 推荐直接用 W4F16**：省 65% 内存、完全无损、速度持平。

### 量化格式支持（BM1684X vs BM1688）

| 格式 | BM1684X | BM1688 | 说明 |
|------|:-------:|:------:|------|
| F16 / F32  | ✅ | ✅ | 无需 calibration |
| W4F16 / W4BF16 | ✅ | ✅ | 无需 calibration，**推荐** |
| INT8 | ⚠️ 精度崩溃 | ✅ | 需要 calibration；对 Whisper encoder 不可用，见下 |
| W4A8 (W4INT8) | ❌ 硬件不支持 | ✅ | BM1684X 架构限制，见下 |

**INT8 实验结论**：encoder INT8 编译成功（625MB，比 F16 小 52%），速度极快（encoder 758ms，比 F16 快 40%），但**精度完全崩溃**——所有输入输出全为 `......`。原因是 Whisper encoder 共 32 层，激活值量化到 8-bit 后误差逐层累积，最终 audio_features 严重失真，decoder 无法正常解码。decoder INT8 编译直接失败（TPU-MLIR v1.28.1 `tpuc-opt` 内部 abort，compiler bug）。**INT8 不适用于 Whisper，W4F16 是 BM1684X 上的最优方案。**

**W4A8 实验结论**：TPU-MLIR 中对应参数名 `W4INT8`，codegen 阶段报错 `tpu_data_type_size Assertion "0" failed`。查阅 Sophgo SDK v26.03.01 文档确认 W4A8 仅支持 BM1688，这是硬件架构限制，升级编译器版本无法解决。calibration 数据（52 条）已保存在 `whisper/calib_data/`，换 BM1688 平台可直接复用。

---

## 踩过的坑

### 1. KV 输入被 constant folding 消除

`[tensor] * n_layer` 创建同一对象的 N 个引用，ONNX tracer 将 KV 全部折叠。必须用列表推导式：
```python
# 正确
dummy_past_self_k = [torch.zeros(1, PADDING_SIZE, n_state) for _ in range(n_layer)]
```

### 2. Conv 节点缺少 `kernel_shape` 属性

新版 torch.onnx.export（opset 17）有时不写入该属性，导致 `model_transform.py` 报 `KeyError`。`export_onnx.py` 中已加 `_fix_conv_kernel_shape` 修复。

### 3. `--disable_layer_group` 必须加

- **Decoder**：图结构复杂（28 输入、25 输出），layer_group 优化不稳定，不加会生成异常 bmodel。
- **turbo Encoder（>500MB）**：不加会在推理时触发 TPU-MLIR v1.28.1 的 layer_group bug，导致板卡 kernel panic 重启。base encoder（46MB）不受影响。

### 4. turbo ONNX 导出 OOM

turbo encoder FP32 ≈2.4GB，WSL 内存不足会 OOM（Exit 137）。需在 `.wslconfig` 设置 `memory=14GB` 以上，并在 `export_onnx.py` 中跳过 onnxsim（protobuf 2GB 上限）、改用 external data 格式保存。

### 5. scp 大文件静默损坏

板卡崩溃期间 scp 传输的大文件会文件大小对、内容错（静默损坏）。**每个上板文件都用 `md5sum` 比对本地**，别假设 scp 成功。positional_embedding.npy 损坏会导致 decoder 死循环输出空格。

### 6. 交叉编译 glibc 版本不兼容

WSL 原生 gcc 15.x 编译的二进制在板卡（Ubuntu 20.04, glibc 2.31）报 `GLIBC_2.34 not found`。用 Ubuntu 20.04 Docker（gcc 9.4）交叉编译，产物依赖 glibc ≤2.29。
