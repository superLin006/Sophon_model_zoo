# SenseVoice Small — BM1684X 移植

SenseVoice Small 语音识别模型完整移植到 Sophon BM1684X，支持 FP32 / FP16，**自动语种识别**（中/英/粤/日/韩），同时输出情感和事件标签。

## 模型说明

SenseVoice Small 是单次前向编码器 + CTC 解码的模型，**无自回归循环**，速度极快。

- 输入：音频特征 `[1, 166, 560]`（Fbank-80 + LFR-7，对应最长 ~10s 音频）
- 输出：logits `[1, 170, 25055]`（前 4 帧为 prompt，后 166 帧为识别结果）
- 前 4 个 prompt 向量是**固定可学习参数**，language_id 等在推理时不使用 → 模型内部自动判断语种

## 目录结构

```
sensevoice/
├── python/
│   ├── export_onnx.py      # Step 1: PyTorch → ONNX（在开发机执行）
│   └── gen_bmodel.sh       # Step 2: ONNX → bmodel（在 TPU-MLIR Docker 内执行）
├── cpp/
│   ├── CMakeLists.txt
│   ├── build.sh            # 交叉编译脚本（依赖 sophon-cross-build Docker 镜像）
│   └── src/
│       ├── main.cpp
│       ├── sensevoice_config.h         # 配置结构体和常量
│       ├── sensevoice_inference.h/.cpp # BMRuntime 推理主类
│       ├── audio_frontend.h/.cpp       # Fbank + LFR 特征提取（依赖 kaldi-native-fbank）
│       └── tokenizer.h/.cpp            # CTC 贪心解码 + token 转文本
├── models/
│   ├── BM1684X/            # bmodel 文件（不入库，本地自行生成）
│   │   ├── sensevoice_small_F32.bmodel
│   │   ├── sensevoice_small_F16.bmodel
│   │   └── tokens.txt
│   └── onnx/               # 中间 ONNX 文件（不入库）
└── test_data/              # 测试音频（不入库）
```

## 快速开始

### 环境准备

1. **开发机**（WSL / Linux x86）
   - Python 3.8+，PyTorch，funasr，onnx，onnxsim
   - Docker（用于 bmodel 转换和交叉编译）

2. **第三方库**（gitignore，需手动准备）
   - `0_Toolkits/soc-sdk-sp4/`：Sophon SOC SDK
   - `1_third_party/kaldi_native_fbank/`：kaldi-native-fbank aarch64 静态库

   kaldi-native-fbank 交叉编译方法：
   ```bash
   git clone https://github.com/csukuangfj/kaldi-native-fbank /tmp/kaldi-native-fbank
   docker run --rm -v /tmp/kaldi-native-fbank:/src sophon-cross-build bash -c '
     mkdir -p /src/build-aarch64 && cd /src/build-aarch64
     cmake .. -DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc \
               -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++ \
               -DCMAKE_INSTALL_PREFIX=/src/install-aarch64 \
               -DKALDI_NATIVE_FBANK_BUILD_TESTS=OFF \
               -DKALDI_NATIVE_FBANK_BUILD_PYTHON=OFF \
               -DBUILD_SHARED_LIBS=OFF
     make -j$(nproc) && make install
   '
   cp /tmp/kaldi-native-fbank/build-aarch64/lib/libkaldi-native-fbank-core.a \
      1_third_party/kaldi_native_fbank/aarch64-linux/
   cp /tmp/kaldi-native-fbank/build-aarch64/lib/libkissfft-float.a \
      1_third_party/kaldi_native_fbank/aarch64-linux/
   cp -r /tmp/kaldi-native-fbank/install-aarch64/include/kaldi-native-fbank \
      1_third_party/kaldi_native_fbank/include/
   ```

### Step 1：导出 ONNX

```bash
cd sensevoice/python
python export_onnx.py
# 首次运行会自动从 ModelScope 下载模型权重
# 产物：models/onnx/sensevoice_small_sim.onnx
```

### Step 2：转换 bmodel

```bash
# 在 TPU-MLIR Docker 内执行（从仓库根目录）
docker run --rm \
  -v $(pwd):/workspace \
  -v $(pwd)/0_Toolkits:/toolkits \
  sophgo/tpuc_dev:latest \
  bash /workspace/sensevoice/python/gen_bmodel.sh F16

# 支持 F32（默认）或 F16
```

### Step 3：交叉编译 C++ 推理程序

```bash
# 先构建交叉编译镜像（只需一次）
docker build -t sophon-cross-build docker/

# 交叉编译
bash sensevoice/cpp/build.sh
# 产物：sensevoice/cpp/build/sensevoice_bm1684
```

### Step 4：部署到 BM1684X 板卡

```bash
# 上传二进制
scp sensevoice/cpp/build/sensevoice_bm1684 root@<board_ip>:/your/path/

# 上传模型文件（首次）
scp sensevoice/models/BM1684X/sensevoice_small_F16.bmodel root@<board_ip>:/your/path/models/
scp /path/to/tokens.txt root@<board_ip>:/your/path/models/

# 在板卡上运行
./sensevoice_bm1684 models/ test.wav F16
# 参数: <model_dir> <audio.wav> [F32|F16]
```

## 性能（BM1684X 实测，~5.6s 音频）

> 统计口径：特征提取 + TPU 推理，不含模型加载（实际部署时模型预加载到内存）

| 精度 | 特征提取 | TPU 推理 | 合计 | RTF |
|------|---------|---------|------|-----|
| F32 | ~34ms | ~155ms | ~189ms | **0.034** |
| F16 | ~34ms | ~20ms | ~54ms | **0.0095** |

- **F16 实时率约 105x**，推荐生产环境使用
- F16 与 F32 识别结果完全一致
- 特征提取（CPU）在 F16 模式下占总耗时 63%，是主要瓶颈

## 输出示例

```
[Init] SenseVoice loaded (F16) from models/
[Timing] audio=5611.5ms  feat=33.7ms  infer=19.5ms  total=53.2ms  RTF=0.0095

--- SenseVoice Result ---
Text     : 对我做了介绍啊，那么我想说的是呢，大家如果对我的研究感兴趣呢。
Language : <|zh|>
Emotion  : <|NEUTRAL|>
Event    : <|Speech|>
```

## 关键技术说明

- **自动语种识别**：4 个 prompt 向量是固定可学习参数，forward 内部不使用 language_id，语种由模型从音频内容自动判断，结果从输出的前 4 帧 token 解码得到
- **音频时长限制**：模型固定输入 166 帧（~10s），超出截断，不足补零
- **tokens.txt 来源**：从 ModelScope `iic/SenseVoiceSmall` 模型目录中获取
- **kaldi-native-fbank**：需要同时链接 `libkaldi-native-fbank-core.a` 和 `libkissfft-float.a`（kissfft 是其内部 FFT 依赖）
