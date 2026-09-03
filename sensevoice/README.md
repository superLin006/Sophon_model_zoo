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
   - 导出环境：独立 conda 环境（实测 Python 3.10 / torch 2.11.0+cpu / transformers 5.14.1 / onnx 1.21 + onnxsim）；从仓库根目录执行 `conda create -n sophon-sensevoice python=3.10 -y`、`conda run -n sophon-sensevoice python -m pip install --upgrade pip`、`conda run -n sophon-sensevoice python -m pip install -r sensevoice/requirements.txt`
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
conda run -n sophon-sensevoice --no-capture-output python sensevoice/python/export_onnx.py
# 首次运行会自动从 ModelScope 下载模型权重
# 产物：sensevoice/models/onnx/sensevoice_small_sim.onnx
```

### Step 2：转换 bmodel

```bash
# 公共 TPU-MLIR 容器已启动后，从仓库根目录执行
docker exec sophon-tpumlir-v128 bash /workspace/sensevoice/python/gen_bmodel.sh F16

# 支持 F32（默认）或 F16
```

### Step 3：交叉编译 C++ 推理程序

```bash
# 先构建交叉编译镜像（只需一次）
docker build -t sophon-cross-build \
  -f 3_docker/Dockerfile.cross-build 3_docker

# 交叉编译
bash sensevoice/cpp/build.sh
# 产物：sensevoice/cpp/build/sensevoice_bm1684
```

### Step 4：部署到 BM1684X 板卡

部署脚本会上传二进制、指定精度的 bmodel、`tokens.txt` 和可选测试音频，并在上传后执行 md5 校验。默认使用 SSH key，也可以通过 `BOARD_PASS` 临时提供密码。

```bash
BOARD_IP=<board_ip> \
  bash sensevoice/deploy_to_board.sh F16 --test
```

可用环境变量：

- `BOARD_IP`：必填，板卡地址
- `BOARD_USER`：默认 `root`
- `BOARD_PORT`：默认 `22`
- `BOARD_DIR`：默认 `/data/sensevoice`
- `MODEL_DIR`：默认 `sensevoice/models/BM1684X`
- `BINARY`：默认 `sensevoice/cpp/build/sensevoice_bm1684`
- `TEST_AUDIO`：默认 `sensevoice/test_data/test_zh.wav`

其中 `tokens.txt` 已随仓库放置在：

```text
sensevoice/models/BM1684X/tokens.txt
```

不执行 smoke test 时去掉 `--test`。板卡上的手动运行命令为：

```bash
./sensevoice_bm1684 models test_data/test.wav F16
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
