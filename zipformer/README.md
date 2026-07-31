# Zipformer bilingual BM1684X 部署

本目录部署 `csukuangfj/k2fsa-zipformer-bilingual-zh-en-t`，采用 **MTK-derived 自定义流式协议**，不是官方 icefall exp/32、64 或 96 ONNX 协议。固定使用 16 kHz、80 维 Fbank；Encoder 窗口 103 帧、步进 96 帧，输出 `[1,24,256]`，并维护 35 个 float32 streaming states。每个 state 的 MTK 内部首维 `2` 是 stack layer 维，不是 batch；Sophon ONNX 边界在其外增加显式 batch=1，wrapper 内以 `squeeze(0)`/`unsqueeze(0)` 保持原始状态语义。

Sophon 支持 Gather，因此 Decoder 不继承 MTK 的 CPU Embedding workaround：逻辑输入为最近两个 token IDs `int64 [1,2]`，图内执行 `Embedding/Gather -> Conv -> ReLU`，输出 `[1,512]`。Joiner 单独保留 `encoder_proj + decoder_proj + output_linear`，不存在重复 projection。Decoder int64 Gather ONNX 已在 TPU-MLIR 1.28.1 编译并通过数值验证；BM1684X bmodel metadata 将该输入降为 `BM_INT32`，manifest 通过 `runtime_dtype: int32` 显式记录转换。Python/C++ 必须在运行时边界转换并校验，不能静默改成 CPU Embedding。

Frontend 固定参数记录在 `configs/tensor_manifest.json`：10 ms 帧移、25 ms 帧长、dither 0、preemphasis 0.97、Povey 窗、`snip_edges=false`、去 DC、low frequency 20、high frequency -400、log power Fbank，以及 1.03 秒尾部静音。

## 模型资产与协议

HF 模型固定 revision：`e2382758de9a0219b4efe682b95af30b399db3b8`。资产分为三层，避免空目录和重复存储：

```text
zipformer/configs/tokens.txt                         # Git 跟踪，运行时词表
zipformer/models/BM1684X/zipformer_*_f16.bmodel    # 本地运行资产，Git 忽略
zipformer/test_data/{test_en.wav,test_zh.wav,golden.json}
$ZIPFORMER_ASSET_ROOT/{checkpoint,onnx-safe-matmul,icefall} # 外部构建资产
```

`ZIPFORMER_ASSET_ROOT` 默认使用 `$HOME/.cache/zipformer-assets`。checkpoint、ONNX 和 icefall 源码不复制进仓库；MLIR、NPZ、profile 和 tensor dump 是可再生成产物，不保留。两个测试 WAV、正式词表及对应 token/text golden 已随项目固定，`golden.json` 同时记录 SHA256。

官方 README 记录的 `600f387-dirty` 不是可解析的公开提交；本次使用同日期最接近的官方 icefall commit `cba6ecc1d1c63da6cc73988cebe0a0189935a8df`（2023-02-09）。

MTK-derived 协议具有 `103 -> 24` 的时间边界，但 attention cache context 为 `[128,64,32,16,64]`，不能替换成官方任一导出：

- exp/32：`39 -> 8`，context 从 64 开始；
- exp/64：`71 -> 16`，context 从 128 开始；
- exp/96：`103 -> 24`，context 从 192 开始。

官方 ONNX 仅用于同 checkpoint 的接口参考和 ORT smoke，不作为 BM1684X 主部署图。

## 导出 MTK-derived ONNX

```sh
conda run -n sophon-whisper --no-capture-output python zipformer/python/export_onnx.py \
  --icefall-root "$HOME/.cache/zipformer-assets/icefall" \
  --mtk-python-dir /home/xh/itc_project/MTK_model_zoo/zipformer/mtk/python \
  --checkpoint "$HOME/.cache/zipformer-assets/checkpoint/pretrained.pt" \
  --output-dir "$HOME/.cache/zipformer-assets/onnx-safe-matmul"
```

导出器会逐一审计 Encoder、Decoder 和 Joiner 的 checkpoint missing/unexpected keys，然后生成静态 batch=1、opset 17 的三个模型。默认资产根目录为：

```sh
export ZIPFORMER_ASSET_ROOT="${ZIPFORMER_ASSET_ROOT:-$HOME/.cache/zipformer-assets}"
```

Encoder 的相对位置 attention 将每个输出行所需的 position slice 前推到 MatMul 输入，再执行等价的 rank-3 MatMul；这是为了规避 TPU-MLIR 1.28.1 `SplitMatMulPattern` 把 Slice 输出 shape 错赋给原 MatMul 的 codegen 缺陷。该重写已通过 PyTorch 原实现/新实现双 chunk、36 个输出回归。

- `encoder_mtk_103_24_256.onnx`
- `decoder_sophon_tokens.onnx`
- `joiner_mtk.onnx`

ONNX 接口必须与 `configs/tensor_manifest.json` 精确一致：

- Encoder：`x, cached_* -> encoder_out, new_cached_*`（36 入、36 出）；
- Decoder：`token_ids -> decoder_out`；
- Joiner：`enc_out, dec_out -> logit`。

检查接口（在公共 TPU-MLIR 容器内执行）：

```sh
ASSET_ROOT="${ZIPFORMER_ASSET_ROOT:-$HOME/.cache/zipformer-assets}"
python3 zipformer/python/tpumlir_args.py --network encoder \
  --manifest zipformer/configs/tensor_manifest.json \
  --onnx "$ASSET_ROOT/onnx-safe-matmul/encoder_mtk_103_24_256.onnx"
python3 zipformer/python/tpumlir_args.py --network decoder \
  --manifest zipformer/configs/tensor_manifest.json \
  --onnx "$ASSET_ROOT/onnx-safe-matmul/decoder_sophon_tokens.onnx"
python3 zipformer/python/tpumlir_args.py --network joiner \
  --manifest zipformer/configs/tensor_manifest.json \
  --onnx "$ASSET_ROOT/onnx-safe-matmul/joiner_mtk.onnx"
```

PyTorch/ORT 数值回归必须覆盖 Encoder 首 chunk 和 35 个状态续传后的第二 chunk、Decoder contexts `[0,0]`、`[1,2]`、`[6253,42]`，以及多组 Joiner 输入。F32 初始阈值为 `atol=1e-4, rtol=1e-3`。随机 Fbank 只验证模型协议，不能替代真实 WAV frontend golden。

## 公共 TPU-MLIR 环境

Zipformer 不维护私有 Dockerfile，统一使用仓库 `3_docker/`：

```sh
./3_docker/build_tpumlir.sh
./3_docker/run_docker.sh
```

公共派生镜像实际工具版本为 `TPU-MLIR v1.28.1-20260429`，基于 `sophgo/tpuc_dev:v3.4` 安装仓库固定 wheel 和 `onnx==1.14.1`。容器内 `/workspace` 对应仓库根目录；镜像不包含 HF token、checkpoint、模型或测试数据。若需挂载外部 cache，只挂具体目录并使用只读模式。

`python/gen_bmodel.sh` 会从 manifest index 生成输入 shapes，并在转换前通过 ONNX checker 严格检查输入和输出的 name、shape、dtype。协议不一致会 fail-fast。

```sh
zipformer/python/gen_bmodel.sh \
  --network joiner \
  --quantize F32 \
  --onnx "$HOME/.cache/zipformer-assets/onnx-safe-matmul/joiner_mtk.onnx" \
  --output /workspace/zipformer/models/BM1684X/zipformer_joiner_F32.bmodel
```

编译顺序固定为 Joiner F32、Decoder Gather F32、Encoder F32 单 chunk、Encoder F32 双 chunk。Encoder 默认添加 `--disable_layer_group`；Decoder 可显式添加 `--decoder-disable-layer-group`；Joiner默认不添加。F32 全部通过后才进入 F16。

当前 F16 三模型已通过 ORT tensor 回归，Encoder 使用独立的 ORT/bmodel 双 chunk 状态链。板上 6 条官方 WAV（3.9–17.6 秒）由 C++ `kaldi-native-fbank` 生成特征，帧数均与同参数 `torchaudio.compliance.kaldi` 一致；六条中的最坏 MAE 为 `4.13e-6`、最大绝对误差为 `5.69e-4`、相对 L2 为 `1.25e-6`。对应 F16 token 均与 F32 ORT 完全一致，因此 C++ `kaldi-native-fbank` 已冻结为 WAV CLI frontend。

17.6 秒样本 warm-up 后各重复 5 次，token 均稳定：F16 模型链平均 RTF 0.0410，chunk P50/P95 为 35.98/37.71 ms；F32 平均 RTF 0.0513，P50/P95 为 45.07/46.55 ms。模型加载和 frontend 不计入这些数字。

## 板上 Python 参考推理

`python/infer_board.py` 加载三个独立 bmodel，并按 manifest 显式维护 35 个 streaming states。真实 bmodel graph 名和结构性输出后缀会被一对一校验，不按输出顺序猜测。

WAV 路径要求板上同时具备 PyTorch 与 `kaldifeat`；缺依赖时程序明确拒绝使用不一致 fallback：

```sh
python3 zipformer/python/infer_board.py \
  --encoder encoder_f32.bmodel \
  --decoder decoder_f32.bmodel \
  --joiner joiner_f32.bmodel \
  --manifest zipformer/configs/tensor_manifest.json \
  --tokens tokens.txt \
  --wav test.wav
```

也可传入外部生成且已包含尾部静音的 Fbank，用于把 frontend 对齐与模型链验证解耦：

```sh
python3 zipformer/python/infer_board.py \
  --encoder encoder_f32.bmodel \
  --decoder decoder_f32.bmodel \
  --joiner joiner_f32.bmodel \
  --manifest zipformer/configs/tensor_manifest.json \
  --tokens tokens.txt \
  --features-npy fbank.npy
```

`--features-npy` 只接受 finite `float32 [N,80]`，不会隐式转换 dtype，也不会重复添加尾部静音。该入口不能证明特征生成器与冻结 frontend 等价；正式 WAV 验收仍需逐元素对齐 `kaldifeat` 或最终 C++ `kaldi-native-fbank` 实现。

## C++ BMRuntime WAV CLI

生产 CLI 只在 `ZIPFORMER_WITH_BM=ON` 时生成，并强制要求 Sophon SDK、aarch64 `kaldi-native-fbank` 和 kissfft。它直接读取 16 kHz PCM16 WAV，在 `ComputeStrictFbank` 内添加一次 1.03 秒时域静音，再按 103 帧窗口、96 帧步进切分特征；不会重复添加特征域静音。

Encoder 的 35 个 `cached_* -> new_cached_*` state 通过名称一一配对，并保留在两组 device buffer 中交替使用。每个 chunk 只下载 `encoder_out`；Decoder 在 host 边界检查 token 范围并显式转换为 BMRuntime `int32`。所有网络的 tensor 名、runtime index、shape、dtype 和字节数均与 manifest、bmodel metadata 逐项校验。

使用公共 Ubuntu 20.04/GCC 9 镜像交叉构建：

```sh
docker build -t sophon-cross-build -f 3_docker/Dockerfile.cross-build .
zipformer/cpp/build.sh cross
```

主机逻辑测试使用：

```sh
zipformer/cpp/build.sh host
```

板端调用方式：

```sh
./zipformer_cli \
  configs/tensor_manifest.json test_data/test_en.wav \
  models/BM1684X/zipformer_encoder_f16.bmodel \
  models/BM1684X/zipformer_decoder_f16.bmodel \
  models/BM1684X/zipformer_joiner_f16.bmodel \
  configs/tokens.txt
```

输出包含 Fbank shape、chunk 数、完整 token IDs、文本、原始音频时长、模型链延迟和 RTF。模型链延迟包含三网首次运行及 BMRuntime 调度，不包含 WAV/Fbank 和 bmodel 加载，因此首轮数值高于 Python warm-up 后的纯模型 benchmark。

板上最终 F16 验收覆盖 6 条官方 WAV 和 2 条中英文补充 WAV，长度 3.9–17.6 秒、最多 20 chunks：**8/8 token 序列与 Python Sail 完全一致**。最长样本 C++ 首轮模型链 RTF 为 0.0636；8 条首轮 RTF 范围为 0.0636–0.2264，均低于实时阈值 1.0。

## 测试

```sh
python3 -m unittest discover -s zipformer/python/test -p 'test_*.py' -v
python3 zipformer/python/state_layout.py \
  --manifest zipformer/configs/tensor_manifest.json

cmake -S zipformer/cpp -B /tmp/zipformer-host \
  -DZIPFORMER_WITH_BM=OFF -DZIPFORMER_BUILD_FBANK=OFF
cmake --build /tmp/zipformer-host -j2
ctest --test-dir /tmp/zipformer-host --output-on-failure
```

模型、大型中间 tensor、MLIR 和构建产物均由 `.gitignore` 排除。正式词表、两条中英文 WAV 与 `test_data/golden.json` 进入版本管理；本机三个 F16 bmodel保留在 `models/BM1684X/`，但不进入 Git。
