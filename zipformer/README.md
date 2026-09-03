# Zipformer bilingual BM1684X 部署

本目录部署 `csukuangfj/k2fsa-zipformer-bilingual-zh-en-t`，采用 **自定义流式协议**（encoder 窗口 103 帧、步进 96 帧，非官方 icefall exp/32、64 或 96 ONNX 协议）。固定 16 kHz、80 维 Fbank；输出 `[1,24,256]`，并维护 35 个 float32 streaming states。每个 state 的内部首维 `2` 是 stack layer 维，不是 batch；Sophon ONNX 边界在其外增加显式 batch=1，wrapper 内以 `squeeze(0)`/`unsqueeze(0)` 保持原始状态语义。

Sophon 支持 Gather，因此 Decoder 不走 CPU Embedding workaround：逻辑输入为最近两个 token IDs `int64 [1,2]`，图内执行 `Embedding/Gather -> Conv -> ReLU`，输出 `[1,512]`。Joiner 单独保留 `encoder_proj + decoder_proj + output_linear`，不存在重复 projection。Decoder int64 Gather ONNX 已在 TPU-MLIR 1.28.1 编译并通过数值验证；BM1684X bmodel metadata 将该输入降为 `BM_INT32`，manifest 通过 `runtime_dtype: int32` 显式记录转换。Python/C++ 必须在运行时边界转换并校验，不能静默改成 CPU Embedding。

Frontend 固定参数记录在 `configs/tensor_manifest.json`：10 ms 帧移、25 ms 帧长、dither 0、preemphasis 0.97、Povey 窗、`snip_edges=false`、去 DC、low frequency 20、high frequency -400、log power Fbank，以及 1.03 秒尾部静音。

## 模型资产与协议

HF 模型固定 revision：`e2382758de9a0219b4efe682b95af30b399db3b8`。资产分为三层，避免空目录和重复存储：

```text
zipformer/configs/tokens.txt                         # Git 跟踪，运行时词表
zipformer/models/BM1684X/zipformer_*_f16.bmodel    # 本地运行资产，Git 忽略
zipformer/test_data/{test_en.wav,test_zh.wav,golden.json}
zipformer/assets/{checkpoint,onnx-safe-matmul,icefall,test_wavs} # 构建资产，Git 忽略
```

`ZIPFORMER_ASSET_ROOT` 默认使用仓库内 **`zipformer/assets/`**（不再散落在 `$HOME/.cache/zipformer-assets`）。checkpoint、ONNX、icefall 源码与官方 6 条测试 WAV 作为构建资产集中于此，不进入 Git；MLIR、NPZ、profile 和 tensor dump 是可再生成产物，不保留。两个测试 WAV、正式词表及对应 token/text golden 已随项目固定，`golden.json` 同时记录 SHA256。

官方 README 记录的 `600f387-dirty` 不是可解析的公开提交；本次使用同日期最接近的官方 icefall commit `cba6ecc1d1c63da6cc73988cebe0a0189935a8df`（2023-02-09）。

流式协议具有 `103 -> 24` 的时间边界，但 attention cache context 为 `[128,64,32,16,64]`，不能替换成官方任一导出：

- exp/32：`39 -> 8`，context 从 64 开始；
- exp/64：`71 -> 16`，context 从 128 开始；
- exp/96：`103 -> 24`，context 从 192 开始。

官方 ONNX 仅用于同 checkpoint 的接口参考和 ORT smoke，不作为 BM1684X 主部署图。

## 导出流式 ONNX

流式模型定义已 vendor 在仓库 `zipformer/python/streaming_zipformer.py`，导出不依赖任何外部私有目录。资产根目录用 `ZIPFORMER_ASSET_ROOT` 指定，默认在仓库内 `zipformer/assets/`：

```sh
export ZIPFORMER_ASSET_ROOT="${ZIPFORMER_ASSET_ROOT:-$(pwd)/zipformer/assets}"

conda run -n sophon-zipformer --no-capture-output python zipformer/python/export_onnx.py \
  --icefall-root "$ZIPFORMER_ASSET_ROOT/icefall" \
  --checkpoint "$ZIPFORMER_ASSET_ROOT/checkpoint/pretrained.pt" \
  --output-dir "$ZIPFORMER_ASSET_ROOT/onnx-safe-matmul"
```

导出器会逐一审计 Encoder、Decoder 和 Joiner 的 checkpoint missing/unexpected keys，然后生成静态 batch=1、opset 17 的三个模型。

Encoder 的相对位置 attention 将每个输出行所需的 position slice 前推到 MatMul 输入，再执行等价的 rank-3 MatMul；这是为了规避 TPU-MLIR 1.28.1 `SplitMatMulPattern` 把 Slice 输出 shape 错赋给原 MatMul 的 codegen 缺陷。该重写已通过 PyTorch 原实现/新实现双 chunk、36 个输出回归。

- `encoder_103_24_256.onnx`
- `decoder_sophon_tokens.onnx`
- `joiner_streaming.onnx`

ONNX 接口必须与 `configs/tensor_manifest.json` 精确一致：

- Encoder：`x, cached_* -> encoder_out, new_cached_*`（36 入、36 出）；
- Decoder：`token_ids -> decoder_out`；
- Joiner：`enc_out, dec_out -> logit`。

检查接口（在公共 TPU-MLIR 容器内执行）：

```sh
ASSET_ROOT="${ZIPFORMER_ASSET_ROOT:-$(pwd)/zipformer/assets}"
python3 zipformer/python/tpumlir_args.py --network encoder \
  --manifest zipformer/configs/tensor_manifest.json \
  --onnx "$ASSET_ROOT/onnx-safe-matmul/encoder_103_24_256.onnx"
python3 zipformer/python/tpumlir_args.py --network decoder \
  --manifest zipformer/configs/tensor_manifest.json \
  --onnx "$ASSET_ROOT/onnx-safe-matmul/decoder_sophon_tokens.onnx"
python3 zipformer/python/tpumlir_args.py --network joiner \
  --manifest zipformer/configs/tensor_manifest.json \
  --onnx "$ASSET_ROOT/onnx-safe-matmul/joiner_streaming.onnx"
```

PyTorch/ORT 数值回归必须覆盖 Encoder 首 chunk 和 35 个状态续传后的第二 chunk、Decoder contexts `[0,0]`、`[1,2]`、`[6253,42]`，以及多组 Joiner 输入。F32 初始阈值为 `atol=1e-4, rtol=1e-3`。随机 Fbank 只验证模型协议，不能替代真实 WAV frontend golden。

## 公共 TPU-MLIR 环境

Zipformer 不维护私有 Dockerfile，统一使用仓库 `3_docker/`：

```sh
./3_docker/build_tpumlir.sh
./3_docker/run_docker.sh          # 默认容器名 sophon-tpumlir-v128
```

公共派生镜像实际工具版本为 `TPU-MLIR v1.28.1-20260429`，基于 `sophgo/tpuc_dev:v3.4` 安装仓库固定 wheel 和 `onnx==1.14.1`。容器内 `/workspace` 对应**仓库根目录**（这是 `run_docker.sh` 创建的唯一挂载）；镜像不包含 HF token、checkpoint、模型或测试数据。

`zipformer/assets/` 虽被 gitignore，但它就在仓库内，因此**在容器里直接可见**（`/workspace/zipformer/assets/`），不需要 `docker cp` 进出容器。

`python/gen_bmodel.sh` 会从 manifest index 生成输入 shapes，并在转换前通过 ONNX checker 严格检查输入和输出的 name、shape、dtype，协议不一致会 fail-fast。中间产物（mlir/npz/prototxt/layer_group_config）由脚本内的 `mktemp -d` + `trap` 落到容器临时目录并在结束时自动清理，不会散落到调用者的当前目录——这是本仓库编译脚本的隔离范本。

```sh
# F16 三模型（交付档）。2026-09-03 从 checkpoint 重编验证：
# encoder 44M / joiner 12M / decoder 11M
docker exec sophon-tpumlir-v128 bash -c 'cd /workspace && \
  bash zipformer/python/gen_bmodel.sh --network joiner  --quantize F16 \
    --onnx zipformer/assets/onnx-safe-matmul/joiner_streaming.onnx \
    --manifest zipformer/configs/tensor_manifest.json \
    --output /workspace/zipformer/models/BM1684X/zipformer_joiner_f16.bmodel && \
  bash zipformer/python/gen_bmodel.sh --network decoder --quantize F16 \
    --onnx zipformer/assets/onnx-safe-matmul/decoder_sophon_tokens.onnx \
    --manifest zipformer/configs/tensor_manifest.json \
    --output /workspace/zipformer/models/BM1684X/zipformer_decoder_f16.bmodel \
    --decoder-disable-layer-group && \
  bash zipformer/python/gen_bmodel.sh --network encoder --quantize F16 \
    --onnx zipformer/assets/onnx-safe-matmul/encoder_103_24_256.onnx \
    --manifest zipformer/configs/tensor_manifest.json \
    --output /workspace/zipformer/models/BM1684X/zipformer_encoder_f16.bmodel'
```

> ⚠️ **输出文件名必须是小写 `_f16`**。磁盘产物、`deploy_to_board.sh`（第 41-43 行的上传清单与第 76 行的板上运行命令）以及本文件的板端调用示例全部使用 `zipformer_{encoder,decoder,joiner}_f16.bmodel`；写成 `_F16` 会导致重编出的产物部署脚本找不到。

**`--disable_layer_group` 由脚本决定，不需要在命令行统一传**（`gen_bmodel.sh:45`）：

| 网络 | 是否添加 | 方式 |
|---|---|---|
| encoder | 总是添加 | 脚本内自动 |
| decoder | 仅当显式传 `--decoder-disable-layer-group` | 命令行 |
| joiner | 不添加 | — |

**两个阶段不要混淆**：

- **F32 验证阶段**（4 步，用于证明协议与数值正确）：Joiner F32 → Decoder Gather F32 → Encoder F32 单 chunk → Encoder F32 双 chunk，全部通过后才进入 F16。
- **F16 交付阶段**（3 步，即上面的命令）：Joiner → Decoder → Encoder，产物为部署用的三个 `_f16.bmodel`。

当前 F16 三模型已通过 ORT tensor 回归，Encoder 使用独立的 ORT/bmodel 双 chunk 状态链。板上 6 条官方 WAV（3.9–17.6 秒）由 C++ `kaldi-native-fbank` 生成特征，帧数均与同参数 `torchaudio.compliance.kaldi` 一致；六条中的最坏 MAE 为 `4.13e-6`、最大绝对误差为 `5.69e-4`、相对 L2 为 `1.25e-6`。对应 F16 token 均与 F32 ORT 完全一致，因此 C++ `kaldi-native-fbank` 已冻结为 WAV CLI frontend。

**RTF 必须区分 warm 与首轮**（数值以 `PERF_SUMMARY.md` 为准）：

- **warm**：17.6 秒样本 warm-up 后各重复 5 次，token 均稳定 —— F16 模型链平均 RTF **0.0410**，chunk P50/P95 为 35.98/37.71 ms；F32 平均 RTF **0.0513**，P50/P95 为 45.07/46.55 ms。模型加载和 frontend 不计入。
- **首轮**：8 条验收样本的 C++ 首轮模型链 RTF 为 **0.0636–0.2264**，含三网首次运行与 BMRuntime 调度，显著高于 warm，均低于实时阈值 1.0。

## 板上 Python 参考推理

`python/infer_board.py` 加载三个独立 bmodel，并按 manifest 显式维护 35 个 streaming states。真实 bmodel graph 名和结构性输出后缀会被一对一校验，不按输出顺序猜测。

> ⚠️ **WAV 入口当前在板卡上不可用**：它要求板上同时具备 PyTorch 与 `kaldifeat`。实测板卡 python3.8 有 torch / torchaudio 2.4.1、numpy、scipy、tokenizers，**但没有 `kaldifeat`**（`import kaldifeat` → ModuleNotFoundError）。缺依赖时程序明确拒绝使用不一致 fallback，因此会直接报错退出。
>
> 两条可行路径：先在板上装 `kaldifeat`，或改用下面的 `--features-npy` 入口（在开发机生成 Fbank 后上传）。**板端正式验收以 C++ CLI 为准**（它自带 `kaldi-native-fbank`，不依赖 Python 侧），Python 参考推理只用于排查 C++ 与 bmodel 之间的分歧。

WAV 入口（需先补 `kaldifeat`）：

```sh
python3 zipformer/python/infer_board.py \
  --encoder zipformer_encoder_f16.bmodel \
  --decoder zipformer_decoder_f16.bmodel \
  --joiner zipformer_joiner_f16.bmodel \
  --manifest zipformer/configs/tensor_manifest.json \
  --tokens tokens.txt \
  --wav test.wav
```

也可传入外部生成且已包含尾部静音的 Fbank，用于把 frontend 对齐与模型链验证解耦（**这条不需要 `kaldifeat`**）：

```sh
python3 zipformer/python/infer_board.py \
  --encoder zipformer_encoder_f16.bmodel \
  --decoder zipformer_decoder_f16.bmodel \
  --joiner zipformer_joiner_f16.bmodel \
  --manifest zipformer/configs/tensor_manifest.json \
  --tokens tokens.txt \
  --features-npy fbank.npy
```

上面用的是交付档的 F16 bmodel（本机与板上部署都只有这三个）。若要复现 F32 验证档，需先按上文「F32 验证阶段」编译出 `_f32.bmodel` 再替换参数。

`--features-npy` 只接受 finite `float32 [N,80]`，不会隐式转换 dtype，也不会重复添加尾部静音。该入口不能证明特征生成器与冻结 frontend 等价；正式 WAV 验收仍需逐元素对齐 `kaldifeat` 或最终 C++ `kaldi-native-fbank` 实现。

## C++ BMRuntime WAV CLI

生产 CLI 只在 `ZIPFORMER_WITH_BM=ON` 时生成，并强制要求 Sophon SDK、aarch64 `kaldi-native-fbank` 和 kissfft。它直接读取 16 kHz PCM16 WAV，在 `ComputeStrictFbank` 内添加一次 1.03 秒时域静音，再按 103 帧窗口、96 帧步进切分特征；不会重复添加特征域静音。

Encoder 的 35 个 `cached_* -> new_cached_*` state 通过名称一一配对，并保留在两组 device buffer 中交替使用。每个 chunk 只下载 `encoder_out`；Decoder 在 host 边界检查 token 范围并显式转换为 BMRuntime `int32`。所有网络的 tensor 名、runtime index、shape、dtype 和字节数均与 manifest、bmodel metadata 逐项校验。

使用公共 Ubuntu 20.04/GCC 9 镜像交叉构建：

```sh
docker build -t sophon-cross-build -f 3_docker/Dockerfile.cross-build 3_docker
bash zipformer/cpp/build.sh cross     # 产物：zipformer/cpp/build-aarch64/zipformer_cli
```

主机逻辑测试使用：

```sh
bash zipformer/cpp/build.sh host      # 产物：zipformer/cpp/build-host/
```

板端调用方式（二进制位于 `cpp/build-aarch64/`）。**注意板上目录布局与仓库内不同**：`deploy_to_board.sh` 把三个 bmodel 平铺上传到 `${BOARD_DIR}/models/`（不带 `BM1684X/` 子目录），configs 上传到 `${BOARD_DIR}/configs/`：

```sh
cd /data/zipformer
./zipformer_cli \
  configs/tensor_manifest.json test_data/test_en.wav \
  models/zipformer_encoder_f16.bmodel \
  models/zipformer_decoder_f16.bmodel \
  models/zipformer_joiner_f16.bmodel \
  configs/tokens.txt
```

（仓库内对应的源文件在 `zipformer/models/BM1684X/` 与 `zipformer/configs/`，不要把这一层路径带到板端命令里。）

输出包含 Fbank shape、chunk 数、完整 token IDs、文本、原始音频时长、模型链延迟和 RTF。模型链延迟包含三网首次运行及 BMRuntime 调度，不包含 WAV/Fbank 和 bmodel 加载，因此首轮数值高于 Python warm-up 后的纯模型 benchmark。

板上最终 F16 验收覆盖 6 条官方 WAV 和 2 条中英文补充 WAV，长度 3.9–17.6 秒、最多 20 chunks：**8/8 token 序列与 Python Sail 完全一致**。最长样本 C++ 首轮模型链 RTF 为 0.0636；8 条首轮 RTF 范围为 0.0636–0.2264，均低于实时阈值 1.0。

> **可复现性**：这 8 条中只有 2 条（`test_data/test_en.wav`、`test_data/test_zh.wav`）连同其 token/文本 golden 一起入库（`test_data/golden.json` 的 `samples` 只有这 2 项）。另外 6 条官方 WAV 位于 `zipformer/assets/test_wavs/`，属 gitignore 的构建资产。因此**从归档能直接复现的是 2/8**；要复现完整 8/8，需先按上文「模型资产与协议」的 HF revision 与 icefall commit 重建 `zipformer/assets/`。

## 部署到 BM1684X 板卡

```sh
BOARD_IP=<board_ip> bash zipformer/deploy_to_board.sh --test
```

脚本上传 C++ CLI、三个 F16 bmodel、`configs/` 和测试 WAV，并逐项执行 md5 校验。支持 `BOARD_USER`、`BOARD_PORT`、`BOARD_DIR`、`MODEL_DIR`、`CONFIG_DIR`、`BINARY`、`TEST_AUDIO` 和 `BOARD_PASS` 环境变量。

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

模型、大型中间 tensor、MLIR 和构建产物均由 `.gitignore` 排除。入库的是：正式词表与 frontend 参数（`configs/tokens.txt`、`configs/tensor_manifest.json`）、两条中英文 WAV 及其 token/文本 golden（`test_data/golden.json`，同时记录资产 SHA256）、以及 `python/streaming_zipformer.py` 等已 vendor 的模型定义。本机三个 F16 bmodel 保留在 `models/BM1684X/`，但不进入 Git。
