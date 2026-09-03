# Whisper — BM1684X 移植

Whisper 语音识别移植到 Sophon BM1684X，支持中英文。
支持 **base**（6 层，512 维）与 **large-v3-turbo**（32 层 encoder / 4 层 decoder，1280 维）两个模型，精度 F32 / F16 / W4F16。
C++ 推理运行时从 bmodel 自动读取维度，一套代码两个模型通用。

## 快速开始

### 环境依赖

- 开发机：独立 conda 环境（Python 3.10、PyTorch、openai-whisper、onnx）；从仓库根目录执行 `conda create -n sophon-whisper python=3.10 -y`、`conda run -n sophon-whisper python -m pip install --upgrade pip`、`conda run -n sophon-whisper python -m pip install -r whisper/requirements.txt`
- `0_Toolkits/soc-sdk-sp4/`：Sophon SOC SDK（bmruntime 头文件 + so）
- `1_third_party/fftw/`：fftw3f aarch64 静态库（Mel 特征计算）

### Step 1：导出 ONNX

```bash
# base（默认）
conda run -n sophon-whisper --no-capture-output python whisper/python/export_onnx.py

# large-v3-turbo（encoder FP32 ≈2.4GB，导出峰值内存高）
# 权重下载：openai 官方 CDN 对某些网络不可达，且 whisper 的 load_model 会校验官方 URL 的 SHA，
# 因此第三方镜像下载的文件若 SHA 不匹配会被直接拒绝。可行做法是从 ModelScope 的
# iic/Whisper-large-v3-turbo 取 large-v3-turbo.pt（标准 OpenAI checkpoint 格式，可 torch.load），
# 再用 --checkpoint 直接加载本地权重，绕过 URL 校验：
curl -fL 'https://www.modelscope.cn/models/iic/Whisper-large-v3-turbo/resolve/master/large-v3-turbo.pt' \
  -o "$HOME/.cache/whisper/large-v3-turbo.pt"
conda run -n sophon-whisper --no-capture-output python whisper/python/export_onnx.py \
  --model large-v3-turbo --checkpoint "$HOME/.cache/whisper/large-v3-turbo.pt" \
  --asset_dir whisper/compile/tmp/turbo_assets
```

内存：建议 `.wslconfig` 配 `[wsl2] memory=14GB`；实机 `free -g` 约 13GB 可用，turbo encoder 全量导出已在此配置下跑通。不足时 OOM killer 报 **Exit 137**（不是 timeout）。

ONNX 产物在 `models/onnx/`。

> ⚠️ **turbo 的部署资产需要手工拷贝一步，脚本不会代劳**（此前无文档记载）：
> - `export_onnx.py --asset_dir` 只把 `mel_128_filters.txt` / `positional_embedding.npy` / `vocab.txt` 写到 `whisper/compile/tmp/turbo_assets/`
> - `gen_bmodel_turbo.sh` **只编译 bmodel，不复制这三个资产**
> - 而 `deploy_to_board.sh` 是从 `models/BM1684X_turbo/` 与 `models/BM1684X_turbo_w4f16/` 读取资产的
>
> 因此导出后必须手工拷贝，否则一旦按规范清理 `compile/tmp/`，重新导出就会让部署缺资产：
> ```bash
> for d in whisper/models/BM1684X_turbo whisper/models/BM1684X_turbo_w4f16; do
>   cp whisper/compile/tmp/turbo_assets/{mel_128_filters.txt,positional_embedding.npy,vocab.txt} "$d/"
> done
> ```
> turbo 用 128 维 mel、base 用 80 维 mel，**两套资产不可混用**（base 的在 `models/BM1684X/`）。

> ⚠️ Docker 生成 `whisper/compile/tmp` 后宿主机可能无写权限，若 export 写 turbo_assets 报 `PermissionError`，先执行 `docker run --rm -v "$PWD":/repo sophon-cross-build bash -lc 'chown -R $(id -u):$(id -g) /repo/whisper/compile/tmp'`（用 `$(id -u)` 而非写死 uid，换机器才成立）。

### Step 2：编译 bmodel

```bash
# base（在 TPU-MLIR Docker 内）
docker exec sophon-tpumlir-v128 bash /workspace/whisper/python/gen_bmodel.sh F16

# turbo F16 / W4F16
docker exec sophon-tpumlir-v128 bash /workspace/whisper/python/gen_bmodel_turbo.sh F16
docker exec sophon-tpumlir-v128 bash /workspace/whisper/python/gen_bmodel_turbo.sh W4F16
```

> ⚠️ **本模型的 encoder 与 decoder 都需要 `--disable_layer_group`**，但两者理由不同：decoder 是图结构复杂（28 输入 / 25 输出），layer_group 优化不稳定、不加会生成异常 bmodel；turbo encoder 是单网络 >500MB（1.3G），不加会在推理时触发 TPU-MLIR v1.28.1 的 layer_group codegen bug 导致**板卡 kernel panic 重启**。base encoder 只有 46MB，不受后一条影响。
>
> 两个脚本都已内置该参数，勿删。**这不是"所有模型都要加"的通用规则**——通用边界是"单网络 bmodel >500MB"，逐层拆分的小网络（如 Qwen3-TTS talker 每层 21MB）不用加，个别模型加了反而 SHA 校验失败（Eureka qwen3）。详见 `.claude/doc/sophon_bm1684_knowledge_base.md` §1.4。

### Step 3：交叉编译 C++

```bash
# 构建交叉编译镜像（只需一次）
docker build -t sophon-cross-build \
  -f 3_docker/Dockerfile.cross-build 3_docker

bash whisper/cpp/build.sh
# 产物：whisper/cpp/build/whisper_bm1684
```

### Step 4：上板运行

使用统一部署脚本上传二进制、模型 bmodel、运行时资产和可选测试音频；上传后自动执行 md5 校验：

```bash
# base F16
BOARD_IP=<board_ip> bash whisper/deploy_to_board.sh --test

# large-v3-turbo W4F16
MODEL_VARIANT=turbo_w4f16 BOARD_IP=<board_ip> \
  bash whisper/deploy_to_board.sh --test
```

脚本支持 `BOARD_USER`、`BOARD_PORT`、`BOARD_DIR`、`MODEL_DIR`、`BINARY`、`TEST_AUDIO` 和 `BOARD_PASS` 环境变量。板卡上的手动运行命令：

```bash
# base
./whisper_bm1684 models/base test_data/test_zh.wav zh F16 base
# turbo F16
./whisper_bm1684 models/turbo test_data/test_zh.wav zh F16 turbo
# turbo W4F16
./whisper_bm1684 models/turbo_w4f16 test_data/test_zh.wav zh W4F16 turbo
# 参数: <model_dir> <audio.wav> [zh|en] [F32|F16|W4F16] [base|turbo]
```

---

## 性能（BM1684X 实测）

### base（~5.8s 音频，5 轮平均）

| 精度 | 总耗时 | 数据来源 |
|------|--------|---------|
| F32  | ~1.86s | ⚠️ 见下方说明 |
| F16  | ~1.01s | ⚠️ 见下方说明 |

> ⚠️ **base 没有板卡实测记录**。这两个耗时数字在仓库内找不到对应的板上日志，`PERF_SUMMARY.md` 也标注 base「板上未部署」。仓库中唯一的 base 耗时证据是 `python/test/outputs/baseline/summary.json` 里的 `elapsed_ms`（test_en 932.4ms、test_zh 763.5ms），那是**开发机 PyTorch CPU baseline，不是板卡 bmodel 数据**，且两者均值 848ms 也与上表的 1.01s 对不上。
>
> base 的 6 个 bmodel（enc/dec × F32/F16/W4F16）与资产在 `models/BM1684X/` 中齐备，可随时上板；**在补做板卡实测之前，不要把上表数字当作 base 的板卡性能引用**。turbo 的数据（下两节）则有明确的 52 条校准集与单条采样两种口径，可正常引用。

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

### 量化格式支持（本平台 BM1684X）

| 格式 | BM1684X | 说明 |
|------|:-------:|------|
| F16 / F32  | ✅ | 无需 calibration |
| W4F16 / W4BF16 | ✅ | 无需 calibration，**turbo 推荐 W4F16** |
| INT8 | ✅ | 需要 calibration；**本模型实测失败**（见下） |
| W4A8 (W4INT8) | ❌ | **硬件不支持**，见下 |

（BM1688 对 W4A8 的支持是平台能力差异，不在本仓验证范围内；本仓全部数据来自 BM1684X。）

**W4A8 实验结论**：尝试在 BM1684X 上编译 W4INT8 turbo bmodel，codegen 阶段报错 `tpu_data_type_size Assertion "0" failed`。查阅 Sophgo SDK v26.03.01 文档确认 W4A8 仅支持 BM1688，这是硬件架构限制，升级 TPU-MLIR 版本无法解决。

**INT8 实验结论**：encoder 32 层误差累积致 audio_features 失真；decoder INT8 编译时 tpuc-opt abort（compiler bug）。详见知识库 §2.2。

> ⚠️ **校准数据不随仓库归档**：`whisper/calib_data/`（52 条，96MB / 104 文件）被 `.gitignore` 排除，属本机产物。它与上文 RTF 统计用的是同一套 52 条校准集，构成为 ChatTTS 合成 20 条 + FLEURS 中文 15 条 + LibriSpeech 英文 15 条 + 测试音频 2 条（共 639s）。目录下同时保存 `wav/` 与预提取的 `encoder_npy/`。换平台或重做 INT8/W4A8 实验时需按此构成重新采集，不要假设归档里有这份数据。

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
