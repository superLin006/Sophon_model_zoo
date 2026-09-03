# AGENTS.md

This file provides guidance to Qoder (qoder.com) when working with code in this repository.

## 仓库概述

Sophon BM1684X 平台深度学习模型移植工作区（SDK-23.09 LTS SP4，TPU-MLIR v1.28.1），已移植 **11 个模型**：Whisper、SenseVoice、Moonshine、Zipformer、ChatTTS、VITS-MeloTTS、Eureka-Audio、QwenLLM、Qwen3-ASR、Qwen3-TTS、HY-MT。全部为 ASR/TTS/LLM 类模型。

**性能与验收数据的唯一正式来源是 [PERF_SUMMARY.md](PERF_SUMMARY.md)**；本文件与各模型 README 不另行给出性能结论，只给流程与踩坑经验。

**开始任何模型相关工作前，必读**：
- [.claude/doc/sophon_bm1684_knowledge_base.md](.claude/doc/sophon_bm1684_knowledge_base.md) — 11 模型移植经验汇总（转换坑、量化决策树、bmruntime 推理模式、调试方法论），每条经验标注来源模型与适用边界；第六章记录了历史结论的演进与裁决，**与本文件或模型 README 不一致时以第六章为准**
- [.claude/doc/sophon_tpumlir_operators.md](.claude/doc/sophon_tpumlir_operators.md) — TPU-MLIR 算子支持表
- [.claude/standards/](.claude/standards/) — bmodel 输出管理、板卡部署、sherpa 交付、models 目录与转换链路，共四份规范
- 目标模型的 `README.md`，以及 `.context/*.md`（**仅 4 个模型有**：moonshine、vits-melo-tts-zh_en、Qwen3-TTS、HY-MT）

## 环境与工具链

| 环境 | 用途 | 说明 |
|---|---|---|
| Conda（各模型独立） | PyTorch → ONNX 导出 | **换机器建环境：`conda create -n <env名> python=<版本> -y && conda run -n <env名> python -m pip install --upgrade pip && conda run -n <env名> python -m pip install -r <模型>/requirements.txt`**；每模型 requirements 锁版本、经实测校验。**一律用 `conda run`，不要 `conda activate` + 裸 pip** |
| Docker `sophon/tpuc_dev:v3.4-tpumlir-1.28.1` | 模型转换 | 由 `./3_docker/build_tpumlir.sh` 从 `0_Toolkits/tpu_mlir-1.28.1-py3-none-any.whl` 构建；`./3_docker/run_docker.sh [容器名]` 启动，**默认容器名 `sophon-tpumlir-v128`**，仓库挂载于 `/workspace`，容器内可用 `model_transform.py`/`model_deploy.py`/`model_runner.py`/`llm_convert.py` |
| Docker `sophon-cross-build` | aarch64 交叉编译 | `docker build -t sophon-cross-build -f 3_docker/Dockerfile.cross-build 3_docker`；Ubuntu 20.04 + GCC 9.4 + glibc 2.31，产物只引用到 `GLIBC_2.29`。**板卡产物只能用此镜像编译**（板卡 glibc **2.31**；WSL GCC15 产物需 2.43、服务器 Buildroot GCC10 需 2.34，均超出 → 运行报 `GLIBC_2.xx not found`）。构建上下文用 `3_docker` 而非 `.`：Dockerfile 无 COPY，用 `.` 会把整个工作树（含数十 GB 产物）交给 docker daemon |

关键前置资源：
- `0_Toolkits/soc-sdk-sp4/`：Sophon SDK（libbmrt/libbmlib 头文件 + .so）。**git 忽略，需本机存在**
- `0_Toolkits/tpu_mlir-1.28.1-py3-none-any.whl`：构建转换镜像用。git 忽略
- `1_third_party/`：fftw、kaldi_native_fbank、nlohmann/json.hpp、sophon、tokenizers-cpp 的 aarch64 预编译静态库与头文件。**大部分 git 忽略（120 个文件中仅 12 个入库，其余靠 `.gitkeep` 占位）；例外是 `tokenizers-cpp/`——2 个 `.a`（36MB）与 2 个头文件已入库**，因为它们是 QwenLLM / Qwen3-ASR 交叉编译的硬依赖且无稳定的上游预编译下载地址。`.gitignore` 的 `*.a` 规则对这两个文件用 `git add -f` 绕过

## 常用命令

### 非 LLM 模型转换（路线 A：PyTorch → ONNX → bmodel）

```bash
conda run -n <该模型的 conda env> --no-capture-output python <model>/python/export_onnx.py  # 导出 ONNX（大模型 >2GB 需跳过 onnxsim、用 external data）
./3_docker/run_docker.sh                          # 启动/进入 TPU-MLIR 容器 sophon-tpumlir-v128
# 容器内（或宿主机用 docker exec）：
docker exec sophon-tpumlir-v128 bash /workspace/<model>/python/gen_bmodel.sh F16  # F32/F16/W4F16 等档位
```

### LLM 模型转换（路线 B：Safetensors → llm_convert → bmodel）

```bash
./3_docker/run_docker.sh
docker exec sophon-tpumlir-v128 bash /workspace/<model>/compile/<脚本>.sh <容器内权重目录> <seq_len> <输出目录>
# 例：HY-MT（权重必须放在仓库里，容器路径即 /workspace/HY-MT/...）
docker exec sophon-tpumlir-v128 bash /workspace/HY-MT/compile/compile_w4bf16_g64.sh \
  /workspace/HY-MT/compile/tmp/HY-MT1.5-1.8B-f16 512 \
  /workspace/HY-MT/models/BM1684X/w4bf16_g64_seq512
```

> ⚠️ **`run_docker.sh` 只创建 `/workspace` 一个挂载点**。历史文档里出现过的 `... compile_xxx.sh /models 512 ...` 中的 `/models` 并不存在，直接用会报找不到目录。确实需要挂载仓库外权重时，必须自己 `docker run` 并追加 `-v <宿主权重目录>:/models`，且在模型 README 中写明这一步。

要点：
- `llm_convert.py -m <dir> -s <seq_len> --quantize w4bf16|w4f16|w8bf16 -g 64 -c bm1684x --out_dir <dir>`
- HY-MT 必须先跑 `patch_tpumlir_hymt.py`（Hunyuan 适配补丁：动态 RoPE、QK-Norm 权重名与 `RoPE -> QK-Norm` 顺序）
- Qwen3-ASR 需先用 `make_models_llm_std.py` 把 transformers 5.14 权重重命名成标准 Qwen3 前缀（`model.language_model.` → `model.`），W4F16 需把 config `dtype` 改为 float16
- 多 bmodel 合并用 `model_tool --combine`（如 Qwen3-TTS 的 3 bmodel 方案、Qwen3-ASR 的 `compile/merge.sh`）

### C++ 交叉编译

**`cpp/build.sh` 有两种风格，不能一概而论**（详见 `3_docker/README.md`）：

| 风格 | 模型 | 调用方式 |
|---|---|---|
| A：`build.sh` 自己 `docker run` 并挂载仓库 | whisper、sensevoice、moonshine、zipformer、chatTTS、vits-melo-tts-zh_en、Eureka-Audio、Qwen3-ASR、Qwen3-TTS | 宿主机仓库根目录：`bash <model>/cpp/build.sh` |
| B：`build.sh` **不起 docker**，只检查 `aarch64-linux-gnu-g++` | **QwenLLM、HY-MT** | 必须先进容器：`docker exec sophon-cross-build bash -lc 'cd <容器内路径>/<model>/cpp && ./build.sh'` |

> ⚠️ 风格 B 的模型在宿主机直接 `bash <model>/cpp/build.sh`：没装交叉编译器时报错退出（安全）；装了（如 WSL gcc 15）则**编译成功但产物不可上板**。上板前用 `file <binary>` 确认是 `ARM aarch64`。

产物目录默认 `<model>/cpp/build/<binary>`，两个例外：**zipformer → `cpp/build-aarch64/`**（`build.sh host` 走 `cpp/build-host/`）、**HY-MT → `cpp/build-aarch64-v2/`**。所有 `build*/` 均不入 Git。

`CMakeLists.txt` 硬编码 `aarch64-linux-gnu-gcc/g++`，链接 `${SOPHON_SDK}/lib/libbmrt.so`、`libbmlib.so` 与 `1_third_party/` 静态库，rpath 为 `-Wl,-rpath,/opt/sophon/libsophon-current/lib`。

**可执行文件命名并不统一**（历史遗留，归档不改名）：`<model>_bm1684`（whisper / sensevoice / moonshine / vits_melo_tts）、`<model>_bm1684x`（eureka_audio / qwen3_asr / qwen3_tts）、无平台后缀（`chattts`、`zipformer_cli`、`hymt_demo`、`qwen_demo`）。

### 板卡部署与测试

```bash
BOARD_IP=<board_ip> bash <model>/deploy_to_board.sh [--test]   # 上传 bmodel/资产/二进制/测试音频，可选上板执行
```

- **QwenLLM 的部署入口是 `QwenLLM/scripts/deploy_to_board.sh`**（无顶层 `deploy_to_board.sh`），其余 10 个模型都在顶层
- 板卡：BM1684X SoC，IP 与凭据由环境变量注入（`BOARD_IP`/`BOARD_USER`/`BOARD_PORT`/`BOARD_PASS`），部署目录默认 `/data/<model>/`；凭据不写入仓库
- **scp 大文件会静默损坏**：上传后必须 md5sum 比对本地与板卡；"板卡行为诡异"先验 md5 再怀疑代码
- RTF 统计口径：只计特征提取 + TPU 推理，不含模型加载；输出格式 `[Timing] audio=<x>ms feat=<x>ms infer=<x>ms total=<x>ms RTF=<x>`
- 板卡回归脚本在各模型 `test_data/`（HY-MT 最全：`board_regression.sh` 5 例、`board_regression_extended.sh` 16 例、`board_regression_full.sh` 61 例、`board_perf_stability.sh` 稳定性、`analyze_board_logs.py` 日志分析）

## 代码架构

### 两条转换路线（先选路线再动手）

- **路线 A（通用算子模型）**：`python/export_onnx.py` → `model_transform.py` → `model_deploy.py` → bmodel。适用于非 LLM 模型（whisper/sensevoice/vits/zipformer/moonshine）、ChatTTS，以及 Qwen3-TTS / Eureka-Audio 的逐层拆分部件
- **路线 B（LLM 标准模型）**：`llm_convert.py`（层融合 + KV 独立）→ 单 bmodel。QwenLLM / Qwen3-ASR / HY-MT 的 LLM 主体全部走此路线，**优先于自定义分块方案**（Qwen3-ASR 迁移后体积 -51%、内存 -49%、decode +83%）；audio-embed 等非标准输入注入会语义有损（Eureka 实测 5-6/9 → 3/9，已放弃）

### 模型目录结构（每个模型独立自包含）

```
<model>/
├── compile/ 或 python/    # 转换脚本（LLM 模型用 compile/，非 LLM 用 python/）
├── cpp/                   # C++ 推理（src/ + CMakeLists.txt + build.sh）
├── models/BM1684X/        # bmodel 产物（git 忽略；只保留最终采用的档位）
├── test_data/             # 测试输入
├── deploy_to_board.sh     # 上板脚本（sshpass + scp + md5 校验）；QwenLLM 在 scripts/ 下
├── requirements.txt       # 该模型 conda 环境的唯一依赖入口
├── README.md              # 模型说明、转换/构建/上板步骤、逐条实测结果与已知限制
└── .context/              # 移植过程记录（仅 4 个模型有）
```

**入库实况**（与 `.gitignore` 一致，勿凭印象判断）：
- bmodel / onnx / 模型权重 / 编译中间物 **全部不入库**
- **小体积验收音频入库**：当前 35 个 wav、13MB（Qwen3-ASR 15、Eureka-Audio 11、moonshine 3、sensevoice 2、whisper 2、zipformer 2）。根 `.gitignore` 的 `test_data/*` 规则不带 `**`，只作用于仓库根层，管不到 `<model>/test_data/`
- **运行时必需的小资产入库**：如 `moonshine/models/tokens.txt`、`moonshine/models/log_k.txt`（靠 `git add -f` 绕过 `**/models/**`）
- `.context/` 链条为 `baseline.md → operator_analysis.md → bmodel_info.md → perf_log.md`（见 `.claude/subagents/README.md`）。**当前无任何模型有 `perf_log.md`**（第 5 阶段 performance-optimizer 的产物），`.context/` 本身也只有 4 个模型具备

验证输出目录约定（见 `bmodel_output_management.md`）：`python/test/outputs/{baseline,onnx,debug}/`，baseline 入库、debug 中间产物不入库。

### C++ 推理模式（bmruntime）

- 一个 `bm_handle` 只能加载一个 bmodel；多 bmodel（如 ChatTTS 三引擎、Qwen3-TTS 三文件）共享单 handle
- **KV 设备常驻**：decode 零拷贝 `with_device` + 层间 d2d 写 KV 槽，是最大性能收益点（whisper decode 3.1×、Eureka 6.2→16 tok/s、Qwen3-TTS RTF 4.37→2.36）
- 层间 launch 是否 sync 取决于产物类型：llm_convert 标准产物靠驱动 FIFO 自然串行（HY-MT 61/61 实测）；model_transform/deploy 通用产物逐层 sync 保守；**CPU 读结果前必须 `bm_thread_sync`**（无争议）。注意 `bmrt_launch_tensor_ex(..., true, false)` 最后两参是 user_mem/user_stmode，**不是 is_sync**
- KV cache 是 head-major `[H,S,D]`，写回必须逐 head memcpy
- 维度（n_mels/n_state/n_layer/vocab/SEQLEN）一律从 bmodel net_info 读取，一套代码多模型通用
- 详细裁决见知识库 §3.1；**模型 README 中若与此冲突，以知识库 §3.1 与 §6 为准**

### 验证流水线（sail 先行原则）

```
PyTorch baseline → ONNX 验证 → bmodel 编译 → 板卡 sail Python 端到端验证 → C++ 实现
```

bmodel 先过板卡 Python/sail 验证再写 C++，C++ 出错时可排除 bmodel 嫌疑。每步产物逐级 cosine/文本对比定位问题（mel→encoder→embeds，block 级 cosine ≥0.9999 为正常）。

板卡 Python 侧 sail 的导入方式是 **`import sophon.sail as sail`**（板上 `sophon-arm-pcie` 提供 `sophon/sail.*.so`），**裸 `import sail` 会失败**。板卡自带 numpy / scipy / tokenizers / torch / torchaudio，**不含 `kaldifeat`**。

## 关键约束与踩坑经验（详细见知识库）

1. **`--disable_layer_group`**：边界是**单网络 bmodel >500MB**（whisper turbo encoder 1.3G 不加 → 板卡 kernel panic 重启）；逐层拆分的小网络不用加（Qwen3-TTS talker 每层 21MB，56 网络未加也 OK）；个别模型加了反而 SHA 校验失败（Eureka qwen3）——**逐模型验证，不能一刀切**。whisper 的 decoder 因图结构复杂（28 入 25 出）也需加，与体积无关
2. **`model_deploy.py` 中间产物写入当前工作目录**（不是 `--mlir` 指定目录）。所有编译脚本必须在隔离目录执行并在结束时清理：zipformer 的 `gen_bmodel.sh` 内置 `mktemp` + `trap` 自动清理，Qwen3-TTS 的 `gen_final.sh` 固定写 `compile/tmp/`。曾因未隔离一次把 ~7.5G 中间物散落到仓库根，且泄漏到仓库外（`../models/`）
3. **量化档位不能套经验**：W4BF16 体积/精度最佳但必须 `-g 64`（group 128 失败）；W4F16 与 W4BF16 是不同激活路径，必须逐模型实测（编译成功 ≠ 质量达标）；生成式模型优先 F32/F16，BF16 对 attention 精度不足（whisper encoder BF16 cosine 仅 0.51）；W4A8 BM1684X 硬件不支持（仅 BM1688）
4. **llm_convert 容器环境**：torch 2.4.1+cpu、transformers 4.57.6+、`huggingface-hub<1.0`；旧镜像 torch 2.1/transformers 5.x 可能在 dtype 初始化失败
5. **ONNX 导出坑**：opset 17 缺 `kernel_shape` 需从权重补全；KV dummy tensor 必须列表推导式（`[t]*n` 会被 constant folding 消除）；>2GB 用 external data 避开 protobuf 上限；内存建议 `.wslconfig` 配 `memory=14GB`（实机 `free -g` 约 13GB 可用，turbo encoder 全量导出已在此配置下跑通），不足时 OOM killer 报 **Exit 137**
6. **git 约定**：bmodel/onnx/权重/编译中间物全部忽略，一切可再生成产物不入库；小体积验收音频与运行时必需小资产入库（见上「入库实况」）。commit 用 conventional 风格（`feat/fix/docs/chore(scope): ...`）
7. **大模型冷读**：板卡 eMMC 冷读大 bmodel 像卡死（2.7G 约 60s，进程进 D 状态），先 `cat *.bmodel >/dev/null` 预热 page cache
8. **产物白名单易被违反**：`models/BM1684X/` 只应保留最终采用的 bmodel 与重建用组件 bmodel。实际会反复出现 `*.bmodel.json`、`*.net_0.profile`、`final.mlir`、`tensor_location.json`、`ref_files.json`、`.modify`、逐网络子目录、实验档目录与构建日志——**每次跑完编译或端到端验证都要复查并清理**，且这些文件被 gitignore，`4_tools/check_repo_structure.py` 默认看不见（需 `--disk` 模式）
9. **root 属主产物**：docker 内生成的文件属 root，宿主机删不掉。用 `docker exec sophon-tpumlir-v128`（镜像内 root，仓库挂 `/workspace`）执行清理，**先 md5/内容核验再删，不用 sudo**
