# Sophon Model Zoo 使用指南

本仓库用于在 Sophon BM1684X 上验证和交付 ASR、TTS、LLM 及音频意图模型。每个算法目录都包含自己的 README、依赖和运行入口，根目录只负责公共工具链和约定。

性能与验收数据统一维护在 [PERF_SUMMARY.md](PERF_SUMMARY.md)，本指南不重复给出任何性能数字。

## 目录约定

```text
Sophon_model_zoo/
├── 0_Toolkits/       # 本地 SDK 和 TPU-MLIR wheel，不入库
├── 1_third_party/    # 本地第三方头文件和静态库（tokenizers-cpp 例外，已入库）
├── 2_utils/          # 可选公共 C/C++ 工具
├── 3_docker/         # TPU-MLIR 与 aarch64 交叉编译镜像
├── 4_tools/          # 仓库结构与约定检查脚本
├── <model>/
│   ├── compile/      # 模型转换脚本；中间物落 compile/tmp/
│   ├── python/       # Python 推理与验证
│   ├── cpp/          # C++ 推理和交叉编译
│   ├── models/       # onnx/、BM1684X/ 本地模型产物
│   ├── assets/       # 小型运行时资产
│   ├── test_data/    # 输入测试数据
│   └── deploy_to_board.sh   # QwenLLM 在 scripts/ 下
└── PERF_SUMMARY.md / AGENTS.md / README.md
```

**入库范围**：bmodel、ONNX、模型权重和编译缓存不入 Git；**小体积验收音频入库**（当前 35 个 wav / 13MB，分布在 Qwen3-ASR、Eureka-Audio、moonshine、sensevoice、whisper、zipformer 的 `test_data/` 或 `test_audios/`），**运行时必需的小资产也入库**（如 `moonshine/models/tokens.txt`）。每个模型 README 会说明其必要例外。

## 环境准备

### 公共 TPU-MLIR 容器

```bash
./3_docker/build_tpumlir.sh    # 首次：从 0_Toolkits/ 的 wheel 构建派生镜像
./3_docker/run_docker.sh       # 启动/进入，默认容器名 sophon-tpumlir-v128
```

这会构建并启动固定的 TPU-MLIR 1.28.1 镜像。容器将**仓库根目录**挂载到 `/workspace`（这是 `run_docker.sh` 创建的唯一挂载），模型编译脚本统一使用该路径。若已有同名容器使用了不同镜像，脚本会拒绝复用并报错——处置方式见 [3_docker/README.md](3_docker/README.md)。

全仓库文档与脚本统一使用容器名 `sophon-tpumlir-v128`。

### aarch64 交叉编译容器

```bash
docker build -t sophon-cross-build \
  -f 3_docker/Dockerfile.cross-build 3_docker
```

BM1684X 板卡二进制必须使用该 Ubuntu 20.04 / GCC 9.4 工具链编译，产物只引用到 `GLIBC_2.29`，在板卡 glibc 2.31 范围内；宿主 GCC 版本过新会导致板卡出现 `GLIBC_2.xx not found`。

构建上下文用 `3_docker` 而不是 `.`——该 Dockerfile 不含 COPY，用 `.` 会把整个工作树（数十 GB 产物）当作上下文传输。

### 模型 Python 环境

模型导出环境以各目录的 `requirements.txt` 为准，**一律用 `conda run`，不要 `conda activate` + 裸 pip**：

```bash
conda create -n <env_name> python=3.10 -y
conda run -n <env_name> python -m pip install --upgrade pip
conda run -n <env_name> python -m pip install -r <model>/requirements.txt
conda run -n <env_name> --no-capture-output python <model>/python/export_onnx.py
```

Qwen3-ASR 使用 Python 3.11。QwenLLM 的 Safetensors 转换在 TPU-MLIR 容器中完成，不需要本地 Python 转换环境。

根目录 `environment.yml` 仅是通用初始化模板，不保证和任一模型的精确版本一致；正式复现时优先使用模型级 requirements。

## 两条转换路线

### 路线 A：PyTorch/外部 ONNX → bmodel

适用于 Whisper、SenseVoice、Moonshine、Zipformer、ChatTTS、VITS，以及 Qwen3-TTS / Eureka-Audio 的非 LLM 部件：

```text
PyTorch 或上游 ONNX
    ↓
python/export_onnx.py 或 compile/export_*.py
    ↓
models/onnx/
    ↓
model_transform.py + model_deploy.py
    ↓
models/BM1684X/
```

ONNX 导出完成后先运行模型目录中的数值验证，再进入 bmodel 编译。MLIR、NPZ、profile 和 JSON 中间物必须写入 `compile/tmp/`，不能混入 `models/onnx/` 或最终 bmodel 目录。

### 路线 B：Safetensors → llm_convert → bmodel

适用于 QwenLLM、Qwen3-ASR 和 HY-MT 的标准 LLM 主体：

```text
HuggingFace/Safetensors
    ↓
必要的权重键或 dtype 预处理
    ↓
llm_convert.py（TPU-MLIR 容器）
    ↓
compile/tmp/
    ↓
models/BM1684X/ 最终 bmodel
```

具体量化档位、group size、序列长度和特殊补丁以模型 README 与编译脚本为准。**转换成功不等于精度达标**，必须进行 baseline、逐层或端到端回归。

容器内引用权重必须走 `/workspace/<模型>/...`。历史文档里的 `/models` 路径不是 `run_docker.sh` 创建的挂载，直接使用会失败。

## C++ 构建

`cpp/build.sh` **有两种风格**，不能一律从宿主机直接跑：

| 风格 | 模型 | 调用方式 |
|---|---|---|
| A：脚本自己起 docker 并挂载仓库 | whisper、sensevoice、moonshine、zipformer、chatTTS、vits-melo-tts-zh_en、Eureka-Audio、Qwen3-ASR、Qwen3-TTS | 宿主机仓库根目录 `bash <model>/cpp/build.sh`（部分模型接受 `host` / `cross` / `install` 参数，见其 README） |
| B：脚本只在容器内运行，不起 docker | **QwenLLM、HY-MT** | `docker exec sophon-cross-build bash -lc 'cd <容器内路径>/<model>/cpp && ./build.sh'` |

> ⚠️ 风格 B 的模型若在宿主机直接执行：宿主没装交叉编译器时报错退出（安全）；宿主装了（如 WSL gcc 15）则**编译成功但产物不可上板**。上板前用 `file <binary>` 确认输出含 `ARM aarch64`。

产物目录默认 `<model>/cpp/build/`，两个例外：**zipformer → `cpp/build-aarch64/`**（`build.sh host` 走 `cpp/build-host/`）、**HY-MT → `cpp/build-aarch64-v2/`**。所有 `build*/` 目录不提交 Git。

CMake 运行库路径统一使用：

```text
/opt/sophon/libsophon-current/lib
```

## 板卡部署

部署脚本统一采用环境变量提供板卡信息，不在源码中保存密码：

```bash
BOARD_IP=<board_ip> \
  bash <model>/deploy_to_board.sh --test
```

必要时可覆盖：

```bash
BOARD_USER=root BOARD_PORT=22 BOARD_DIR=/data/<model> \
  bash <model>/deploy_to_board.sh
```

**QwenLLM 的入口是 `QwenLLM/scripts/deploy_to_board.sh`**，其余 10 个模型在各自顶层目录。

部署流程必须遵循：

```text
交叉编译 → 上传 → md5 校验 → 单样本 smoke test → 完整回归
```

大 bmodel 上传后必须在本地和板卡分别执行 md5 校验。板卡输出异常时，先排除传输损坏，再排查 C++ 代码。

## 测试与产物

推荐验证顺序：

```text
PyTorch baseline
    → ONNX 数值验证
    → bmodel 编译
    → Python/sail 板卡验证
    → C++ 板卡验证
```

板卡 Python 侧 sail 的导入方式是 `import sophon.sail as sail`，裸 `import sail` 会失败。

通用测试输出位置：

```text
<model>/python/test/outputs/
├── baseline/       # 可提交的参考结果
├── onnx/           # ONNX 对比结果
└── debug/          # 可重新生成的中间 tensor
```

编译与验证结束后，逐模型按 `.claude/standards/bmodel_output_management.md` 的目录白名单清理：`models/onnx/` 只留 `.onnx`/`.onnx.data`，`models/BM1684X/` 只留最终采用的 bmodel，`compile/tmp/` 可整目录删除，`cpp/build*/` 可重新编译。**不要使用 `git clean` 做全局清除**（这些产物大多被 gitignore，`git clean` 的行为不可预期且无法只针对某一模型）。

被 docker 生成的 root 属主文件用 `docker exec sophon-tpumlir-v128` 清理，先 md5/内容核验再删，不用 sudo。

## 仓库结构检查

提交前运行：

```bash
python3 4_tools/check_repo_structure.py            # 只检查 git 跟踪的内容
python3 4_tools/check_repo_structure.py --disk     # 额外扫描磁盘上的产物目录
```

检查覆盖：模型目录与入口完整性、模型产物层级白名单、内部文档链接、文档中引用的路径是否存在或被 gitignore、Shell/Python 语法、本机路径与认证信息硬编码、容器名与镜像约定、模型覆盖对账、精度声明与产物一致性。

**默认模式只看 git 跟踪的文件**，而 `models/**` 整体被 gitignore——因此产物目录里的中间物必须用 `--disk` 模式才能发现。
