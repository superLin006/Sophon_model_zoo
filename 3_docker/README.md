# 公共 Docker 环境

本目录统一维护模型转换和 aarch64 交叉编译环境。仓库根目录在 TPU-MLIR 容器中固定挂载为 `/workspace`，各模型目录不再自行维护 Dockerfile。

## TPU-MLIR 1.28.1

基础镜像 `sophgo/tpuc_dev:v3.4` 本身不包含完整 TPU-MLIR 命令。使用仓库中的固定 wheel 构建公共派生镜像：

```bash
./3_docker/build_tpumlir.sh
```

默认输入和输出：

```text
wheel: 0_Toolkits/tpu_mlir-1.28.1-py3-none-any.whl
image: sophon/tpuc_dev:v3.4-tpumlir-1.28.1
```

也可显式覆盖：

```bash
TPUMLIR_IMAGE=my-tpumlir:1.28.1 \
  ./3_docker/build_tpumlir.sh /path/to/tpu_mlir-1.28.1-py3-none-any.whl
```

启动或进入公共转换容器：

```bash
./3_docker/run_docker.sh
# 指定容器名
./3_docker/run_docker.sh my-tpumlir
# 使用自定义镜像
TPUMLIR_IMAGE=my-tpumlir:1.28.1 ./3_docker/run_docker.sh my-tpumlir
```

默认容器名为 **`sophon-tpumlir-v128`**，全仓库文档与脚本统一使用该名字（可用 `TPUMLIR_CONTAINER` 环境变量在个别脚本中覆盖）。容器内可直接使用：

```bash
model_transform.py --help
model_deploy.py --help
model_runner.py --help
llm_convert.py --help
python3 -c "import tpu_mlir"   # 1.28.1
```

### 镜像守卫：同名容器镜像不符时会拒绝复用

`run_docker.sh` 在容器已存在时校验其镜像是否等于 `$TPUMLIR_IMAGE`，**不符则报错退出 2**，避免误进入一个 TPU-MLIR 版本不同的旧环境：

```text
[ERROR] 容器 <name> 使用镜像 <actual>，期望 <expected>；请换容器名或先手动删除旧容器。
```

处置：换一个容器名（`./3_docker/run_docker.sh my-tpumlir`），或确认旧容器无需保留后 `docker rm <name>` 再重跑。**不要为了让脚本通过而把 `TPUMLIR_IMAGE` 改成旧容器的镜像**——那会静默使用错误版本的工具链编译。

### 挂载约定

`run_docker.sh` **只创建一个挂载**：仓库根目录 → `/workspace`（`-w /workspace`）。

因此容器内引用任何仓库文件都必须走 `/workspace/<模型>/...`。历史文档中出现过的 `docker exec <容器> bash <脚本> /models <seq_len> ...` 里的 `/models` **不是 `run_docker.sh` 创建的挂载点**，直接使用会失败。若确实需要在容器内访问仓库外的权重，必须自行 `docker run` 并追加 `-v <宿主权重目录>:/models`，且要在对应模型 README 中写明这一步。

## aarch64 交叉编译

`Dockerfile.cross-build` 提供 Ubuntu 20.04（glibc 2.31）、`g++-aarch64-linux-gnu` / `gcc-aarch64-linux-gnu` 与 CMake 3.28。该 Dockerfile **不含任何 `COPY`/`ADD`**，构建上下文的内容不被使用，因此上下文应指向本目录而不是仓库根，避免把整个工作树（含数十 GB 模型产物）交给 docker daemon：

```bash
docker build -t sophon-cross-build -f 3_docker/Dockerfile.cross-build 3_docker
```

镜像内**不安装宿主架构的 `gcc`/`g++`**，只有 aarch64 交叉编译器——这正是各模型 `cpp/build.sh` 用 `command -v aarch64-linux-gnu-g++` 判断"是否在容器内"的依据。

### 两种 `cpp/build.sh` 风格（不统一，按模型区分）

| 风格 | 模型 | 行为 | 正确调用方式 |
|---|---|---|---|
| **A：脚本自己起容器并挂载仓库** | whisper、sensevoice、moonshine、zipformer、chatTTS、vits-melo-tts-zh_en、Eureka-Audio、Qwen3-ASR、Qwen3-TTS（9 个） | `build.sh` 内部 `docker run sophon-cross-build` 并挂载仓库与 SDK，然后执行 cmake + make | 宿主机仓库根目录直接 `bash <model>/cpp/build.sh` |
| **B：脚本只在容器内运行** | **QwenLLM、HY-MT**（2 个） | `build.sh` **不起 docker**，仅检查 `aarch64-linux-gnu-g++` 是否存在 | 先进容器再执行，例如 `docker exec sophon-cross-build bash -lc 'cd <容器内仓库路径>/QwenLLM/cpp && ./build.sh'` |

> ⚠️ 风格 B 的模型在宿主机直接 `bash <model>/cpp/build.sh` 有两种结局：宿主没装交叉编译器时报错退出（安全）；宿主装了（如 WSL 的 gcc 15）则**编译成功但产物不可上板**（`GLIBC_2.xx not found`）。这是 QwenLLM 与 HY-MT 各自的 README 必须显式给出容器内调用方式的原因。

产物目录默认为 `<model>/cpp/build/`，两个例外以各自 README 为准：**zipformer → `cpp/build-aarch64/`**（`build.sh host` 走 `cpp/build-host/`）、**HY-MT → `cpp/build-aarch64-v2/`**。所有 `build*/` 目录均不入 Git。

各模型的 `cpp/build.sh` 负责传入该模型所需的 Sophon SDK 与 `1_third_party/` 第三方库路径；SDK 和模型资产不复制进镜像。

## 安全与产物约定

- Docker 镜像中不复制模型、测试音频、Hugging Face token 或板卡凭据。
- checkpoint、ONNX、bmodel 等资产的存放位置由各模型 README 指定。
- MLIR、NPZ、profile 和构建目录均为可再生成产物，不作为源码保留；转换脚本必须把中间产物写入各模型自己的隔离临时目录（通常是 `<model>/compile/tmp/`，VITS 等例外以各自 README 为准）。
- 板卡连接信息由 `BOARD_IP` / `BOARD_USER` / `BOARD_PORT` / `BOARD_PASS` 环境变量注入，不写入镜像、脚本或文档。
