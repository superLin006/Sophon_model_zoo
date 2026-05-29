# sherpa-onnx × Sophon BM1684X 交付规范 v1.0

> 用于把基于 sherpa-onnx 框架、跑在 BM1684X TPU 上的模型（如 SenseVoice ASR、
> ChatTTS TTS）打包成开发/测试人员可直接使用的交付物。
> 参考 MTK 的 `android_deliver_workflow.md`，但适配 Sophon（Ubuntu aarch64 板卡 + ssh）。

---

## 流程概览

```
确认编译产物 → 整理 deliver/ → 板卡本地验证 → 打包交付
```

与 MTK 版的关键差异：
- 板卡是 **Ubuntu aarch64 + ssh/scp**（不是 Android/adb）
- 必须 **glibc 版本匹配**（见下，最易踩的坑）
- 交付物是 **sherpa-onnx SDK**（头文件 + 两个 .so + 模型），不是单可执行文件

---

## ⚠️ 头号铁律：glibc 工具链必须匹配板卡

板卡（Ubuntu 20.04）的 glibc 符号最高到 `GLIBC_2.30`。用过新的编译器交叉编译，
产物会要求板卡没有的新符号，一运行就 `GLIBC_2.xx not found`。

| 工具链 | 产物需要 | 板卡 2.30 | 结论 |
|---|---|---|---|
| WSL 系统 GCC 15 | GLIBC_2.43 | ✗ | 跑不了 |
| 服务器 Buildroot GCC 10 | GLIBC_2.34 | ✗ | 跑不了 |
| **`sophon-cross-build` 镜像（Ubuntu 20.04 + GCC 9.4 + glibc 2.31）** | **GLIBC_2.29** | ✓ | **用这个** |

镜像定义：`3_docker/Dockerfile.cross-build`（`ubuntu:20.04` + `g++-aarch64-linux-gnu`）。
板卡 native GCC 9.3 也能编，但 `-flto` 单核会 OOM 重启 —— **务必 `-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF`**。

---

## 编译要点（docker 交叉编译）

1. 在 `sophon-cross-build` 容器内编译，挂载源码 + soc-sdk：
   ```bash
   docker run --rm -v "$PWD":/repo -v <soc-sdk>:/sdk -w /repo \
       sophon-cross-build:latest bash build-xxx-docker.sh
   ```
2. 关键 CMake 开关：
   - `-DSHERPA_ONNX_ENABLE_SOPHON=ON -DSHERPA_ONNX_SOPHON_SDK_DIR=/sdk`（链 libbmrt/libbmlib）
   - `-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF`（关 LTO）
   - `-DCMAKE_POLICY_VERSION_MINIMUM=3.5`（兼容 CMake 4）
   - `-DBUILD_SHARED_LIBS=ON`，自写 toolchain 指向 `aarch64-linux-gnu-g++`
   - ChatTTS 额外需 `SHERPA_ONNX_CHATTTS_DEPS_DIR`（含 fftw/ 和 nlohmann/json.hpp）+ `SHERPA_ONNX_ENABLE_TTS=ON`
3. **依赖下载会卡**（espeak-ng 等）：复用已有 build 目录的 `_deps/*-src`，
   配 `-DFETCHCONTENT_FULLY_DISCONNECTED=ON` + `-DFETCHCONTENT_SOURCE_DIR_<NAME>=...` 离线复用。

产物：`build-xxx/install/{include,lib}`。验证：
```bash
aarch64-linux-gnu-objdump -T install/lib/libsherpa-onnx-c-api.so | grep -oE 'GLIBC_2\.[0-9]+' | sort -uV | tail
# 最高应 <= 2.30；并确认 NEEDED 含 libbmrt.so / libbmlib.so
```

---

## deliver/ 标准目录结构

```
deliver/
├── bin/        # 测试程序（aarch64）
├── include/    # SDK 头文件，保留 sherpa-onnx/c-api/ 前缀路径
│   └── sherpa-onnx/c-api/{cxx-api.h, c-api.h}
├── lib/        # libsherpa-onnx-cxx-api.so + libsherpa-onnx-c-api.so + libonnxruntime.so
├── models/     # 按模型分子目录（bmodel + tokens/vocab/资产）
├── test_data/  # 测试输入 + run_*.sh
└── README.md
```

**规则：**
- 脚本用 `#!/bin/sh`（板卡可能无 bash）
- 头文件**必须带**（开发人员要查 API + 编译），保留 `sherpa-onnx/c-api/` 目录层级
- `LD_LIBRARY_PATH` 要含 `lib/` **和** `/opt/sophon/libsophon-current/lib`（驱动）
- 每次从源头重新生成，不手改 deliver/
- 大文件（模型/.so/打包）进 .gitignore；只把 README + 脚本提交入库

---

## run_*.sh 模板

**ASR（音频→文字）/ TTS（文字→音频）通用骨架：**
```sh
#!/bin/sh
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
EXEC="$ROOT_DIR/bin/<可执行文件>"
export LD_LIBRARY_PATH="$ROOT_DIR/lib:/opt/sophon/libsophon-current/lib:$LD_LIBRARY_PATH"
"$EXEC" <模型参数...>
```

---

## README.md 必含章节

1. 环境要求（板卡型号、Ubuntu 版本、glibc 匹配说明、ssh 访问）
2. 交付目录结构
3. 快速开始：`scp -r deliver root@<ip>:<dir>` → `chmod +x` → `sh test_data/run_*.sh`，附**预期输出**
4. **给开发人员的 SDK 调用示例**：cxx-api 的 C++ 代码片段、`provider="sophon"` 选 TPU、链接/编译命令
5. 命令行参数
6. 常见问题表（bash/库路径/glibc/截断等）

---

## 执行步骤

```bash
# 1. 整理
rm -rf deliver && mkdir -p deliver/{bin,include,lib,models,test_data}
cp <build>/install/lib/*.so deliver/lib/
cp -r <build>/install/include/* deliver/include/
cp <build>/test 程序 deliver/bin/ && chmod +x deliver/bin/*
cp -r <模型> deliver/models/ ; cp <测试数据> deliver/test_data/
# 写 run_*.sh + README.md

# 2. 板卡本地验证（必做！照 README 步骤实跑一遍）
scp -r deliver root@<ip>:/root/sophon_demo
ssh root@<ip> "cd /root/sophon_demo && chmod +x bin/* test_data/*.sh && sh test_data/run_asr.sh"

# 3. 打包（板卡无 zip 时用 tar.gz；模型大，tar.gz 更合适）
tar -czf <项目>_$(date +%Y%m%d).tar.gz deliver/
```

---

## 常见问题

| 问题 | 解决 |
|---|---|
| `GLIBC_2.xx not found` | 工具链太新；用 `sophon-cross-build` 镜像重编 |
| 编译链接 OOM / 板卡重启 | 关 LTO：`-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF` |
| `libbmrt.so.1.0 not found` | `LD_LIBRARY_PATH` 加 `/opt/sophon/libsophon-current/lib` |
| 依赖下载卡死 | 复用 `_deps/*-src` + `FETCHCONTENT_FULLY_DISCONNECTED=ON` |
| `bash: not found` | 脚本用 `sh` |
| `zip: not found` | 用 `tar -czf` |

---

## 项目配置（执行时填充）

| 配置项 | 值 |
|---|---|
| 源码仓库 | （如 `sherpa-onnx-2025-1217`，分支） |
| 编译镜像 | `sophon-cross-build:latest` |
| soc-sdk | `0_Toolkits/soc-sdk-sp4` |
| 板卡 | `root@<ip>:22` |
| 模型列表 | （bmodel + 资产路径） |
| 后端选择 | ASR `provider="sophon"`；TTS 设 `model.chattts.gpt` 等 |
