# models 目录结构与转换链路标准 v2.0

> 目标：任何新加入的模型都能从仓库脚本独立复现 pytorch → onnx → bmodel 全链路。
> 本文件只记**标准与守则**，不记带日期的清理流水账——执行记录在 git 历史里，写进标准只会随产物再生成而失效（v1.0 的 §4/§5/§6 即因此移除，见文末版本说明）。

## 1. models/ 目录布局（磁盘应如下组织）

```
<model>/
├── models/
│   ├── <HF原始权重目录>/        # 可选：HuggingFace/ModelScope 原始权重（本地备份）
│   ├── onnx/                   # 只保留 gen_bmodel*.sh 实际消费的 .onnx / .onnx.data
│   │   └── .gitkeep
│   ├── BM1684X/                # 该芯片最终采用的 bmodel（按芯片目录划分）
│   │   └── .gitkeep
│   └── <小资产>.txt/.npy       # C++ 运行必需且体积小的文件可入库（如 moonshine/tokens.txt）
```

约束：

- `models/**` 已在根 `.gitignore` 全局忽略；目录结构靠 `.gitkeep` 占位入库（新增需 `git add -f`）。
- **bmodel 只保留最终采用的档位**。未采用的实验档位、中间 combine 产物、逐网络子目录一律放 `compile/tmp/`，不留在 `models/BM1684X/`。
- **`models/BM1684X/` 内禁止出现任何非 bmodel 文件**：`*.bmodel.json`、`*.net_0.profile`、`final.mlir`、`tensor_location.json`、`ref_files.json`、`.modify`、构建日志、逐网络子目录都是 `model_deploy.py` 的副产物，每次编译后必须清理。
- **`models/onnx/` 收敛原则**：只留 `gen_bmodel*.sh` 实际消费的 onnx（本仓库统一为 `_sim` 版）。raw 版由 `export_onnx.py` 可随时重新生成，**不落盘**。禁止存放 mlir/npz/json/prototxt/ref_files 等转换中间产物。
- 辅助资产（vocab / mel_filters / positional_embedding 等）随各自模型目录存放，避免跨目录重复拷贝。
- **本仓库不单独维护转换环境总表**；各模型以自身根目录的 `requirements.txt` 为唯一依赖入口。

> ⚠️ 上述约束在磁盘上会**反复被违反**：每跑一次编译或端到端验证，`model_deploy.py` 就会重新生成 `.bmodel.json` / `.net_0.profile` / 逐网络子目录。这些文件全部被 gitignore，`tools/check_repo_structure.py` 的默认模式看不见它们，**必须用 `--disk` 模式复查**。

## 2. 转换链路：两条路线

| 路线 | 适用 | 幂等入口 | 中间产物落点 |
|---|---|---|---|
| **A. PyTorch → ONNX → bmodel** | 非 LLM 模型（whisper / sensevoice / moonshine / zipformer / vits / chatTTS，以及 Qwen3-TTS、Eureka-Audio 的逐层拆分部件） | `<model>/python/export_onnx.py` + `gen_bmodel.sh`（或 `compile/` 下对应脚本） | onnx → `models/onnx/`；mlir/deploy 中间物 → `compile/tmp/` |
| **B. Safetensors → llm_convert → bmodel** | 标准大 LLM（QwenLLM / Qwen3-ASR / HY-MT 的 LLM 主体） | `<model>/compile/compile_*.sh`（QwenLLM 另有 `scripts/`） | mlir/npz 即产物，无需保留 onnx |

判定原则：

- 非 LLM 部件**必须**有 python onnx 导出脚本（禁止 placeholder 或在 README 里一笔带过）；LLM 主体走 llm_convert 视为等价（其内部即 torch→onnx→mlir→bmodel），不强制导出并保留数 GB onnx。
- 多 bmodel 合并（`model_tool --combine`）必须固化成脚本（如 `Qwen3-ASR/compile/merge.sh`），不得只留手册命令或 log。
- 容器内引用仓库文件一律走 `/workspace/<模型>/...`。`run_docker.sh` **只创建 `/workspace` 这一个挂载**，脚本参数里出现的 `/models` 之类路径不存在，除非文档同时写明额外的 `-v` 挂载。

## 3. 脚本守则（防止回归）

1. **转换中间产物一律写 `compile/tmp/`（或同层 tmp 目录），禁止 cd 进 `models/onnx` 就地生成**。`model_deploy.py` 把 npz/mlir/json 写入**当前工作目录**而非 `--mlir` 指定目录，因此编译脚本必须自带隔离与清理：zipformer 的 `gen_bmodel.sh` 用 `mktemp` + `trap` 自动清理，Qwen3-TTS 的 `gen_final.sh` 固定写 `compile/tmp/gen_final/`。未隔离的后果是单次编译把 ~7.5G 中间物散落到仓库根，甚至泄漏到仓库外。
2. **导出脚本禁止硬编码本机绝对路径**。模型权重路径用 `--model-path` / 环境变量提供；输出目录用 `--out` 或相对脚本位置推导。需要宿主 uid 时用 `$(id -u):$(id -g)`，不要写死数字。
3. **每条 onnx 导出应带数值自检**（onnxruntime 对比 torch，或与真实模型前向对齐，容差明确），防止改算子时静默漂移。
4. 大模型 kernel panic 相关的 `--disable_layer_group` 等参数**写死在编译脚本里**，README 注明原因与适用边界（边界见知识库 §1.4，不是"所有模型都加"）。
5. 遇到 transformers 4.57 vmap mask 在静态导出下报错时，用**预计算 attention_mask（dict 形式）绕过**（实证：`Qwen3-TTS/python/export_codec.py`）。
6. **回归/benchmark 脚本的汇总行必须由实际计数生成**，不得硬编码用例数。`HY-MT/test_data/board_regression_full.sh` 曾把收尾行写死成 `x46 cases` 而实际定义 61 个用例，导致 4 份板上日志长期少报 15 条并被分析文档反复引用；已改为动态计数并附带失败数。
7. **验收日志必须能独立复核**：逐条输出要带用例标识（文件名或用例名）。Qwen3-ASR 的 benchmark 日志只打 `[ASR] N tokens, total Xs (...)`，不带音频名，导致 RTF 中位数无法从日志复算——这是待修项，新脚本不要重犯。
8. **root 属主产物的清理方式**：docker 内生成的文件属 root，宿主机删不掉。用 `docker exec sophon-tpumlir-v128`（镜像内 root，仓库挂 `/workspace`）执行，**先 md5/内容核验再删，不用 sudo**，也不要用全局 `git clean`（产物大多被 gitignore，`git clean` 行为不可预期且无法只针对单个模型）。

---

**版本**

- v2.0（2026-09-03）：移除 v1.0 的 §4「转换链路缺口登记」、§5「清理记录」、§6「onnx 收敛与验证记录」——三节都是带日期的执行流水账，且其中的"已删除"声明在后续端到端验证重新生成产物后全部失效，留在标准里会误导。移除前已迁移其中**仅存于该处**的内容：
  - onnx 收敛原则（只留 `_sim`、raw 不落盘）→ 并入本文件 §1
  - 「不维护转换环境总表、以各模型 requirements.txt 为入口」→ 并入本文件 §1
  - root 属主产物的 docker 清理手段 → 并入本文件 §3.8，同时见 `AGENTS.md` 关键约束第 9 条
  - Qwen3-TTS `embedding_text.onnx` 的 bf16 Sigmoid workaround → 迁入知识库 §1.3 T10
  - whisper turbo external-data onnx 的 `onnx.checker` 误报 → 迁入知识库 §1.3 T11
  - 各模型 onnx 收敛后的具体留存清单与验证结果 → 由各模型 README 承载（如 `Qwen3-TTS/README.md` 的 100 个 onnx 分目录表、`zipformer/README.md` 的资产三层表）
  
  新增 §3.6/§3.7（脚本汇总行与日志可复核性），来自本轮归档梳理中发现的两个实测缺陷。
- v1.1（2026-08-27）
- v1.0（2026-08-14）
