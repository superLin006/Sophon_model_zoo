# bmodel 输出文件管理规范 v1.1

> 保存代码示例见 project-initializer Step 4 / python-converter Step 6；本规范只记**目录约定 + 入库规则**。

## 标准目录结构

```
{model}/python/
├── export_onnx.py
├── gen_bmodel.sh
└── test/
    ├── test_pytorch.py       # PyTorch baseline
    ├── test_onnx.py          # ONNX 精度验证
    └── outputs/
        ├── baseline/         # PyTorch ground truth（保留入库）
        ├── onnx/             # ONNX 验证输出
        └── debug/            # 中间输出（给 C++ 对比用，不入库）
```

## 各子目录内容

### `baseline/` — ground truth（保留入库）
`result.json`（完整输出）+ `result.txt`（纯文本输出）+ 生成式模型的 codes/hidden npy。

### `onnx/` — ONNX 验证输出
`result.json` + `diff_vs_baseline.txt`（精度对比报告）。

### `debug/` — C++ 对比用中间输出（不入库，统一 .npy 格式）
- `input_features.npy`（预处理后的模型输入，必须）
- `model_output.npy`（模型原始输出，必须）
- 多段式/Encoder-Decoder 模型分模块各存一份（encoder/decoder/CP/talker 等）

## 模型产物目录约定

- `models/onnx/`：只保留 `gen_bmodel*.sh` 实际消费的 `.onnx` / `.onnx.data`（本仓库统一为 `_sim` 版）。mlir / profile / npz / json / prototxt 等转换产物一律不放在这里，可随时重新生成
- `models/BM1684X/`：保留**最终采用**的 bmodel + 重建用组件 bmodel。**禁止出现任何非 bmodel 文件**——以下都是 `model_deploy.py` 的副产物，每轮编译后必须清理：
  - `*.bmodel.json`、`*.net_0.profile`
  - 逐网络子目录（内含 `final.mlir`、`tensor_location.json`、`ref_files.json`、`.modify`）
  - 构建日志（`build_*.log` 等）
  - 实验档目录（未采用的量化档位、中间 combine 产物，例如 `w4f16_experiment/`）
- 板卡 `models/`：只放最终 bmodel（见 board_deploy_workflow.md）

> ⚠️ **上述杂物全部被 `.gitignore` 排除**（根 `.gitignore` 的 `**/models/**`），因此 `tools/check_repo_structure.py` 的默认模式看不见它们，会给出「0 个问题」的假绿灯。每轮编译或端到端验证后必须用 `--disk` 模式复查。

## .gitignore 配置

仓库根 `.gitignore` 的实际规则：

```gitignore
# 产物与中间物整体排除
*.onnx
*.onnx.data
*.bmodel
*.npz
**/models/**
!**/models/**/.gitkeep
```

验证输出的规则写在**各模型自己的 `.gitignore`** 里（路径相对该模型目录）。以 moonshine 为例：

```gitignore
python/test/outputs/debug/
!python/test/outputs/baseline/
```

注意：目录级排除后反选不生效，`.gitkeep` / `README` / `tokens.txt` 等需要入库的小文件必须 `git add -f`。**不要在 `.gitignore` 里写 `{model}/` 或 `<model>/` 这类占位符**——那不是合法模式，不会匹配任何路径。

---

**版本**: v1.2（2026-09-03：具体列出 `models/BM1684X/` 内禁止出现的产物类型；点明这些杂物被 gitignore、默认检查模式看不见，需 `--disk` 复查；`.gitignore` 片段换成仓库真实规则，移除 `{model}` 占位符写法）
v1.1（2026-08-14：代码模板移交 subagent、补模型产物目录约定）
