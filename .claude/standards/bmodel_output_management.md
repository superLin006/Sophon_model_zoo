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

- `models/onnx/`：只保留 `.onnx` 文件（中间 mlir/profile 等转换产物不保留，可随时重新生成）
- `models/BM1684X/`：保留**最终采用**的 bmodel + 重建用组件 bmodel；实验产物（未采用的量化档位、中间 combine）清理掉
- 板卡 `models/`：只放最终 bmodel（见 board_deploy_workflow.md）

## .gitignore 配置

```gitignore
# debug 输出可重新生成，不入库
{model}/python/test/outputs/debug/

# 保留 baseline 作为验证基准
!{model}/python/test/outputs/baseline/
```

---

**版本**: v1.1（2026-08-14：代码模板移交 subagent、补模型产物目录约定）
