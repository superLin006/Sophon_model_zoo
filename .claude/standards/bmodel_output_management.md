# bmodel 输出文件管理规范 v1.0

---

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

---

## 各子目录内容

### `baseline/` — ground truth

```
baseline/
├── result.json       # 完整输出（text、tokens、shape 等）
└── result.txt        # 纯文本输出（ASR/NLP 用）
```

### `onnx/` — ONNX 验证输出

```
onnx/
├── result.json
└── diff_vs_baseline.txt   # 精度对比报告（mean_abs_diff 等）
```

### `debug/` — C++ 对比用中间输出

```
debug/
├── input_features.npy     # 预处理后的模型输入（必须保存）
├── model_output.npy       # 模型原始输出（必须保存）
├── encoder_output.npy     # encoder 输出（Encoder-Decoder 模型）
├── decoder_logits.npy     # decoder logits（Encoder-Decoder 模型）
└── *.npy                  # 其他需要 C++ 对比的中间结果
```

**格式**：统一使用 `.npy`（numpy 和 C++ 都能读）

---

## 保存规范

直接在测试脚本中保存，不要创建独立的工具文件：

```python
import numpy as np, json
from pathlib import Path

output_dir   = Path(__file__).parent / "outputs"
baseline_dir = output_dir / "baseline"
debug_dir    = output_dir / "debug"
for d in [baseline_dir, debug_dir]:
    d.mkdir(parents=True, exist_ok=True)

# 保存关键中间输出（给 C++ 对比用）
np.save(debug_dir / "input_features.npy", input_features)
np.save(debug_dir / "model_output.npy",   model_output.numpy())

# 保存最终结果
with open(baseline_dir / "result.json", "w", encoding="utf-8") as f:
    json.dump({"text": result_text}, f, ensure_ascii=False, indent=2)
with open(baseline_dir / "result.txt", "w", encoding="utf-8") as f:
    f.write(result_text + "\n")
```

---

## .gitignore 配置

```gitignore
# debug 输出可重新生成，不入库
{model}/python/test/outputs/debug/

# 保留 baseline 作为验证基准
!{model}/python/test/outputs/baseline/
```

---

**版本**: v1.0
