# Sophon BM1684 移植知识库

**针对平台**: BM1684 (SDK-23.09 LTS SP4)
**转换工具**: TPU-MLIR (Docker: sophgo/tpuc_dev:latest)
**更新日期**: 2026-05-09

---

## 🔗 转换链路

```
PyTorch (.pt)
    ↓  [conda: sophon-export]  export_onnx.py
ONNX (.onnx)
    ↓  [Docker]  model_transform.py  →  .mlir + 精度基准
    ↓  [Docker]  model_deploy.py     →  .bmodel
    ↓  [板端]    bmrt_test 验证
```

## ⚠️ BM1684 精度限制

- **仅支持 FP32**，不支持 FP16 / BF16
- INT8 理论支持，但需要校准数据集（run_calibration.py）
- BM1684X 才支持 FP16，注意区分

## ✅ TPU-MLIR 支持的 ONNX 算子（相比 MTK 的重大改进）

以下算子 MTK MDLA 不支持，但 Sophon TPU-MLIR 支持：
- **GATHER** — Embedding 层可以在 NPU 上跑
- **逻辑运算** — Equal, Greater, Less, Where 等全支持
- **TOPK** — Beam Search 可在 NPU 上跑
- **LOG / CEIL / FLOOR / ROUND** — 数学函数全支持
- **LOG_SOFTMAX** — 直接支持
- **LEAKY_RELU** — 直接支持
- **5D Tensor** — 支持（KV Cache 设计更自由）

## ❌ 已知不兼容 / 需注意的情况

### 1. 动态 Shape
- TPU-MLIR 支持有限的动态 shape（--dynamic 参数）
- 建议仍然固定输入 shape，性能更好
- 处理方式同 MTK：padding 到固定长度

### 2. 自定义算子
- PyTorch 自定义 op 无法直接转换
- 解决：在导出前用标准算子替换，或拆分模型

### 3. 控制流 (If/Loop)
- 能转换但部分情况会 fallback 到 CPU 执行
- 用 bmrt_test 检查各层运行设备

### 4. CPU Fallback 检测
```bash
# 检查是否有层在 CPU 上跑（性能杀手）
bmrt_test --bmodel model.bmodel
# 观察输出中每层的设备信息
```

## 📐 标准转换命令

### FP32（BM1684 主要精度）
```bash
# Step 1: ONNX → MLIR（带精度基准）
model_transform.py \
    --model_name mymodel \
    --model_def mymodel.onnx \
    --input_shapes [[1,3,224,224]] \
    --mean 0.0,0.0,0.0 \
    --scale 0.0039216,0.0039216,0.0039216 \
    --test_input input.npy \
    --test_result mymodel_top_outputs.npz \
    --mlir mymodel.mlir

# Step 2: MLIR → BModel
model_deploy.py \
    --mlir mymodel.mlir \
    --quantize F32 \
    --chip bm1684 \
    --test_input mymodel_in_f32.npz \
    --test_reference mymodel_top_outputs.npz \
    --tolerance 0.99,0.99 \
    --model mymodel_bm1684_fp32.bmodel
```

### INT8（需要校准数据）
```bash
# 先生成校准表
run_calibration.py mymodel.mlir \
    --dataset ./calibration_data \
    --input_num 100 \
    -o mymodel_cali_table

# 再编译
model_deploy.py \
    --mlir mymodel.mlir \
    --quantize INT8 \
    --calibration_table mymodel_cali_table \
    --chip bm1684 \
    --model mymodel_bm1684_int8.bmodel
```

## 📁 每个模型目录标准结构

```
<model_name>/
├── README.md
├── python/
│   ├── export_onnx.py        # PyTorch → ONNX（conda 环境跑）
│   ├── gen_bmodel.sh         # ONNX → bmodel（Docker 内跑）
│   ├── test_onnx.py          # ONNX 精度验证
│   └── test_bmodel.py        # bmodel 精度验证（需 sophon-sail）
├── models/
│   └── BM1684/               # 输出的 bmodel（不上传 git）
└── test_data/                # 测试样本（不上传 git）
```

## 🎯 移植检查清单

### 导出前
- [ ] 确认模型可在 PyTorch 正常推理
- [ ] 梳理输入输出 shape，确认是否需要固定
- [ ] 检查是否有自定义算子

### ONNX 导出
- [ ] 用 `onnx.checker.check_model()` 验证合法性
- [ ] 用 `onnxsim` 简化图（减少冗余节点）
- [ ] 对比 PyTorch 和 ONNX 输出数值（差异 < 1e-5）

### MLIR 转换
- [ ] 指定正确的 `--input_shapes`
- [ ] 提供 `--test_input` 做精度基准
- [ ] 检查 tolerance 是否通过（> 0.99）

### bmodel 编译
- [ ] BM1684 使用 `--quantize F32`
- [ ] 用 `bmrt_test` 确认无 CPU fallback
- [ ] 记录推理耗时和内存占用

## 🔗 相关资源

- SDK 版本: SDK-23.09 LTS SP4
- Docker 镜像: sophgo/tpuc_dev:latest
- TPU-MLIR 文档: https://github.com/sophgo/tpu-mlir/blob/master/README_cn.md
- 参考 Demo: https://github.com/sophgo/sophon-demo/tree/release/sample

---

**维护者**: 算法工程师 + Claude Code
**最后更新**: 2026-05-09
