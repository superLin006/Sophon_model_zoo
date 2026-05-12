# Sophon BM1684X 移植知识库

**目标芯片**: BM1684X (SDK-23.09 LTS SP4)
**转换工具**: TPU-MLIR v1.28.1 (Docker: `sophgo/tpuc_dev:latest`)
**更新日期**: 2026-05-12

---

## 转换链路

```
PyTorch (.pt)
    ↓  export_onnx.py  [conda: sophon-export]
ONNX (.onnx)
    ↓  model_transform.py  [Docker: sophgo/tpuc_dev:latest]
.mlir
    ↓  model_deploy.py --chip bm1684x --quantize F32|F16
.bmodel
    ↓  交叉编译 C++ 推理程序  [Docker: sophon-cross-build]
aarch64 可执行文件  →  scp 到 BM1684X 板卡
```

---

## BM1684X 支持的精度

| 精度 | 支持 | 说明 |
|------|------|------|
| FP32 | ✅ | 精度最高，文件较大 |
| FP16 | ✅ | 推荐生产使用，速度约 4-8x，精度损失极小 |
| INT8 | ✅ | 需要校准数据集（run_calibration.py），本项目暂未使用 |

---

## 标准转换命令

### FP32 / FP16

```bash
# Step 1: ONNX → MLIR
model_transform.py \
    --model_name mymodel \
    --model_def mymodel_sim.onnx \
    --input_shapes [[1,3,224,224]] \
    --mlir mymodel.mlir

# Step 2: MLIR → bmodel (F32)
model_deploy.py \
    --mlir mymodel.mlir \
    --quantize F32 \
    --chip bm1684x \
    --model mymodel_F32.bmodel

# Step 2: MLIR → bmodel (F16)
model_deploy.py \
    --mlir mymodel.mlir \
    --quantize F16 \
    --chip bm1684x \
    --model mymodel_F16.bmodel
```

### INT8（需要校准数据）

```bash
run_calibration.py mymodel.mlir \
    --dataset ./calibration_data \
    --input_num 100 \
    -o mymodel_cali_table

model_deploy.py \
    --mlir mymodel.mlir \
    --quantize INT8 \
    --calibration_table mymodel_cali_table \
    --chip bm1684x \
    --model mymodel_INT8.bmodel
```

---

## 常见问题 & 已验证的 Workaround

### 1. Conv 节点缺少 `kernel_shape` 属性

**触发**：新版 `torch.onnx.export`（opset 17）导出时 Conv 节点不写 `kernel_shape`，`model_transform.py` 报错。

**修复**：导出后用以下函数补全：
```python
def _fix_conv_kernel_shape(model):
    init_map = {init.name: init for init in model.graph.initializer}
    for node in model.graph.node:
        if node.op_type != "Conv":
            continue
        if any(attr.name == "kernel_shape" for attr in node.attribute):
            continue
        weight_name = node.input[1]
        if weight_name not in init_map:
            continue
        kernel_shape = list(init_map[weight_name].dims[2:])
        node.attribute.append(onnx.helper.make_attribute("kernel_shape", kernel_shape))
    return model
```

### 2. 含 KV Cache 的模型：dummy tensor 不能用 `* n` 乘法创建

**触发**：`[tensor] * n_layer` 创建的是同一个对象的 n 个引用，ONNX tracer 视为同一张量，constant folding 将 layer 1-N 的 KV 输入消除，bmodel 输入数量比预期少。

**修复**：必须用列表推导式，每层独立创建：
```python
# 错误
dummy_k = [torch.zeros(1, 448, 512)] * 6

# 正确
dummy_k = [torch.zeros(1, 448, 512) for _ in range(6)]
```

### 3. 复杂 Decoder 图需加 `--disable_layer_group`

**触发**：输入输出数量多（如 Whisper Decoder 28 输入 25 输出）时，`model_deploy.py` 的 layer group 优化不稳定，报错或生成异常 bmodel。

**修复**：
```bash
model_deploy.py --mlir ... --disable_layer_group --chip bm1684x ...
```

### 4. tpu_mlir pip install 在无外网容器内超时

**触发**：容器内 `pip install tpu_mlir*.whl` 尝试拉取网络依赖，无外网时挂起。

**修复**：先 `--no-deps` 安装，失败再正常安装（依赖已预装在容器内）：
```bash
WHL=$(ls /toolkits/tpu_mlir*.whl | head -1)
pip install "$WHL" -q --no-deps 2>/dev/null || pip install "$WHL" -q
```

### 5. 交叉编译产物 glibc 版本过高

**触发**：WSL 原生 gcc（15.x）编译的 aarch64 二进制在 BM1684X 板卡（Ubuntu 20.04, glibc 2.31）上报 `GLIBC_2.34 not found`。

**修复**：使用 Ubuntu 20.04 Docker 镜像（gcc 9.4）做交叉编译，产物最高依赖 glibc 2.29：
```dockerfile
FROM ubuntu:20.04
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    cmake g++-aarch64-linux-gnu gcc-aarch64-linux-gnu make
```

### 6. 确认板卡型号：BM1684 vs BM1684X

在板卡上执行：
```bash
bm-smi
# 输出包含 "1684X-SOC" 即为 BM1684X
```

`gen_bmodel.sh` 中必须使用 `--chip bm1684x`（非 `bm1684`），否则运行时报：
```
runtime arch[BM1684X] is not the same with bmodel arch[BM1684]
```

---

## 项目目录规范

```
<model_name>/
├── README.md               # 用法、性能、踩坑记录
├── python/
│   ├── export_onnx.py      # PyTorch → ONNX（开发机执行）
│   └── gen_bmodel.sh       # ONNX → bmodel（TPU-MLIR Docker 内执行）
├── cpp/
│   ├── CMakeLists.txt
│   ├── build.sh            # 交叉编译（sophon-cross-build Docker）
│   └── src/
├── models/
│   ├── BM1684X/            # bmodel + 资产文件（不入库，.gitkeep 占位）
│   └── onnx/               # 中间 ONNX（不入库，.gitkeep 占位）
└── test_data/              # 测试音频/图片（不入库，.gitkeep 占位）
```

---

## 移植检查清单

### ONNX 导出
- [ ] 模型可在 PyTorch 正常推理，输出 shape 符合预期
- [ ] dummy tensor 用列表推导式创建（KV Cache 场景必须）
- [ ] `onnx.checker.check_model()` 通过
- [ ] `_fix_conv_kernel_shape()` 处理 Conv 节点
- [ ] onnxsim 简化后验证精度一致

### bmodel 转换
- [ ] `--chip bm1684x`（注意不是 bm1684）
- [ ] `--quantize F32` 或 `F16`
- [ ] 复杂图（多输入输出）加 `--disable_layer_group`
- [ ] `bmrt_test --bmodel xxx.bmodel` 确认输入数量正确

### 交叉编译
- [ ] 使用 `sophon-cross-build` 镜像（Ubuntu 20.04 + gcc 9.4）
- [ ] 链接 `.so` 时设置 rpath：`-Wl,-rpath,/opt/sophon/libsophon-0.5.1/lib`
- [ ] 静态库依赖全部列出（如 kaldi-native-fbank 需同时链接 libkissfft-float.a）

### 板卡部署
- [ ] `bm-smi` 确认芯片型号
- [ ] 模型加载正常，输入输出数量与导出时一致
- [ ] 统计 RTF 时只计特征提取 + TPU 推理（不含模型加载）

---

## 相关资源

- SDK: SDK-23.09 LTS SP4
- TPU-MLIR: https://github.com/sophgo/tpu-mlir
- Sophon Demo 参考: https://github.com/sophgo/sophon-demo
