# 移植教程

## 环境准备

### 1. Conda 环境（模型导出用）

```bash
conda env create -f environment.yml
conda activate sophon-export
```

### 2. Docker 环境（模型转换用）

```bash
# 拉取 TPU-MLIR 官方镜像
docker pull sophgo/tpuc_dev:latest

# 启动容器（自动挂载当前仓库到 /workspace）
./docker/run_docker.sh
```

---

## 标准移植流程

### Step 1：导出 ONNX（Conda 环境）

```bash
conda activate sophon-export
cd <model_name>/python

python export_onnx.py \
    --weights model.pt \
    --output model.onnx

# 验证 ONNX 合法性
python -c "import onnx; onnx.checker.check_model('model.onnx'); print('OK')"

# 简化图（推荐）
python -m onnxsim model.onnx model_sim.onnx
```

### Step 2：ONNX → MLIR（Docker 容器内）

```bash
# 进入容器
./docker/run_docker.sh

# 容器内执行
cd /workspace/<model_name>/python

model_transform.py \
    --model_name <name> \
    --model_def model_sim.onnx \
    --input_shapes [[1,3,224,224]] \
    --test_input input.npy \
    --test_result top_outputs.npz \
    --mlir <name>.mlir
```

### Step 3：MLIR → BModel（Docker 容器内，BM1684 用 FP32）

```bash
model_deploy.py \
    --mlir <name>.mlir \
    --quantize F32 \
    --chip bm1684 \
    --test_input <name>_in_f32.npz \
    --test_reference top_outputs.npz \
    --tolerance 0.99,0.99 \
    --model <name>_bm1684_fp32.bmodel
```

### Step 4：验证（Docker 容器内）

```bash
bmrt_test --bmodel <name>_bm1684_fp32.bmodel
```

---

## 新模型目录模板

每个模型按以下结构建目录：

```
<model_name>/
├── README.md               # 模型说明、性能指标
├── python/
│   ├── export_onnx.py      # PyTorch → ONNX
│   ├── gen_bmodel.sh       # 一键转换脚本（Docker 内执行）
│   ├── test_onnx.py        # ONNX 精度验证
│   └── test_bmodel.py      # bmodel 精度验证
├── models/
│   └── BM1684/             # 输出 bmodel（不上传 git）
└── test_data/              # 测试样本（不上传 git）
```

---

## BM1684 注意事项

| 项目 | 说明 |
|------|------|
| 支持精度 | FP32（主要），INT8（需校准数据） |
| 不支持精度 | FP16、BF16 |
| 动态 Shape | 有限支持，建议固定 shape |
| 算子覆盖 | 比 MTK/RK 宽，GATHER/逻辑运算/TOPK 均支持 |
| CPU Fallback | 用 bmrt_test 确认各层在 TPU 上运行 |

---

## INT8 量化（可选）

```bash
# 准备 100~200 条校准数据（npy 格式）
mkdir calibration_data

# 生成校准表
run_calibration.py <name>.mlir \
    --dataset ./calibration_data \
    --input_num 100 \
    -o <name>_cali_table

# 编译 INT8 bmodel
model_deploy.py \
    --mlir <name>.mlir \
    --quantize INT8 \
    --calibration_table <name>_cali_table \
    --chip bm1684 \
    --model <name>_bm1684_int8.bmodel
```
