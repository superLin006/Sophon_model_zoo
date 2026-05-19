# Docker 环境说明

## 镜像

`sophgo/tpuc_dev:latest` — TPU-MLIR 官方开发镜像，内含所有转换工具：
- `model_transform.py`
- `model_deploy.py`
- `run_calibration.py`
- `bmrt_test`

## 拉取镜像

```bash
docker pull sophgo/tpuc_dev:latest
```

如果拉取失败，可手动下载 `tpuc_dev_v3.4.tar.gz` 后加载：
```bash
docker load -i tpuc_dev_v3.4.tar.gz
```

## 启动容器

```bash
./docker/run_docker.sh
# 或指定容器名
./docker/run_docker.sh my-sophon
```

容器内 `/workspace` 对应本仓库根目录。

## 标准转换流程（容器内执行）

```bash
# Step 1: ONNX → MLIR
model_transform.py \
    --model_name <name> \
    --model_def <model>.onnx \
    --input_shapes [[1,3,224,224]] \
    --mlir <name>.mlir

# Step 2: MLIR → bmodel (BM1684 只支持 FP32)
model_deploy.py \
    --mlir <name>.mlir \
    --quantize F32 \
    --chip bm1684 \
    --model <name>_bm1684_fp32.bmodel

# Step 3: 验证
bmrt_test --bmodel <name>_bm1684_fp32.bmodel
```

## 注意事项

- **BM1684 仅支持 FP32**，不支持 FP16/BF16/INT8（INT8 理论支持但需校准）
- 转换必须在 x86 主机上进行，不需要连接实体板卡
- 每个模型目录下的 `python/` 存放转换脚本，`models/BM1684/` 存放输出的 bmodel
