# Sophon Model Zoo

Sophon BM1684 平台深度学习模型移植工作区，基于 SDK-23.09 LTS SP4。

## 项目结构

```
Sophon_model_zoo/
├── 0_Toolkits/               # SDK-23.09 LTS SP4（不上传 Git）
├── 1_third_party/            # 第三方库
├── 2_utils/                  # 公共工具脚本
├── docker/                   # Docker 环境配置
│   ├── run_docker.sh         # 一键启动 TPU-MLIR 容器
│   └── README.md             # Docker 使用说明
├── environment.yml           # Conda 环境（用于模型导出）
└── TUTORIAL.md               # 完整移植教程
```

## 支持的模型

| 模型 | 类型 | 精度 | 状态 |
|------|------|------|------|
| (待添加) | - | FP32 | - |

## 技术栈

- **平台**: Sophon BM1684
- **SDK**: SDK-23.09 LTS SP4
- **转换工具**: TPU-MLIR (Docker: `sophgo/tpuc_dev:latest`)
- **支持精度**: FP32（BM1684 主要精度）、INT8（需校准数据）

## 快速开始

### 1. 准备 Conda 环境（用于模型导出）

```bash
conda env create -f environment.yml
conda activate sophon-export
```

### 2. 启动 TPU-MLIR Docker 容器（用于模型转换）

```bash
# 首次使用先拉取镜像
docker pull sophgo/tpuc_dev:latest

# 启动容器
./docker/run_docker.sh
```

### 3. 转换流程

```
PyTorch → ONNX（Conda 环境）→ bmodel（Docker 容器内）
```

详见 [TUTORIAL.md](TUTORIAL.md)。

## 转换链路

```
PyTorch (.pt)
    ↓  export_onnx.py  [conda: sophon-export]
ONNX (.onnx)
    ↓  model_transform.py  [Docker]
MLIR (.mlir)
    ↓  model_deploy.py --chip bm1684 --quantize F32  [Docker]
BModel (.bmodel)
```
