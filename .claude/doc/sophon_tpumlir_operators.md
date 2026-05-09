# Sophon TPU-MLIR ONNX 算子支持列表

**数据来源**: TPU-MLIR 源码 `python/transform/OnnxConverter.py` (onnxop_factory)
**工具版本**: sophgo/tpuc_dev:latest
**目标芯片**: BM1684 / BM1684X / BM1688
**更新日期**: 2026-05-09

---

## ✅ 支持的 ONNX 算子（约 130+）

### A - C
| 算子 | 说明 |
|------|------|
| Abs | 绝对值 |
| Acos | 反余弦 |
| Add | 加法 |
| And | 逻辑与 |
| ArgMax | 最大值索引 |
| ArgMin | 最小值索引 |
| Atan | 反正切 |
| Atanh | 反双曲正切 |
| AveragePool | 平均池化 |
| BatchNormalization | 批归一化 |
| Cast | 类型转换 |
| Ceil | 向上取整 |
| Clip | 数值截断 |
| Concat | 张量拼接 |
| Constant | 常量 |
| ConstantOfShape | 固定值张量 |
| Conv | 2D 卷积 |
| ConvTranspose | 转置卷积 |
| Cos | 余弦 |
| CumSum | 累积求和 |

### D - G
| 算子 | 说明 |
|------|------|
| DepthToSpace | Depth 转 Space |
| DequantizeLinear | 线性反量化 |
| Div | 除法 |
| Dropout | Dropout（推理时透传）|
| Einsum | 爱因斯坦求和 |
| Elu | ELU 激活 |
| Equal | 等于比较 |
| Erf | 误差函数 |
| Exp | 指数 |
| Expand | 张量广播扩展 |
| Flatten | 展平 |
| Floor | 向下取整 |
| Gather | 索引查找（✅ 支持 Embedding）|
| GatherElements | 按元素索引 |
| GatherND | 多维索引 |
| GELU | GELU 激活 |
| Gemm | 通用矩阵乘（含 bias）|
| GlobalAveragePool | 全局平均池化 |
| GlobalMaxPool | 全局最大池化 |
| Greater | 大于比较 |
| GreaterOrEqual | 大于等于 |
| GridSample | 网格采样 |
| GroupNormalization | 组归一化 |
| GRU | GRU 循环单元 |

### H - L
| 算子 | 说明 |
|------|------|
| HardSigmoid | Hard Sigmoid |
| HardSwish | Hard Swish |
| Identity | 恒等映射 |
| If | 条件控制流 |
| InstanceNormalization | 实例归一化 |
| LayerNormalization | 层归一化 |
| LeakyRelu | Leaky ReLU |
| Less | 小于比较 |
| LessOrEqual | 小于等于 |
| Log | 对数 |
| LogSoftmax | Log Softmax |
| Loop | 循环控制流 |
| LRN | 局部响应归一化 |
| LSTM | LSTM 循环单元 |

### M - P
| 算子 | 说明 |
|------|------|
| MatMul | 矩阵乘法 |
| Max | 逐元素最大值 |
| MaxPool | 最大池化 |
| Min | 逐元素最小值 |
| Mod | 取模 |
| Mul | 乘法 |
| Neg | 取负 |
| NonMaxSuppression | NMS |
| NonZero | 非零元素索引 |
| Not | 逻辑非 |
| OneHot | One-Hot 编码 |
| Or | 逻辑或 |
| Pad | 填充 |
| PRelu | 参数化 ReLU |
| Pow | 幂运算 |

### Q - Z
| 算子 | 说明 |
|------|------|
| QuantizeLinear | 线性量化 |
| Range | 序列生成 |
| Reciprocal | 倒数 |
| ReduceL1 | L1 归约 |
| ReduceL2 | L2 归约 |
| ReduceLogSumExp | LogSumExp 归约 |
| ReduceMax | 最大值归约 |
| ReduceMean | 均值归约 |
| ReduceMin | 最小值归约 |
| ReduceProd | 乘积归约 |
| ReduceSum | 求和归约 |
| Relu | ReLU 激活 |
| Reshape | 形状变换 |
| Resize | 尺寸调整（插值）|
| ReverseSequence | 序列翻转 |
| RoiAlign | ROI 对齐（检测用）|
| Round | 四舍五入 |
| ScatterElements | 按元素散射 |
| ScatterND | 多维散射 |
| Shape | 获取形状 |
| Sigmoid | Sigmoid 激活 |
| Sign | 符号函数 |
| Sin | 正弦 |
| Slice | 切片 |
| Softmax | Softmax |
| Softplus | Softplus 激活 |
| SpaceToDepth | Space 转 Depth |
| Split | 分割 |
| Sqrt | 平方根 |
| Squeeze | 压缩维度 |
| Sub | 减法 |
| Sum | 多输入求和 |
| Tanh | Tanh 激活 |
| Tile | 张量重复 |
| TopK | Top-K 选择（✅ 支持 Beam Search）|
| Transpose | 转置 |
| Trilu | 上/下三角掩码（✅ 支持 causal mask）|
| Unsqueeze | 增加维度 |
| Upsample | 上采样 |
| Where | 条件选择（✅ 支持 masked_fill）|
| Xor | 逻辑异或 |

---

## 🆚 与 MTK MDLA 5.3 关键差异对比

| 算子 | MTK MDLA 5.3 | Sophon TPU-MLIR | 影响场景 |
|------|:-----------:|:---------------:|---------|
| **Gather** | ❌ | ✅ | Embedding 层可在 NPU 跑 |
| **Equal / Greater / Less** | ❌ | ✅ | Attention mask 动态计算 |
| **Where** | ❌ | ✅ | masked_fill 直接支持 |
| **TopK** | ❌ | ✅ | Beam Search 在 NPU 跑 |
| **Log / Ceil / Floor / Round** | ❌ | ✅ | 数学函数完整支持 |
| **LogSoftmax** | ❌ | ✅ | 直接使用无需拆分 |
| **LeakyRelu** | ❌ | ✅ | 检测模型常用 |
| **Trilu** | ❌ | ✅ | causal mask 直接支持 |
| **NonZero** | ❌ | ✅ | 动态索引 |
| **If / Loop** | ❌ | ⚠️ 有限支持 | 控制流部分回退 CPU |

---

## ⚠️ 需要注意的情况

### 1. 动态 Shape
- TPU-MLIR 支持 `--dynamic` 参数，但性能弱于固定 shape
- **建议**：编译时仍固定 `--input_shapes`，运行时 padding

### 2. 控制流算子（If / Loop）
- 能转换，但复杂控制流可能 fallback 到 CPU
- **验证方式**：`bmrt_test --bmodel xxx.bmodel` 观察各层设备

### 3. BM1684 精度限制
- BM1684：仅 **FP32**（不支持 FP16/BF16）
- BM1684X / BM1688：支持 FP16、BF16、INT8

### 4. 自定义算子
- PyTorch 自定义 op 无法转换，需在导出前替换为标准算子

### 5. 大 Tensor
- 超出芯片显存限制时会报错
- BM1684 显存 12GB，单层 Tensor 不能超限

---

## 🔍 算子兼容性检查方法

```bash
# 方法1：直接尝试 model_transform，报错即不支持
model_transform.py --model_def model.onnx --mlir model.mlir ...

# 方法2：用 onnx 工具提前检查图中算子
python3 -c "
import onnx
model = onnx.load('model.onnx')
ops = set(n.op_type for n in model.graph.node)
print('算子列表:', sorted(ops))
"
```

---

## 📚 参考资料

- TPU-MLIR 源码: https://github.com/sophgo/tpu-mlir
- OnnxConverter: `python/transform/OnnxConverter.py`
- 官方快速开始文档: https://github.com/sophgo/tpu-mlir/blob/master/README_cn.md

---

**维护者**: 算法工程师 + Claude Code
**最后更新**: 2026-05-09
