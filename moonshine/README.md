# Moonshine — BM1684X 移植

[Moonshine](https://github.com/usefulsensors/moonshine) 轻量流式语音识别(ASR)移植到 Sophon BM1684X。
本仓库使用 **streaming-small** 变体(10 层 decoder / 10 层 encoder,hidden 512/620,vocab 32768),精度 F32 / F16,纯 BMRuntime C++ 推理,无第三方推理框架依赖。

> 移植流程遵循 `../.claude/subagents/README.md` 四阶段(initializer → operator-analyst → python-converter → cpp-implementer),中间产物与验证记录见 `.context/`。

## 特性

- **轻量英文流式 ASR**:模型总 140M 参数,bmodel F16 仅 280MB(enc 109M + dec 171M)
- **特征提取极简**:无 STFT/mel filterbank,预处理 = 分帧 + CMVN + asinh(纯数学,C++ 实现与 HF bit 级一致,无需 fftw3)
- **单步自回归 decoder**:KV cache 增量输出(23 入 21 出),全部逻辑(embed 查表 / pos_emb / RoPE / causal mask)在模型内
- **固定输入 10s**:160000 samples → 2000 帧 → encoder 输出 [1, 500, 620]

## 快速开始

### 环境依赖

- 开发机:conda `sophon-whisper`(Python 3.10、PyTorch 2.11、transformers ≥5.14、onnx、onnxsim)
- HF 权重: `models/moonshine-streaming-small/`(UsefulSensors/moonshine-streaming-small,如缺失先下载)
- Docker: `sophgo/tpuc_dev:latest`(TPU-MLIR v1.28.1)、`sophon-cross-build`(aarch64 交叉编译)

### Step 1:导出 ONNX

```bash
cd moonshine
conda activate sophon-whisper
python python/export_onnx.py        # 导出 encoder + decoder,含内置自检(decoder 27 步 token 100% 一致)
```

产物在 `models/onnx/`(encoder/decoder sim onnx)。导出同时生成 `models/log_k.txt`(特征提取资产,已在仓库中)。

### Step 2:编译 bmodel

```bash
# 仓库根目录执行;F32 先,F16 后
docker run --rm \
  -v $(pwd)/moonshine:/workspace \
  -v $(pwd)/0_Toolkits:/toolkits \
  sophgo/tpuc_dev:latest bash /workspace/python/gen_bmodel.sh F32
docker run --rm \
  -v $(pwd)/moonshine:/workspace \
  -v $(pwd)/0_Toolkits:/toolkits \
  sophgo/tpuc_dev:latest bash /workspace/python/gen_bmodel.sh F16
```

产物在 `models/BM1684X/`。**decoder 必须 `--disable_layer_group`**(23 输入复杂图,脚本已内置)。

### Step 3:交叉编译 C++

```bash
bash moonshine/cpp/build.sh
# 产物: moonshine/cpp/build/moonshine_bm1684
```

### Step 4:上板运行

```bash
# 板卡上部署目录需含: moonshine_bm1684、models/(4 个 bmodel + tokens.txt)
./moonshine_bm1684 models/ test_data/0.wav F32
./moonshine_bm1684 models/ test_data/0.wav F16
# 参数: <model_dir> <audio.wav> [F32|F16]
```

更多 C++ 细节见 [cpp/README.md](cpp/README.md)。

## 性能(BM1684X 实测,RTF 只计特征提取 + TPU 推理)

| 精度 | 模型大小 | 0.wav(6.6s, 27 步) | RTF | 8k.wav(4.8s, 19 步) | RTF |
|------|---------|--------------------|-----|---------------------|-----|
| F32 | 548MB | 653ms(feat 14.4 + infer 639) | **0.099** | 457ms | 0.095 |
| F16 | 280MB | 296ms(feat 14.3 + infer 282) | **0.045** | 206ms | 0.043 |

- **F16 比 F32 快 ~2.2 倍,转写文本完全一致,推荐默认 F16**
- 解码精度: 板卡 27 步自回归与 Python 参考 **token 100% 一致**(F32/F16),argmax 0 步偏差
- 文本验证: 0.wav / 8k.wav 与 ORT/HF 参考转写完全一致

## 模型结构

```
Encoder: 分帧(80) + CMVN + asinh[C++ 侧] → Linear(80→620)+SiLU → CausalConv1d×2(stride 2)
         → 10×Layer(sliding-window self-attn + GELU MLP) → LayerNorm → [1, 500, 620]
Decoder(单步): token + encoder_out + cache_len + past_k/v×10(旋转后 K)
         → 模型内: embed 查表、encoder pos_emb+proj(620→512)、RoPE(partial 0.5)、causal mask
         → 10×Layer(self-attn + cross-attn + SiLU-GLU MLP) → proj_out → logits [1,1,32768]
         + new_k/v×10 增量输出(C++ 写回 cache 槽位)
解码: greedy,sos=1, eos=2,max_dec_len=128(10s 音频实测 ~27 步)
```

固定 shape 参数: `T=2000 帧(frame_len=80)`,`T_enc=500`,`max_dec_len=128`,`vocab=32768`。

## 踩过的坑

1. **TPU-MLIR 1.28.1 的 Or 算子不支持常量输入**(onnxsim 提升的 initializer 报 `operand eq not found`)→ causal mask 改嵌套 `Where` 等价实现,export_onnx.py 已加 Or 断言防回归。
2. **Conv 缺 kernel_shape**: torch 2.11 导出不写该属性,`_fix_conv_kernel_shape` 从权重推断补上。
3. **HF decoder 原地修改 `encoder_hidden_states`**(`+= pos_emb`):验证/基准循环必须每步传 clone,否则从第 2 步起双重 pos_emb。
4. **int64 输入自动降 INT32**: bmodel 内 token/cache_len 为 INT32,C++ 上传需 cast。
5. **板卡 libsophon 0.5.1**: `bmrt_tensor` 收 `bm_shape_t` 结构体(与新版 int* 签名不同)。
6. **边界效应**: 短音频尾部补零后,encoder 输出最后 ~12 帧受填充影响;与未填充 ground truth 对比时只比前 `T_enc-12` 帧。
7. **头文件命名**: 勿用 `features.h`(与 glibc 同名导致 include-guard 递归),C++ 用 `moonshine_features.h`。

## 目录结构

```
moonshine/
├── .context/               # subagent 流程文档(baseline / operator_analysis / bmodel_info)
├── python/
│   ├── export_onnx.py      # PyTorch → ONNX(方案 B: C++ 侧 CMVN+Asinh)
│   ├── gen_bmodel.sh       # ONNX → bmodel(F32/F16 通用)
│   └── test/               # test_pytorch.py(baseline) / test_onnx.py(精度验证)
├── models/                 # 本地备份: HF 权重 / onnx / bmodel(均不入库)
│   └── tokens.txt          # 词表(C++ 运行必需,入库)
├── cpp/                    # BMRuntime C++ 推理(交叉编译 + 板卡部署)
└── test_data/              # 测试音频(0.wav / 1.wav / 8k.wav)
```

## 已知限制

- **英文为主**: moonshine-streaming-small 为英文模型;10s 固定输入,更长音频需切段(16.7s 的 1.wav 超上限,未验证)。
- bmodel/ONNX/权重体积大,不入库,本地重新生成即可(脚本齐全)。
