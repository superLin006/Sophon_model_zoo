# VITS-MeloTTS BM1684X — Python 工具说明

本目录包含从原始 ONNX 模型到 BM1684X bmodel 的完整转换流程。

---

## 前提条件

| 工具 | 说明 |
|------|------|
| `sophgo/tpuc_dev:v3.2` | TPU-MLIR Docker，用于编译 bmodel |
| `sophon-cross-build` | 交叉编译 Docker，用于编译 C++ 推理程序 |
| `0_Toolkits/soc-sdk-sp4` | Sophon SOC SDK（头文件 + 库） |
| `0_Toolkits/tpu_mlir*.whl` | TPU-MLIR Python 包 |

所有命令从**仓库根目录** `Sophon_model_zoo/` 执行。

---

## 文件说明

```
python/
├── make_tpu_model.py    # Step 1: 生成 model_tpu.onnx（去除 TPU 不支持的算子）
├── make_split_models.py # Step 2: 拆分出 part_a_encoder.onnx + part_c_flow_decoder.onnx
├── gen_bmodel.sh        # Step 3: 编译 bmodel（在 TPU-MLIR Docker 内执行）
├── export_onnx.py       # 仅用于从 PyTorch 重新导出 model.onnx（通常不需要）
└── fix_onnx.py          # 历史遗留，已被 make_tpu_model.py 取代，不再使用
```

---

## 转换步骤

### Step 1：生成 model_tpu.onnx

去除原始 `model.onnx` 中 TPU 不支持的算子（NonZero、RandomNormalLike），固化 sid=1。

```bash
docker run --rm \
    -v $(pwd)/vits-melo-tts-zh_en:/workspace \
    sophgo/tpuc_dev:v3.2 \
    python3 /workspace/python/make_tpu_model.py
```

输出：`models/onnx/vits-melo-tts-zh_en/model_tpu.onnx`

### Step 2：拆分子图

将 `model_tpu.onnx` 拆成两个静态 shape 的子图：
- **Part A**（enc_p + DP）：输入 tokens/tones，输出 dp_w、h、x_mask
- **Part C**（Flow + Decoder）：输入 z_p[1,192,256]，输出 audio[1,1,131072]

```bash
docker run --rm \
    -v $(pwd)/vits-melo-tts-zh_en:/workspace \
    sophgo/tpuc_dev:v3.2 \
    python3 /workspace/python/make_split_models.py
```

输出：
- `models/onnx/vits-melo-tts-zh_en/part_a_encoder.onnx`
- `models/onnx/vits-melo-tts-zh_en/part_c_flow_decoder.onnx`

### Step 3：编译 bmodel

```bash
docker run --rm \
    -v $(pwd):/repo \
    sophgo/tpuc_dev:v3.2 \
    bash /repo/vits-melo-tts-zh_en/python/gen_bmodel.sh F32
```

将 `F32` 替换为 `F16` 可生成半精度版本。

输出：
- `models/BM1684X/vits_part_a_F32.bmodel`
- `models/BM1684X/vits_part_c_F32.bmodel`

---

## 为什么要拆成三段？

原始 `model.onnx` 无法整图编译到 TPU，原因：

1. **SDP 含 NonZero×21**：算子输出 shape 依赖运行时数值，TPU-MLIR 无法静态推断
2. **Flow 含 RandomNormalLike**：TPU-MLIR v1.28.1 未实现该算子
3. **MAS 含 Range**：输出 shape 依赖 T_mel（运行时决定），无法静态编译

解决方案：
- SDP 替换为零常量（只用 DP，确定性时长预测）
- RandomNormalLike 分支绕过（noise_scale=0 时贡献为零）
- MAS 保留在 CPU（仅约 8ms）

最终推理链路：**Part A（TPU 6ms）→ MAS（CPU 8ms）→ Part C（TPU 305ms）= RTF 0.12**

---

## 注意事项

- bmodel 的 T_mel 固定为 256（最多生成约 3s 音频），推理时 z_p 会 pad 到 256，输出截取实际有效帧
- BM1684X SDK 无 `BM_INT64`，Part A 的 token/tone 输入需在 C++ 侧 cast 为 int32 再上传
- `matmul_ht` 中操作 Part A 输出的 h 时，行步长必须用 `L_MAX=128`，而非 `seq_len`（详见知识库）
