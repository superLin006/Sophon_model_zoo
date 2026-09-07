# VITS-MeloTTS BM1684X — Python 工具说明

本目录包含从原始 ONNX 模型到 BM1684X bmodel 的完整转换流程。

---

## 前提条件

| 工具 | 说明 |
|------|------|
| 公共 TPU-MLIR 容器 | 用于编译 bmodel（容器名默认为 `sophon-tpumlir-v128`） |
| `sophon-cross-build` | 交叉编译 Docker，用于编译 C++ 推理程序 |
| `0_Toolkits/soc-sdk-sp4` | Sophon SOC SDK（头文件 + 库） |
| `0_Toolkits/tpu_mlir*.whl` | TPU-MLIR Python 包 |

- ONNX 导出/处理使用独立 conda 环境；从仓库根目录执行 `conda create -n sophon-vits-melo-tts-zh-en python=3.10 -y`、`conda run -n sophon-vits-melo-tts-zh-en python -m pip install -r vits-melo-tts-zh_en/requirements.txt`。需要重新导出时，另准备 MeloTTS 源码目录并通过 `PYTHONPATH` 指定；公共 TPU-MLIR 容器负责 bmodel 编译。

所有命令从**仓库根目录** `Sophon_model_zoo/` 执行。

---

## 文件说明

```
python/
├── make_tpu_model.py    # Step 1: 生成 model_tpu.onnx（去除 TPU 不支持的算子）
├── make_split_models.py # Step 2: 拆分 Part A/C1/C2，并生成 C2 window 子图
├── gen_bmodel.sh        # Step 3: 编译 bmodel（在 TPU-MLIR Docker 内执行）
├── export_onnx.py       # 使用 MeloTTS 权重导出中英混合单体 model.onnx（L 动态，仅作上游输入）
```

> **原始 `model.onnx` 来源**：`export_onnx.py` 使用上游 MeloTTS 的中英混合模型结构导出单体 ONNX。推荐先从 ModelScope 下载 `myshell-ai/MeloTTS-Chinese` 的 `checkpoint.pth` 与 `config.json`，再准备与导出脚本兼容的 MeloTTS 源码：
>
> ```bash
> conda run -n sophon-vits-melo-tts-zh-en python -m pip install modelscope==1.37.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
> conda run -n sophon-vits-melo-tts-zh-en python -c "from modelscope import snapshot_download; snapshot_download('myshell-ai/MeloTTS-Chinese', local_dir='/tmp/melotts_chinese')"
> git clone --depth 1 https://github.com/myshell-ai/MeloTTS.git /tmp/MeloTTS
> git -C /tmp/MeloTTS checkout 209145371cff8fc3bd60d7be902ea69cbdb7965a
> conda run -n sophon-vits-melo-tts-zh-en python -m pip install -r /tmp/MeloTTS/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
> conda run -n sophon-vits-melo-tts-zh-en env PYTHONPATH=/tmp/MeloTTS MELOTTS_CHECKPOINT=/tmp/melotts_chinese/checkpoint.pth MELOTTS_CONFIG=/tmp/melotts_chinese/config.json python vits-melo-tts-zh_en/python/export_onnx.py
> ```
>
> 导出脚本会生成中英混合词典、`tokens.txt` 和 `model.onnx`；随后按 Step 1→2→3 执行。`model.onnx`、`model_tpu.onnx` 和中间文件不属于最终交付目录。

---

## 转换步骤

### Step 1：生成 model_tpu.onnx

去除原始 `model.onnx` 中 TPU 不支持的算子（NonZero、RandomNormalLike），固化 sid=1。

```bash
conda run -n sophon-vits-melo-tts-zh-en --no-capture-output python vits-melo-tts-zh_en/python/make_tpu_model.py
```

输出：`vits-melo-tts-zh_en/models/onnx/vits-melo-tts-zh_en/model_tpu.onnx`

### Step 2：拆分子图

将 `model_tpu.onnx` 拆成三个静态 shape 子图：
- **Part A**（enc_p + DP）：输入 tokens/tones，输出 dp_w、h、x_mask
- **Part C1**（Flow）：输入 z_p、y_mask，输出 flow 结果
- **Part C2**（Decoder）：输入 flow 结果，输出 audio

```bash
conda run -n sophon-vits-melo-tts-zh-en --no-capture-output python vits-melo-tts-zh_en/python/make_split_models.py
```

输出：
- `vits-melo-tts-zh_en/models/onnx/vits-melo-tts-zh_en/part_a_encoder.onnx`
- `vits-melo-tts-zh_en/models/onnx/vits-melo-tts-zh_en/part_c1_flow.onnx`
- `vits-melo-tts-zh_en/models/onnx/vits-melo-tts-zh_en/part_c2_decoder_stream_W128_R16.onnx`

### Step 3：编译 bmodel

```bash
# 仅编译 C2 window bmodel（复用已有 Part A/C1）
docker exec sophon-tpumlir-v128 bash /workspace/vits-melo-tts-zh_en/python/gen_bmodel.sh F16 --stream
```

输出：
- `vits-melo-tts-zh_en/models/BM1684X/vits_part_a_F16.bmodel`
- `vits-melo-tts-zh_en/models/BM1684X/vits_part_c1_F16.bmodel`
- `vits-melo-tts-zh_en/models/BM1684X/vits_part_c2_F16.bmodel`
- `vits-melo-tts-zh_en/models/BM1684X/vits_part_c2_stream_W128_R16_F16.bmodel`（执行 `--stream` 生成）

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

最终推理链路：**Part A（TPU 3.9ms）→ MAS（CPU 127.5ms，256 token）→ Part C（TPU 169.3ms，T_mel=1024）**；256 token 中英混合样本总耗时 309.9ms，RTF 0.0267

---

## 注意事项

- bmodel 的 `L=256`、`T_mel=1024` 均为固定形状（最多约 11.9s 音频）；推理时 z_p 会 pad 到 1024，超过上限的有效帧会被 C++ 截断并打印警告
- BM1684X SDK 无 `BM_INT64`，Part A 的 token/tone 输入需在 C++ 侧 cast 为 int32 再上传
- `matmul_ht` 中操作 Part A 输出的 h 时，行步长必须用 `L_MAX=256`，而非 `seq_len`（详见知识库）
- C2 window 模型输入为 `[1,192,160]`，其中每个窗口交付中间 128 个 mel 帧，左右 16 帧作为上下文；C++ 运行时以 32 帧 overlap-add 交付连续 PCM。C1 含全局注意力，仍按完整 mel 序列运行。
- C2 window 模型通过显式 `gen_bmodel.sh F16 --stream` 生成，旧 `gen_bmodel.sh F16` 和旧三件 bmodel 不会被覆盖。
