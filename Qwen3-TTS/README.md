# Qwen3-TTS-12Hz-0.6B — BM1684X 移植（纯 bmruntime C++）

Qwen3-TTS-12Hz-0.6B-CustomVoice 语音合成在 Sophon BM1684X 上的推理实现。
纯 C++ bmruntime（无 sail/Python），模型常驻内存批量合成。

## 模型（3 bmodel，`models/BM1684X/`）

| 文件 | 大小 | 说明 |
|---|---|---|
| `qwen3_tts_talker_w8s192.bmodel` | 1.5G | talker W8BF16，SEQLEN=192（56 层网络 + 2 embedding + codec_head） |
| `qwen3_tts_cp_allf32.bmodel` | 696M | code_predictor 40 网络：5 prefill F32 + 5 cache **F16 16槽** + 15 lm_head F32 + 15 embedding F32 |
| `codec_decoder.bmodel` | 235M | codec 解码 F16 |

bmodel 不入库，用 `python/gen_final.sh` 重建（docker `sophon/tpuc_dev:v3.4-tpumlir-1.28.1`）：
前置 `models/onnx/` 由 `python/export_talker.py` 导出（99 个 ONNX，~40 分钟）。

## 构建

```bash
cd cpp && bash build.sh      # sophon-cross-build 容器交叉编译 → cpp/build/qwen3_tts_bm1684x
bash deploy_to_board.sh      # 上传 3 bmodel + 二进制 + tokenizer 到板卡 /data/Qwen3-TTS（md5 校验）
```

## 用法（板卡 /data/Qwen3-TTS）

单条合成：

```bash
./qwen3_tts_bm1684x --talker_bmodel models/qwen3_tts_talker_w8s192.bmodel \
  --cp_bmodel models/qwen3_tts_cp_allf32.bmodel \
  --codec_bmodel models/codec_decoder.bmodel \
  --model_dir /data/Qwen3-TTS/assets \
  --text '你好。' --speaker Vivian --language Chinese --sample --seed 42 --out out.wav
```

批量合成（模型常驻，`name|speaker|lang|text` 每行）：

```bash
./qwen3_tts_bm1684x ... --batch batch_verify.txt --out out/
```

## 关键配置（与 C++ 对齐，勿随意改）

- SEQLEN=192：prefill 上限 192 帧（~15.4s 语音），decode 超限自动截断
- CP cache 16 槽：history `[1,8,16,128]` + mask `[1,1,1,17]`，F16 量化（F32 组件精度下限）
- 采样模式（--sample）默认 seed 42，每句重新播种可复现；NaN/全零 logits 回退 argmax
- KV cache 设备常驻（talker 28 层 + CP 5 层），decode 零拷贝 + d2d 写槽

## 性能（板卡实测）

batch 模式 RTF 均值 ~1.85（14 条混合语料），decode 上限 ~192 帧。
已知限制：TPU 量化网络层内激活精度不足（bf16 尾数 7 位）→ talker/CP 生成式路径量化退化，
talker 最低 W8BF16、CP cache 最低 F16（CP 组件保持 F32）。

## 验证

`test_outputs/final_20260814/`（README 含 RTF 表）、`test_outputs/batch_verify.txt`（14 条验证输入）。
2026-08-14：3 文件合并版（cp_allf32 内置 F16 cache）14/14 通过，RTF 1.79–2.46。
