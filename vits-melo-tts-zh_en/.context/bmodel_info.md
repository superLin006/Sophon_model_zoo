# VITS-MeloTTS BM1684X 转换与验收记录

## 当前交付

当前采用三段式 F16 推理链路：

```text
Part A（TPU）：文本编码 + 确定性 DP 时长预测，固定 L=256
    → dp_w [1,1,256]、h [1,192,256]、x_mask [1,1,256]
Part B（CPU）：MAS 对齐 + z_p 计算
    → z_p [1,192,T_mel]，T_mel 上限 1024
Part C1（TPU）：Flow，输入 [1,192,1024] + y_mask [1,1,1024]
Part C2（TPU）：Decoder，输出 [1,1,524288]
```

交付 bmodel：

| 文件 | 约大小 | 输入/输出 | 精度 |
|---|---:|---|---|
| `models/BM1684X/vits_part_a_F16.bmodel` | 20 MB | `[1,256]` token/tone → Part A 输出 | F16 |
| `models/BM1684X/vits_part_c1_F16.bmodel` | 53 MB | `[1,192,1024]` + `[1,1,1024]` → flow | F16 |
| `models/BM1684X/vits_part_c2_F16.bmodel` | 34 MB | `[1,192,1024]` → `[1,1,524288]` | F16 |

三件 bmodel 合计约 107 MB。转换中间产物写入 `compile/tmp/vits_F16/`，该目录不属于交付内容。

## 复现流程

原始单体 ONNX 来自 sherpa-onnx 对 MeloTTS 的预转换，仓库提供 `python/export_onnx.py` 重新导出入口。推荐从 ModelScope 下载 `myshell-ai/MeloTTS-Chinese` 的 `checkpoint.pth` 与 `config.json`，准备与导出脚本兼容的 MeloTTS 源码后执行：

```bash
conda create -n sophon-vits-melo-tts-zh-en python=3.10 -y
conda run -n sophon-vits-melo-tts-zh-en python -m pip install -r vits-melo-tts-zh_en/requirements.txt
git clone --depth 1 https://github.com/myshell-ai/MeloTTS.git /tmp/MeloTTS
conda run -n sophon-vits-melo-tts-zh-en env \
  PYTHONPATH=/tmp/MeloTTS \
  MELOTTS_CHECKPOINT=/tmp/melotts_chinese/checkpoint.pth \
  MELOTTS_CONFIG=/tmp/melotts_chinese/config.json \
  python vits-melo-tts-zh_en/python/export_onnx.py
conda run -n sophon-vits-melo-tts-zh-en python vits-melo-tts-zh_en/python/make_tpu_model.py
conda run -n sophon-vits-melo-tts-zh-en python vits-melo-tts-zh_en/python/make_split_models.py
docker exec sophon-tpumlir-v128 bash /workspace/vits-melo-tts-zh_en/python/gen_bmodel.sh F16
```

导出脚本会生成中英混合词典、`tokens.txt` 和 `model.onnx`；`model.onnx`、`model_tpu.onnx` 和中间文件不属于最终交付目录。

## C++ 运行时约束

- `seq_len` 支持 1~256；短输入由 C++ padding 到 256，`x_lengths` 保留实际长度。
- `T_mel` 超过 1024 帧时，CPU MAS 在分配 `z_p` 前截断到 1024，并输出警告；更长文本应在前端分句合成。
- `sid=1` 已固化在模型方案中；BM1684X 无 `BM_INT64`，token/tone 在上传前转为 int32。
- 输出采样率为 44100 Hz，最多约 11.9 秒。
- Part C1 和 C2 必须在同一个 BMRuntime 中加载，以避免多个 runtime 的 DDR 地址冲突。

## 板卡验收

BM1684X 上使用仓库当前交叉编译的 C++ 程序和三件 F16 bmodel 验证：

| 输入 | seq_len | T_mel | 输出时长 | 总耗时 | RTF | 结果 |
|---|---:|---:|---:|---:|---:|---|
| 中文 smoke | 61 | 232 | 2.694 s | 182.4 ms | 0.0677 | PASS |
| 中英混合 smoke | 53 | 223 | 2.589 s | 179.9 ms | 0.0695 | PASS |
| 中文长输入 | 256 | 933 | 10.832 s | 300.8 ms | 0.0278 | PASS |
| 中英混合长输入 | 256 | 999 | 11.598 s | 309.9 ms | 0.0267 | PASS |

两组 256 token 长输入均为单次 L=256 推理，不是分块拼接；输出 WAV 均为 44100Hz、16-bit、mono，板卡 uptime 未变化，没有出现卡死、重启或空音频。

## 音频内容回听

将 VITS 输出重采样为 SenseVoice 所需的 16kHz 后进行回听转写。另补充 4 组 183~199 token、7.8~8.4 秒的完整音频，避免 SenseVoice 的约 10 秒输入上限影响判断：

| 类型 | SenseVoice 回听结论 |
|---|---|
| 中文 | 与原文基本逐字一致，句首、句中和句尾完整 |
| 英文 | 句子结构和大部分词语完整；少量普通 ASR 误识别，如 `every` 识别为 `Harry` |
| 中英混合 | 中文、英文均被识别，语言切换完整；个别专有词被转写为近似音 |
| 数字/标点 | 数字内容基本保留；标点和版本号格式存在 ASR 归一化或省略 |

结论：8 组音频均有有效语音信号，未发现静音、循环重复、半途消失或异常截断。SenseVoice 对专有名词、数字和英文模型名的误识别属于回听工具识别误差，不能单独判定为 VITS 发音错误；需要最终音质确认时仍应人工试听。

## 设计取舍与历史失败方案

- 原始模型中的 SDP 包含 21 个 `NonZero`，其输出 shape 与运行时数据相关；Flow 含 `RandomNormalLike`，MAS 含动态 `Range`，无法直接编译成单体 bmodel。
- 当前方案将 SDP 的随机分支替换为零常量，仅使用 DP 生成确定性时长；Flow 的 `RandomNormalLike` 分支在 `noise_scale=0` 时贡献为零，因此绕过；MAS 保留在 CPU。
- Part A 当前固定 L=256；Part C1/C2 当前固定 T_mel=1024。L=256 解决输入上限，T_mel=1024 解决 256 token 样本的输出截断，同时增加了 C1/C2 的计算量和 bmodel 体积。

## 文件对应关系

| 文件 | 作用 |
|---|---|
| `python/export_onnx.py` | 使用 MeloTTS 权重导出中英混合单体 `model.onnx` |
| `python/make_tpu_model.py` | 去除 TPU 不支持算子并固化 sid |
| `python/make_split_models.py` | 生成 Part A/C1/C2 三段静态 ONNX |
| `python/gen_bmodel.sh` | 在 TPU-MLIR 容器内编译三件 bmodel |
| `python/test/test_onnx.py` | 检查三段 ONNX 结构和固定形状可执行性 |
| `cpp/src/tts_inference.cpp` | BMRuntime 三段推理、CPU MAS 和长度截断 |
| `deploy_to_board.sh` | 上传、MD5 校验并执行板卡 smoke test |

*记录更新：2026-09-03；TPU-MLIR v1.28.1；Chip: BM1684X。*
