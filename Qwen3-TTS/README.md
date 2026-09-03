# Qwen3-TTS-12Hz-0.6B — BM1684X 移植（纯 bmruntime C++）

Qwen3-TTS-12Hz-0.6B-CustomVoice 语音合成在 Sophon BM1684X 上的推理实现。
纯 C++ bmruntime（无 sail/Python），模型常驻内存批量合成。

## 模型（3 bmodel，`models/BM1684X/`）

| 文件 | 大小 | 说明 |
|---|---|---|
| `qwen3_tts_talker_w8s192.bmodel` | 1.5G | talker W8BF16，SEQLEN=192（56 层网络 + 2 embedding + codec_head） |
| `qwen3_tts_cp_allf32.bmodel` | 696M | code_predictor 40 网络：5 prefill F32 + 5 cache **F16 16槽** + 15 lm_head F32 + 15 embedding F32 |
| `codec_decoder.bmodel` | 235M | codec 解码 F16 |

> `models/BM1684X/` 只保留上表 3 个合并 bmodel。逐网络 bmodel、`.bmodel.json`、`.net_0.profile` 和 W4F16 实验档均已清理；它们都可由 `python/gen_final.sh` 重新生成，不属于运行时交付文件。

bmodel 不入库，用 `python/gen_final.sh` 重建（docker `sophon/tpuc_dev:v3.4-tpumlir-1.28.1`）：
前置 `models/onnx/` 由 `python/export_talker.py`（talker/cp/embedding 部分）+ `python/export_codec.py`（codec 部分）导出，共 **100 个 ONNX**，按用途分目录存放：

```
models/onnx/
├── talker/      56 个：talker_block_0..27 + talker_block_cache_0..27（28 层 prefill/decode 分图）
├── cp/          40 个：cp_block/cp_block_cache 5 层 + cp_lm_head_0..14 + cp_embedding_0..14（15 code head）
├── embedding/   3 个：embedding_code / embedding_text / codec_head
└── codec/       1 个：codec_decoder_T325.onnx
```

> 为什么是 100 个而非几个大图：TPU-MLIR 编译 bmodel 需要**静态 shape**，LLM 的 decode 逐 token 循环
> 无法作为动态图整体编译，必须把每层拆成 prefill（整段）+ cache（单步+KV）两个图；15 个 code head
> 同理各为独立输出头。**运行时已用 `model_tool --combine` 合并成 3 个 bmodel**（talker 56 网络合并、
> cp 40 网络合并、codec 单独），板上部署只见 3 个文件。onnx 只是编译输入，可随时用 export 脚本重导。
>
> 完整性校验：`conda run -n sophon-qwen3-tts --no-capture-output python Qwen3-TTS/python/verify_onnx.py`（确认 gen_final.sh 消费的 100 个 onnx 齐全且结构正确）。
> 导出环境：`sophon-qwen3-tts` conda env（Python 3.10 / torch 2.6.0 / transformers 4.57.3 / onnx 1.22），依赖见 `Qwen3-TTS/requirements.txt`。
>
> 新机器统一建环境：
> ```bash
> conda create -n sophon-qwen3-tts python=3.10 -y
> conda run -n sophon-qwen3-tts python -m pip install --upgrade pip
> conda run -n sophon-qwen3-tts python -m pip install -r Qwen3-TTS/requirements.txt
> conda run -n sophon-qwen3-tts --no-capture-output python Qwen3-TTS/python/export_talker.py
> conda run -n sophon-qwen3-tts --no-capture-output python Qwen3-TTS/python/export_codec.py
> conda run -n sophon-qwen3-tts --no-capture-output python Qwen3-TTS/python/verify_onnx.py
> docker exec sophon-tpumlir-v128 bash /workspace/Qwen3-TTS/python/gen_final.sh
> ```

## 构建

```bash
cd cpp && bash build.sh      # sophon-cross-build 容器交叉编译 → cpp/build/qwen3_tts_bm1684x
BOARD_IP=<board_ip> bash deploy_to_board.sh --test  # 上传、md5 校验并执行单句 smoke test
```

## 用法（板卡 /data/Qwen3-TTS）

单条合成：

```bash
./qwen3_tts_bm1684x --talker_bmodel models/qwen3_tts_talker_w8s192.bmodel \
  --cp_bmodel models/qwen3_tts_cp_allf32.bmodel \
  --codec_bmodel models/codec_decoder.bmodel \
  --model_dir assets \
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

## 性能与量化实测（BM1684X 板卡）

验证按时间线分两轮，**两轮用例数不同，不要混为一谈**：

| 日期 | 轮次 | 用例数 | 结果 | 逐条数据 |
|---|---|---|---|---|
| 2026-08-14 | 3 文件合并版首次验收（cp_allf32 内置 F16 cache），W8BF16 SEQLEN=192 采样模式 seed42 | **14** | 14/14 通过，RTF 1.79–2.46，均值 ~1.9 | ✅ `test_outputs/final_20260814/README.md`（逐条 case / dur / RTF 表，已入库） |
| 2026-08-17 | W8BF16 与 W4F16 g64 的 A/B 扩展集，同一随机种子、模型常驻 | **54** | 两档均 54/54 成功 | ⚠️ 见下方证据缺口 |

### 08-17 的 A/B 结果（54 条）

| 版本 | 成功率 | 加权 RTF | Talker 大小 | 三文件总大小 |
|---|---:|---:|---:|---:|
| W8BF16 | 54/54 | 1.900 | 1,582,284,800 B | 2,557,792,256 B |
| W4F16 group64 | 54/54 | **1.818** | **1,141,886,976 B** | **2,117,394,432 B** |

W4F16 相比 W8BF16 的 talker 体积减少 27.8%、三文件总大小减少 17.2%。人工试听确认 W4F16 扩展集总体稳定；`en_03`、`zh_03` 两条样本存在明确音质问题，保留为已知限制。**当前 W4F16 是体积、速度和质量之间的最佳候选，W8BF16 作为保守回退。**

### 「54 条」到底是哪 54 条

**54 = `test_outputs/batch_verify.txt`（14 行）+ `test_outputs/batch_verify_54.txt`（40 行）的并集**。两个文件零重叠，合计去重后正好 54 行（已实测核对）。08-14 那轮只用了前 14 条，08-17 扩展到全部 54 条。

> ⚠️ `batch_verify_54.txt` 这个文件名有误导性——它只有 **40** 行，不是 54 行。跑完整 54 条需要把两个清单拼起来：
> ```bash
> cat test_outputs/batch_verify.txt test_outputs/batch_verify_54.txt > /tmp/batch_all54.txt
> ./qwen3_tts_bm1684x ... --batch /tmp/batch_all54.txt --out out/
> ```

### 证据缺口（已登记为待修项）

- **54 条的加权 RTF（1.900 / 1.818）在全仓没有逐条原始数据**，只以汇总数字形式存在于本文件与知识库。要复核只能重跑上面的 54 条 batch。
- `test_outputs/ab_20260817/` 下的 4 份 `.log` **不是 RTF 记录**：它们是**用 Qwen3-ASR 转写 Qwen3-TTS 合成音频**的结果（内容为 `[ASR] ... language Chinese<asr_text>`，文件名 `asr_w4_batch` / `asr_w8_batch` / `asr_w8_extra_{w4,w8}audio`），交叉了 TTS 档位 × ASR 档位，属**合成音频可懂度的间接证据**。同目录的 `w4/`、`w8/`、`w4_16k/`、`w8_16k/`、`w4_extra/`、`w8_extra/` 等子目录是两轮合成的 wav（体积大，未入库）。
- 逐条 RTF 表只有 08-14 的 14 条那份（`final_20260814/README.md`）。
