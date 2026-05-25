# BM1684X Qwen 系列模型意图识别 Benchmark 结果

> 测试日期：2026-05-23 ~ 2026-05-25
> 板卡：BM1684X SoC 模式 (6GB DDR, 9GB DevMem)
> SDK：SophonSDK v23.09 LTS-SP4
> 测试场景：语音助手意图识别（ASR文本 → action + params）

## 模型对比总览

| 指标 | **qwen2.5-3b** ✅ | qwen3-4b ✅ |
|------|:------------:|:------:|
| bmodel 编译时间 | 2025-06-20 (TPU-MLIR v1.19) | 2026-05-25 (TPU-MLIR v1.28.1) |
| 量化方式 | W4BF16 (GPTQ-INT4) | W4F16 (AWQ) |
| seq_len | 2048 | 2048 |
| bmodel 大小 | 2.0 GB | 2.6 GB |
| DevMem 占用 | **2311 MB** | ~3640 MB |
| 模型加载耗时 | 44.6 s | ~58 s（冷） / ~7 s（热缓存） |
| **FTL（首字延迟）** | **1.49 s** | 2.05 s（冷） / 4.6 s（热降频） |
| **Decode TPS** | **20.9 tok/s** | 14.8（冷） / 5.8（热降频） |
| **端到端延迟** | **2.04 s** | 2.95 s（冷） / 6.5 s（热降频） |
| 意图识别准确率 | **10/10** | **10/10** |
| thinking 模式 | 不适用 | 需关闭（`--no_think`） |

> **热降频说明**：连续推理多条后板卡降频明显，qwen3-4b 受影响更大（FTL 从 2.05s 升至 4.6s，TPS 从 14.8 降至 5.8）。qwen2.5-3b 在 10 条连续测试中性能稳定。

**结论：推荐 qwen2.5-3b，降频后 qwen3-4b 的 E2E 延迟达 6.5s，不适合实时场景。**

---

## qwen2.5-3b（推荐 ✅）

### 环境

- Python 3.8 + C++ pybind11 (`chat.cpython-38-aarch64-linux-gnu.so`)
- System Prompt（prefill ~70 tokens）：
  ```
  把用户指令归类为以下动作之一：open_whiteboard, close_window, set_volume,
  open_camera, set_pen, draw_shape, set_tool, save_file, screenshot。
  输出JSON：{"action":"动作名","params":{}}。
  ```

### 性能

| 指标 | 数值 |
|------|------|
| 模型加载耗时 | 44.58 s |
| DevMem 占用 | 2311 MB / 9070 MB |
| 平均 FTL | **1.487 s** |
| 平均 Prefill 速度 | **46.1 token/s** |
| 平均 Decode TPS | **20.9 token/s** |
| 平均端到端延迟 | **2.043 s** |

### C++ Demo 性能对比

| 指标 | Python Demo | C++ Demo |
|------|:-----------:|:--------:|
| 模型加载 | 44.58 s | ~44 s |
| FTL | 1.487 s | 1.488 s |
| Decode TPS | 20.9 tok/s | 19.1 tok/s |
| 端到端 | ~2.0 s | ~2.0 s |

> C++ 与 Python 性能持平，C++ 零 Python 依赖更适合嵌入式部署。代码路径：`LLM-TPU/models/Qwen2/cpp_demo/`

### 10 条意图识别：10/10 ✅

| # | 输入 | 输出 | 正确 |
|---|------|------|:---:|
| 1 | 帮我打开白板 | `{"action":"open_whiteboard","params":{}}` | ✅ |
| 2 | 我要写字，用马克笔，笔迹大小调到12 | `{"action":"set_pen","params":{"pen_type":"马克笔","size":12}}` | ✅ |
| 3 | 关闭当前窗口 | `{"action":"close_window","params":{}}` | ✅ |
| 4 | 把音量调大一点 | `{"action":"set_volume","params":{}}` | ✅ |
| 5 | 打开摄像头 | `{"action":"open_camera","params":{}}` | ✅ |
| 6 | 我想画一个圆形 | `{"action":"draw_shape","params":{}}` | ✅ |
| 7 | 切换到橡皮擦模式 | `{"action":"set_tool","params":{"tool":"橡皮擦"}}` | ✅ |
| 8 | 保存当前文件 | `{"action":"save_file","params":{}}` | ✅ |
| 9 | 帮我截图 | `{"action":"screenshot","params":{}}` | ✅ |
| 10 | 退出程序 | `{"action":"close_window","params":{}}` | ✅ |

---

## qwen3-4b（可用，热降频明显 ⚠️）

### bmodel 重新编译记录

原始 bmodel（2025-05-14，TPU-MLIR v1.18）因内嵌 kernel module 与板卡 SDK 不兼容导致 SHA 校验失败，decode 阶段崩溃。

用 TPU-MLIR v1.28.1 从 ModelScope 下载权重重新编译：

```bash
# 下载权重（ModelScope，HF不可达）
bash QwenLLM/download_qwen3_weights.sh

# 转换（容器内已修复 transformers==4.51.1）
bash QwenLLM/compile_qwen3_bmodel.sh
# 注意：AWQ 只支持 --quantize w4f16（不是 w4bf16）
```

新 bmodel：`qwen3-4b-awq_w4f16_seq2048_bm1684x_1dev_static_20260525_143511.bmodel`

### Thinking 模式

Qwen3 默认开启思维链（输出 `<think>...</think>` + JSON），必须关闭：

```python
# benchmark_intent.py 使用 --no_think 参数
text = tokenizer.apply_chat_template(history, ..., enable_thinking=False)
```

| 模式 | 输出 tokens | E2E 延迟 |
|------|:-----------:|:--------:|
| thinking 开（默认） | ~200 tok | **16.1 s** ❌ |
| thinking 关（`--no_think`） | ~15 tok | **2.95 s**（冷） |

### 性能（no_think 模式）

| 条件 | FTL | Prefill | TPS | E2E |
|------|:---:|:-------:|:---:|:---:|
| 冷启动 / 前 3 条 | 2.05 s | 36.6 tok/s | 14.8 tok/s | **2.95 s** |
| 连续 10 条（热降频后） | 4.61 s | 15.7 tok/s | 5.8 tok/s | **6.49 s** |

### 10 条意图识别：10/10 ✅

| # | 输入 | 输出 | 正确 |
|---|------|------|:---:|
| 1 | 帮我打开白板 | `{"action":"open_whiteboard","params":{}}` | ✅ |
| 2 | 我要写字，用马克笔，笔迹大小调到12 | `{"action":"set_pen","params":{"pen_type":"marker","line_width":12}}` | ✅ |
| 3 | 关闭当前窗口 | `{"action":"close_window","params":{}}` | ✅ |
| 4 | 把音量调大一点 | `{"action":"set_volume","params":{}}` | ✅ |
| 5 | 打开摄像头 | `{"action":"open_camera","params":{}}` | ✅ |
| 6 | 我想画一个圆形 | `{"action":"draw_shape","params":{}}` | ✅ |
| 7 | 切换到橡皮擦模式 | `{"action":"set_tool","params":{}}` | ✅ |
| 8 | 保存当前文件 | `{"action":"save_file","params":{}}` | ✅ |
| 9 | 帮我截图 | `{"action":"screenshot","params":{}}` | ✅ |
| 10 | 退出程序 | `{"action":"close_window","params":{}}` | ✅ |

---

## qwen2.5-1.5b（不可用 ❌）

模型输出垃圾 token，1.5B 参数量不足以完成结构化意图识别任务。

---

## 后续方向

1. **热降频问题**：qwen3-4b 在持续推理下降频严重，考虑增加推理间隔或改进散热
2. **KV cache 模式**：用 `--use_block_with_kv` 编译模型，多轮对话复用 KV cache，降低 FTL
3. **qwen3-4b cpp_demo**：`LLM-TPU/models/Qwen3/cpp_demo_v7/` 提供纯 C++ 推理，待测
