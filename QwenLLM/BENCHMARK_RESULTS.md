# BM1684X Qwen 系列模型意图识别 Benchmark 结果

> 测试日期：2026-05-23 ~ 2026-05-25
> 板卡：BM1684X SoC 模式 (6GB DDR, 9GB DevMem)
> SDK：SophonSDK v23.09 LTS-SP4
> 测试场景：语音助手意图识别（ASR文本 → action + params）

## 模型对比总览

| 指标 | qwen2.5-3b ✅ | **qwen3-1.7b ⭐** | qwen3-0.6b | qwen3-4b |
|------|:---:|:---:|:---:|:---:|
| 编译工具 | TPU-MLIR v1.19 | TPU-MLIR v1.28.1 | TPU-MLIR v1.28.1 | TPU-MLIR v1.28.1 |
| 量化方式 | W4BF16 (GPTQ) | W4BF16 (PTQ) | W4BF16 (PTQ) | W4F16 (AWQ) |
| bmodel 大小 | 2.0GB | **1.4GB** | 562MB | 2.6GB |
| DevMem 占用 | 2311MB | **1731MB** | 871MB | 3132MB |
| 模型加载耗时 | 44.6s | **2.36s** | 1.19s | 57.5s（冷） |
| **FTL（首字延迟）** | 1.487s | **0.878s** | 0.511s | 2.049s |
| **Decode TPS** | 20.9 tok/s | **29.1 tok/s** | 52.6 tok/s | 14.8 tok/s |
| **端到端延迟** | 2.04s | **1.20s** | 0.92s | 2.95s |
| 意图识别准确率 | **10/10** | **9/10** ⚠️ | 8/10 ❌ | **10/10** |
| thinking 模式 | 不适用 | 需关闭（`--no_think`） | 需关闭（`--no_think`） | 需关闭（`--no_think`） |

**结论：推荐 qwen3-1.7b。** 比 qwen2.5-3b 快 1.7x，DevMem 减少 25%，加载仅 2.36s（vs 44.6s）；唯一失败的 case 07（橡皮擦 → `set_pen` 而非 `set_tool`）可通过系统提示词调整修复。

0.6B 速度最快但准确率不足（8/10）且输出格式不稳定（带 markdown 代码块），不建议用于生产。

---

## qwen3-1.7b（推荐 ⭐）

### 环境

- bmodel：`qwen3-1.7b_w4bf16_seq2048_bm1684x_1dev_static_20260525_151623.bmodel`
- 编译：`llm_convert.py --quantize w4bf16 -s 2048 -c bm1684x`（无 AWQ 预量化，直接 PTQ）
- Python 3.8 + pybind11 chat.so（Qwen3 python_demo）

### 性能

| 指标 | 数值 |
|------|------|
| bmodel 大小 | 1.4GB |
| DevMem 占用 | 1731MB / 9070MB |
| 模型加载耗时 | **2.36s** |
| 平均 FTL | **0.878s** |
| 平均 Prefill 速度 | **82.6 token/s** |
| 平均 Decode TPS | **29.1 token/s** |
| 平均端到端延迟 | **1.202s** |

### 10 条意图识别：9/10

| # | 输入 | 输出 | 正确 |
|---|------|------|:---:|
| 1 | 帮我打开白板 | `{"action":"open_whiteboard"}` | ✅ |
| 2 | 我要写字，用马克笔，笔迹大小调到12 | `{"action":"set_pen","params":{"pen":"马克笔","size":"12"}}` | ✅ |
| 3 | 关闭当前窗口 | `{"action":"close_window"}` | ✅ |
| 4 | 把音量调大一点 | `{"action":"set_volume","params":{"volume":100}}` | ✅ |
| 5 | 打开摄像头 | `{"action":"open_camera"}` | ✅ |
| 6 | 我想画一个圆形 | `{"action":"draw_shape","params":{"shape":"circle"}}` | ✅ |
| 7 | 切换到橡皮擦模式 | `{"action":"set_pen","params":{"mode":"eraser"}}` | ❌（应为 `set_tool`） |
| 8 | 保存当前文件 | `{"action":"save_file"}` | ✅ |
| 9 | 帮我截图 | `{"action":"screenshot"}` | ✅ |
| 10 | 退出程序 | `{"action":"close_window"}` | ✅ |

**Case 07 修复方向**：在系统提示词中将 `set_tool` 的说明改为"切换工具/模式（橡皮擦、选择框等）"，明确区分 `set_pen`（笔的参数）与 `set_tool`（工具切换）。

---

## qwen3-0.6b（不建议生产使用 ❌）

### 性能

| 指标 | 数值 |
|------|------|
| bmodel 大小 | 562MB |
| DevMem 占用 | 871MB / 9070MB |
| 模型加载耗时 | 1.19s |
| 平均 FTL | 0.511s |
| 平均 Prefill 速度 | 142.0 token/s |
| 平均 Decode TPS | 52.6 token/s |
| 平均端到端延迟 | 0.920s |

### 10 条意图识别：8/10 ❌

| # | 输入 | 结果 | 正确 |
|---|------|------|:---:|
| 1 | 帮我打开白板 | `open_whiteboard` | ✅ |
| 2 | 我要写字，用马克笔，笔迹大小调到12 | `draw_shape`（应为 `set_pen`） | ❌ |
| 3 | 关闭当前窗口 | `close_window` | ✅ |
| 4 | 把音量调大一点 | `set_volume` | ✅ |
| 5 | 打开摄像头 | `open_camera` | ✅ |
| 6 | 我想画一个圆形 | `draw_shape` | ✅ |
| 7 | 切换到橡皮擦模式 | `draw_shape`（应为 `set_tool`） | ❌ |
| 8 | 保存当前文件 | `save_file` | ✅ |
| 9 | 帮我截图 | `screenshot` | ✅ |
| 10 | 退出程序 | `close_window` | ✅ |

额外问题：输出包裹 markdown 代码块（````json ... ````），JSON 解析需要额外剥离。

---

## qwen2.5-3b（稳定基准 ✅）

### 性能

| 指标 | 数值 |
|------|------|
| bmodel 大小 | 2.0GB |
| DevMem 占用 | 2311MB / 9070MB |
| 模型加载耗时 | 44.6s |
| 平均 FTL | **1.487s** |
| 平均 Prefill 速度 | **46.1 token/s** |
| 平均 Decode TPS | **20.9 token/s** |
| 平均端到端延迟 | **2.043s** |

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

### 编译记录

原始 bmodel（2025-05-14，TPU-MLIR v1.18）因内嵌 kernel module 与板卡 SDK 不兼容导致 SHA 校验失败。
用 TPU-MLIR v1.28.1 重编：`--quantize w4f16`（AWQ 模型专用，不能用 w4bf16）

### 性能（no_think 模式）

| 条件 | FTL | TPS | E2E |
|------|:---:|:---:|:---:|
| 冷启动 / 前 3 条 | 2.05s | 14.8 tok/s | **2.95s** |
| 连续 10 条（热降频后） | 4.61s | 5.8 tok/s | **6.49s** |

> 热降频说明：BM1684X 连续推理后降频明显，qwen3-4b 受影响最大（E2E 从 2.95s 升至 6.49s）。

### 10 条意图识别：10/10 ✅

---

## Qwen3.5-0.8B（已编译，待测 🔄）

### 编译记录

- 架构：GatedDeltaNet hybrid（75% 线性注意力 + 25% 全注意力）
- `head_dim=256`, `partial_rotary_factor=0.25`, 24 层
- 编译约束：
  - `ChunkGatedDeltaRule` 仅支持 dynamic codegen → `--dynamic` 必须
  - W4/W8BF16 的 BM1684X A16MatMul 要求 N≥对齐值；DeltaNet 门控投影 N=16 不满足
  - 最终方案：`--quantize bf16 --dynamic`
- bmodel：`qwen3.5-0.8b_bf16_seq2048_bm1684x_1dev_dynamic_*.bmodel`，**1.5GB**
- 需要三个 Python 补丁（见 `compile_qwen35_text_bmodel.sh`）：
  1. `max_pixels=0` 纯文本路径 + `rotary_dim`/`mrope_section` 设置
  2. 绕过 `AutoConfig`（transformers 4.51.1 不识别 `qwen3_5` 模型类型）
  3. `mrope()`/`mrope_batch()`/`get_mrope_index()` 支持 `partial_rotary_factor<1`

> 性能数据待板卡实测更新。预期 BF16 dynamic 相比 W4BF16 static 会更慢，但架构新颖。

---

## 后续方向

1. **qwen3.5-0.8b 实测**：部署到板卡，测量 FTL/TPS/E2E 及意图识别准确率
2. **qwen3-1.7b 系统提示词优化**：明确 `set_tool` 覆盖"橡皮擦/选择框等工具切换"，预计可达 10/10
3. **qwen3-0.6b 格式问题**：需在 prompt 中强调"输出纯 JSON，不加代码块"，准确率问题更难解决
4. **热降频问题（qwen3-4b）**：考虑推理间隔或改进散热
