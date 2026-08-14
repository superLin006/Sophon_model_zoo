# HY-MT W8BF16 61 用例全量回归分析（2026-08-14）

**测试对象**：板卡 `/data/hymt`（W8BF16 seq512，bmodel 20260812_173234，hymt_demo_async）
**用例集**：`test_data/board_regression_full.sh`（16 原有 + 45 新增 = 61 用例）
**原始日志**：`outputs/board_extended/w8_full_61cases.log`

## 结论

- **61/61 全部成功**，无空输出、无 0 token、无重复退化
- **性能稳定**：prefill ~205ms 恒定，decode 22.2–24.0 token/s（均值 ~23.4）
- **准确性良好**：翻译在短句/中长文/长文、中英/英中/日中/中日语对、数字单位、人名日期场景均准确

## 准确性瑕疵（3 处）

| 用例 | 现象 | 评估 |
|---|---|---|
| terminology_short | 要求"边缘推理→edge inference"，输出 "edge computing" | 术语未完全遵循参考翻译，语义可接受 |
| formatting_short | `<sn>…<sn>…</sn>…</sn>` 标签嵌套错位 | 格式类用例的老弱点（16 用例版同样存在） |
| names_products | "customize the voice tone" 未用 voice cloning 表述 | 语义正确，措辞非最优 |

## 性能摘要

```
prefill: 204.6 – 210.4 ms（恒定，与输入长度无关）
decode : 22.2 – 24.0 token/s（短句略快，长句略慢）
最长输出: long_instruction_en 127 tokens（@256 上限内正常截断）
```

## 对比 W4BF16（2026-08-12 记录）

| 指标 | W8BF16 | W4BF16 g64 |
|---|---:|---:|
| decode | ~23.4 tok/s | ~34.5 tok/s |
| prefill | ~205 ms | ~202 ms |
| 术语遵循 | 部分 | 部分 |

W4BF16 速度优势明显（+47% decode），准确性与 W8 相当；本次 W4 版本已同步部署于板卡 `/data/hymt_w4g64`（bmodel 20260812_175057），可用 `board_regression_full.sh /data/hymt_w4g64` 复测。
