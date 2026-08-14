# 最终验证（2026-08-14 17:3x，SEQLEN=192 采样模式 seed42）

**配置**：talker W8BF16 SEQLEN192 + CP F16 16槽 cache + CP F32 组件 + codec F16
**全部 14 条成功，RTF 均值 ~1.9**

| case | dur | RTF |
|---|---:|---:|
| zh_short 你好。 | 1.92s | 2.380 |
| new_english（Ryan） | 10.24s | 1.795 |
| baseline_zh | 4.64s | 1.947 |
| spk_vivian | 4.00s | 2.009 |
| spk_serena | 3.36s | 2.080 |
| spk_unclefu | 6.40s | 1.879 |
| spk_aiden | 4.16s | 1.970 |
| spk_ryan_zh | 10.24s | 1.785 |
| type_mixed 中英混 | 6.72s | 1.856 |
| type_number 数字 | 8.32s | 1.825 |
| type_punct 标点 | 10.24s | 1.818 |
| len_long2 长句 | 10.24s | 1.813 |
| en_serena | 5.12s | 1.924 |
| en_vivian | 5.76s | 1.916 |

注：10.24s 的为达到 codec 128 帧上限（内容长或音色拖音），非失败。
