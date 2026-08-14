# BM1684X 板卡部署规范 v1.1

> 部署命令模板见 cpp-implementer subagent Step 6/7；本规范只记**判断标准 + 踩坑经验**。

## 流程概览

```
交叉编译（sophon-cross-build Docker）→ scp 上传 → md5 校验 → 板卡运行 → RTF 统计
```

## 板卡目录结构

```
/data/<model>/
├── <model>_bm1684          # 可执行文件
├── models/                 # bmodel（只放最终采用的，实验产物不留在板卡）
├── assets/                 # 资产文件（tokenizer.json、vocab 等）
└── test_data/              # 测试输入
```

## 上传完整性校验（必做）

scp 传输大文件偶发损坏（Qwen3-TTS 移植中 bmodel 传坏导致板卡行为异常，靠 md5 定位）：

```bash
md5sum models/BM1684X/*.bmodel                                  # 本地
sshpass -p "<pwd>" ssh root@<ip> "md5sum /data/<model>/models/*" # 板卡
# 逐一比对，不一致的重传
```

**任何"板卡行为诡异"（输出错乱、加载失败、段错误）时，先验 md5 排除文件损坏，再怀疑代码。**

## RTF 统计口径

> **只计特征提取 + TPU 推理，不含模型加载**（实际部署时模型预加载到内存）

程序输出格式（必须遵循）：
```
[Timing] audio=<x>ms  feat=<x>ms  infer=<x>ms  total=<x>ms  RTF=<x>
```
生成式模型（TTS/LLM）额外记录每帧/每 token 耗时和分模块分解。

## 常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| `GLIBC_2.xx not found` | 工具链 glibc 太新 | 用 `sophon-cross-build`（Ubuntu 20.04 gcc 9.4）；板卡 native 编译关 LTO（单核 OOM 重启） |
| `libbmrt.so not found` | rpath 未设置 | CMake 加 `-Wl,-rpath,/opt/sophon/libsophon-current/lib` |
| `runtime arch[BM1684] != bmodel arch[BM1684X]` | chip 写错 | 确认 `--chip bm1684x` |
| 推理结果与 Python 不一致 | 预处理逻辑不一致 | save_debug + npy 逐子步骤对比 |
| 依赖下载卡死（sherpa 类） | 网络问题 | 复用 `_deps/*-src` + `FETCHCONTENT_FULLY_DISCONNECTED=ON` |

## 性能输出参考（已验证案例）

| 模型 | 精度 | 特征提取 | TPU 推理 | 合计 | RTF |
|------|------|---------|---------|------|-----|
| SenseVoice Small | F32 | ~34ms | ~155ms | ~189ms | 0.034 |
| SenseVoice Small | F16 | ~34ms | ~20ms | ~54ms | 0.0095 |
| Qwen3-TTS 0.6B（talker W8+CP F32，长句） | 混合 | — | — | ~12.7s/5.04s 音频 | 2.51 |

---

**版本**: v1.1（2026-08-14：md5 校验、rpath 修正 libsophon-current、命令模板移交 subagent、补 Qwen3-TTS 案例）
