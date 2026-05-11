# Whisper (BM1684)

Whisper 语音识别模型移植到 Sophon BM1684 平台的完整实现。

## 目录结构

```
whisper/
├── python/                  # 模型转换相关脚本（在开发机/WSL 执行）
│   ├── analyze/             # 算子兼容性分析
│   │   ├── analyze_operators.py   # 算子扫描脚本
│   │   └── onnx_analysis/         # 分析输出（ONNX 文件，不上传）
│   ├── export_onnx.py       # Step1: PyTorch → ONNX
│   ├── gen_bmodel.sh        # Step2: ONNX → bmodel（Docker 内执行）
│   └── test_onnx.py         # ONNX 精度验证
├── cpp/                     # C++ 推理代码（在 BM1684 服务器上编译运行）
├── models/
│   └── BM1684/              # 编译好的 bmodel（不上传 git）
└── test_data/               # 测试音频（不上传 git）
```

## 移植状态

| 步骤 | 状态 | 说明 |
|------|------|------|
| 算子兼容性分析 | ✅ 完成 | Encoder/Decoder 全部兼容，无需修改网络结构 |
| ONNX 导出 | 🔄 进行中 | |
| bmodel 转换 | ⏳ 待完成 | |
| C++ 推理 | ⏳ 待完成 | |

## 算子分析结论

- **Encoder**: 10 种算子，全部支持（Add/Conv/MatMul/LayerNorm/Softmax 等）
- **Decoder**: 11 种算子，全部支持（含 Gather，Sophon 支持 NPU 跑 Embedding）
- **无需修改模型网络结构**，直接导出 ONNX 转换即可

## 模型规格（whisper-base）

| 参数 | 值 |
|------|----|
| n_mels | 80 |
| n_audio_ctx | 1500 |
| n_audio_state | 512 |
| n_audio_layer | 6 |
| n_vocab | 51865 |
| n_text_ctx | 448 |
| n_text_state | 512 |
| n_text_layer | 6 |

## 参考

- 官方 Demo（仅供参考）: `../sophon_demo/whisper/`
- 算子支持列表: `../.claude/doc/sophon_tpumlir_operators.md`
