# Moonshine streaming-small BM1684X C++ 推理

Moonshine streaming-small ASR(10 层, vocab 32768)的纯 BMRuntime C++ 推理实现,
交叉编译部署到 BM1684X 板卡。

## 目录结构

```
cpp/
├── CMakeLists.txt
├── build.sh                # sophon-cross-build Docker 交叉编译
├── src/
│   ├── main.cpp            # ./moonshine_bm1684 <model_dir> <wav_path> [F32|F16] [debug_dir]
│   ├── moonshine_inference.h/.cpp   # BMRuntime 封装 + 解码循环 + token 解码
│   └── utils/
│       ├── moonshine_features.h     # 特征提取: 补零 + 分帧 + CMVN + asinh(与 Python 逐行对应)
│       └── wav_reader.h/.cpp        # WAV 读取(PCM16/float32, 多声道平均, 8k→16k 线性插值)
└── tools/
    ├── feat_dump.cpp       # 宿主机特征提取验证工具(原生编译)
    └── compare_xframes.py  # C++ 特征 vs numpy 参考对比
```

## 用法

```bash
./moonshine_bm1684 <model_dir> <wav_path> [F32|F16] [debug_dir]
# model_dir 需含: moonshine_encoder_<prec>.bmodel、moonshine_decoder_<prec>.bmodel、tokens.txt
```

示例:

```bash
cd /root/moonshine
./moonshine_bm1684 models/ test_data/0.wav F32
./moonshine_bm1684 models/ test_data/0.wav F16
```

## 特征提取(与 Python 完全一致)

1. WAV 16kHz mono float32(不足 10s 尾部补零到 160000 samples)
2. 分帧 reshape [2000, 80]
3. CMVN: mean = x.mean(-1); rms = sqrt(centered².mean(-1) + 1e-6); normed = centered/rms
4. x_frames = asinh(exp(log_k) * normed), LOG_K = -0.4875200987

验证: C++ 特征 vs numpy 参考 max_abs = 2.4e-07(float32 1-2 ulp)。

## 解码

- encoder: x_frames [1,2000,80] → encoder_out [1,500,620]
- decoder: 23 入 21 出, token/cache_len 为 INT32, KV cache [10][128][512]
- 贪心 argmax, sos=1, eos=2, 最多 128 步;new_k_i/new_v_i 写入 cache 槽位 t
- tokens.txt 按 id 直接查表拼接,▁(U+2581)开头 token 前加空格,跳过特殊 token

## 测试结果

板卡: 172.16.25.248(BM1684X, libsophon 0.5.1),部署目录 /root/moonshine/

### 转写文本(与 test_onnx.py / HF 填充版参考 100% 一致)

| 音频 | F32 | F16 | 参考 |
|------|-----|-----|------|
| 0.wav (16k 原生) | After early nightfall, the yellow lamps would light up here and there the squalid quarter of the brothels. | 相同 | 一致 |
| 8k.wav (线性插值重采样) | Yet these thoughts affected Hester Prynne less with hope than apprehension. | 相同 | 一致(HF 填充版参考) |

### 数值精度(板卡 C++ vs 本地 ORT 参考, 27 步全部 argmax 一致)

| 精度 | encoder max_abs | decoder logits max_abs | argmax 不一致 |
|------|-----------------|------------------------|---------------|
| F32  | 2.87e-5         | 3.82e-5                | 0 / 27        |
| F16  | 3.27e-2         | 4.33e-2                | 0 / 27        |

特征提取: 板卡 C++ x_frames vs numpy 参考 max_abs = 2.38e-7(float32 1-2 ulp, 69% 元素完全一致)。

### 性能(RTF = (feat+infer)/audio, 不含模型加载)

| 精度 | 0.wav (6.625s, 27 步) | 8k.wav (4.825s, 19 步) |
|------|----------------------|------------------------|
| F32  | feat 14.4ms + infer 638.6ms = 653ms, RTF 0.099 | feat 12.3ms + infer 444.6ms = 457ms, RTF 0.095 |
| F16  | feat 14.3ms + infer 282.1ms = 296ms, RTF 0.045 | feat 12.3ms + infer 193.5ms = 206ms, RTF 0.043 |

## 构建

```bash
# 交叉编译(产物 moonshine/cpp/build/moonshine_bm1684)
bash moonshine/cpp/build.sh
```
