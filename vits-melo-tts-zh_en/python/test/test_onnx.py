#!/usr/bin/env python3
"""
test_onnx.py - vits-melo-tts-zh_en ONNX 验证脚本
验证 decoder_T256.onnx（方案 D: 解码器子模型）的输出
与原始 model.onnx 的输出对比

运行方式（从项目根目录）:
  python python/test/test_onnx.py [--test zh|en_zh]

输出:
  python/test/outputs/debug/ 下的 npy 文件（供 C++ 端参考）
  python/test/outputs/baseline/ 下的 WAV 文件

背景:
  完整 model.onnx 包含以下动态算子，无法直接编译为 bmodel：
  - RandomNormalLike ×2: TPU-MLIR 不支持 (RandnLike.cpp crash)
  - NonZero ×21 (SDP模块内): 数据相关 shape，shape-infer 崩溃
  - T_mel 动态维度: duration predictor 决定，无法静态推断

  解决方案: 模型拆分
  - Part 1 (CPU): 文本编码器 + SDP + 对齐 + Flow -> z_hat[1,192,T_mel]
  - Part 2 (bmodel): HiFi-GAN 解码器 -> audio[1,1,T_audio]

  当前 bmodel: decoder_T256.onnx (T_mel_fixed=256, T_audio_fixed=131072)
"""

import argparse
import sys
import os
import numpy as np
import onnxruntime as ort

# 路径设置：从项目根目录运行
PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
DEBUG_DIR = os.path.join(PROJ_ROOT, 'python/test/outputs/debug')
BASELINE_DIR = os.path.join(PROJ_ROOT, 'python/test/outputs/baseline')
os.makedirs(DEBUG_DIR, exist_ok=True)
os.makedirs(BASELINE_DIR, exist_ok=True)

T_MEL_FIXED = 256  # 与 decoder_T256.onnx 对应

# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────
def save_wav(audio_np, path, sr=44100):
    """保存 float32 audio 为 WAV 文件"""
    try:
        import soundfile as sf
        sf.write(path, audio_np.squeeze(), sr)
    except ImportError:
        try:
            import scipy.io.wavfile as wav
            audio_int16 = (audio_np.squeeze() * 32767).clip(-32768, 32767).astype(np.int16)
            wav.write(path, sr, audio_int16)
        except ImportError:
            print(f"  [WAV] 跳过（无 soundfile/scipy）")
            return
    print(f"  WAV saved: {path}")


def load_debug_inputs(test_name):
    """加载 baseline debug 输入"""
    tokens = np.load(os.path.join(DEBUG_DIR, f'{test_name}_input_tokens.npy'))
    tones = np.load(os.path.join(DEBUG_DIR, f'{test_name}_input_tones.npy'))
    return tokens, tones


def prepare_padded_inputs(tokens, tones, L_fixed=128):
    """将 tokens/tones 填充到 L_fixed"""
    L_actual = len(tokens)
    x_padded = np.zeros((1, L_fixed), dtype=np.int64)
    tones_padded = np.zeros((1, L_fixed), dtype=np.int64)
    x_padded[0, :L_actual] = tokens
    tones_padded[0, :L_actual] = tones
    x_lengths = np.array([L_actual], dtype=np.int64)
    return x_padded, x_lengths, tones_padded


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', default='zh', choices=['zh', 'en_zh'],
                        help='使用哪个测试用例')
    parser.add_argument('--noise_scale', type=float, default=0.667)
    parser.add_argument('--length_scale', type=float, default=1.0)
    parser.add_argument('--noise_scale_w', type=float, default=0.8)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    test_name = f'test_{args.test}'
    print("=" * 60)
    print(f"  vits-melo-tts-zh_en ONNX 验证: {test_name}")
    print("=" * 60)

    # ── 1. 加载输入
    print(f"\n[1] 加载 {test_name} 输入...")
    tokens, tones = load_debug_inputs(test_name)
    L_actual = len(tokens)
    x_padded, x_lengths, tones_padded = prepare_padded_inputs(tokens, tones, L_fixed=128)
    print(f"  L_actual={L_actual}, x_padded={x_padded.shape}")

    # ── 2. 运行完整 model.onnx (baseline)
    print("\n[2] 运行 model.onnx (baseline)...")
    model_onnx_path = os.path.join(PROJ_ROOT, 'models/onnx/vits-melo-tts-zh_en/model.onnx')
    if not os.path.exists(model_onnx_path):
        print(f"  [Error] 找不到 {model_onnx_path}")
        sys.exit(1)

    from onnx import helper, TensorProto
    import onnx

    # 获取 flow 输出 z_hat 和 y_mask（用于 Part 1 到 Part 2 的传递）
    model_full = onnx.load(model_onnx_path)
    model_copy = onnx.ModelProto()
    model_copy.CopyFrom(model_full)
    # 添加中间输出
    flow_out_name = '/flow/flows.0/Concat_output_0'
    ymask_name = '/Cast_4_output_0'
    g_emb_name = '/Unsqueeze_6_output_0'
    for name in [flow_out_name, ymask_name, g_emb_name]:
        model_copy.graph.output.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, None))

    sess_full = ort.InferenceSession(model_copy.SerializeToString())

    full_outputs = sess_full.run(None, {
        'x': x_padded,
        'x_lengths': x_lengths,
        'tones': tones_padded,
        'sid': np.array([1], dtype=np.int64),  # speaker_id=1 per model metadata
        'noise_scale': np.array([args.noise_scale], dtype=np.float32),
        'length_scale': np.array([args.length_scale], dtype=np.float32),
        'noise_scale_w': np.array([args.noise_scale_w], dtype=np.float32),
    })

    audio_full = full_outputs[0]           # [1, 1, T_audio]
    z_hat_full = full_outputs[1]           # [1, 192, T_mel] (actual)
    y_mask_full = full_outputs[2]          # [1, 1, T_mel]
    g_emb = full_outputs[3]               # [1, 256, 1]

    T_mel_actual = z_hat_full.shape[2]
    T_audio_actual = audio_full.shape[2]

    print(f"  model.onnx output: audio={audio_full.shape}")
    print(f"  z_hat (flow output): {z_hat_full.shape}")
    print(f"  y_mask: {y_mask_full.shape}")
    print(f"  g_emb: {g_emb.shape}")
    print(f"  T_mel_actual={T_mel_actual}, T_audio_actual={T_audio_actual}")

    # ── 3. 运行 decoder_T256.onnx (Part 2)
    print(f"\n[3] 运行 decoder_T{T_MEL_FIXED}.onnx...")
    decoder_onnx_path = os.path.join(
        PROJ_ROOT, f'models/onnx/vits-melo-tts-zh_en/decoder_T{T_MEL_FIXED}.onnx'
    )
    if not os.path.exists(decoder_onnx_path):
        print(f"  [Error] 找不到 {decoder_onnx_path}")
        print("  请先运行: python python/fix_onnx.py --generate_decoder")
        sys.exit(1)

    sess_dec = ort.InferenceSession(decoder_onnx_path)

    # 准备 decoder 输入: pad/trim z_hat 到 T_mel_fixed
    if T_mel_actual <= T_MEL_FIXED:
        # Pad
        z_hat_padded = np.zeros((1, 192, T_MEL_FIXED), dtype=np.float32)
        z_hat_padded[:, :, :T_mel_actual] = z_hat_full
        # 对 padding 部分乘以 0（实际上已经是0，y_mask会处理）
    else:
        # Trim (不应该发生 for T_mel_fixed=256 with L=128)
        z_hat_padded = z_hat_full[:, :, :T_MEL_FIXED]
        print(f"  WARNING: T_mel_actual ({T_mel_actual}) > T_mel_fixed ({T_MEL_FIXED}), trimming!")

    print(f"  z_hat_padded: {z_hat_padded.shape}")

    dec_outputs = sess_dec.run(None, {
        '/Mul_10_output_0': z_hat_padded,
        '/Unsqueeze_6_output_0': g_emb,
    })
    audio_dec = dec_outputs[0]  # [1, 1, T_mel_fixed * 512]

    print(f"  decoder output: {audio_dec.shape}")

    # 有效音频 = 前 T_mel_actual * 512 个采样
    T_audio_valid = T_mel_actual * 512
    audio_dec_valid = audio_dec[:, :, :T_audio_valid]

    print(f"  有效音频（前 {T_audio_valid} samples）: {audio_dec_valid.shape}")

    # ── 4. 质量对比
    print("\n[4] 质量验证...")

    # 基本健康检查
    assert not np.isnan(audio_full).any(), "model.onnx audio has NaN!"
    assert not np.isnan(audio_dec).any(), "decoder audio has NaN!"
    assert not np.isinf(audio_dec).any(), "decoder audio has Inf!"

    print(f"  model.onnx  - range: [{audio_full.min():.4f}, {audio_full.max():.4f}]")
    print(f"  decoder     - range: [{audio_dec_valid.min():.4f}, {audio_dec_valid.max():.4f}]")

    # 对比数值（由于使用相同的 z_hat，输出应该非常接近）
    min_len = min(audio_full.shape[2], audio_dec_valid.shape[2])
    diff = np.abs(audio_full[:, :, :min_len] - audio_dec_valid[:, :, :min_len])
    print(f"  mean_diff (first {min_len} samples): {diff.mean():.6f}")
    print(f"  max_diff:  {diff.max():.6f}")

    if diff.mean() < 1e-4:
        print("  PASS: decoder 与 model.onnx 输出高度一致 (mean_diff < 1e-4)")
    elif diff.mean() < 1e-2:
        print("  OK: decoder 与 model.onnx 输出基本一致 (mean_diff < 1e-2)")
    else:
        print(f"  WARNING: 差异较大 (mean_diff={diff.mean():.4f})")

    # ── 5. 保存 debug 输出
    print("\n[5] 保存 debug 文件...")

    # bmodel 推理所需的输入文件（padded）
    np.save(os.path.join(DEBUG_DIR, f'{test_name}_input_x_padded.npy'), x_padded)
    np.save(os.path.join(DEBUG_DIR, f'{test_name}_input_tones_padded.npy'), tones_padded)
    np.save(os.path.join(DEBUG_DIR, f'{test_name}_input_x_lengths.npy'), x_lengths)

    # flow 输出（Part 1 -> Part 2 的中间结果）
    np.save(os.path.join(DEBUG_DIR, f'{test_name}_z_hat.npy'), z_hat_full)
    np.save(os.path.join(DEBUG_DIR, f'{test_name}_z_hat_padded.npy'), z_hat_padded)
    np.save(os.path.join(DEBUG_DIR, f'{test_name}_g_emb.npy'), g_emb)

    # 音频输出参考
    np.save(os.path.join(DEBUG_DIR, f'{test_name}_audio_full.npy'), audio_full)
    np.save(os.path.join(DEBUG_DIR, f'{test_name}_audio_decoder.npy'), audio_dec)

    saved_files = [
        f'{test_name}_input_x_padded.npy       [1,128] int64',
        f'{test_name}_input_tones_padded.npy    [1,128] int64',
        f'{test_name}_input_x_lengths.npy       [1] int64  (实际长度={L_actual})',
        f'{test_name}_z_hat.npy                 {z_hat_full.shape} (动态T_mel={T_mel_actual})',
        f'{test_name}_z_hat_padded.npy          {z_hat_padded.shape} (T_mel_fixed={T_MEL_FIXED})',
        f'{test_name}_g_emb.npy                 {g_emb.shape}',
        f'{test_name}_audio_full.npy            {audio_full.shape} (baseline)',
        f'{test_name}_audio_decoder.npy         {audio_dec.shape} (decoder bmodel input)',
    ]
    for f in saved_files:
        print(f"  {f}")

    # ── 6. 保存 WAV
    print("\n[6] 保存 WAV 文件...")
    save_wav(audio_full, os.path.join(BASELINE_DIR, f'{test_name}_audio_full.wav'))
    save_wav(audio_dec_valid, os.path.join(BASELINE_DIR, f'{test_name}_audio_decoder_valid.wav'))

    # ── 7. 摘要
    print("\n" + "=" * 60)
    print("  验证摘要")
    print("=" * 60)
    print(f"  测试用例: {test_name}")
    print(f"  L_actual: {L_actual}")
    print(f"  T_mel_actual: {T_mel_actual} (动态)")
    print(f"  T_mel_fixed: {T_MEL_FIXED} (bmodel 固定)")
    print(f"  T_audio_actual: {T_audio_actual} ({T_audio_actual/44100:.2f}s)")
    print(f"  T_audio_fixed: {T_MEL_FIXED*512} ({T_MEL_FIXED*512/44100:.2f}s bmodel 输出)")
    print(f"  audio range: [{audio_dec_valid.min():.4f}, {audio_dec_valid.max():.4f}]")
    print(f"  NaN: {np.isnan(audio_dec).any()}")
    print(f"  mean_diff vs baseline: {diff.mean():.6f}")
    print()
    print("  bmodel 推理方式（C++ 端）:")
    print(f"    1. 用 ONNX 运行 model.onnx 直到 /flow/flows.0/Concat_output_0")
    print(f"       得到 z_hat[1,192,T_mel_actual]，g_emb[1,256,1]")
    print(f"    2. Pad z_hat 到 [1,192,{T_MEL_FIXED}]")
    print(f"    3. 运行 bmodel decoder")
    print(f"    4. 截取 audio[:,:,:{T_mel_actual}*512] 作为输出")
    print()
    print("  (或: 分离 SDP+文本编码部分到更小的 ONNX，bmodel 仅做解码)")
    print("=" * 60)


if __name__ == '__main__':
    main()
