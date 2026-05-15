#!/usr/bin/env python3
"""
make_split_models.py  -- 三段式拆分，最大化 TPU 利用率

架构:
  Part A (TPU, ~364 nodes):
      x[1,L], x_lengths[1], tones[1,L]
      → enc_p (文本编码器) + dp (确定性时长预测器)
      → dp_w[1,1,L], h[1,192,L], attn_mask[1,1,1,L], x_mask[1,1,L]
      注: g_emb(speaker embedding) 已被 constant-fold 进 Part C 权重

  Part B (CPU, ~39 nodes, <5ms):
      dp_w + h + attn_mask + x_mask
      → T_mel 计算 (15节点: Exp/Mul/Ceil/ReduceSum/Clip/Cast/ReduceMax)
      → MAS 对齐 (24节点: Range/Less/Reshape/Pad/Slice/Sub/Mul/MatMul/Transpose)
      → z_p[1,192,T_mel], y_mask[1,1,T_mel]

  Part C (TPU, ~1942 nodes):
      z_p[1,192,T_fixed], y_mask[1,1,T_fixed]
      → Flow (归一化流逆向, ~1695 nodes)
      → HiFi-GAN Decoder (~245 nodes)
      → audio[1,1,T_fixed*512]

用法: python python/make_split_models.py
      (从项目根目录运行)
"""

import os
import numpy as np
import onnx
from onnx import shape_inference
from onnx.utils import extract_model
from onnxsim import simplify
import onnxruntime as ort

PROJ     = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TPU_ONNX = os.path.join(PROJ, 'models/onnx/vits-melo-tts-zh_en/model_tpu.onnx')
OUT_A    = os.path.join(PROJ, 'models/onnx/vits-melo-tts-zh_en/part_a_encoder.onnx')
OUT_C    = os.path.join(PROJ, 'models/onnx/vits-melo-tts-zh_en/part_c_flow_decoder.onnx')

L_FIXED     = 128
T_MEL_FIXED = 256   # ~3s 上限（256 * 512 / 44100 ≈ 2.97s）

def main():
    print(f'Loading {TPU_ONNX}')
    m = onnx.load(TPU_ONNX)

    # shape inference 以填充 value_info
    m_si = shape_inference.infer_shapes(m, check_type=False, strict_mode=False)
    tmp_si = TPU_ONNX.replace('.onnx', '_si.onnx')
    onnx.save(m_si, tmp_si)

    # ── Part A ──────────────────────────────────────
    print('\n=== Part A (enc_p + dp) ===')
    part_a_outputs = [
        '/dp/Mul_3_output_0',                # dp_w  [1,1,128]
        '/enc_p/Split_output_0',              # h     [1,192,128]
        '/enc_p/encoder/Unsqueeze_output_0',  # attn_mask [1,1,1,128]
        '/enc_p/Cast_1_output_0',             # x_mask    [1,1,128]
    ]
    extract_model(tmp_si, OUT_A,
        input_names=['x', 'x_lengths', 'tones'],
        output_names=part_a_outputs)

    # verify
    sess_a = ort.InferenceSession(OUT_A)
    x  = np.zeros((1, L_FIXED), dtype=np.int64); x[0, :10] = 1
    xl = np.array([10], dtype=np.int64)
    t  = np.zeros((1, L_FIXED), dtype=np.int64)
    out_a = sess_a.run(None, {'x': x, 'x_lengths': xl, 'tones': t})
    print(f'  outputs: {[o.shape for o in out_a]}')

    ops_a = {}
    for n in onnx.load(OUT_A).graph.node:
        ops_a[n.op_type] = ops_a.get(n.op_type, 0) + 1
    bad_a = {k: v for k, v in ops_a.items() if k in ('RandomNormalLike', 'NonZero', 'Range')}
    print(f'  nodes={sum(ops_a.values())}  problem_ops={bad_a}')

    # ── Part C ──────────────────────────────────────
    print(f'\n=== Part C (Flow + Decoder, T_mel_fixed={T_MEL_FIXED}) ===')
    # Flow 输入: z_p = /Transpose_3_output_0, y_mask = /Cast_4_output_0
    # g_emb 已 constant-fold 进权重
    extract_model(tmp_si, OUT_C,
        input_names=['/Transpose_3_output_0', '/Cast_4_output_0'],
        output_names=['y'])

    mc = onnx.load(OUT_C)
    mc_sim, ok = simplify(mc, overwrite_input_shapes={
        '/Transpose_3_output_0': [1, 192, T_MEL_FIXED],
        '/Cast_4_output_0':      [1, 1, T_MEL_FIXED],
    }, perform_optimization=True)
    if ok:
        mc = mc_sim
    onnx.save(mc, OUT_C)

    # verify
    sess_c = ort.InferenceSession(OUT_C)
    zp   = np.random.randn(1, 192, T_MEL_FIXED).astype(np.float32)
    mask = np.ones((1, 1, T_MEL_FIXED), dtype=np.float32)
    out_c = sess_c.run(None, {'/Transpose_3_output_0': zp, '/Cast_4_output_0': mask})
    print(f'  outputs: {[o.shape for o in out_c]}')

    ops_c = {}
    for n in mc.graph.node:
        ops_c[n.op_type] = ops_c.get(n.op_type, 0) + 1
    bad_c = {k: v for k, v in ops_c.items() if k in ('RandomNormalLike', 'NonZero', 'Range')}
    print(f'  nodes={sum(ops_c.values())}  problem_ops={bad_c}')

    # ── Summary ─────────────────────────────────────
    print(f'\n{"="*60}')
    if not bad_a and not bad_c:
        print('✅ 两个子模型无问题算子，可以编译 bmodel')
        print(f'\n  Part A: {OUT_A}')
        print(f'  Part C: {OUT_C}')
        print(f'\n编译命令 (在 Docker 内 /tmp/compile 目录):')
        print(f'  model_transform.py --model_name vits_part_a \\')
        print(f"    --model_def {OUT_A} --input_shapes '[[1,{L_FIXED}],[1],[1,{L_FIXED}]]' \\")
        print(f'    --mlir vits_part_a.mlir')
        print(f'  model_deploy.py --mlir vits_part_a.mlir --quantize F32 --chip bm1684x \\')
        print(f'    --disable_layer_group --model vits_part_a_F32.bmodel')
        print(f'  (同理 Part C: input_shapes [[1,192,{T_MEL_FIXED}],[1,1,{T_MEL_FIXED}]])')
    else:
        print(f'⚠️  仍有问题算子: A={bad_a}, C={bad_c}')

if __name__ == '__main__':
    main()
