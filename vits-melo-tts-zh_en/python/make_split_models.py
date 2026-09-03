#!/usr/bin/env python3
"""
make_split_models.py  -- 三段式拆分，最大化 TPU 利用率

架构:
  Part A (TPU, ~364 nodes):
      x[1,L], x_lengths[1], tones[1,L]
      → enc_p (文本编码器) + dp (确定性时长预测器)
      → dp_w[1,1,L], h[1,192,L], attn_mask[1,1,1,L], x_mask[1,1,L]
      注: g_emb(speaker embedding) 已被 constant-fold 进 Part C 权重

  Part B (CPU, ~39 nodes, <10ms):
      dp_w + h + attn_mask + x_mask
      → T_mel 计算 (15节点: Exp/Mul/Ceil/ReduceSum/Clip/Cast/ReduceMax)
      → MAS 对齐 (24节点: Range/Less/Reshape/Pad/Slice/Sub/Mul/MatMul/Transpose)
      → z_p[1,192,T_mel], y_mask[1,1,T_mel]

  Part C1 (TPU, Flow):
      z_p[1,192,T_fixed], y_mask[1,1,T_fixed]
      → flow output [1,192,T_fixed]
  Part C2 (TPU, Decoder):
      flow output [1,192,T_fixed]
      → audio[1,1,T_fixed*512]

  注意: T_MEL_FIXED=512 是当前 BM1684X C++ 运行时约定。

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
TMP_DIR  = os.path.join(PROJ, 'compile/tmp/vits_split')
TPU_ONNX = os.path.join(PROJ, 'models/onnx/vits-melo-tts-zh_en/model_tpu.onnx')
OUT_A    = os.path.join(PROJ, 'models/onnx/vits-melo-tts-zh_en/part_a_encoder.onnx')
OUT_C1   = os.path.join(PROJ, 'models/onnx/vits-melo-tts-zh_en/part_c1_flow.onnx')
OUT_C2   = os.path.join(PROJ, 'models/onnx/vits-melo-tts-zh_en/part_c2_decoder.onnx')

L_FIXED     = 128
T_MEL_FIXED = 512


def main():
    os.makedirs(TMP_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(OUT_A), exist_ok=True)
    print(f'Loading {TPU_ONNX}')
    m = onnx.load(TPU_ONNX)

    m_si = shape_inference.infer_shapes(m, check_type=False, strict_mode=False)
    tmp_si = os.path.join(TMP_DIR, 'model_tpu_si.onnx')
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

    # ── Part C1 (Flow) ──────────────────────────────────
    print(f'\n=== Part C1 (Flow, T_mel_fixed={T_MEL_FIXED}) ===')
    extract_model(tmp_si, OUT_C1,
        input_names=['/Transpose_3_output_0', '/Cast_4_output_0'],
        output_names=['/Mul_10_output_0'])

    mc1 = onnx.load(OUT_C1)
    mc1_sim, ok = simplify(mc1, overwrite_input_shapes={
        '/Transpose_3_output_0': [1, 192, T_MEL_FIXED],
        '/Cast_4_output_0':      [1, 1, T_MEL_FIXED],
    }, perform_optimization=True)
    if ok:
        mc1 = mc1_sim
    onnx.save(mc1, OUT_C1)

    # ── Part C2 (Decoder) ──────────────────────────────
    print(f'\n=== Part C2 (Decoder, T_mel_fixed={T_MEL_FIXED}) ===')
    extract_model(tmp_si, OUT_C2,
        input_names=['/Mul_10_output_0'],
        output_names=['y'])

    mc2 = onnx.load(OUT_C2)
    mc2_sim, ok = simplify(mc2, overwrite_input_shapes={
        '/Mul_10_output_0': [1, 192, T_MEL_FIXED],
    }, perform_optimization=True)
    if ok:
        mc2 = mc2_sim
    onnx.save(mc2, OUT_C2)

    # ── ORT 冒烟 ───────────────────────────────────────
    zp = np.random.randn(1, 192, T_MEL_FIXED).astype(np.float32)
    mask = np.ones((1, 1, T_MEL_FIXED), dtype=np.float32)
    out_c1 = ort.InferenceSession(OUT_C1).run(
        None, {'/Transpose_3_output_0': zp, '/Cast_4_output_0': mask})[0]
    out_c2 = ort.InferenceSession(OUT_C2).run(
        None, {'/Mul_10_output_0': out_c1})[0]
    print(f'  C1 output: {out_c1.shape}; C2 output: {out_c2.shape}')

    ops = {}
    for path in (OUT_C1, OUT_C2):
        for n in onnx.load(path).graph.node:
            ops[n.op_type] = ops.get(n.op_type, 0) + 1
    bad_c = {k: v for k, v in ops.items() if k in ('RandomNormalLike', 'NonZero', 'Range')}
    print(f'  C1+C2 nodes={sum(ops.values())}  problem_ops={bad_c}')

    # ── Summary ─────────────────────────────────────
    print(f'\n{"="*60}')
    if not bad_a and not bad_c:
        print('✅ 三个子模型无问题算子，可以编译 bmodel')
        print(f'\n  Part A: {OUT_A}')
        print(f'  Part C1: {OUT_C1}')
        print(f'  Part C2: {OUT_C2}')
    else:
        print(f'⚠️  仍有问题算子: A={bad_a}, C1+C2={bad_c}')

if __name__ == '__main__':
    main()
