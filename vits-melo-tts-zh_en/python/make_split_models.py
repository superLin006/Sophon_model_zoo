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
      z_p[1,192,1024], y_mask[1,1,1024]
      → flow output [1,192,1024]
  Part C2 (TPU, Decoder):
      flow output [1,192,1024]
      → audio[1,1,524288]

  注意: T_MEL_FIXED=1024 是当前 BM1684X C++ 运行时约定。

用法: python python/make_split_models.py
      (从项目根目录运行)
"""

import os
from copy import deepcopy
import numpy as np
import onnx
from onnx import shape_inference, helper, TensorProto
from onnxsim import simplify
import onnxruntime as ort

PROJ     = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TMP_DIR  = os.path.join(PROJ, 'compile/tmp/vits_split')
TPU_ONNX = os.path.join(PROJ, 'models/onnx/vits-melo-tts-zh_en/model_tpu.onnx')
OUT_A    = os.path.join(PROJ, 'models/onnx/vits-melo-tts-zh_en/part_a_encoder.onnx')
OUT_C1   = os.path.join(PROJ, 'models/onnx/vits-melo-tts-zh_en/part_c1_flow.onnx')
OUT_C2   = os.path.join(PROJ, 'models/onnx/vits-melo-tts-zh_en/part_c2_decoder.onnx')

L_FIXED     = 256
T_MEL_FIXED = 1024
STREAM_WINDOW = 128
STREAM_OVERLAP = 32
STREAM_CONTEXT = 16
STREAM_INPUT = STREAM_WINDOW + 2 * STREAM_CONTEXT


def set_static_shape(value_info, shape):
    value_info.type.tensor_type.shape.ClearField('dim')
    for value in shape:
        value_info.type.tensor_type.shape.dim.add(dim_value=value)


def make_c2_stream_model():
    path = os.path.join(
        os.path.dirname(OUT_C2),
        f'part_c2_decoder_stream_W{STREAM_WINDOW}_R{STREAM_CONTEXT}.onnx',
    )
    model = onnx.load(OUT_C2)
    set_static_shape(model.graph.input[0], [1, 192, STREAM_INPUT])
    set_static_shape(model.graph.output[0], [1, 1, STREAM_INPUT * 512])
    model.graph.ClearField('value_info')
    model = shape_inference.infer_shapes(model, check_type=False, strict_mode=False)
    onnx.save(model, path)
    print(f'  C2 stream model: input=[1,192,{STREAM_INPUT}] '
          f'output=[1,1,{STREAM_INPUT * 512}]')
    return path



def extract_subgraph(model, input_shapes, output_shapes, output_names, path, input_types=None):
    input_types = input_types or {}
    producers = {}
    for node in model.graph.node:
        for output in node.output:
            producers[output] = node

    selected = []
    selected_ids = set()
    boundary = set(input_shapes)

    def visit(value):
        if value in boundary:
            return
        node = producers.get(value)
        if node is None or id(node) in selected_ids:
            return
        for inp in node.input:
            if inp:
                visit(inp)
        selected_ids.add(id(node))
        selected.append(node)

    for output in output_names:
        visit(output)

    initializer_map = {item.name: item for item in model.graph.initializer}
    used_inputs = {inp for node in selected for inp in node.input if inp}
    initializers = [
        deepcopy(initializer_map[name])
        for name in used_inputs
        if name in initializer_map and name not in boundary
    ]
    inputs = [
        helper.make_tensor_value_info(
            name, input_types.get(name, TensorProto.FLOAT), shape)
        for name, shape in input_shapes.items()
    ]
    outputs = [
        helper.make_tensor_value_info(name, TensorProto.FLOAT, output_shapes[name])
        for name in output_names
    ]
    graph = helper.make_graph(
        [deepcopy(node) for node in selected],
        f"{model.graph.name}_subgraph",
        inputs,
        outputs,
        initializer=initializers,
    )
    subgraph = helper.make_model(graph, producer_name="Sophon Model Zoo")
    subgraph.opset_import.clear()
    subgraph.opset_import.extend(deepcopy(item) for item in model.opset_import)
    subgraph.ir_version = model.ir_version
    onnx.checker.check_model(subgraph)
    onnx.save(subgraph, path)


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
        '/dp/Mul_3_output_0',                # dp_w  [1,1,256]
        '/enc_p/Split_output_0',              # h     [1,192,256]
        '/enc_p/encoder/Unsqueeze_output_0',  # attn_mask [1,1,1,256]
        '/enc_p/Cast_1_output_0',             # x_mask    [1,1,256]
    ]
    extract_subgraph(
        m_si,
        {'x': [1, L_FIXED], 'x_lengths': [1], 'tones': [1, L_FIXED]},
        {
            part_a_outputs[0]: [1, 1, L_FIXED],
            part_a_outputs[1]: [1, 192, L_FIXED],
            part_a_outputs[2]: [1, 1, 1, L_FIXED],
            part_a_outputs[3]: [1, 1, L_FIXED],
        },
        part_a_outputs,
        OUT_A,
        {'x': TensorProto.INT64, 'x_lengths': TensorProto.INT64, 'tones': TensorProto.INT64},
    )

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
    extract_subgraph(
        m_si,
        {
            '/Transpose_3_output_0': [1, 192, T_MEL_FIXED],
            '/Cast_2_output_0': [1, 1, T_MEL_FIXED],
        },
        {'/Mul_10_output_0': [1, 192, T_MEL_FIXED]},
        ['/Mul_10_output_0'],
        OUT_C1,
    )

    mc1 = onnx.load(OUT_C1)
    mc1_sim, ok = simplify(mc1, overwrite_input_shapes={
        '/Transpose_3_output_0': [1, 192, T_MEL_FIXED],
        '/Cast_2_output_0':      [1, 1, T_MEL_FIXED],
    }, perform_optimization=True)
    if ok:
        mc1 = mc1_sim
    onnx.save(mc1, OUT_C1)

    # ── Part C2 (Decoder) ──────────────────────────────
    print(f'\n=== Part C2 (Decoder, T_mel_fixed={T_MEL_FIXED}) ===')
    extract_subgraph(
        m_si,
        {'/Mul_10_output_0': [1, 192, T_MEL_FIXED]},
        {'y': [1, 1, T_MEL_FIXED * 512]},
        ['y'],
        OUT_C2,
    )

    mc2 = onnx.load(OUT_C2)
    mc2_sim, ok = simplify(mc2, overwrite_input_shapes={
        '/Mul_10_output_0': [1, 192, T_MEL_FIXED],
    }, perform_optimization=True)
    if ok:
        mc2 = mc2_sim
    onnx.save(mc2, OUT_C2)

    stream_c2 = make_c2_stream_model()

    # ── ORT 冒烟 ───────────────────────────────────────
    zp = np.random.randn(1, 192, T_MEL_FIXED).astype(np.float32)
    mask = np.ones((1, 1, T_MEL_FIXED), dtype=np.float32)
    out_c1 = ort.InferenceSession(OUT_C1).run(
        None, {'/Transpose_3_output_0': zp, '/Cast_2_output_0': mask})[0]
    out_c2 = ort.InferenceSession(OUT_C2).run(
        None, {'/Mul_10_output_0': out_c1})[0]
    stream_input = np.random.randn(1, 192, STREAM_INPUT).astype(np.float32)
    stream_out = ort.InferenceSession(stream_c2).run(
        None, {'/Mul_10_output_0': stream_input})[0]
    assert stream_out.shape == (1, 1, STREAM_INPUT * 512)
    print(f'  C1 output: {out_c1.shape}; C2 output: {out_c2.shape}')
    print(f'  C2 stream output: {stream_out.shape}')

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
