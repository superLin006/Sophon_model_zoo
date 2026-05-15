#!/usr/bin/env python3
"""
make_tpu_model.py  -- 生成可完整编译为 bmodel 的 ONNX

修改策略:
  1. 把 SDP 输出 /Mul_2_output_0 替换为全零常量
     => 消除 SDP 内全部 NonZero×21 + /sdp/RandomNormalLike
     => 推理时只用 DP (确定性时长预测), 效果与 noise_scale_w=0 等价
  2. 把 noise_scale 输入替换为常量 0
     => /Mul_9_output_0 = 0, flow 的 /RandomNormalLike 分支被 constant-fold 消除
  3. 用 onnxsim 做 constant folding, 自动删除死节点
  4. 输出: model_tpu.onnx  -- 可直接用 model_transform.py 编译 bmodel

用法:
  python python/make_tpu_model.py
  (在项目根目录运行)
"""

import os, sys
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
IN_ONNX  = os.path.join(PROJ, 'models/onnx/vits-melo-tts-zh_en/model.onnx')
OUT_ONNX = os.path.join(PROJ, 'models/onnx/vits-melo-tts-zh_en/model_tpu.onnx')

def make_zero_initializer(name, shape, dtype=np.float32):
    data = np.zeros(shape, dtype=dtype)
    return numpy_helper.from_array(data, name=name)

def main():
    print(f'Loading {IN_ONNX} ...')
    model = onnx.load(IN_ONNX)
    graph = model.graph

    # ── 收集图信息 ──────────────────────────────────
    # value_info shape map
    shape_map = {}
    for vi in list(graph.value_info) + list(graph.input) + list(graph.output):
        dims = []
        for d in vi.type.tensor_type.shape.dim:
            dims.append(d.dim_value if d.dim_value > 0 else None)
        shape_map[vi.name] = dims

    init_names = {i.name for i in graph.initializer}

    # ── 修改 1: SDP 输出替换为全零 ──────────────────
    # /sdp/Split_output_0 [1,1,L] -> /Mul_2 * constant -> /Mul_2_output_0
    # 把 /Mul_2_output_0 替换为零常量, shape [1,1,128]
    L_FIXED = 128
    sdp_zero_name = 'sdp_zero_const'
    sdp_zero_init = make_zero_initializer(sdp_zero_name, [1, 1, L_FIXED])
    graph.initializer.append(sdp_zero_init)

    # 找 /Mul_2 节点，把它的 output 名重定向，
    # 更简单：直接加一个新 initializer，然后把下游 /Add_1 的输入改掉
    # /Add_1 inputs = ['/Mul_2_output_0', '/dp/Mul_3_output_0']
    replaced_add1 = False
    for node in graph.node:
        if node.name == '/Add_1' and '/Mul_2_output_0' in node.input:
            idx = list(node.input).index('/Mul_2_output_0')
            node.input[idx] = sdp_zero_name
            replaced_add1 = True
            print(f'  [1] /Add_1 input[{idx}] <- {sdp_zero_name}  (SDP输出替换为全零)')
            break
    assert replaced_add1, '/Add_1 node not found!'

    # ── 修改 2: 删除 flow RandomNormalLike 分支 ──────
    # /RandomNormalLike -> /Mul_8(* sigma) -> /Mul_9(* noise_scale) -> /Add_2(+ /Transpose_3_output_0)
    # noise_scale=0 时，/Mul_9=0，/Add_2 = /Transpose_3_output_0
    # 步骤 a: 把所有使用 noise_scale 的节点替换为零常量（先断开 graph input 引用）
    ns_zero_name = 'noise_scale_zero'
    ns_zero_init = make_zero_initializer(ns_zero_name, [1])
    graph.initializer.append(ns_zero_init)
    ns_replaced = 0
    for node in graph.node:
        for i, inp in enumerate(node.input):
            if inp == 'noise_scale':
                node.input[i] = ns_zero_name
                ns_replaced += 1
    # 步骤 b: 把所有使用 /Add_2_output_0 的节点替换为 /Transpose_3_output_0
    # RNL/Mul_8/Mul_9/Add_2 变成死代码，onnxsim 自动清除
    add2_replaced = 0
    for node in graph.node:
        for i, inp in enumerate(node.input):
            if inp == '/Add_2_output_0':
                node.input[i] = '/Transpose_3_output_0'
                add2_replaced += 1
    print(f'  [2] noise_scale->0 ({ns_replaced}处), /Add_2_output_0->/Transpose_3_output_0 ({add2_replaced}处)')
    print(f'      RNL+Mul_8+Mul_9+Add_2 变死代码，onnxsim 自动清除')

    # noise_scale 从 graph inputs 移除（不再需要）
    to_remove = [i for i in graph.input if i.name == 'noise_scale']
    for inp in to_remove:
        graph.input.remove(inp)
    print(f'      noise_scale 从 graph inputs 中移除')

    # ── 修改 3: sid 固定为 1 ────────────────────────
    sid_init = numpy_helper.from_array(np.array([1], dtype=np.int64), name='sid_const_1')
    graph.initializer.append(sid_init)
    sid_replaced = 0
    for node in graph.node:
        for i, inp in enumerate(node.input):
            if inp == 'sid':
                node.input[i] = 'sid_const_1'
                sid_replaced += 1
    to_remove = [i for i in graph.input if i.name == 'sid']
    for inp in to_remove:
        graph.input.remove(inp)
    print(f'  [3] sid=1 固定为常量 (替换 {sid_replaced} 处, 从inputs移除)')

    # ── 保存中间结果 ─────────────────────────────────
    tmp_path = OUT_ONNX.replace('.onnx', '_pre_sim.onnx')
    onnx.save(model, tmp_path)
    print(f'\n中间文件保存: {tmp_path}')

    # ── 修改 4: onnxsim constant folding ────────────
    print('\nRunning onnxsim (constant folding + dead node elimination)...')
    try:
        from onnxsim import simplify
        # 固定 L=128 做 shape inference
        input_shapes = {
            'x':           [1, L_FIXED],
            'x_lengths':   [1],
            'tones':       [1, L_FIXED],
            'length_scale':[1],
            'noise_scale_w':[1],
        }
        model_sim, ok = simplify(
            model,
            overwrite_input_shapes=input_shapes,
            perform_optimization=True,
        )
        if ok:
            print('  onnxsim: OK')
            model = model_sim
        else:
            print('  onnxsim: simplification check failed, using pre-sim model')
    except Exception as e:
        print(f'  onnxsim error: {e}')
        print('  continuing without simplification...')

    # ── 保存最终结果 ─────────────────────────────────
    onnx.save(model, OUT_ONNX)
    print(f'\n输出: {OUT_ONNX}')

    # 验证
    nodes_by_op = {}
    for n in model.graph.node:
        nodes_by_op[n.op_type] = nodes_by_op.get(n.op_type, 0) + 1

    rnl = nodes_by_op.get('RandomNormalLike', 0)
    nz  = nodes_by_op.get('NonZero', 0)
    print(f'\n验证:')
    print(f'  RandomNormalLike: {rnl}  (目标: 0)')
    print(f'  NonZero:          {nz}  (目标: 0)')
    print(f'  总节点数:         {sum(nodes_by_op.values())}')
    print(f'  Graph inputs:     {[i.name for i in model.graph.input]}')

    if rnl == 0 and nz == 0:
        print('\n✅ 模型已准备好，可以编译 bmodel')
        print(f'\n下一步:')
        print(f'  model_transform.py \\')
        print(f'    --model_name vits_melo_tts_L128 \\')
        print(f'    --model_def {OUT_ONNX} \\')
        print(f'    --input_shapes [[1,128],[1],[1,128],[1],[1]] \\')
        print(f'    --mlir vits_melo_tts_L128.mlir')
    else:
        print(f'\n⚠️  仍有不支持算子，需要进一步处理')

if __name__ == '__main__':
    main()
