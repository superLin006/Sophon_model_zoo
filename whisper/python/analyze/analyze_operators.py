"""
Whisper 模型算子兼容性分析脚本
分析 Whisper Encoder/Decoder 导出的 ONNX 中所有算子，
对照 Sophon TPU-MLIR 支持列表，输出兼容性报告。

用法:
    python analyze_operators.py --model base
    python analyze_operators.py --model small
"""

import argparse
import os
import sys
import torch
import whisper
import onnx
from collections import defaultdict

# ============================================================
# Sophon TPU-MLIR 支持的 ONNX 算子（来自 OnnxConverter.py）
# ============================================================
SOPHON_SUPPORTED_OPS = {
    "Abs", "Acos", "Add", "And", "ArgMax", "ArgMin", "Atan", "Atanh",
    "AveragePool", "BatchNormalization", "Cast", "Ceil", "Clip", "Concat",
    "Constant", "ConstantOfShape", "Conv", "ConvTranspose", "Cos", "CumSum",
    "DepthToSpace", "DequantizeLinear", "Div", "Dropout", "Einsum", "Elu",
    "Equal", "Erf", "Exp", "Expand", "Flatten", "Floor", "Gather",
    "GatherElements", "GatherND", "GELU", "Gemm", "GlobalAveragePool",
    "GlobalMaxPool", "Greater", "GreaterOrEqual", "GridSample",
    "GroupNormalization", "GRU", "HardSigmoid", "HardSwish", "Identity",
    "If", "InstanceNormalization", "LayerNormalization", "LeakyRelu",
    "Less", "LessOrEqual", "Log", "LogSoftmax", "Loop", "LRN", "LSTM",
    "MatMul", "Max", "MaxPool", "Min", "Mod", "Mul", "Neg",
    "NonMaxSuppression", "NonZero", "Not", "OneHot", "Or", "Pad", "PRelu",
    "Pow", "QuantizeLinear", "Range", "Reciprocal", "ReduceL1", "ReduceL2",
    "ReduceLogSumExp", "ReduceMax", "ReduceMean", "ReduceMin", "ReduceProd",
    "ReduceSum", "Relu", "Reshape", "Resize", "ReverseSequence", "RoiAlign",
    "Round", "ScatterElements", "ScatterND", "Shape", "Sigmoid", "Sign",
    "Sin", "Slice", "Softmax", "Softplus", "SpaceToDepth", "Split", "Sqrt",
    "Squeeze", "Sub", "Sum", "Tanh", "Tile", "TopK", "Transpose", "Trilu",
    "Unsqueeze", "Upsample", "Where", "Xor",
}

# 需要额外关注的算子（支持但有限制）
SOPHON_LIMITED_OPS = {
    "If":   "控制流，复杂情况可能 CPU fallback",
    "Loop": "控制流，复杂情况可能 CPU fallback",
}


def export_encoder_onnx(model, model_name: str, output_path: str):
    """导出 Whisper Encoder 为 ONNX"""
    print(f"\n[Encoder] 导出 ONNX -> {output_path}")

    class EncoderWrapper(torch.nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder

        def forward(self, mel):
            return self.encoder(mel)

    n_mels = model.dims.n_mels
    dummy_mel = torch.zeros(1, n_mels, 3000)
    wrapper = EncoderWrapper(model.encoder)
    wrapper.eval()

    torch.onnx.export(
        wrapper,
        dummy_mel,
        output_path,
        input_names=["mel"],
        output_names=["encoder_output"],
        dynamic_axes=None,
        opset_version=14,
        do_constant_folding=True,
    )
    print(f"[Encoder] ONNX 导出完成")


def export_decoder_onnx(model, model_name: str, output_path: str):
    """导出 Whisper Decoder 为 ONNX（单步推理，不含 KV Cache）"""
    print(f"\n[Decoder] 导出 ONNX -> {output_path}")

    dims = model.dims
    n_text_ctx = dims.n_text_ctx
    n_text_state = dims.n_text_state

    class DecoderWrapper(torch.nn.Module):
        def __init__(self, decoder):
            super().__init__()
            self.decoder = decoder

        def forward(self, tokens, encoder_output):
            return self.decoder(tokens, encoder_output)

    dummy_tokens = torch.zeros(1, 4, dtype=torch.long)
    dummy_encoder_output = torch.zeros(1, 1500, n_text_state)
    wrapper = DecoderWrapper(model.decoder)
    wrapper.eval()

    torch.onnx.export(
        wrapper,
        (dummy_tokens, dummy_encoder_output),
        output_path,
        input_names=["tokens", "encoder_output"],
        output_names=["logits"],
        dynamic_axes=None,
        opset_version=14,
        do_constant_folding=True,
    )
    print(f"[Decoder] ONNX 导出完成")


def analyze_onnx(onnx_path: str, part_name: str):
    """分析 ONNX 文件中的算子，返回兼容性信息"""
    print(f"\n{'='*60}")
    print(f"  分析: {part_name}  ({os.path.basename(onnx_path)})")
    print(f"{'='*60}")

    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    print(f"[OK] ONNX 格式验证通过")

    # 统计所有算子
    op_count = defaultdict(int)
    for node in model.graph.node:
        op_count[node.op_type] += 1

    all_ops = set(op_count.keys())
    supported = all_ops & SOPHON_SUPPORTED_OPS
    limited = all_ops & set(SOPHON_LIMITED_OPS.keys())
    unsupported = all_ops - SOPHON_SUPPORTED_OPS

    print(f"\n[算子统计]")
    print(f"  总算子种类: {len(all_ops)}")
    print(f"  ✅ 完全支持: {len(supported)}")
    print(f"  ⚠️  有限支持: {len(limited)}")
    print(f"  ❌ 不支持:   {len(unsupported)}")

    print(f"\n[完全支持的算子]")
    for op in sorted(supported):
        print(f"  ✅ {op:<30} (出现 {op_count[op]} 次)")

    if limited:
        print(f"\n[有限支持的算子 - 需关注]")
        for op in sorted(limited):
            print(f"  ⚠️  {op:<30} (出现 {op_count[op]} 次) -> {SOPHON_LIMITED_OPS[op]}")

    if unsupported:
        print(f"\n[不支持的算子 - 需要处理]")
        for op in sorted(unsupported):
            print(f"  ❌ {op:<30} (出现 {op_count[op]} 次)")
    else:
        print(f"\n[结论] ✅ 所有算子均被 Sophon TPU-MLIR 支持！")

    # 输入输出信息
    print(f"\n[模型输入输出]")
    for inp in model.graph.input:
        shape = [d.dim_value if d.dim_value > 0 else "?"
                 for d in inp.type.tensor_type.shape.dim]
        dtype = inp.type.tensor_type.elem_type
        print(f"  输入: {inp.name:<30} shape={shape}, dtype={dtype}")
    for out in model.graph.output:
        shape = [d.dim_value if d.dim_value > 0 else "?"
                 for d in out.type.tensor_type.shape.dim]
        dtype = out.type.tensor_type.elem_type
        print(f"  输出: {out.name:<30} shape={shape}, dtype={dtype}")

    return {
        "all_ops": all_ops,
        "supported": supported,
        "limited": limited,
        "unsupported": unsupported,
        "op_count": op_count,
    }


def print_model_dims(model):
    dims = model.dims
    print(f"\n[模型参数]")
    print(f"  n_mels:       {dims.n_mels}")
    print(f"  n_audio_ctx:  {dims.n_audio_ctx}")
    print(f"  n_audio_state:{dims.n_audio_state}")
    print(f"  n_audio_head: {dims.n_audio_head}")
    print(f"  n_audio_layer:{dims.n_audio_layer}")
    print(f"  n_vocab:      {dims.n_vocab}")
    print(f"  n_text_ctx:   {dims.n_text_ctx}")
    print(f"  n_text_state: {dims.n_text_state}")
    print(f"  n_text_head:  {dims.n_text_head}")
    print(f"  n_text_layer: {dims.n_text_layer}")


def main():
    parser = argparse.ArgumentParser(description="Whisper 算子兼容性分析")
    parser.add_argument("--model", default="base",
                        choices=["tiny", "base", "small", "medium"],
                        help="Whisper 模型大小")
    parser.add_argument("--output_dir", default="./onnx_analysis",
                        help="ONNX 输出目录")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Whisper 算子兼容性分析 (Sophon BM1684)")
    print(f"  模型: whisper-{args.model}")
    print(f"{'='*60}")

    # 加载模型
    print(f"\n[加载模型] whisper-{args.model} ...")
    model = whisper.load_model(args.model)
    model.eval()
    print_model_dims(model)

    # 导出 Encoder ONNX
    enc_onnx = os.path.join(args.output_dir, f"whisper_{args.model}_encoder.onnx")
    export_encoder_onnx(model, args.model, enc_onnx)

    # 导出 Decoder ONNX
    dec_onnx = os.path.join(args.output_dir, f"whisper_{args.model}_decoder.onnx")
    export_decoder_onnx(model, args.model, dec_onnx)

    # 分析算子
    enc_result = analyze_onnx(enc_onnx, "Whisper Encoder")
    dec_result = analyze_onnx(dec_onnx, "Whisper Decoder")

    # 汇总报告
    print(f"\n{'='*60}")
    print(f"  汇总报告")
    print(f"{'='*60}")

    all_unsupported = enc_result["unsupported"] | dec_result["unsupported"]
    all_limited = enc_result["limited"] | dec_result["limited"]

    if not all_unsupported and not all_limited:
        print(f"\n✅ 完全兼容！可直接进行 ONNX → bmodel 转换")
        print(f"   无需修改模型网络结构")
    elif not all_unsupported and all_limited:
        print(f"\n⚠️  基本兼容，但有以下算子需要关注:")
        for op in sorted(all_limited):
            print(f"   - {op}: {SOPHON_LIMITED_OPS.get(op, '')}")
        print(f"\n   建议：先直接转换，测试是否 CPU fallback")
    else:
        print(f"\n❌ 存在不支持的算子，需要修改模型网络结构:")
        for op in sorted(all_unsupported):
            print(f"   - {op}")
        print(f"\n   处理方案请参考 .claude/doc/sophon_bm1684_knowledge_base.md")

    print(f"\n[ONNX 文件已保存至]")
    print(f"  {enc_onnx}")
    print(f"  {dec_onnx}")


if __name__ == "__main__":
    main()
