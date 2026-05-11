"""
SenseVoice Small ONNX 导出脚本
目标芯片: BM1684X (FP32/FP16)

模型结构:
  输入: audio_features [1, 166, 560]  (10s 音频, Fbank80 + LFR7)
  输出: logits [1, 170, 25055]        (170 = 166 + 4 prompt tokens)

prompt embedding 已在 MTK torch_model.py 中固定为 4 个可学习向量，
无需 Gather/Embedding lookup，对 BM1684X 算子友好。

用法:
    python export_onnx.py
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import onnx
import onnxsim

# 复用 MTK 的模型定义和工具函数
MTK_MODEL_DIR = os.path.join(
    os.path.dirname(__file__),
    "../../../MTK_model_zoo/sense-voice/SenseVoice_workspace/model_prepare"
)
sys.path.insert(0, os.path.abspath(MTK_MODEL_DIR))

from torch_model import SenseVoiceSmall
from model_utils import load_cmvn, load_pretrained_weights

MODEL_DIR  = os.path.join(os.path.dirname(__file__), "../../../MTK_model_zoo/sense-voice/SenseVoice_workspace/models/sensevoice-small")


def _fix_conv_kernel_shape(model):
    """新版 torch.onnx exporter 可能漏写 Conv 的 kernel_shape，从 weight 推导补全。"""
    init_map = {init.name: init for init in model.graph.initializer}
    for node in model.graph.node:
        if node.op_type != "Conv":
            continue
        if any(attr.name == "kernel_shape" for attr in node.attribute):
            continue
        if len(node.input) < 2:
            continue
        weight_name = node.input[1]
        if weight_name not in init_map:
            continue
        kernel_shape = list(init_map[weight_name].dims[2:])
        node.attribute.append(onnx.helper.make_attribute("kernel_shape", kernel_shape))
    return model
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../models/onnx")
FIXED_FRAMES   = 166   # 10s 音频对应的 LFR 帧数
INPUT_DIM      = 560   # Fbank(80) × LFR窗口(7)
VOCAB_SIZE     = 25055


def load_model(model_dir: str) -> SenseVoiceSmall:
    """用 FunASR 下载并转换权重，再加载到自定义模型"""
    model_pt = os.path.join(model_dir, "model.pt")

    if not os.path.exists(model_pt):
        print("[Setup] model.pt 不存在，从 FunASR/ModelScope 下载...")
        try:
            from funasr import AutoModel
            funasr_model = AutoModel(model="iic/SenseVoiceSmall",
                                     model_revision="master",
                                     disable_update=True)
            # FunASR 会把权重缓存到 ~/.cache/modelscope
            # 找到实际权重文件
            import glob
            pt_files = glob.glob(os.path.expanduser(
                "~/.cache/modelscope/hub/iic/SenseVoiceSmall/model.pt"))
            if not pt_files:
                pt_files = glob.glob(os.path.expanduser(
                    "~/.cache/modelscope/hub/*/SenseVoiceSmall/model.pt"))
            if pt_files:
                import shutil
                shutil.copy(pt_files[0], model_pt)
                print(f"[Setup] 已复制权重到: {model_pt}")
            else:
                # 直接从 funasr_model 的内部 model 提取 state_dict
                print("[Setup] 直接从 FunASR 模型提取权重...")
                inner = funasr_model.model
                torch.save(inner.state_dict(), model_pt)
                print(f"[Setup] 权重已保存到: {model_pt}")
        except Exception as e:
            print(f"[Error] 自动下载失败: {e}")
            print("请手动运行: ")
            print("  from funasr import AutoModel")
            print("  AutoModel(model='iic/SenseVoiceSmall')")
            sys.exit(1)

    print(f"[Load] 加载 CMVN 和权重...")
    neg_mean, inv_stddev = load_cmvn(os.path.join(model_dir, "am.mvn"))
    model = SenseVoiceSmall(neg_mean, inv_stddev)
    model = load_pretrained_weights(model, model_dir)
    model.eval()
    return model


def export(model: nn.Module, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    onnx_path = os.path.join(output_dir, "sensevoice_small.onnx")
    sim_path  = os.path.join(output_dir, "sensevoice_small_sim.onnx")

    # dummy 输入：每次独立创建，避免 constant folding
    dummy = torch.zeros(1, FIXED_FRAMES, INPUT_DIM)

    print(f"\n[Export] 导出 ONNX -> {onnx_path}")
    print(f"  输入: audio_features {list(dummy.shape)}")
    print(f"  预期输出: logits [1, {FIXED_FRAMES+4}, {VOCAB_SIZE}]")

    # 验证前向
    with torch.no_grad():
        # forward 只用第一个参数，其余为兼容性占位
        dummy_id = torch.tensor([0], dtype=torch.long)
        out = model(dummy, dummy_id, dummy_id, dummy_id, dummy_id)
    print(f"  实际输出 shape: {list(out.shape)}")

    # 包一层只接受单输入的 wrapper，去掉无用的 id 参数
    class ExportWrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            dummy_id = torch.zeros(1, dtype=torch.long)
            return self.m(x, dummy_id, dummy_id, dummy_id, dummy_id)

    wrapper = ExportWrapper(model)
    wrapper.eval()

    torch.onnx.export(
        wrapper,
        dummy,
        onnx_path,
        input_names=["audio_features"],
        output_names=["logits"],
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"[Export] 完成: {os.path.getsize(onnx_path)/1024/1024:.1f} MB")

    # 验证
    m = onnx.load(onnx_path)
    m = _fix_conv_kernel_shape(m)
    onnx.save(m, onnx_path)
    onnx.checker.check_model(m)
    print("[Export] ONNX 格式验证通过")

    # onnxsim 简化
    print("[Simplify] 运行 onnxsim...")
    model_sim, ok = onnxsim.simplify(onnx.load(onnx_path))
    if ok:
        model_sim = _fix_conv_kernel_shape(model_sim)
        onnx.save(model_sim, sim_path)
        print(f"[Simplify] 完成: {os.path.getsize(sim_path)/1024/1024:.1f} MB -> {sim_path}")
    else:
        print("[Simplify] 失败，使用原始 ONNX")
        sim_path = onnx_path

    return sim_path


def main():
    print("=" * 60)
    print("  SenseVoice Small ONNX 导出  (BM1684X)")
    print("=" * 60)

    model = load_model(MODEL_DIR)
    sim_path = export(model, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print(f"  产物: {sim_path}")
    print("  下一步: 运行 gen_bmodel.sh 转换为 bmodel")
    print("=" * 60)


if __name__ == "__main__":
    main()
