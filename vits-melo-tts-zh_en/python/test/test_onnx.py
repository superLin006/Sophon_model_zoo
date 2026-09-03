#!/usr/bin/env python3
"""
vits-melo-tts-zh_en ONNX 验证（当前三段式 scheme）
==============================================
验证 models/onnx/vits-melo-tts-zh_en/ 下三段拆分 ONNX 的结构完整性 + ORT 可执行性：

  part_a_encoder.onnx  : x[1,256] + x_lengths[1] + tones[1,256]
                         → dp_w[1,1,256]、h[1,192,256]、attn_mask[1,1,1,256]、x_mask[1,1,256]
  part_c1_flow.onnx    : z_p[1,192,1024] + y_mask[1,1,1024] → flow 输出
  part_c2_decoder.onnx : flow 输出 [1,192,1024] → audio[1,1,524288]

历史说明：旧 test_onnx.py 引用已废弃的 model.onnx / decoder_T256.onnx（方案 D，整图无法编译
——含 SDP 的 NonZero×21、Flow 的 RandomNormalLike、动态 T_mel）。当前为三段式拆分方案，
产出的 3 个 onnx 由 make_tpu_model.py / make_split_models.py 生成，经本脚本做 shape/数值
可执行冒烟；端到端精度由 C++ 板卡的中英文 256 token 实测与 TTS 试听验证。

原始 model.onnx 来自上游 sherpa-onnx 对 MeloTTS 的导出（见 python/README.md），不在仓库内。

运行方式（从项目根目录，sophon-export 或任意含 onnxruntime 的 env）:
  python python/test/test_onnx.py              # 跑三段冒烟
环境: conda sophon-vits-melo-tts-zh-en（onnx 1.21 + onnxruntime）
"""
import os
import sys
import numpy as np

import onnx
import onnxruntime as ort

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ONNX_DIR = os.path.join(PROJ_ROOT, "models", "onnx", "vits-melo-tts-zh_en")


def _mk_session(path):
    so = ort.SessionOptions()
    so.log_severity_level = 3
    return ort.InferenceSession(path, so)


def _zeros_like(inp, int_dtype=np.int64):
    shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
    return np.zeros(tuple(shape), int_dtype if inp.type.startswith("tensor(int") else np.float32)


def _check(path, feeds):
    print(f"\n[{path.split('/')[-1]}]")
    sess = _mk_session(path)
    m = onnx.load(path)
    onnx.checker.check_model(m)          # 结构完整性
    print(f"  checker OK ({len(m.graph.node)} nodes)")
    for name, val in feeds.items():
        print(f"  in  {name} {tuple(val.shape)} {val.dtype}")
    outs = sess.run(None, feeds)
    ok = all(np.isfinite(o).all() and o.size > 0 for o in outs)
    for o in outs:
        print(f"  out {tuple(o.shape)} finite={np.isfinite(o).all()}")
    print(f"  => {'PASS' if ok else 'FAIL (NaN/空输出)'}")
    return ok


def main():
    L = 256
    T_MEL_FIXED = 1024
    all_ok = True

    a = _check(
        os.path.join(ONNX_DIR, "part_a_encoder.onnx"),
        {
            "x": np.zeros((1, L), np.int64),
            "x_lengths": np.full((1,), L, np.int64),
            "tones": np.zeros((1, L), np.int64),
        },
    )

    c1 = _check(
        os.path.join(ONNX_DIR, "part_c1_flow.onnx"),
        {
            "/Transpose_3_output_0": np.zeros((1, 192, T_MEL_FIXED), np.float32),
            "/Cast_2_output_0": np.ones((1, 1, T_MEL_FIXED), np.float32),
        },
    )

    # part_c2 输入是 part_c1 的 flow 输出，形状 [1,192,T_MEL_FIXED]
    c2 = _check(
        os.path.join(ONNX_DIR, "part_c2_decoder.onnx"),
        {
            "/Mul_10_output_0": np.zeros((1, 192, T_MEL_FIXED), np.float32),
        },
    )

    all_ok = a and c1 and c2
    print(f"\n结果: {'全部 PASS' if all_ok else '存在 FAIL'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()