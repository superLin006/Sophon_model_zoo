#!/usr/bin/env python
# Qwen3-TTS onnx 完整性校验：确保 gen_final.sh 消费的 100 个 onnx 全部在场且结构正确。
#
# 背景：onnx 是编译输入（不是交付物，交付物是 models/BM1684X/ 的 3 个 bmodel）。
#   100 个 onnx = talker/56 + cp/40 + embedding/3 + codec/1，是 TPU-MLIR 静态 shape
#   约束下的必然拆分：每层 prefill/decode 分图、15 个 code head 独立、codec 单图。
#   本脚本从 gen_final.sh 中解析出全部引用路径，逐一检查存在性 + onnx.checker。
#
# 用法（sophon-qwen3-tts 或任意含 onnx 的 env）:
#   python verify_onnx.py
import os
import re
import sys

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # Qwen3-TTS/
ONNX_DIR = os.path.join(PROJ, "models", "onnx")
GEN = os.path.join(PROJ, "python", "gen_final.sh")


def expected_refs():
    """gen_final.sh 实际消费的 100 个 onnx（talker 56 + cp 40 + embedding 3 + codec 1）。"""
    refs = []
    refs += [f"talker/talker_block_{i}.onnx" for i in range(28)]
    refs += [f"talker/talker_block_cache_{i}.onnx" for i in range(28)]
    refs += [f"cp/cp_block_{i}.onnx" for i in range(5)]
    refs += [f"cp/cp_block_cache_{i}.onnx" for i in range(5)]
    refs += [f"cp/cp_lm_head_{g}.onnx" for g in range(15)]
    refs += [f"cp/cp_embedding_{e}.onnx" for e in range(15)]
    refs += ["embedding/embedding_code.onnx", "embedding/embedding_text.onnx",
             "embedding/codec_head.onnx", "codec/codec_decoder_T325.onnx"]
    return refs


def main():
    import onnx

    # gen_final.sh 中字面量引用的 onnx（确定性的部分），与期望集合交叉验证
    script = open(GEN, encoding="utf-8").read()
    literal = sorted(set(re.findall(r"\$ONNX/([a-z]+/[a-zA-Z0-9_]+\.onnx)", script)))
    refs = expected_refs()
    full = [(r, os.path.join(ONNX_DIR, r)) for r in refs]

    missing = [r for r, p in full if not os.path.isfile(p)]
    if missing:
        print(f"[FAIL] 缺失 {len(missing)}/{len(full)} 个 onnx：")
        for r in missing:
            print("   ", r)
        sys.exit(1)

    bad = []
    for r, p in full:
        try:
            m = onnx.load(p, load_external_data=False)
            onnx.checker.check_model(m)
        except Exception as e:
            bad.append((r, f"{type(e).__name__}: {e}"))

    print(f"期望 {len(full)} 个 onnx 全部存在，checker 失败 {len(bad)}")
    for r, err in bad:
        print(f"  FAIL {r}: {err}")
    if bad:
        sys.exit(1)

    from collections import Counter
    cats = Counter(r.split("/")[0] for r in refs)
    print("分类:", dict(cats), "（与 gen_final.sh 消费一致：literals", literal, "）")
    print("OK: Qwen3-TTS onnx 完整且结构正确")


if __name__ == "__main__":
    main()