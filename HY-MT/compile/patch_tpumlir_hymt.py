#!/usr/bin/env python3
"""Reproducibly add HY-MT1.5 support to TPU-MLIR 1.28.1's llm_convert.

Run inside the TPU-MLIR container. Original files are retained with a
`.pre_hymt` suffix. Re-running the script is idempotent.
"""

from pathlib import Path
import shutil


ROOT = Path("/usr/local/lib/python3.10/dist-packages/tpu_mlir/python")


def patch_file(path: Path, replacements: list[tuple[str, str]]) -> None:
    backup = path.with_suffix(path.suffix + ".pre_hymt")
    if not backup.exists():
        shutil.copy2(path, backup)
    text = path.read_text(encoding="utf-8")
    changed = False
    for old, new in replacements:
        if new in text:
            continue
        count = text.count(old)
        if count == 0:
            raise RuntimeError(f"Patch anchor not found in {path}: {old[:100]!r}")
        text = text.replace(old, new)
        changed = True
    if changed:
        path.write_text(text, encoding="utf-8")
        print(f"patched {path}")
    else:
        print(f"already patched {path}")


def main() -> None:
    info = ROOT / "llm" / "LlmInfo.py"
    patch_file(info, [
        (
            '    QWEN3 = "qwen3"\n',
            '    QWEN3 = "qwen3"\n    HUNYUAN_V1_DENSE = "hunyuan_v1_dense"\n',
        ),
        (
            "# janus\nJANUS_INFO = ModelInfo(",
            '''# HY-MT1.5 / HunYuanDenseV1\nHUNYUAN_DENSE_INFO = ModelInfo(\n    ModelConfig(),\n    weights={\n        LlmList.LAYERS: "model.layers",\n        LlmList.EMBEDING: "model.embed_tokens",\n        LlmList.INPUT_LN: "input_layernorm",\n        LlmList.Q_PROJ: "self_attn.q_proj",\n        LlmList.Q_NORM: "self_attn.query_layernorm",\n        LlmList.K_PROJ: "self_attn.k_proj",\n        LlmList.K_NORM: "self_attn.key_layernorm",\n        LlmList.V_PROJ: "self_attn.v_proj",\n        LlmList.O_PROJ: "self_attn.o_proj",\n        LlmList.POST_ATTN_LN: "post_attention_layernorm",\n        LlmList.MLP_GATE: "mlp.gate_proj",\n        LlmList.MLP_UP: "mlp.up_proj",\n        LlmList.MLP_DOWN: "mlp.down_proj",\n        LlmList.NORM: "model.norm",\n        LlmList.LMHEAD: "lm_head",\n    })\n\n# janus\nJANUS_INFO = ModelInfo(''',
        ),
    ])

    converter = ROOT / "llm" / "LlmConverter.py"
    patch_file(converter, [
        (
            "        self.model_info = COMMON_INFO\n",
            "        self.model_info = (HUNYUAN_DENSE_INFO if self.config.model_type == LlmType.HUNYUAN_V1_DENSE else COMMON_INFO)\n",
        ),
        (
            "        from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding\n        rotary_embed = LlamaRotaryEmbedding(config=self.llm_config)\n",
            "        if self.llm_type == LlmType.HUNYUAN_V1_DENSE:\n            from transformers.models.hunyuan_v1_dense.modeling_hunyuan_v1_dense import HunYuanDenseV1RotaryEmbedding\n            rotary_embed = HunYuanDenseV1RotaryEmbedding(config=self.llm_config)\n        else:\n            from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding\n            rotary_embed = LlamaRotaryEmbedding(config=self.llm_config)\n",
        ),
        (
            "            # rotary cos/sin\n            q_op, k_op = self.apply_rotary_pos(block_mlir, in1_op, q_op, k_op, rotary_cos,\n                                               rotary_sin)\n",
            "            # rotary cos/sin\n            q_op, k_op = self.apply_rotary_pos(block_mlir, in1_op, q_op, k_op, rotary_cos,\n                                               rotary_sin)\n            # Hunyuan applies learned per-head QK RMSNorm after RoPE. This order is\n            # not interchangeable with Qwen3 because the norm has learned weights.\n            if self.llm_type == LlmType.HUNYUAN_V1_DENSE:\n                q_op = self.rms_norm(block_mlir, q_op, q_norm)\n                k_op = self.rms_norm(block_mlir, k_op, k_norm)\n",
        ),
        (
            "        if self.llm_type in [LlmType.QWEN3, LlmType.GEMMA3]:\n            self.set_common_weight(q_norm, weight_dict, self.rmsnorm_type)\n",
            "        if self.llm_type in [LlmType.QWEN3, LlmType.GEMMA3, LlmType.HUNYUAN_V1_DENSE]:\n            self.set_common_weight(q_norm, weight_dict, self.rmsnorm_type)\n",
        ),
    ])

    entry = ROOT / "tools" / "llm_convert.py"
    patch_file(entry, [
        (
            'if config.model_type in ["qwen3", "qwen2", "llama", "minicpm"]:',
            'if config.model_type in ["qwen3", "qwen2", "llama", "minicpm", "hunyuan_v1_dense"]:',
        ),
    ])


if __name__ == "__main__":
    main()
