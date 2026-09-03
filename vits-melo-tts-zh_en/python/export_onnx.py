#!/usr/bin/env python3
"""Export the Chinese-English MeloTTS checkpoint to a dynamic ONNX model."""

import json
import os
from pathlib import Path
from typing import Any

import onnx
import torch
from melo.models import SynthesizerTrn
from melo.text import language_id_map

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "models" / "onnx" / "vits-melo-tts-zh_en"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
os.chdir(OUTPUT_DIR)


class HParams(dict):
    def __getattr__(self, name: str) -> Any:
        value = self[name]
        return HParams(value) if isinstance(value, dict) else value


class ModelWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model
        self.lang_id = language_id_map["ZH"]

    def forward(self, x, x_lengths, tones, sid, noise_scale, length_scale, noise_scale_w):
        bert = torch.zeros(x.shape[0], 1024, x.shape[1], dtype=torch.float32)
        ja_bert = torch.zeros(x.shape[0], 768, x.shape[1], dtype=torch.float32)
        language = torch.zeros_like(x)
        language[:, 1::2] = self.lang_id
        return self.model.infer(
            x=x,
            x_lengths=x_lengths,
            sid=sid,
            tone=tones,
            language=language,
            bert=bert,
            ja_bert=ja_bert,
            noise_scale=noise_scale,
            noise_scale_w=noise_scale_w,
            length_scale=length_scale,
        )[0]


def load_model(config_path: Path, checkpoint_path: Path) -> tuple[HParams, torch.nn.Module]:
    hps = HParams(json.loads(config_path.read_text(encoding="utf-8")))
    model = SynthesizerTrn(
        len(hps.symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        num_tones=hps.num_tones,
        num_languages=hps.num_languages,
        **hps.model,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return hps, model


def write_tokens(symbols: list[str]) -> None:
    (OUTPUT_DIR / "tokens.txt").write_text(
        "".join(f"{symbol} {index}\n" for index, symbol in enumerate(symbols)),
        encoding="utf-8",
    )


def add_metadata(path: Path, hps: HParams) -> None:
    model = onnx.load(path)
    model.metadata_props.clear()
    metadata = {
        "model_type": "melo-vits",
        "comment": "melo",
        "version": 2,
        "language": "Chinese + English",
        "add_blank": int(hps.data.add_blank),
        "n_speakers": 1,
        "jieba": 1,
        "sample_rate": hps.data.sampling_rate,
        "bert_dim": 1024,
        "ja_bert_dim": 768,
        "speaker_id": 1,
        "lang_id": language_id_map["ZH"],
        "tone_start": 0,
        "url": "https://github.com/myshell-ai/MeloTTS",
        "license": "MIT license",
        "description": "MeloTTS Chinese-English model",
    }
    for key, value in metadata.items():
        item = model.metadata_props.add()
        item.key = key
        item.value = str(value)
    onnx.save(model, path)


def main() -> None:
    checkpoint_value = os.environ.get("MELOTTS_CHECKPOINT")
    config_value = os.environ.get("MELOTTS_CONFIG")
    if not checkpoint_value or not config_value:
        raise SystemExit("请设置 MELOTTS_CHECKPOINT 和 MELOTTS_CONFIG")

    hps, model = load_model(Path(config_value), Path(checkpoint_value))
    write_tokens(hps.symbols)
    wrapper = ModelWrapper(model)
    x = torch.randint(low=0, high=10, size=(1, 256), dtype=torch.int64)
    x_lengths = torch.tensor([256], dtype=torch.int64)
    tones = torch.zeros_like(x)
    sid = torch.tensor([1], dtype=torch.int64)
    noise_scale = torch.tensor([1.0], dtype=torch.float32)
    length_scale = torch.tensor([1.0], dtype=torch.float32)
    noise_scale_w = torch.tensor([1.0], dtype=torch.float32)

    output = OUTPUT_DIR / "model.onnx"
    torch.onnx.export(
        wrapper,
        (x, x_lengths, tones, sid, noise_scale, length_scale, noise_scale_w),
        output,
        opset_version=18,
        input_names=[
            "x", "x_lengths", "tones", "sid", "noise_scale",
            "length_scale", "noise_scale_w",
        ],
        output_names=["y"],
        dynamic_axes={
            "x": {0: "N", 1: "L"},
            "x_lengths": {0: "N"},
            "tones": {0: "N", 1: "L"},
            "y": {0: "N", 1: "S", 2: "T"},
        },
    )
    add_metadata(output, hps)
    print(f"输出: {output}")


if __name__ == "__main__":
    main()
