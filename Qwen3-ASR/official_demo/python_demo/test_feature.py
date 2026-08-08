import torch
import numpy as np
from transformers import AutoProcessor
import librosa

processor = AutoProcessor.from_pretrained("/mnt/data/zjw/source/llm/LLM-TPU/models/Qwen3_ASR/config", fix_mistral_regex=True)
y, _ = librosa.load("asr_zh.wav", sr=16000)
y = y[:(len(y) // 16000) * 16000]

inputs = processor(text="test", audio=[y], return_tensors="pt", padding=True)
features = inputs.input_features
print("HuggingFace Features Shape:", features.shape)
print("HuggingFace Features Mean:", features.mean().item())
print("HuggingFace Features Max:", features.max().item())
print("HuggingFace Features Min:", features.min().item())
