#!/usr/bin/env python3
"""
Utility functions for SenseVoice model loading and preparation
"""

import torch
import numpy as np
import os
import sys


def load_cmvn(cmvn_file):
    """
    Load CMVN (Cepstral Mean and Variance Normalization) parameters

    Args:
        cmvn_file: Path to am.mvn file

    Returns:
        neg_mean: Negative mean tensor [560]
        inv_stddev: Inverse standard deviation tensor [560]
    """
    print(f"Loading CMVN from: {cmvn_file}")

    means = []
    variance = []

    with open(cmvn_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Parse Kaldi nnet format
    # Format:
    # <AddShift> 560 560
    # <LearnRateCoef> 0 [ value1 value2 ... ]
    # <Rescale> 560 560
    # <LearnRateCoef> 0 [ value1 value2 ... ]

    # Find AddShift section (means)
    addshift_idx = content.find("<AddShift>")
    if addshift_idx >= 0:
        # Find the bracket after <LearnRateCoef> 0
        bracket_start = content.find("[", addshift_idx)
        bracket_end = content.find("]", bracket_start)
        if bracket_start >= 0 and bracket_end >= 0:
            values_str = content[bracket_start+1:bracket_end].strip()
            means = [float(v) for v in values_str.split()]

    # Find Rescale section (variance/stddev)
    rescale_idx = content.find("<Rescale>")
    if rescale_idx >= 0:
        # Find the bracket after <LearnRateCoef> 0
        bracket_start = content.find("[", rescale_idx)
        bracket_end = content.find("]", bracket_start)
        if bracket_start >= 0 and bracket_end >= 0:
            values_str = content[bracket_start+1:bracket_end].strip()
            variance = [float(v) for v in values_str.split()]

    if not means or not variance:
        raise ValueError(f"Failed to parse CMVN file: {cmvn_file}")

    means = np.array(means, dtype=np.float32)
    variance = np.array(variance, dtype=np.float32)

    # In Kaldi format:
    # AddShift contains the negative mean values
    # Rescale contains the inverse standard deviation values
    # So we use them directly
    neg_mean = means  # Already negative mean in Kaldi format
    inv_stddev = variance  # Already inverse stddev in Kaldi format

    print(f"CMVN loaded: neg_mean shape={neg_mean.shape}, inv_stddev shape={inv_stddev.shape}")

    return torch.from_numpy(neg_mean), torch.from_numpy(inv_stddev)


def load_pretrained_weights(model, model_dir):
    """
    Load pretrained weights from FunASR model directory

    Args:
        model: SenseVoiceSmall model instance
        model_dir: Path to sensevoice-small directory

    Returns:
        model: Model with loaded weights
    """
    model_pt = os.path.join(model_dir, "model.pt")

    if not os.path.exists(model_pt):
        raise FileNotFoundError(f"Model file not found: {model_pt}")

    print(f"Loading pretrained weights from: {model_pt}")

    # Load state dict
    state_dict = torch.load(model_pt, map_location='cpu')

    # FunASR model.pt might be wrapped in a dict
    if isinstance(state_dict, dict):
        if "model" in state_dict:
            state_dict = state_dict["model"]
        elif "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

    # Try to load the state dict
    # Note: Some keys might not match exactly, we'll handle missing keys
    try:
        # Check if embed.weight exists in state_dict (old Embedding layer)
        if "embed.weight" in state_dict:
            print("Converting Embedding weights to 4 separate prompt vectors...")
            # Original: embed.weight [vocab_size, embed_dim]
            # We need to extract specific rows for the 4 prompt positions
            embedding_weight = state_dict.pop("embed.weight")  # [16, 560]

            # Default prompt indices (used during training/tracing)
            # These should match the create_prompt() defaults
            default_language_idx = 0   # auto language (自动检测语言)
            default_event_idx = 1      # HAPPY event
            default_event_type_idx = 2 # Speech type
            default_text_norm_idx = 14 # withitn (启用标点符号)

            # Extract the 4 vectors from the embedding table
            language_prompt = embedding_weight[default_language_idx:default_language_idx+1, :]  # [1, 560]
            event_prompt = embedding_weight[default_event_idx:default_event_idx+1, :]        # [1, 560]
            event_type_prompt = embedding_weight[default_event_type_idx:default_event_type_idx+1, :]  # [1, 560]
            text_norm_prompt = embedding_weight[default_text_norm_idx:default_text_norm_idx+1, :]  # [1, 560]

            # Load into the new parameters
            state_dict["language_prompt"] = language_prompt
            state_dict["event_prompt"] = event_prompt
            state_dict["event_type_prompt"] = event_type_prompt
            state_dict["text_norm_prompt"] = text_norm_prompt

            print(f"  Original embedding: {embedding_weight.shape}")
            print(f"  Extracted 4 prompt vectors: {language_prompt.shape}, {event_prompt.shape}, {event_type_prompt.shape}, {text_norm_prompt.shape}")
            print(f"  Using prompt indices: language={default_language_idx}, event={default_event_idx}, type={default_event_type_idx}, norm={default_text_norm_idx}")

        incompatible = model.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys:
            print(f"Warning: Missing keys: {len(incompatible.missing_keys)}")
            if len(incompatible.missing_keys) <= 10:
                for key in incompatible.missing_keys:
                    print(f"  - {key}")
        if incompatible.unexpected_keys:
            print(f"Warning: Unexpected keys: {len(incompatible.unexpected_keys)}")
            if len(incompatible.unexpected_keys) <= 10:
                for key in incompatible.unexpected_keys:
                    print(f"  - {key}")
    except Exception as e:
        print(f"Error loading state dict: {e}")
        print("Attempting to match keys manually...")

        # Try to match keys manually
        model_state = model.state_dict()
        matched = 0
        for k, v in state_dict.items():
            if k in model_state:
                if model_state[k].shape == v.shape:
                    model_state[k] = v
                    matched += 1
                else:
                    print(f"Shape mismatch for {k}: model={model_state[k].shape}, ckpt={v.shape}")

        model.load_state_dict(model_state)
        print(f"Manually matched {matched} keys")

    print("✅ Pretrained weights loaded successfully")
    return model


def create_sensevoice_model(model_dir):
    """
    Create SenseVoice model with pretrained weights and CMVN

    Args:
        model_dir: Path to sensevoice-small directory

    Returns:
        model: Complete SenseVoiceSmall model ready for inference
    """
    from torch_model import SenseVoiceSmall

    # Load CMVN parameters
    cmvn_file = os.path.join(model_dir, "am.mvn")
    neg_mean, inv_stddev = load_cmvn(cmvn_file)

    # Create model
    print("Creating SenseVoiceSmall model...")
    model = SenseVoiceSmall(neg_mean, inv_stddev)

    # Load pretrained weights
    model = load_pretrained_weights(model, model_dir)

    # Set to eval mode
    model.eval()

    return model


def save_torchscript(model, save_path, example_inputs):
    """
    Save model as TorchScript

    Args:
        model: PyTorch model
        save_path: Output .pt file path
        example_inputs: Tuple of example inputs for tracing
    """
    print(f"Tracing model with example inputs...")
    model = model.cpu()
    model.eval()

    with torch.no_grad():
        traced = torch.jit.trace(model, example_inputs, strict=False)

    print(f"Saving TorchScript to: {save_path}")
    torch.jit.save(traced, save_path)

    print("✅ TorchScript saved successfully")
