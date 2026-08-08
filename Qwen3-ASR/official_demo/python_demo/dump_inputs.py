import sys
import torch
import numpy as np
import qwen_asr
from transformers import AutoProcessor
import librosa

def get_rope_index(attention_mask):
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
    return position_ids

def dump_audio_inputs(wav_path, config_dir, language="English"):
    processor = AutoProcessor.from_pretrained(config_dir, fix_mistral_regex=True)
    tokenizer = processor.tokenizer

    ID_AUDIO_BOS = tokenizer.convert_tokens_to_ids(processor.audio_bos_token)

    messages = [
        {"role": "user", "content": [{"type": "audio", "audio": ""}]}
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    text = text + f"language {language}<asr_text>"

    y, _ = librosa.load(wav_path, sr=16000)
    y = y[:(len(y) // 16000) * 16000]
    inputs = processor(text=text, audio=[y], return_tensors="pt", padding=True)

    input_ids = inputs.input_ids.squeeze(0).numpy().astype(np.int32)
    position_ids = get_rope_index(inputs.attention_mask).squeeze(1).numpy().astype(np.int32)

    start_offset = int(torch.where(inputs.input_ids == ID_AUDIO_BOS)[1][0])
    audio_tokens_per_chunk = 13
    audio_dims = inputs.feature_attention_mask.sum().item()
    audio_features = inputs.input_features[:, :, :audio_dims].reshape(128, -1, 200).transpose(0, 1).numpy().astype(np.float32)
    n_times = audio_features.shape[0]
    audio_offset_list = np.array([i * audio_tokens_per_chunk + start_offset + 1 for i in range(n_times)], dtype=np.int32)

    input_ids.tofile("input_ids.bin")
    position_ids.tofile("position_ids.bin")
    audio_features.tofile("audio_features.bin")
    audio_offset_list.tofile("audio_offsets.bin")
    print(f"[Dump Successful] Dumped input_ids ({len(input_ids)}), position_ids shape {position_ids.shape}, audio_features shape {audio_features.shape}, audio_offsets ({len(audio_offset_list)}) for {wav_path} (Language: {language})")

if __name__ == "__main__":
    wav_path = sys.argv[1] if len(sys.argv) > 1 else "asr_en.wav"
    config_dir = sys.argv[2] if len(sys.argv) > 2 else "../config"
    language = sys.argv[3] if len(sys.argv) > 3 else "English"
    dump_audio_inputs(wav_path, config_dir, language)
