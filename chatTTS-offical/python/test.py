
import json
import torch

# 从 JSON 文件中读取数据
with open('slct_voice_240605.json', 'r', encoding='utf-8') as json_file:
    slct_idx_loaded = json.load(json_file)

# 将包含 Tensor 数据的部分转换回 Tensor 对象
for key in slct_idx_loaded:
    tensor_list = slct_idx_loaded[key]["tensor"]
    slct_idx_loaded[key]["tensor"] = torch.tensor(tensor_list)

# 将音色 tensor 打包进params_infer_code，固定使用此音色发音，调低temperature
speak_tensor = slct_idx_loaded["6"]["tensor"] # female


import ChatTTS
import torch
import torchaudio
import numpy as np
import time as time
import os
os.environ["OPENBLAS_NUM_THREADS"] = "16"
chat = ChatTTS.Chat()
chat.load(local_path='../models')
torch.set_num_threads(4)
inputs_en = """
大家好，我是一个文本转语音模型，[uv_break]专为对话场景设计。
[uv_break]我支持中英文混合输入，[uv_break]并提供多种音色选择，
[uv_break]可以精确控制语速、[uv_break]停顿和语调等韵律元素。
[uv_break]希望大家使用愉快。[uv_break]
""".replace('\n', '')

torch.manual_seed(1222)
start = time.time()
wavs = chat.infer(inputs_en,
                  skip_refine_text=True, use_decoder=True, 
                  params_refine_text = ChatTTS.Chat.RefineTextParams(
                      prompt='[oral_2][laugh_0][break_4]',
                      ),
                  params_infer_code = ChatTTS.Chat.InferCodeParams(
                      prompt="[speed_5]",
                      temperature=0.0001,
                      spk_emb=speak_tensor))
time_cost = time.time() - start 

sample_rate = 24000
print(wavs[0].shape)
wav_len = wavs[0].shape[0] / sample_rate 

print("Real-Time Factor(RTF): ", time_cost /wav_len )
torchaudio.save("test.wav", torch.unsqueeze(torch.from_numpy(wavs[0]), 0), sample_rate=sample_rate)