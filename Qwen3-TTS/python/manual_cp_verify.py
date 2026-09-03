import torch
from transformers.cache_utils import DynamicCache
from qwen_tts import Qwen3TTSModel

import os

MODEL = os.environ.get(
    "QWEN3_TTS_MODEL",
    "请通过环境变量 QWEN3_TTS_MODEL 设置权重目录",
)
tts=Qwen3TTSModel.from_pretrained(MODEL,device_map="cpu",dtype=torch.float32,attn_implementation="eager",local_files_only=True)
m=tts.model
cp=m.talker.code_predictor
cp.config._attn_implementation="eager"
import numpy as np
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
hid=np.load(os.path.join(REPO, "python", "test", "outputs", "baseline", "greedy_hidden.npy"))[0].astype(np.float32)
past_hidden=torch.from_numpy(hid)
code0=1995
c0=m.talker.model.codec_embedding(torch.tensor([code0]))
pf=torch.cat([past_hidden.unsqueeze(0), c0], dim=0).unsqueeze(0)

# prefill as original: attention_mask 2D [1,2] ones, cache_position [0,1]
att=torch.ones(1,2,dtype=torch.long)
cache=DynamicCache()
out=cp(inputs_embeds=pf, attention_mask=att, past_key_values=cache, use_cache=True, cache_position=torch.tensor([0,1]), output_hidden_states=True)
# out.logits shape [1,15,2048]? generation_steps=0 -> forward logits all heads
print('out keys', out.keys())
print('logits shape', out.logits.shape)
# In prefill forward, logits = stacked for i 1..15 hidden[:,i]. We only need head0 on hidden[:,1]
print('prefill code1', int(out.logits[0,1].argmax()), '(expect 1642)')
seq=[code0, int(out.logits[0,1].argmax())]
# now decode steps g=1..14 using cp forward with input_ids and generation_steps
for g in range(1,15):
    x=torch.tensor([[seq[g]]], dtype=torch.long)
    cp_pos=g+1  # generation_steps g => position g+1? record g=1 pos2
    att=torch.ones(1, cp_pos+1, dtype=torch.long)
    out=cp(input_ids=x, attention_mask=att, past_key_values=cache, use_cache=True, cache_position=torch.tensor([cp_pos]), output_hidden_states=True, generation_steps=g)
    tok=int(out.logits[0,0].argmax())
    seq.append(tok)
print('codes16', seq)
print('expected', [1995,1642,519,22,793,1485,422,1902,1728,1446,743,1377,914,344,1772,1177])
