"""
Qwen3.5-0.8B 文本推理 Pipeline（纯 Python + sophon.sail）

架构：GatedDeltaNet hybrid，24 层
  - 线性注意力层 (18/24): block 0,1,2, 4,5,6, 8,9,10, 12,13,14, 16,17,18, 20,21,22
  - 全注意力层   (6/24):  block 3,7,11,15,19,23

bmodel tensor 名称（由 model_tool --info 确认）：
  embedding     : input=input_ids[1,2048] int32, output=embedding[1,2048,1024] bf16
  embedding_cache: input=input_ids[1,1] int32, output=embedding_cache[1,1,1024] bf16
  lm_head       : input=hidden_states[1,1024] bf16, output=token_id[1,1] int32（已含 argmax）

  block_cache_{i} (线性层):
    input : input_states[1,1,1024], conv_state[1,6144,4], recurrent_state[1,16,128,128] — 全 bf16
    output: output_states[1,1,1024]
            model.language_model.layers.{i}.linear_attn.conv1d.slice  [1,6144,4]
            model.language_model.layers.{i}.linear_attn.in_proj_a.recurrent_update [1,16,128,128]

  block_cache_{i} (全注意力层):
    input : input_states[1,1,1024], position_ids[1,1] int32,
            attention_mask[1,1,1,2049] bf16, history_k[1,2048,2,256], history_v[1,2048,2,256]
    output: output_states[1,1,1024], k_cache[1,1,2,256], v_cache[1,1,2,256]

  block_{i} prefill (线性层, dynamic):
    input : input_states[max:1,max:2048,max:1024], recurrent_states[max:1,max:16,max:128,max:128]
    output: output_states[1,2048,1024]
            model.language_model.layers.{i}.linear_attn.conv1d.conv_state_slice [1,6144,4]
            model.language_model.layers.{i}.chunk_gated_delta_rule [1,16,128,128]

  block_{i} prefill (全注意力层, dynamic):
    input : input_states, position_ids[max:1,max:2048], attention_mask[max:1,max:1,max:2048,max:2048]
    output: output_states[1,2048,1024], k_cache[1,2048,2,256], v_cache[1,2048,2,256]
"""

import numpy as np
import sophon.sail as sail

FULL_ATTN_LAYERS = {3, 7, 11, 15, 19, 23}
NUM_LAYERS = 24
SEQLEN = 2048
KV_HEADS = 2
HEAD_DIM = 256
HIDDEN = 1024
BF16 = np.float32   # sail SYSIO 返回 bf16 映射为 float32 numpy array

CONV_STATE_SHAPE  = (1, 6144, 4)
RECUR_STATE_SHAPE = (1, 16, 128, 128)
KV_CACHE_SHAPE    = (1, SEQLEN, KV_HEADS, HEAD_DIM)


def _conv_out_name(i):
    return f"model.language_model.layers.{i}.linear_attn.conv1d.slice"

def _recur_out_name(i):
    return f"model.language_model.layers.{i}.linear_attn.in_proj_a.recurrent_update"

def _prefill_conv_out_name(i):
    return f"model.language_model.layers.{i}.linear_attn.conv1d.conv_state_slice"

def _prefill_recur_out_name(i):
    return f"model.language_model.layers.{i}.chunk_gated_delta_rule"


class Qwen35Pipeline:
    def __init__(self, bmodel_path: str, dev_id: int = 0):
        self.SEQLEN = SEQLEN
        self.MAX_INPUT_LENGTH = SEQLEN - 128

        self._engine = sail.Engine(bmodel_path, dev_id, sail.IOMode.SYSIO)
        self._graph = {n: n for n in self._engine.get_graph_names()}
        self._alloc_states()

        self.history_length = 0
        self._last_token = 0

    # ------------------------------------------------------------------

    def _alloc_states(self):
        self._conv_states  = {}
        self._recur_states = {}
        self._k_caches     = {}
        self._v_caches     = {}
        for i in range(NUM_LAYERS):
            if i in FULL_ATTN_LAYERS:
                self._k_caches[i] = np.zeros(KV_CACHE_SHAPE, dtype=np.float32)
                self._v_caches[i] = np.zeros(KV_CACHE_SHAPE, dtype=np.float32)
            else:
                self._conv_states[i]  = np.zeros(CONV_STATE_SHAPE,  dtype=np.float32)
                self._recur_states[i] = np.zeros(RECUR_STATE_SHAPE, dtype=np.float32)

    def _reset_states(self):
        for arr in list(self._conv_states.values()) + list(self._recur_states.values()) \
                 + list(self._k_caches.values()) + list(self._v_caches.values()):
            arr[:] = 0

    def _run(self, graph: str, feeds: dict) -> dict:
        return self._engine.process(graph, feeds)

    # ------------------------------------------------------------------

    def forward_first(self, tokens: list) -> int:
        self._reset_states()
        seq = len(tokens)

        # 1. Embedding (static, pad to SEQLEN)
        input_ids = np.zeros((1, SEQLEN), dtype=np.int32)
        input_ids[0, :seq] = tokens
        out = self._run("embedding", {"input_ids": input_ids})
        hidden = out["embedding"][:, :seq, :]  # [1, seq, 1024] bf16

        # 2. 每层 prefill
        pos_ids   = np.arange(seq, dtype=np.int32).reshape(1, seq)
        # attention_mask shape: [1,1,seq,seq]（max:1,max:1,max:2048,max:2048）
        causal = np.triu(np.full((seq, seq), -10000.0, dtype=np.float32), k=1)
        attn_mask = causal.reshape(1, 1, seq, seq)

        for i in range(NUM_LAYERS):
            if i in FULL_ATTN_LAYERS:
                feeds = {
                    "input_states":   hidden,
                    "position_ids":   pos_ids,
                    "attention_mask": attn_mask,
                }
                out = self._run(f"block_{i}", feeds)
                hidden = out["output_states"][:, :seq, :]
                self._k_caches[i][:, :seq, :, :] = out["k_cache"]
                self._v_caches[i][:, :seq, :, :] = out["v_cache"]
            else:
                feeds = {
                    "input_states":    hidden,
                    "recurrent_states": self._recur_states[i],
                }
                out = self._run(f"block_{i}", feeds)
                hidden = out["output_states"][:, :seq, :]
                self._conv_states[i][:]  = out[_prefill_conv_out_name(i)]
                self._recur_states[i][:] = out[_prefill_recur_out_name(i)]

        # 3. lm_head — 直接输出 token_id int32
        last_hidden = hidden[:, -1, :].reshape(1, HIDDEN)
        out = self._run("lm_head", {"hidden_states": last_hidden})
        token = int(out["token_id"].flat[0])

        self.history_length = seq + 1
        self._last_token = token
        return token

    def forward_next(self) -> int:
        token_id = self._last_token
        pos = self.history_length - 1

        # 1. Embedding cache (single token)
        input_ids = np.array([[token_id]], dtype=np.int32)
        out = self._run("embedding_cache", {"input_ids": input_ids})
        hidden = out["embedding_cache"]  # [1, 1, 1024]

        # 2. 每层 decode
        pos_id    = np.array([[pos]], dtype=np.int32)
        # attention_mask: [1,1,1,2049]，有效位置填 0，其余 -inf
        attn_mask = np.full((1, 1, 1, SEQLEN + 1), -10000.0, dtype=np.float32)
        attn_mask[0, 0, 0, :pos + 1] = 0.0

        for i in range(NUM_LAYERS):
            if i in FULL_ATTN_LAYERS:
                feeds = {
                    "input_states":   hidden,
                    "position_ids":   pos_id,
                    "attention_mask": attn_mask,
                    "history_k":      self._k_caches[i],
                    "history_v":      self._v_caches[i],
                }
                out = self._run(f"block_cache_{i}", feeds)
                hidden = out["output_states"]
                self._k_caches[i][:, pos:pos+1, :, :] = out["k_cache"]
                self._v_caches[i][:, pos:pos+1, :, :] = out["v_cache"]
            else:
                feeds = {
                    "input_states":    hidden,
                    "conv_state":      self._conv_states[i],
                    "recurrent_state": self._recur_states[i],
                }
                out = self._run(f"block_cache_{i}", feeds)
                hidden = out["output_states"]
                self._conv_states[i][:]  = out[_conv_out_name(i)]
                self._recur_states[i][:] = out[_recur_out_name(i)]

        # 3. lm_head
        last_hidden = hidden.reshape(1, HIDDEN)
        out = self._run("lm_head", {"hidden_states": last_hidden})
        token = int(out["token_id"].flat[0])

        self.history_length += 1
        self._last_token = token
        return token

    def deinit(self):
        pass

    def init(self, devices: list, bmodel_path: str):
        pass


# ---------------------------------------------------------------------------
# 兼容 benchmark_intent.py 的惰性封装（模拟 chat.Qwen()）
# ---------------------------------------------------------------------------

class _LazyQwen35:
    def __init__(self):
        self._pipeline = None
        self.SEQLEN = SEQLEN
        self.MAX_INPUT_LENGTH = SEQLEN - 128
        self.history_length = 0

    def init(self, devices: list, bmodel_path: str):
        dev_id = devices[0] if devices else 0
        self._pipeline = Qwen35Pipeline(bmodel_path, dev_id=dev_id)
        self.SEQLEN           = self._pipeline.SEQLEN
        self.MAX_INPUT_LENGTH = self._pipeline.MAX_INPUT_LENGTH

    def forward_first(self, tokens):
        tok = self._pipeline.forward_first(tokens)
        self.history_length = self._pipeline.history_length
        return tok

    def forward_next(self):
        tok = self._pipeline.forward_next()
        self.history_length = self._pipeline.history_length
        return tok

    def deinit(self):
        if self._pipeline:
            self._pipeline.deinit()

    def clear_kv(self):
        if self._pipeline:
            self._pipeline._reset_states()
            self._pipeline.history_length = 0
            self.history_length = 0


Qwen = _LazyQwen35


# ---------------------------------------------------------------------------
# 独立运行入口（板子上直接测试单条）
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, time
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path",  required=True)
    parser.add_argument("-c", "--config_path", required=True)
    parser.add_argument("-d", "--devid", type=int, default=0)
    parser.add_argument("-p", "--prompt", default="帮我打开白板")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.config_path, trust_remote_code=True)

    SYSTEM_PROMPT = (
        "把用户指令归类为以下动作之一：open_whiteboard, close_window, set_volume, "
        "open_camera, set_pen, draw_shape, set_tool, save_file, screenshot。"
        '输出JSON：{"action":"动作名","params":{}}。'
    )
    history = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": args.prompt},
    ]
    try:
        text = tokenizer.apply_chat_template(
            history, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        text = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
    tokens = tokenizer(text).input_ids
    print(f"Prompt tokens: {len(tokens)}")

    pipeline = Qwen35Pipeline(args.model_path, dev_id=args.devid)

    t0 = time.time()
    token = pipeline.forward_first(tokens)
    t1 = time.time()
    print(f"FTL: {t1-t0:.3f}s")

    EOS = [tokenizer.eos_token_id]
    output = ""
    tok_num = 0
    full_word_tokens = []
    while token not in EOS and pipeline.history_length < SEQLEN:
        full_word_tokens.append(token)
        word = tokenizer.decode(full_word_tokens, skip_special_tokens=True)
        if "â" in word or "�" in word:
            token = pipeline.forward_next()
            tok_num += 1
            continue
        output += word
        full_word_tokens = []
        tok_num += 1
        token = pipeline.forward_next()
        if output.strip().endswith("}") and tok_num > 3:
            break

    t2 = time.time()
    decode_time = t2 - t1
    tps = tok_num / decode_time if decode_time > 0 else 0
    print(f"Output: {output}")
    print(f"Tokens: {tok_num}  TPS: {tps:.1f}  E2E: {t2-t0:.3f}s")
