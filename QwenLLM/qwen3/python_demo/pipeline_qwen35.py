"""
Qwen3.5-0.8B 文本推理 Pipeline（纯 Python + sophon.sail）

架构：GatedDeltaNet hybrid，24 层
  - 线性注意力层 (18/24): block 0,1,2, 4,5,6, 8,9,10, 12,13,14, 16,17,18, 20,21,22
    状态: conv_state [1,6144,4] + recurrent_state [1,16,128,128]
  - 全注意力层 (6/24): block 3,7,11,15,19,23
    状态: KV cache history_k/v [1,2048,2,256]

bmodel net 索引（根据 model_tool --info 输出）：
  block_cache_0..23  → net 1..24   (静态 decode，单 token)
  embedding_cache    → net 25      (静态，单 token embedding)
  lm_head            → net 26      (静态，[1,1024] → logits)
  embedding          → net 0       (静态，[1,2048] → [1,2048,1024])
  block_0..23        → net 27..50  (动态 prefill)

接口与 benchmark_intent.py 兼容：
  forward_first(tokens) → int (第一个 token id)
  forward_next()        → int (下一个 token id)
  deinit()
  SEQLEN, MAX_INPUT_LENGTH, history_length
"""

import numpy as np
import sophon.sail as sail

# 哪些层是全注意力层（其余为线性注意力层）
FULL_ATTN_LAYERS = {3, 7, 11, 15, 19, 23}
NUM_LAYERS = 24
SEQLEN = 2048
HEAD_DIM = 256
KV_HEADS = 2
HIDDEN = 1024
CONV_STATE_SHAPE  = (1, 6144, 4)
RECUR_STATE_SHAPE = (1, 16, 128, 128)
KV_CACHE_SHAPE    = (1, SEQLEN, KV_HEADS, HEAD_DIM)


class Qwen35Pipeline:
    """
    Qwen3.5-0.8B 混合架构推理 Pipeline。
    对外接口与 benchmark_intent.py 期望的 chat.Qwen() 相同。
    """

    def __init__(self, bmodel_path: str, dev_id: int = 0):
        self.SEQLEN = SEQLEN
        self.MAX_INPUT_LENGTH = SEQLEN - 128  # 留给输出的空间

        self._dev_id = dev_id
        self._handle = sail.Handle(dev_id)
        self._engine = sail.Engine(bmodel_path, dev_id, sail.IOMode.SYSIO)

        self._net_names = self._engine.get_network_names()
        self._build_net_index()
        self._alloc_states()

        self.history_length = 0
        self._token_length = 0   # prefill 后填充的位置数

    # ------------------------------------------------------------------
    # 初始化辅助
    # ------------------------------------------------------------------

    def _build_net_index(self):
        """把 net 名称映射成有意义的 key，并为每个 net 缓存 graph_name。"""
        names = set(self._net_names)

        self._net_embedding        = "embedding"
        self._net_embedding_cache  = "embedding_cache"
        self._net_lm_head          = "lm_head"
        self._net_blocks           = [f"block_{i}"       for i in range(NUM_LAYERS)]
        self._net_block_caches     = [f"block_cache_{i}" for i in range(NUM_LAYERS)]

        all_nets = ([self._net_embedding, self._net_embedding_cache, self._net_lm_head]
                    + self._net_blocks + self._net_block_caches)
        for net in all_nets:
            if net not in names:
                raise RuntimeError(f"bmodel 缺少 net: {net}\n已有: {sorted(names)}")

        # sail.Engine 中 graph_name 即 net_name（一个 net = 一个 graph）
        graph_names = self._engine.get_graph_names()
        self._graph = {n: n for n in graph_names}
        # 验证所有需要的 net 都有对应 graph
        for net in all_nets:
            if net not in self._graph:
                raise RuntimeError(f"graph 缺少 net: {net}\n已有: {sorted(graph_names)}")

    def _alloc_states(self):
        """为每层预分配状态 numpy 数组（零初始化）。"""
        self._conv_states   = {}
        self._recur_states  = {}
        self._k_caches      = {}
        self._v_caches      = {}

        for i in range(NUM_LAYERS):
            if i in FULL_ATTN_LAYERS:
                self._k_caches[i] = np.zeros(KV_CACHE_SHAPE, dtype=np.float32)
                self._v_caches[i] = np.zeros(KV_CACHE_SHAPE, dtype=np.float32)
            else:
                self._conv_states[i]  = np.zeros(CONV_STATE_SHAPE,  dtype=np.float32)
                self._recur_states[i] = np.zeros(RECUR_STATE_SHAPE, dtype=np.float32)

    def _reset_states(self):
        for i in self._conv_states:
            self._conv_states[i][:] = 0
            self._recur_states[i][:] = 0
        for i in self._k_caches:
            self._k_caches[i][:] = 0
            self._v_caches[i][:] = 0

    # ------------------------------------------------------------------
    # 推理接口
    # ------------------------------------------------------------------

    def forward_first(self, tokens: list) -> int:
        """Prefill：处理完整输入 token 序列，返回第一个输出 token。"""
        self._reset_states()
        self.history_length = len(tokens)
        self._token_length  = len(tokens)

        # 1. Embedding (静态 net，pad 到 SEQLEN)
        input_ids = np.zeros((1, SEQLEN), dtype=np.int32)
        input_ids[0, :len(tokens)] = tokens
        emb_out = self._infer(self._net_embedding, {"input_ids": input_ids})
        # hidden: [1, SEQLEN, HIDDEN] → 取有效长度部分 [1, seq, HIDDEN]
        hidden = emb_out["embedding"][:, :len(tokens), :]  # [1, seq, HIDDEN]

        # 2. 每层 prefill（动态 block_i）
        seq = len(tokens)
        position_ids    = np.arange(seq, dtype=np.int32).reshape(1, seq)
        attention_mask  = np.zeros((1, 1, seq, seq), dtype=np.float32)
        # causal mask: 上三角为 -inf
        causal = np.triu(np.full((seq, seq), -10000.0, dtype=np.float32), k=1)
        attention_mask[0, 0] = causal

        for i in range(NUM_LAYERS):
            if i in FULL_ATTN_LAYERS:
                feeds = {
                    "input_states":  hidden,
                    "position_ids":  position_ids,
                    "attention_mask": attention_mask,
                }
                out = self._infer(self._net_blocks[i], feeds)
                hidden = out["output_states"]
                # 保存 KV cache（prefill 输出整段 kv）
                self._k_caches[i][:, :seq, :, :] = out["k_cache"]
                self._v_caches[i][:, :seq, :, :] = out["v_cache"]
            else:
                # 线性注意力层 prefill：传入零初始化 recurrent_state
                feeds = {
                    "input_states":   hidden,
                    "recurrent_states": self._recur_states[i],
                }
                out = self._infer(self._net_blocks[i], feeds)
                hidden = out["output_states"]
                self._conv_states[i][:]  = out["conv_state"]
                self._recur_states[i][:] = out["recurrent_state"]

        # 3. lm_head：取最后一个 token 的 hidden [1, HIDDEN]
        last_hidden = hidden[:, -1:, :].reshape(1, HIDDEN)  # [1, HIDDEN]
        lm_out = self._infer(self._net_lm_head, {"hidden_states": last_hidden})
        token_id = int(np.argmax(lm_out["logits"]))

        self.history_length = len(tokens) + 1
        self._last_decode_token = token_id
        return token_id

    def forward_next(self) -> int:
        """Decode：单步推理，返回下一个 token。"""
        # 当前 token 就是上一步输出的 token，但 pipeline 只返回 id
        # benchmark_intent.py 并不传 token 进来，我们需要自己记录
        # 用 _last_token 在 forward_first/forward_next 间传递
        token_id = self._last_decode_token
        pos = self.history_length - 1  # 当前 decode 位置

        # 1. Embedding cache（单 token）
        input_ids = np.array([[token_id]], dtype=np.int32)
        emb_out = self._infer(self._net_embedding_cache, {"input_ids": input_ids})
        hidden = emb_out["embedding"]  # [1, 1, HIDDEN]

        # 2. 每层 decode（静态 block_cache_i）
        position_id = np.array([[pos]], dtype=np.int32)
        # attention_mask for decode: [1, 1, 1, pos+1]，全 0（attend to all past）
        attn_mask = np.zeros((1, 1, 1, pos + 1), dtype=np.float32)
        # pad 到 SEQLEN+1（bmodel 期望固定形状 [1,1,1,SEQLEN+1]）
        attn_mask_padded = np.full((1, 1, 1, SEQLEN + 1), -10000.0, dtype=np.float32)
        attn_mask_padded[0, 0, 0, :pos + 1] = 0.0

        for i in range(NUM_LAYERS):
            if i in FULL_ATTN_LAYERS:
                feeds = {
                    "input_states":   hidden,
                    "position_ids":   position_id,
                    "attention_mask": attn_mask_padded,
                    "history_k":      self._k_caches[i],
                    "history_v":      self._v_caches[i],
                }
                out = self._infer(self._net_block_caches[i], feeds)
                hidden = out["output_states"]
                # 将新 kv 写入 cache 的 pos 位置
                self._k_caches[i][:, pos:pos+1, :, :] = out["k_cache"]
                self._v_caches[i][:, pos:pos+1, :, :] = out["v_cache"]
            else:
                feeds = {
                    "input_states":   hidden,
                    "conv_state":     self._conv_states[i],
                    "recurrent_state": self._recur_states[i],
                }
                out = self._infer(self._net_block_caches[i], feeds)
                hidden = out["output_states"]
                self._conv_states[i][:]  = out["conv_state_update"]
                self._recur_states[i][:] = out["recurrent_state_update"]

        # 3. lm_head
        last_hidden = hidden.reshape(1, HIDDEN)
        lm_out = self._infer(self._net_lm_head, {"hidden_states": last_hidden})
        next_token = int(np.argmax(lm_out["logits"]))

        self.history_length += 1
        self._last_decode_token = next_token
        return next_token

    def deinit(self):
        pass  # sail.Engine 无需显式释放

    # ------------------------------------------------------------------
    # 内部：sail 推理封装
    # ------------------------------------------------------------------

    def _infer(self, net_name: str, feeds: dict) -> dict:
        """运行一个 net，返回所有输出的 numpy dict。

        sail.Engine(SYSIO 模式) 的 process() 接受 {name: np.ndarray} 并返回同格式 dict。
        """
        graph = self._graph[net_name]
        out = self._engine.process(graph, feeds)
        return out

    # ------------------------------------------------------------------
    # 兼容 benchmark_intent.py 的 init() 接口
    # ------------------------------------------------------------------

    def init(self, devices: list, bmodel_path: str):
        """benchmark_intent.py 调用 model.init(devices, path)，此处为空（构造时已完成）。"""
        pass


# ---------------------------------------------------------------------------
# 让 benchmark_intent.py 可以直接 import 本文件并当作 chat 模块使用
# 用法：在 benchmark_intent.py 同目录下放一个 chat.py 软链或直接修改 import
# 这里提供一个顶层工厂，模拟 chat.Qwen()
# ---------------------------------------------------------------------------

class _LazyQwen35:
    """
    模拟 chat.Qwen() 的惰性初始化对象。
    benchmark_intent.py 先 model = chat.Qwen()，再 model.init(devices, path)。
    """
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


# 作为 chat 模块替代时暴露 Qwen 类
Qwen = _LazyQwen35


# ---------------------------------------------------------------------------
# 独立运行入口（板子上直接测试）
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
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

    import time
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
        if "â" in word or "?" in word:
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
    print(f"Output: {output}")
    print(f"TPS: {tok_num / decode_time:.1f} tok/s  E2E: {t2-t0:.3f}s")
