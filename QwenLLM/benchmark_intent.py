#!/usr/bin/env python3
"""
意图识别 Benchmark 脚本
在板子上运行：python3 benchmark_intent.py -m <bmodel> -c <config_dir>

测量指标：
  - FTL  首字延迟 (s)
  - TPS  decode 速度 (token/s)
  - Prefill速度 (token/s)
  - 输出 token 数
  - 每轮耗时

用法示例（板子上）：
  python3 benchmark_intent.py \
      -m qwen2.5/qwen2.5-1.5b_int4_seq2048_1dev.bmodel \
      -c qwen2.5/config \
      --model_name qwen2.5-1.5b
"""

import argparse
import time
import json
import sys
import os
import builtins

# 把脚本所在目录和调用目录都加入 sys.path，确保能找到 chat.cpython*.so
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 意图识别专用测试用例（模拟真实语音助手场景）
INTENT_TEST_CASES = [
    "帮我打开白板",
    "我要写字，用马克笔，笔迹大小调到12",
    "关闭当前窗口",
    "把音量调大一点",
    "打开摄像头",
    "我想画一个圆形",
    "切换到橡皮擦模式",
    "保存当前文件",
    "帮我截图",
    "退出程序",
]

SYSTEM_PROMPT = (
    "把用户指令归类为以下动作之一：open_whiteboard, close_window, set_volume, "
    "open_camera, set_pen, draw_shape, set_tool, save_file, screenshot。"
    '输出JSON：{"action":"动作名","params":{}}。'
)


def run_benchmark(args):
    from transformers import AutoTokenizer, GenerationConfig

    # 强制行刷新，SSH 管道下不丢输出
    import functools
    print = functools.partial(builtins.print, flush=True)

    print(f"\n{'='*60}")
    print(f"模型: {args.model_name}")
    print(f"bmodel: {args.model_path}")
    print(f"config: {args.config_path}")
    print(f"{'='*60}")

    # 加载 tokenizer
    print("加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.config_path, trust_remote_code=True)
    tokenizer.decode([0])  # warm up

    # 加载模型
    print("加载模型...")
    if args.pipeline:
        # 纯 Python sail pipeline（用于 Qwen3.5 等无 chat.so 的模型）
        import importlib.util, pathlib
        spec = importlib.util.spec_from_file_location("_pipeline", args.pipeline)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        model = mod.Qwen()
    else:
        import chat
        model = chat.Qwen() if hasattr(chat, 'Qwen') else chat.Qwen3() if hasattr(chat, 'Qwen3') else None
        if model is None:
            for cls_name in ['Qwen', 'Qwen3', 'Qwen2', 'LLM']:
                if hasattr(chat, cls_name):
                    model = getattr(chat, cls_name)()
                    break
    if model is None:
        print("ERROR: 找不到模型类，请检查 chat.cpython*.so 或 --pipeline 参数")
        sys.exit(1)

    devices = [int(d) for d in args.devid.split(",")]
    load_start = time.time()
    model.init(devices, args.model_path)
    load_time = time.time() - load_start
    print(f"模型加载耗时: {load_time:.2f}s")
    SEQLEN = model.SEQLEN
    # 兼容不同版本 chat 模块的属性名
    MAX_INPUT = getattr(model, 'MAX_INPUT_LENGTH', SEQLEN // 2)
    print(f"SEQLEN: {SEQLEN}, MAX_INPUT_LENGTH: {MAX_INPUT}")

    EOS = [tokenizer.eos_token_id]

    test_cases = INTENT_TEST_CASES[:3] if args.quick else INTENT_TEST_CASES
    results = []
    print(f"\n开始测试 {len(test_cases)} 条意图识别用例{'（快速模式）' if args.quick else ''}...\n")

    for i, asr_text in enumerate(test_cases):
        history = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": asr_text},
        ]
        try:
            text = tokenizer.apply_chat_template(
                history, tokenize=False, add_generation_prompt=True,
                enable_thinking=not args.no_think,
            )
        except TypeError:
            text = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
        tokens = tokenizer(text).input_ids
        prefill_len = len(tokens)

        print(f"[{i+1:02d}] 输入: {asr_text!r}  (prefill tokens: {prefill_len})")

        # 推理
        tok_num = 0
        answer = ""
        full_word_tokens = []

        first_start = time.time()
        token = model.forward_first(tokens)
        first_end = time.time()
        ftl = first_end - first_start
        prefill_tps = prefill_len / ftl

        # history_length / token_length 兼容不同版本
        def cur_len():
            if hasattr(model, 'history_length'):
                return model.history_length
            if hasattr(model, 'token_length'):
                return model.token_length
            return tok_num

        while token not in EOS and cur_len() < SEQLEN:
            full_word_tokens.append(token)
            word = tokenizer.decode(full_word_tokens, skip_special_tokens=True)
            if "â" in word or "�" in word:
                token = model.forward_next()
                tok_num += 1
                continue
            answer += word
            full_word_tokens = []
            tok_num += 1
            token = model.forward_next()
            # 意图识别输出一般很短，JSON 闭合后可提前退出
            if answer.strip().endswith("]") and tok_num > 3:
                break

        decode_time = time.time() - first_end
        tps = tok_num / decode_time if decode_time > 0 else 0
        total_time = time.time() - first_start

        # 清除历史
        if hasattr(model, 'clear_kv'):
            model.clear_kv()

        result = {
            "input": asr_text,
            "prefill_tokens": prefill_len,
            "output_tokens": tok_num,
            "ftl_s": round(ftl, 3),
            "prefill_tps": round(prefill_tps, 1),
            "decode_tps": round(tps, 1),
            "total_s": round(total_time, 3),
            "output": answer.strip(),
        }
        results.append(result)

        print(f"     FTL={ftl:.3f}s  Prefill={prefill_tps:.0f}tok/s  TPS={tps:.1f}tok/s  输出={tok_num}tok")
        print(f"     结果: {answer.strip()[:120]}")

    model.deinit()

    # 汇总统计
    avg_ftl      = sum(r["ftl_s"]       for r in results) / len(results)
    avg_pre_tps  = sum(r["prefill_tps"] for r in results) / len(results)
    avg_dec_tps  = sum(r["decode_tps"]  for r in results) / len(results)
    avg_total    = sum(r["total_s"]      for r in results) / len(results)

    print(f"\n{'='*60}")
    print(f"  模型: {args.model_name}")
    print(f"  平均 FTL          : {avg_ftl:.3f} s")
    print(f"  平均 Prefill 速度  : {avg_pre_tps:.1f} token/s")
    print(f"  平均 Decode TPS   : {avg_dec_tps:.1f} token/s")
    print(f"  平均端到端延迟     : {avg_total:.3f} s")
    print(f"  模型加载耗时       : {load_time:.2f} s")
    print(f"{'='*60}\n")

    # 保存结果
    out_file = f"benchmark_{args.model_name}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "model": args.model_name,
            "bmodel": args.model_path,
            "summary": {
                "avg_ftl_s":        round(avg_ftl, 3),
                "avg_prefill_tps":  round(avg_pre_tps, 1),
                "avg_decode_tps":   round(avg_dec_tps, 1),
                "avg_total_s":      round(avg_total, 3),
                "load_time_s":      round(load_time, 2),
            },
            "details": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"结果已保存到: {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="意图识别 Benchmark")
    parser.add_argument("-m", "--model_path",  required=True, help="bmodel 文件路径")
    parser.add_argument("-c", "--config_path", required=True, help="config 目录路径（含 tokenizer）")
    parser.add_argument("-n", "--model_name",  default="model",  help="模型名称（用于结果文件命名）")
    parser.add_argument("-d", "--devid",       default="0",      help="device ID")
    parser.add_argument("--quick",    action='store_true', help="只测前3条用例，快速验证流程")
    parser.add_argument("--no_think", action='store_true', help="禁用 Qwen3 thinking 模式（enable_thinking=False）")
    parser.add_argument("--pipeline", default=None, help="纯 Python pipeline 文件路径（用于无 chat.so 的模型，如 Qwen3.5）")
    args = parser.parse_args()
    run_benchmark(args)
