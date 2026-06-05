#!/usr/bin/env python3
"""
生成 Whisper large-v3-turbo W4A8 量化所需的 calibration 数据。

数据来源：
  1. ChatTTS 合成（多音色，中英文混合）
  2. Mozilla CommonVoice（streaming 模式，少量真实录音）

输出：
  whisper/calib_data/wav/          WAV 文件（各自原始采样率）
  whisper/calib_data/encoder_npy/  Encoder 输入 .npy，形状 [1,128,3000]

运行方式（分两步，避免 ChatTTS 模型驻留内存时再跑 Mel 转换）：
  conda run -n sophon-chatTTS python whisper/python/gen_calib_data.py --step synth
  conda run -n sophon-chatTTS python whisper/python/gen_calib_data.py --step mel
  # 或一次跑完（内部用 subprocess 隔离）：
  conda run -n sophon-chatTTS python whisper/python/gen_calib_data.py
"""

import argparse
import os
import random
import subprocess
import sys
import numpy as np
import soundfile as sf
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CALIB_DIR = REPO_ROOT / "whisper" / "calib_data"
WAV_DIR   = CALIB_DIR / "wav"
NPY_DIR   = CALIB_DIR / "encoder_npy"
WAV_DIR.mkdir(parents=True, exist_ok=True)
NPY_DIR.mkdir(parents=True, exist_ok=True)

CHATTTS_WEIGHTS = Path.home() / ".cache/huggingface/hub/models--2Noise--ChatTTS/snapshots/1a3c04a8b0651689bd9242fbb55b1f4b5a9aef84"
SAMPLE_RATE_WHISPER = 16000
SAMPLE_RATE_CHATTTS = 24000
N_MELS    = 128
N_SAMPLES = 480000  # 30s @ 16kHz

ZH_TEXTS = [
    "今天天气怎么样？外面是不是很冷？",
    "请帮我查一下明天上午十点的航班信息。",
    "这款手机的电池容量是四千八百毫安时，支持六十五瓦快充。",
    "会议将于下午两点半在三楼会议室准时召开，请各部门负责人准时出席。",
    "根据最新数据显示，今年第三季度国内生产总值同比增长百分之五点二。",
    "小明每天早上七点起床，先做三十分钟运动，然后吃早饭去上班。",
    "这道菜需要放适量的盐、少许糖和一勺老抽，大火翻炒两分钟即可。",
    "他在图书馆找到了一本关于人工智能的书，花了整整三天时间读完了它。",
    "请在系统设置中找到无障碍选项，开启屏幕朗读功能后重启设备。",
    "列车将于十五分钟后进站，请旅客提前做好上车准备，注意携带好随身行李。",
]

EN_TEXTS = [
    "The weather forecast shows heavy rain this afternoon. Please bring an umbrella.",
    "Could you please check the flight status for tomorrow morning at ten o'clock?",
    "This smartphone features a four thousand eight hundred milliamp battery with sixty-five watt fast charging.",
    "The quarterly earnings report shows a revenue increase of twelve percent compared to last year.",
    "Please navigate to the settings menu and enable the accessibility features for screen reading.",
    "The train will arrive at the platform in fifteen minutes. Please have your tickets ready.",
    "To complete the installation, restart your computer and follow the on-screen instructions.",
    "Scientists have discovered a new species of deep-sea fish living at depths of over two thousand meters.",
    "The conference room is booked for the entire afternoon, so we will need to find an alternative meeting space.",
    "Turn left at the traffic light, continue for three hundred meters, and the destination will be on your right.",
]


# ---------------------------------------------------------------------------
# Step 1: 合成音频（ChatTTS + CommonVoice + 测试音频）
# 此步骤加载大量模型，运行完后进程退出，确保内存完全释放
# ---------------------------------------------------------------------------
def step_synth():
    import shutil

    # 内置测试音频
    test_dir = REPO_ROOT / "whisper" / "test_data"
    for f in test_dir.glob("*.wav"):
        dst = WAV_DIR / f"testdata_{f.name}"
        shutil.copy2(f, dst)
        print(f"  [test] {dst.name}")

    # ChatTTS 合成
    _synth_chattts(n_zh=10, n_en=10)

    # CommonVoice
    _synth_commonvoice(n_zh=15, n_en=15)

    wav_count = len(list(WAV_DIR.glob("*.wav")))
    print(f"\n[synth] 完成，WAV 总数: {wav_count}  →  {WAV_DIR}")


def _synth_chattts(n_zh, n_en):
    print(f"\n[ChatTTS] 合成 {n_zh} 条中文 + {n_en} 条英文...")
    try:
        import ChatTTS
        import torch
    except ImportError as e:
        print(f"[ChatTTS] 跳过: {e}")
        return

    chat = ChatTTS.Chat()
    if not chat.load(source="custom", custom_path=str(CHATTTS_WEIGHTS)):
        print("[ChatTTS] 权重加载失败，跳过")
        return

    texts = [(t, "zh") for t in ZH_TEXTS[:n_zh]] + [(t, "en") for t in EN_TEXTS[:n_en]]
    random.shuffle(texts)

    saved = 0
    for idx, (text, lang) in enumerate(texts):
        spk = chat.sample_random_speaker()
        try:
            wavs = chat.infer(
                text,
                skip_refine_text=True,
                use_decoder=True,
                params_infer_code=ChatTTS.Chat.InferCodeParams(
                    spk_emb=spk,
                    prompt="[speed_5]",
                    temperature=0.3,
                ),
            )
        except Exception as e:
            print(f"  [{idx}] 合成失败: {e}")
            continue

        if not wavs:
            continue

        audio_np = wavs[0]
        if isinstance(audio_np, torch.Tensor):
            audio_np = audio_np.numpy()
        wav_out = WAV_DIR / f"chattts_{lang}_{idx:03d}.wav"
        sf.write(str(wav_out), audio_np, SAMPLE_RATE_CHATTTS)
        saved += 1
        print(f"  [{idx:2d}] {lang}: {wav_out.name}  ({len(audio_np)/SAMPLE_RATE_CHATTTS:.1f}s)")

    # 显式释放模型，减少 peak RSS
    del chat
    print(f"[ChatTTS] 完成 {saved} 条，模型已释放")


def _synth_commonvoice(n_zh, n_en):
    print(f"\n[CommonVoice] 下载 {n_zh} 条 zh-CN + {n_en} 条 en...")
    try:
        from datasets import load_dataset
    except ImportError:
        print("[CommonVoice] datasets 未安装，跳过")
        return

    for lang_code, n in [("zh-CN", n_zh), ("en", n_en)]:
        try:
            ds = load_dataset(
                "mozilla-foundation/common_voice_13_0",
                lang_code,
                split="validation",
                streaming=True,
                trust_remote_code=True,
            )
            count = 0
            for sample in ds:
                if count >= n:
                    break
                audio = sample["audio"]
                arr = np.array(audio["array"], dtype=np.float32)
                sr  = audio["sampling_rate"]
                wav_out = WAV_DIR / f"cv_{lang_code.replace('-','_')}_{count:03d}.wav"
                # CommonVoice 一般已是 48kHz，直接存原始采样率；Mel 步骤用 whisper load_audio 做 resample
                sf.write(str(wav_out), arr, sr)
                count += 1
                print(f"  [{count:2d}] {lang_code}: {wav_out.name}  (sr={sr})")
        except Exception as e:
            print(f"  [{lang_code}] 失败: {e}")

    print(f"[CommonVoice] 完成")


# ---------------------------------------------------------------------------
# Step 2: WAV → Encoder .npy
# 用 whisper.audio.load_audio（内部 ffmpeg 流式 decode + resample），极省内存
# 此步骤不加载任何 torch 大模型
# ---------------------------------------------------------------------------
def step_mel():
    from whisper.audio import load_audio, log_mel_spectrogram, pad_or_trim

    wav_files = sorted(WAV_DIR.glob("*.wav"))
    print(f"\n[mel] 转换 {len(wav_files)} 条 WAV → encoder .npy (n_mels={N_MELS})")
    ok = 0
    for wav_path in wav_files:
        npy_out = NPY_DIR / (wav_path.stem + ".npy")
        if npy_out.exists():
            ok += 1
            continue
        try:
            # load_audio: ffmpeg decode → resample to 16kHz → float32 mono
            audio = load_audio(str(wav_path))           # ndarray float32 @ 16kHz
            audio = pad_or_trim(audio, N_SAMPLES)        # 固定 30s
            mel   = log_mel_spectrogram(audio, n_mels=N_MELS)   # [128, 3000]
            mel_np = mel.numpy()[np.newaxis, :, :].astype(np.float32)  # [1,128,3000]
            np.save(str(npy_out), mel_np)
            ok += 1
            print(f"  [ok] {wav_path.name}")
        except Exception as e:
            print(f"  [!] {wav_path.name}: {e}")

    print(f"[mel] 完成 {ok}/{len(wav_files)} 条 → {NPY_DIR}")


# ---------------------------------------------------------------------------
# Main：默认用 subprocess 把 synth 和 mel 分成两个进程，避免峰值叠加
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", choices=["synth", "mel", "all"], default="all")
    args = parser.parse_args()

    random.seed(42)

    if args.step == "synth":
        step_synth()
    elif args.step == "mel":
        step_mel()
    else:
        # all: 先 synth（子进程，退出后内存全释放），再 mel
        print("=== Step 1/2: 音频合成（独立子进程）===")
        ret = subprocess.run(
            [sys.executable, __file__, "--step", "synth"],
            check=False,
        )
        if ret.returncode != 0:
            print(f"[!] synth 子进程退出码 {ret.returncode}，检查上方日志")
            sys.exit(ret.returncode)

        print("\n=== Step 2/2: Mel 特征提取 ===")
        step_mel()

        npy_count = len(list(NPY_DIR.glob("*.npy")))
        print(f"\n完成！encoder calibration .npy: {npy_count} 条")
        print(f"  位置: {NPY_DIR}")
        print(f"\n下一步: 在 TPU-MLIR Docker 内运行 gen_calib_table_turbo.sh")
