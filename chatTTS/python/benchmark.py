import json
import time
import os
import torch
import torchaudio
import numpy as np

os.environ["OPENBLAS_NUM_THREADS"] = "16"

import ChatTTS

# 50条测试样本：25条中文 + 25条英文
TEST_SAMPLES = [
    # ===== 中文 25条 =====
    "今天天气真不错，适合出去散步。",
    "人工智能正在改变我们的生活方式。",
    "请问附近有没有好吃的餐厅推荐？",
    "这个项目的截止日期是下周五。",
    "学习一门新技能需要持续的练习和耐心。",
    "深度学习在图像识别领域取得了突破性进展。",
    "今天开会讨论的主要议题是产品路线图。",
    "语音合成技术让机器能够像人类一样说话。",
    "我们需要在保证质量的前提下提高效率。",
    "这款芯片在边缘计算领域具有很强的竞争力。",
    "自然语言处理是人工智能的重要研究方向。",
    "请在下班前把报告发送到我的邮箱。",
    "算能科技专注于高性能人工智能芯片的研发。",
    "语音识别和语音合成是智能助手的核心能力。",
    "这次测试的目的是评估模型的推理性能。",
    "边缘计算可以有效降低数据传输的延迟。",
    "我们的产品已经在多个行业得到了广泛应用。",
    "模型量化可以在保证精度的同时减小模型体积。",
    "实时语音合成对延迟有非常严格的要求。",
    "硬件加速是提升推理速度的有效手段。",
    "中文文本转语音需要处理复杂的语调和韵律。",
    "这套系统已经通过了严格的性能测试和验证。",
    "智能语音助手可以帮助用户提高工作效率。",
    "我们正在研究如何进一步优化推理延迟。",
    "BM1684X是一款面向人工智能推理的高性能芯片。",
    # ===== 英文 25条 =====
    "Hello, how are you doing today?",
    "Artificial intelligence is transforming the world.",
    "The weather is great for a morning run.",
    "Please send the report before end of day.",
    "Deep learning has achieved remarkable results in vision tasks.",
    "This chip delivers exceptional performance for edge AI applications.",
    "Natural language processing enables machines to understand human speech.",
    "The inference speed of this model exceeds real-time requirements.",
    "We need to optimize the pipeline for lower latency.",
    "Speech synthesis technology has advanced significantly in recent years.",
    "The project deadline is set for next Friday afternoon.",
    "Edge computing reduces the need for cloud connectivity.",
    "Our system supports both Chinese and English voice synthesis.",
    "Model quantization reduces memory footprint without significant accuracy loss.",
    "Real-time text to speech is critical for interactive applications.",
    "The benchmark results show consistent performance across test cases.",
    "Hardware acceleration is key to achieving low latency inference.",
    "This solution is designed for deployment on embedded devices.",
    "We are evaluating multiple models to find the best trade-off.",
    "The neural network was trained on a large multilingual dataset.",
    "Please make sure all dependencies are installed before running the script.",
    "The average real-time factor reflects overall system performance.",
    "Sophon BM1684X provides powerful computing capabilities for AI workloads.",
    "Voice assistants rely on accurate and fast speech synthesis engines.",
    "This test covers a variety of sentence lengths and complexities.",
    # ===== 中文长文本 10条 =====
    "算能科技自主研发的BM1684X芯片，采用先进的工艺制程，集成了强大的张量处理单元，能够高效完成深度学习推理任务。该芯片支持多种精度格式，包括INT4、INT8、BF16和FP32，可以根据不同应用场景灵活调整，在保证精度的同时最大化吞吐量。",
    "语音合成技术经过多年发展，已经从早期的拼接式合成进化到如今基于深度学习的端到端模型。现代TTS系统能够生成自然流畅、富有情感的语音，支持多种语言和方言，在智能客服、有声读物、导航播报等众多场景中得到了广泛应用。",
    "边缘计算是一种将数据处理能力下沉到数据源附近的计算架构。通过在本地完成推理任务，边缘计算可以显著降低数据传输延迟，减少对云端服务器的依赖，同时也能更好地保护用户隐私数据。这对于实时性要求较高的应用场景尤为重要。",
    "大语言模型的兴起为人工智能带来了新的突破，这些模型通过在海量文本数据上进行预训练，获得了强大的语言理解和生成能力。然而，将大模型部署到资源受限的边缘设备上仍然面临巨大挑战，需要通过模型压缩、量化和知识蒸馏等技术来降低计算量和内存占用。",
    "在智能制造领域，人工智能技术正在发挥越来越重要的作用。通过对生产线上的传感器数据进行实时分析，AI系统能够预测设备故障、优化生产参数、提高产品良率。这种数据驱动的智能化改造正在推动传统制造业向工业4.0时代加速迈进。",
    "自然语言处理技术的核心任务包括文本分类、命名实体识别、情感分析、机器翻译和问答系统等。这些任务在金融、医疗、法律和教育等行业有着广泛的应用前景。随着预训练语言模型的快速发展，NLP系统的性能已经在多个基准测试中超越了人类水平。",
    "模型量化是一种常见的模型压缩技术，通过将浮点数权重转换为低比特整数来减小模型体积和计算量。INT4量化可以将模型大小压缩到原始FP32模型的八分之一，同时借助专用硬件加速，推理速度可以提升数倍。这使得在内存受限的嵌入式设备上部署大规模神经网络成为可能。",
    "实时语音交互系统对端到端延迟有严格要求，通常需要在300毫秒以内完成从语音识别到语音合成的全部处理流程。为了满足这一要求，系统需要在流式处理框架下工作，即在接收到部分输入时便开始处理，而不是等待完整输入后再进行计算。这需要ASR、NLU和TTS模块之间的紧密配合。",
    "BM1684X芯片内置了32TOPS的INT8算力和16TOPS的FP16算力，配备了高带宽的LPDDR4X内存和PCIE5.0接口。其独特的多核张量处理器架构能够高效执行卷积、矩阵乘法等深度学习算子，同时支持动态形状推理，为多样化的AI应用场景提供了灵活而强大的计算平台。",
    "语音情感识别是人机交互领域的重要研究课题，通过分析语音信号中的韵律特征、声学特征和语言特征，系统能够判断说话人的情绪状态。这项技术在心理健康评估、客服质量监控和人机协作等领域具有重要价值，但同时也面临跨语言、跨文化泛化能力不足等挑战。",
    # ===== 英文长文本 10条 =====
    "The Sophon BM1684X is a high-performance AI inference chip developed by SOPHGO, featuring a proprietary tensor processing unit architecture that delivers up to 32 TOPS of INT8 computing power. The chip is designed for edge AI deployment scenarios where low latency, low power consumption, and high throughput are simultaneously required, making it ideal for applications in smart manufacturing, autonomous driving, and intelligent surveillance.",
    "Text-to-speech synthesis has undergone a remarkable transformation over the past decade, evolving from rule-based concatenative systems to end-to-end neural network models. Modern TTS systems like ChatTTS leverage transformer architectures and vector quantization to generate highly natural and expressive speech. These models can capture subtle prosodic variations, emotional nuances, and speaking styles that were previously impossible to replicate with traditional methods.",
    "Edge computing represents a paradigm shift in how we process and analyze data, moving computation closer to the data source rather than relying on centralized cloud infrastructure. This architectural approach offers significant advantages in terms of latency reduction, bandwidth conservation, and privacy preservation. For real-time AI applications such as speech recognition and synthesis, edge deployment can reduce response times from hundreds of milliseconds to just tens of milliseconds.",
    "Model quantization is a critical technique for deploying large neural networks on resource-constrained devices. By representing model weights and activations with lower bit precision, such as INT4 or INT8, we can dramatically reduce memory footprint and accelerate inference without significantly degrading model accuracy. Modern quantization-aware training methods further close the accuracy gap between quantized and full-precision models, enabling practical deployment of billion-parameter models on embedded hardware.",
    "The development of multilingual speech synthesis systems presents unique challenges, as different languages have fundamentally different phonological structures, prosodic patterns, and writing systems. A robust multilingual TTS system must handle tonal languages like Chinese, agglutinative languages like Turkish, and complex script systems like Arabic, while maintaining consistent voice quality and naturalness across all supported languages. This requires careful design of the text frontend, acoustic model, and vocoder components.",
    "Real-time speech processing pipelines must carefully balance computational efficiency with output quality. In a typical streaming TTS system, the text encoder processes input tokens incrementally, the acoustic model generates mel-spectrograms in chunks, and the vocoder synthesizes audio frames that can be played back immediately without waiting for the entire utterance to be processed. This streaming architecture reduces the time-to-first-audio from several seconds to under 200 milliseconds, dramatically improving the perceived responsiveness of voice interfaces.",
    "The integration of large language models with speech synthesis creates a powerful foundation for next-generation conversational AI systems. By combining the contextual understanding capabilities of LLMs with the natural speech generation of neural TTS, these systems can engage in fluid, context-aware spoken conversations. However, the computational demands of running both components simultaneously pose significant challenges for real-time deployment, requiring careful optimization of model architectures and inference pipelines.",
    "Hardware-software co-design is increasingly important for efficient AI inference at the edge. By tailoring neural network architectures to the specific computational patterns of target hardware accelerators, developers can achieve substantial improvements in throughput and energy efficiency. This approach involves close collaboration between algorithm researchers, compiler engineers, and hardware designers to identify and eliminate computational bottlenecks throughout the full stack from model training to production deployment.",
    "Benchmarking AI inference systems requires careful attention to methodology to ensure results are both reproducible and representative of real-world performance. Key metrics include throughput measured in samples per second, latency for individual inference requests, and the real-time factor for streaming applications. Proper benchmarking must account for hardware warm-up effects, memory bandwidth limitations, and the statistical variation in processing times across different input lengths and complexities.",
    "The deployment of AI models on specialized inference hardware like the BM1684X requires a comprehensive toolchain that handles model conversion, optimization, and runtime management. The TPU-MLIR compiler framework transforms models from standard formats like ONNX and PyTorch into optimized bmodel binaries that can efficiently utilize the chip's tensor processing units. This compilation process includes graph-level optimizations such as operator fusion, memory layout transformation, and precision calibration to maximize inference performance.",
]

def run_benchmark():
    print("=" * 60)
    print("ChatTTS Benchmark - 70 samples (35 ZH + 35 EN, short+long)")
    print("=" * 60)

    # 加载模型
    print("\n[1/2] Loading model...")
    chat = ChatTTS.Chat()
    chat.load(local_path='../models')
    torch.set_num_threads(4)

    # 加载音色
    with open('slct_voice_240605.json', 'r', encoding='utf-8') as f:
        slct = json.load(f)
    spk = torch.tensor(slct['6']['tensor'])

    params_infer = ChatTTS.Chat.InferCodeParams(
        prompt="[speed_5]",
        temperature=0.0001,
        spk_emb=spk,
    )

    # 预热 5 次（确保模型完全加载进 TPU 内存，不计入结果）
    print("[2/2] Warming up (5 rounds)...")
    warmup_texts = [
        "warm up one",
        "系统预热中，请稍候。",
        "this is the third warm up sentence.",
        "第四次预热，模型加载完成。",
        "final warm up round, ready to benchmark.",
    ]
    for i, wt in enumerate(warmup_texts):
        chat.infer(wt, skip_refine_text=True, use_decoder=True,
                   params_infer_code=params_infer)
        print(f"  warm up {i+1}/5 done")
    print("Warm up done.\n")

    # 开始测试
    results = []
    errors = []

    n_total = len(TEST_SAMPLES)
    for i, text in enumerate(TEST_SAMPLES):
        if i < 25:
            lang, group = "ZH", "short"
        elif i < 50:
            lang, group = "EN", "short"
        elif i < 60:
            lang, group = "ZH", "long"
        else:
            lang, group = "EN", "long"
        short_text = text[:40] + "..." if len(text) > 40 else text
        try:
            t0 = time.time()
            wavs = chat.infer(text, skip_refine_text=True, use_decoder=True,
                              params_infer_code=params_infer)
            t1 = time.time()

            infer_time = t1 - t0
            wav_len = wavs[0].shape[0] / 24000
            rtf = infer_time / wav_len

            results.append({
                "idx": i + 1,
                "lang": lang,
                "group": group,
                "text": short_text,
                "infer_time": infer_time,
                "wav_len": wav_len,
                "rtf": rtf,
            })
            print(f"[{i+1:02d}/{n_total}] [{lang}/{group}] RTF={rtf:.3f} | infer={infer_time:.2f}s | audio={wav_len:.2f}s | {short_text}")

        except Exception as e:
            errors.append((i + 1, str(e)))
            print(f"[{i+1:02d}/{n_total}] [{lang}/{group}] ERROR: {e}")

    # 统计结果
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    rtf_all     = [r["rtf"] for r in results]
    rtf_zh      = [r["rtf"] for r in results if r["lang"] == "ZH"]
    rtf_en      = [r["rtf"] for r in results if r["lang"] == "EN"]
    rtf_zh_s    = [r["rtf"] for r in results if r["lang"] == "ZH" and r["group"] == "short"]
    rtf_zh_l    = [r["rtf"] for r in results if r["lang"] == "ZH" and r["group"] == "long"]
    rtf_en_s    = [r["rtf"] for r in results if r["lang"] == "EN" and r["group"] == "short"]
    rtf_en_l    = [r["rtf"] for r in results if r["lang"] == "EN" and r["group"] == "long"]

    def stats(name, arr):
        if not arr:
            return
        print(f"\n  {name} ({len(arr)} samples):")
        print(f"    平均 RTF : {np.mean(arr):.3f}")
        print(f"    最大 RTF : {np.max(arr):.3f}")
        print(f"    最小 RTF : {np.min(arr):.3f}")
        print(f"    中位 RTF : {np.median(arr):.3f}")
        print(f"    标准差   : {np.std(arr):.3f}")
        realtime = sum(1 for r in arr if r < 1.0)
        print(f"    RTF < 1  : {realtime}/{len(arr)} ({100*realtime/len(arr):.0f}%)")

    stats("全部样本", rtf_all)
    stats("中文-短句", rtf_zh_s)
    stats("中文-长文", rtf_zh_l)
    stats("英文-短句", rtf_en_s)
    stats("英文-长文", rtf_en_l)
    stats("中文汇总", rtf_zh)
    stats("英文汇总", rtf_en)

    total_infer = sum(r["infer_time"] for r in results)
    total_audio = sum(r["wav_len"] for r in results)
    print(f"\n  总推理时间: {total_infer:.1f}s")
    print(f"  总音频时长: {total_audio:.1f}s")
    print(f"  整体 RTF  : {total_infer/total_audio:.3f}")

    if errors:
        print(f"\n  失败样本: {len(errors)}")
        for idx, err in errors:
            print(f"    [{idx}] {err}")

    print("=" * 60)

if __name__ == "__main__":
    run_benchmark()
