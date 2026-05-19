// ChatTTS C++ Benchmark — mirrors python/benchmark.py
// 70 samples: 25 ZH-short + 25 EN-short + 10 ZH-long + 10 EN-long
// Tests both non-streaming (RTF) and streaming (RTF + TTFA).

#include "chattts.h"
#include <iostream>
#include <numeric>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>

static const std::vector<std::string> TEST_SAMPLES = {
    // ===== 中文短句 25条 =====
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
    // ===== 英文短句 25条 =====
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
    // ===== 中文长文 10条 =====
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
    // ===== 英文长文 10条 =====
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
};

using Clock = std::chrono::steady_clock;
using Ms    = std::chrono::duration<double, std::milli>;

static void print_stats(const std::string& label, const std::vector<double>& arr,
                        const std::string& unit = "", double threshold = 0.0) {
    if (arr.empty()) { printf("\n  %s: (no data)\n", label.c_str()); return; }
    double sum  = std::accumulate(arr.begin(), arr.end(), 0.0);
    double mean = sum / arr.size();
    std::vector<double> s = arr;
    std::sort(s.begin(), s.end());
    double median = s[s.size() / 2];
    double var = 0;
    for (double v : arr) var += (v - mean) * (v - mean);
    double stddev = std::sqrt(var / arr.size());

    printf("\n  %s (%d samples):\n", label.c_str(), (int)arr.size());
    printf("    均值  : %.3f %s\n", mean,   unit.c_str());
    printf("    最大  : %.3f %s\n", s.back(), unit.c_str());
    printf("    最小  : %.3f %s\n", s.front(), unit.c_str());
    printf("    中位  : %.3f %s\n", median, unit.c_str());
    printf("    标准差: %.3f %s\n", stddev, unit.c_str());
    if (threshold > 0) {
        int ok = 0;
        for (double v : arr) if (v < threshold) ok++;
        printf("    < %.3g: %d/%d (%.0f%%)\n", threshold,
               ok, (int)arr.size(), 100.0 * ok / arr.size());
    }
}

int main(int argc, char* argv[]) {
    std::string model_dir  = "/data/chatTTS-offical/models";
    std::string spk_file   = "spk_emb.bin";
    int   tpu_id           = 0;
    int   warmup           = 3;
    int   stream_batch     = 24;
    bool  no_stream        = false;  // skip streaming benchmark
    bool  stream_only      = false;  // skip non-streaming benchmark

    for (int i = 1; i < argc; ++i) {
        if      (!std::strcmp(argv[i], "--model-dir")    && i+1 < argc) model_dir    = argv[++i];
        else if (!std::strcmp(argv[i], "--spk-emb")      && i+1 < argc) spk_file     = argv[++i];
        else if (!std::strcmp(argv[i], "--tpu-id")       && i+1 < argc) tpu_id       = std::stoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--warmup")       && i+1 < argc) warmup       = std::stoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--stream-batch") && i+1 < argc) stream_batch = std::stoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--no-stream"))                   no_stream    = true;
        else if (!std::strcmp(argv[i], "--stream-only"))                 stream_only  = true;
    }
    if (stream_only) no_stream = false;  // stream-only implies streaming is enabled

    printf("============================================================\n");
    printf("ChatTTS C++ Benchmark\n");
    printf("  70 samples: 25 ZH-short + 25 EN-short + 10 ZH-long + 10 EN-long\n");
    printf("  stream_batch = %d\n", stream_batch);
    printf("============================================================\n\n");

    printf("[1/3] Loading model from: %s\n", model_dir.c_str());
    ChatTTSConfig cfg;
    cfg.gpt_model_path      = model_dir + "/chattts-llama_int4_1dev_1024_bm1684x.bmodel";
    cfg.decoder_model_path  = model_dir + "/decoder_1-768-1024_bm1684x.bmodel";
    cfg.vocos_model_path    = model_dir + "/vocos_1-100-2048_bm1684x.bmodel";
    cfg.vocab_path          = model_dir + "/asset/tokenizer/vocab.txt";
    cfg.homophones_map_path = model_dir + "/asset/homophones_map.json";
    cfg.tpu_id              = tpu_id;

    InferParams params;
    params.temperature   = 0.0001f;
    params.speed         = 5;
    params.max_new_token = 2048;

    StreamParams sparams;
    sparams.stream_batch         = stream_batch;
    sparams.pass_first_n_batches = 2;

    struct Sample {
        std::string lang, group;
        double infer_s, audio_s, rtf;
        bool ok;
    };
    std::vector<Sample> ns_results;
    int N = (int)TEST_SAMPLES.size();

    // ── Non-streaming benchmark ──────────────────────────────────────────────
    // Scoped block: ChatTTS is destroyed at end of block, calling bmrt_destroy()
    // which reclaims all BMRT heap blocks. The streaming instance created after
    // this block then starts with a completely clean device memory state.
    if (!stream_only) {
        auto tts = std::make_unique<ChatTTS>(cfg);
        if (!spk_file.empty()) {
            if (!tts->load_speaker(spk_file))
                fprintf(stderr, "Warning: failed to load speaker: %s\n", spk_file.c_str());
            else
                printf("Speaker loaded: %s\n", spk_file.c_str());
        }

        printf("\n[2/3] Warming up (%d rounds, non-stream)...\n", warmup);
        static const char* WU[] = {
            "warm up one", "系统预热中，请稍候。",
            "this is the third warm up.", "第四次预热完成。", "final warm up.",
        };
        for (int i = 0; i < warmup && i < 5; ++i) {
            tts->infer(WU[i], params, false);
            printf("  warm up %d/%d done\n", i+1, warmup);
        }
        printf("Warm up done.\n\n");

        printf("[3/3] Non-streaming benchmark...\n");
        printf("------------------------------------------------------------\n");

        for (int i = 0; i < N; ++i) {
            std::string lang  = (i < 25) ? "ZH" : (i < 50) ? "EN" : (i < 60) ? "ZH" : "EN";
            std::string group = (i < 50) ? "short" : "long";
            const std::string& text = TEST_SAMPLES[i];
            std::string disp = text.size() > 38 ? text.substr(0, 38) + "..." : text;

            Sample s; s.lang = lang; s.group = group;
            try {
                auto t0 = Clock::now();
                auto pcm = tts->infer(text, params, false);
                auto t1  = Clock::now();
                s.infer_s = std::chrono::duration<double>(t1-t0).count();
                s.audio_s = (double)pcm.size() / 24000.0;
                s.rtf     = s.infer_s / s.audio_s;
                s.ok      = true;
                printf("[%02d/%d] [%s/%-5s] RTF=%.3f  infer=%.2fs  audio=%.2fs  %s\n",
                       i+1, N, lang.c_str(), group.c_str(),
                       s.rtf, s.infer_s, s.audio_s, disp.c_str());
            } catch (const std::exception& e) {
                s.ok = false;
                printf("[%02d/%d] [%s/%-5s] ERROR: %s\n", i+1, N, lang.c_str(), group.c_str(), e.what());
            }
            ns_results.push_back(s);
        }
        // tts destroyed here → bmrt_destroy() → clean heap for streaming
        printf("\nNon-streaming done. BMRT heap reset.\n");
    }

    // ── Streaming benchmark ──────────────────────────────────────────────────
    struct StreamSample {
        std::string lang, group;
        double infer_s, audio_s, rtf, ttfa_ms;
        bool ok;
    };
    std::vector<StreamSample> st_results;

    if (!no_stream) {
        printf("\n[3/3] Streaming benchmark (stream_batch=%d)...\n", stream_batch);
        printf("------------------------------------------------------------\n");

        auto stts = std::make_unique<ChatTTS>(cfg);
        if (!spk_file.empty()) stts->load_speaker(spk_file);

        // Warmup for stream-only mode (otherwise warmup was done in non-stream block above)
        if (stream_only) {
            printf("\n[2/3] Warming up (%d rounds, stream)...\n", warmup);
            static const char* WU[] = {
                "warm up one", "系统预热中，请稍候。",
                "this is the third warm up.", "第四次预热完成。", "final warm up.",
            };
            for (int i = 0; i < warmup && i < 5; ++i) {
                stts->infer(WU[i], params, false);
                printf("  warm up %d/%d done\n", i+1, warmup);
            }
            printf("Warm up done.\n\n");
        }

        for (int i = 0; i < N; ++i) {
            std::string lang  = (i < 25) ? "ZH" : (i < 50) ? "EN" : (i < 60) ? "ZH" : "EN";
            std::string group = (i < 50) ? "short" : "long";
            const std::string& text = TEST_SAMPLES[i];
            std::string disp = text.size() > 30 ? text.substr(0, 30) + "..." : text;

            StreamSample s; s.lang = lang; s.group = group;
            try {
                std::vector<float> pcm;
                bool first = true;
                double ttfa_ms = 0;
                auto t0 = Clock::now();

                stts->infer_stream(text, params, sparams,
                    [&](const std::vector<float>& chunk) {
                        if (first) {
                            ttfa_ms = Ms(Clock::now() - t0).count();
                            first = false;
                        }
                        pcm.insert(pcm.end(), chunk.begin(), chunk.end());
                    }, false);

                auto t1 = Clock::now();
                s.infer_s = std::chrono::duration<double>(t1-t0).count();
                s.audio_s = (double)pcm.size() / 24000.0;
                s.rtf     = s.audio_s > 0 ? s.infer_s / s.audio_s : 0;
                s.ttfa_ms = ttfa_ms;
                s.ok      = true;
                printf("[%02d/%d] [%s/%-5s] RTF=%.3f  TTFA=%5.0fms  infer=%.2fs  audio=%.2fs  %s\n",
                       i+1, N, lang.c_str(), group.c_str(),
                       s.rtf, s.ttfa_ms, s.infer_s, s.audio_s, disp.c_str());
            } catch (const std::exception& e) {
                s.ok = false;
                printf("[%02d/%d] [%s/%-5s] ERROR: %s\n", i+1, N, lang.c_str(), group.c_str(), e.what());
            }
            st_results.push_back(s);
        }
    }

    // ── Summary ──────────────────────────────────────────────────────────────
    printf("\n============================================================\n");
    printf("RESULTS SUMMARY\n");
    printf("============================================================\n");

    auto collect_rtf = [&](const std::vector<Sample>& res,
                           const std::string& lang, const std::string& group) {
        std::vector<double> v;
        for (auto& r : res)
            if (r.ok && (lang.empty() || r.lang == lang) && (group.empty() || r.group == group))
                v.push_back(r.rtf);
        return v;
    };
    auto collect_st_rtf = [&](const std::string& lang, const std::string& group) {
        std::vector<double> v;
        for (auto& r : st_results)
            if (r.ok && (lang.empty() || r.lang == lang) && (group.empty() || r.group == group))
                v.push_back(r.rtf);
        return v;
    };
    auto collect_ttfa = [&](const std::string& lang, const std::string& group) {
        std::vector<double> v;
        for (auto& r : st_results)
            if (r.ok && (lang.empty() || r.lang == lang) && (group.empty() || r.group == group))
                v.push_back(r.ttfa_ms);
        return v;
    };

    printf("\n━━ 非流式 RTF ━━\n");
    print_stats("全部样本",  collect_rtf(ns_results, "", ""),      "", 1.0);
    print_stats("中文-短句", collect_rtf(ns_results, "ZH", "short"), "", 1.0);
    print_stats("中文-长文", collect_rtf(ns_results, "ZH", "long"),  "", 1.0);
    print_stats("英文-短句", collect_rtf(ns_results, "EN", "short"), "", 1.0);
    print_stats("英文-长文", collect_rtf(ns_results, "EN", "long"),  "", 1.0);

    if (!no_stream && !st_results.empty()) {
        printf("\n━━ 流式 RTF ━━\n");
        print_stats("全部样本",  collect_st_rtf("", ""),      "", 1.0);
        print_stats("中文-短句", collect_st_rtf("ZH", "short"), "", 1.0);
        print_stats("中文-长文", collect_st_rtf("ZH", "long"),  "", 1.0);
        print_stats("英文-短句", collect_st_rtf("EN", "short"), "", 1.0);
        print_stats("英文-长文", collect_st_rtf("EN", "long"),  "", 1.0);

        printf("\n━━ 流式 TTFA (ms) ━━\n");
        print_stats("全部样本",  collect_ttfa("", ""),      "ms");
        print_stats("中文-短句", collect_ttfa("ZH", "short"), "ms");
        print_stats("中文-长文", collect_ttfa("ZH", "long"),  "ms");
        print_stats("英文-短句", collect_ttfa("EN", "short"), "ms");
        print_stats("英文-长文", collect_ttfa("EN", "long"),  "ms");
    }

    // Totals
    printf("\n━━ 整体统计 ━━\n");
    {
        double ti = 0, ta = 0; int err = 0;
        for (auto& r : ns_results) { if (r.ok) { ti += r.infer_s; ta += r.audio_s; } else err++; }
        printf("\n  非流式 — 总推理: %.1fs  总音频: %.1fs  整体RTF: %.3f",
               ti, ta, ta > 0 ? ti/ta : 0);
        if (err) printf("  失败: %d", err);
        printf("\n");
    }
    if (!no_stream && !st_results.empty()) {
        double ti = 0, ta = 0, ttfa_sum = 0; int err = 0, cnt = 0;
        for (auto& r : st_results) {
            if (r.ok) { ti += r.infer_s; ta += r.audio_s; ttfa_sum += r.ttfa_ms; cnt++; }
            else err++;
        }
        printf("  流式   — 总推理: %.1fs  总音频: %.1fs  整体RTF: %.3f  均值TTFA: %.0fms",
               ti, ta, ta > 0 ? ti/ta : 0, cnt > 0 ? ttfa_sum/cnt : 0);
        if (err) printf("  失败: %d", err);
        printf("\n");
    }
    printf("============================================================\n");
    return 0;
}
