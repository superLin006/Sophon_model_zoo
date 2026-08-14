#pragma once

#include <random>
#include <string>
#include <vector>

#include "bmruntime_interface.h"
#include "memory.h"

/**
 * @brief Qwen 系列模型的 bmruntime 推理引擎（适用于 BM1684X SoC）
 *
 * 支持 Qwen2 / Qwen3 系列 bmodel（static / prefill-KV 两种导出格式）。
 * 移植自 LLM-TPU demo/python_demo/chat.cpp，去除 pybind11 依赖。
 */
class QwenEngine
{
public:
    QwenEngine() : sgen(std::random_device()()) {}

    void init(const std::vector<int>& devids, const std::string& model_path);
    void deinit();

    int  forward_first(std::vector<int>& tokens);
    int  forward_next();
    void clear_kv();

    // ---- 模型参数（init 后有效）----
    int  SEQLEN{0};
    int  MAX_INPUT_LENGTH{0};
    int  PREFILL_KV_LENGTH{0};
    int  NUM_LAYERS{0};
    int  token_length{0};
    int  history_length{0};
    bool support_prefill_kv{false};
    bool lmhead_with_topk{false};
    bool is_dynamic{false};
    bool is_same_addr{false};

    std::vector<int> visited_tokens;

    // ---- 生成参数（init 前可写）----
    std::string generation_mode{"greedy"};
    float       temperature{1.0f};
    float       top_p{1.0f};
    int         top_k{1};
    float       penalty{1.0f};

    std::mt19937 sgen;

private:
    void init_by_names();
    void net_launch(const bm_net_info_t* net, int stage_idx = 0);
    void net_launch_dyn(const bm_net_info_t* net, int real_len, int stage_idx = 0);
    void net_launch_decode(int block_idx, int kv_offset, bm_device_mem_t& input_mem,
                           const int* position_id, std::vector<uint16_t>& attention_mask);
    void d2d(bm_device_mem_t& dst, bm_device_mem_t& src, int offset = 0, int size = 0);
    int  greedy_search(bm_device_mem_t& logits_mem);
    int  penalty_sample(bm_device_mem_t& logits_mem);
    int  forward_first_with_kv(std::vector<int>& tokens);
    void empty_mem(bm_device_mem_t& mem);
    void empty_net(const bm_net_info_t* net, int stage_idx = 0);

    std::vector<bm_handle_t>         handles;
    bm_handle_t                       bm_handle{nullptr};
    void*                             p_bmrt{nullptr};
    std::vector<const bm_net_info_t*> net_blocks;
    std::vector<const bm_net_info_t*> net_blocks_cache;
    const bm_net_info_t*              net_embed{nullptr};
    const bm_net_info_t*              net_embed_cache{nullptr};
    const bm_net_info_t*              net_lm{nullptr};
    const bm_net_info_t*              net_greedy_head{nullptr};
    const bm_net_info_t*              net_sample_head{nullptr};

    bm_device_mem_t              dev_buffer{};
    std::vector<bm_device_mem_t> past_key;
    std::vector<bm_device_mem_t> past_value;

    int      hidden_bytes{0};
    int      kv_bytes{0};
    uint16_t mask_value{0};
};
