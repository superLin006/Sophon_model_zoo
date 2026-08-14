#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "bmruntime_interface.h"
#include "memory.h"

/**
 * @brief HY-MT 混元模型的 bmruntime 推理引擎（适用于 BM1684X SoC）
 *
 * 适配 llm_tpu 导出的 static 3-input prefill bmodel（embedding / block_* /
 * block_cache_* / lm_head）。lm_head 输出为 top-k 后的标量 token_id
 * （lmhead_with_topk=true），无 greedy/sample 采样头。
 *
 * 注意事项：
 * - 层间 launch 依赖驱动对同线程 d2d/launch 的队列串行化（与官方 llm_tpu demo
 *   一致），CPU 读结果前必须 bm_thread_sync；换 SDK 版本时需复核。
 * - 非线程安全：单线程使用（shared mutable 状态 visited_tokens/token_length）。
 */
class QwenEngine
{
public:
    QwenEngine() = default;

    void init(const std::vector<int>& devids, const std::string& model_path);
    void deinit();

    int  forward_first(std::vector<int>& tokens);
    int  forward_next();

    // ---- 模型参数（init 后有效）----
    int  SEQLEN{0};
    int  MAX_INPUT_LENGTH{0};
    int  NUM_LAYERS{0};
    int  token_length{0};
    int  history_length{0};
    bool lmhead_with_topk{false};

    std::vector<int> visited_tokens;

private:
    void init_by_names();
    void net_launch(const bm_net_info_t* net, int stage_idx = 0);
    void net_launch_decode(int block_idx, int kv_offset, bm_device_mem_t& input_mem,
                           const int* position_id, std::vector<uint16_t>& attention_mask);
    void d2d(bm_device_mem_t& dst, bm_device_mem_t& src, int offset = 0, int size = 0);
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

    bm_device_mem_t              dev_buffer{};
    std::vector<bm_device_mem_t> past_key;
    std::vector<bm_device_mem_t> past_value;

    int      hidden_bytes{0};
    int      kv_bytes{0};
    uint16_t mask_value{0};
    bool     is_same_addr{false};
};
