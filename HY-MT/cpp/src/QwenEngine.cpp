#include "QwenEngine.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

// ---- helpers ----

void QwenEngine::empty_mem(bm_device_mem_t& mem)
{
    int value = 0;
    bm_memset_device_ext(bm_handle, &value, 1, mem);
}

void QwenEngine::empty_net(const bm_net_info_t* net, int stage_idx)
{
    int value = 0;
    for (int i = 0; i < net->input_num; i++) {
        bm_memset_device_ext(bm_handle, &value, 1, net->stages[stage_idx].input_mems[i]);
    }
    for (int i = 0; i < net->output_num; i++) {
        bm_memset_device_ext(bm_handle, &value, 1, net->stages[stage_idx].output_mems[i]);
    }
}

void QwenEngine::d2d(bm_device_mem_t& dst, bm_device_mem_t& src, int offset, int size)
{
    if (!size)
        size = static_cast<int>(bm_mem_get_device_size(src));
    bm_memcpy_d2d_byte(bm_handle, dst, offset, src, 0, size);
}

// ---- init / deinit ----

void QwenEngine::init_by_names()
{
    auto is_exist = [](const char* name, const char** names, int num) {
        for (int i = 0; i < num; i++) {
            if (strcmp(name, names[i]) == 0)
                return true;
        }
        return false;
    };

    net_embed       = bmrt_get_network_info(p_bmrt, "embedding");
    net_embed_cache = bmrt_get_network_info(p_bmrt, "embedding_cache");
    net_lm          = bmrt_get_network_info(p_bmrt, "lm_head");
    if (!net_embed || !net_embed_cache || !net_lm) {
        throw std::runtime_error("[QwenEngine] bmodel 缺网络 embedding/embedding_cache/lm_head");
    }

    const char** net_names = nullptr;
    int          num_nets  = bmrt_get_network_number(p_bmrt);
    bmrt_get_network_names(p_bmrt, &net_names);

    int num_blocks = num_nets - 3; // embed, embed_cache, lm_head

    lmhead_with_topk = net_lm->stages[0].output_shapes[0].dims[1] == 1;

    NUM_LAYERS = num_blocks / 2;
    for (int i = 0; i < NUM_LAYERS; i++) {
        std::string bname = "block_" + std::to_string(i);
        std::string cname = "block_cache_" + std::to_string(i);
        if (!is_exist(bname.c_str(), net_names, num_nets) ||
            !is_exist(cname.c_str(), net_names, num_nets)) {
            NUM_LAYERS = i;
            break;
        }
        net_blocks.emplace_back(bmrt_get_network_info(p_bmrt, bname.c_str()));
        net_blocks_cache.emplace_back(bmrt_get_network_info(p_bmrt, cname.c_str()));
    }
    free(net_names);

    if (net_embed_cache->output_dtypes[0] == BM_FLOAT16) {
        mask_value = 0xF0E2;
    } else if (net_embed_cache->output_dtypes[0] == BM_BFLOAT16) {
        mask_value = 0xC61C;
    } else {
        throw std::runtime_error("[QwenEngine] Invalid attention dtype (not fp16/bf16)");
    }

    if (net_blocks.empty() || net_blocks_cache.empty()) {
        throw std::runtime_error("[QwenEngine] No block networks found in bmodel "
                                 "(block_0/block_cache_0 missing)");
    }

    MAX_INPUT_LENGTH = net_embed->stages[0].input_shapes[0].dims[1];
    SEQLEN           = net_blocks_cache[0]->stages[0].input_shapes[3].dims[1];
    history_length   = 0;

    // M8：net_launch_decode 用 block_cache_0 的 position/mask 缓冲给所有层（launch 时
    // 以 in_t 传入的 device_mem 为准，各层自身 input_mems 地址可以不同），因此这里
    // 校验的是各层 shape 一致而非地址相同；shape 不一致会静默错数据，提前报错。
    const auto& base = net_blocks_cache[0]->stages[0];
    for (int i = 1; i < NUM_LAYERS; i++) {
        const auto& s = net_blocks_cache[i]->stages[0];
        if (s.input_shapes[1].dims[0] != base.input_shapes[1].dims[0] ||
            s.input_shapes[2].dims[3] != base.input_shapes[2].dims[3]) {
            throw std::runtime_error(
                "[QwenEngine] block_cache position/mask shape 不一致（工具链约定破坏）");
        }
    }
    is_same_addr = true;
    for (int i = 0; i < NUM_LAYERS; i++) {
        const auto& s = net_blocks[i]->stages[0];
        if (i == 0) {
            is_same_addr = s.input_mems[0].u.device.device_addr ==
                           s.output_mems[0].u.device.device_addr;
        } else if (s.input_mems[0].u.device.device_addr !=
                   net_blocks[i - 1]->stages[0].output_mems[0].u.device.device_addr) {
            is_same_addr = false;  // 层间非直连，每层都需 d2d
        }
    }
}

void QwenEngine::init(const std::vector<int>& devices, const std::string& model_path)
{
    try {
        for (auto d : devices) {
            bm_handle_t h;
            bm_status_t s = bm_dev_request(&h, d);
            if (BM_SUCCESS != s) {
                throw std::runtime_error("[QwenEngine] bm_dev_request failed: " +
                                         std::to_string(d));
            }
            handles.push_back(h);
        }
        bm_handle = handles[0];

#ifdef SOC_TARGET
        p_bmrt = bmrt_create(handles[0]);
#else
        p_bmrt = bmrt_create_ex(handles.data(), static_cast<int>(handles.size()));
#endif
        if (p_bmrt == nullptr) {
            throw std::runtime_error("[QwenEngine] bmrt_create failed");
        }
        bmrt_set_flags(p_bmrt, BM_RUNTIME_SHARE_MEM);

        bool ok = bmrt_load_bmodel(p_bmrt, model_path.c_str());
        if (!ok) {
            throw std::runtime_error("[QwenEngine] Failed to load bmodel: " + model_path);
        }

        init_by_names();

        visited_tokens.resize(SEQLEN);

        hidden_bytes =
            static_cast<int>(bm_mem_get_device_size(net_blocks_cache[0]->stages[0].output_mems[0]));
        kv_bytes =
            static_cast<int>(bm_mem_get_device_size(net_blocks_cache[0]->stages[0].output_mems[1]));

        auto buffer_size = bm_mem_get_device_size(net_embed->stages[0].output_mems[0]);
        if (bm_malloc_device_byte(bm_handle, &dev_buffer, buffer_size) != BM_SUCCESS) {
            throw std::runtime_error("[QwenEngine] bm_malloc_device_byte failed");
        }

        past_key.resize(NUM_LAYERS);
        past_value.resize(NUM_LAYERS);

        for (int i = 0; i < NUM_LAYERS; i++) {
            past_key[i]   = net_blocks_cache[i]->stages[0].input_mems[3];
            past_value[i] = net_blocks_cache[i]->stages[0].input_mems[4];
            empty_mem(past_key[i]);
            empty_mem(past_value[i]);
        }
    } catch (...) {
        deinit();  // M2：失败路径清理已分配资源后重抛
        throw;
    }
}

void QwenEngine::deinit()
{
    // 幂等：init 失败路径也会调用
    if (dev_buffer.u.device.device_addr && bm_handle) {
        bm_free_device(bm_handle, dev_buffer);
        dev_buffer.u.device.device_addr = 0;
    }
    if (p_bmrt) {
        bmrt_destroy(p_bmrt);
        p_bmrt = nullptr;
    }
    for (auto h : handles) {
        bm_dev_free(h);
    }
    handles.clear();
    bm_handle = nullptr;
}

// ---- inference helpers ----

void QwenEngine::net_launch(const bm_net_info_t* net, int stage_idx)
{
    std::vector<bm_tensor_t> in_t(net->input_num);
    std::vector<bm_tensor_t> out_t(net->output_num);

    for (int i = 0; i < net->input_num; i++) {
        bmrt_tensor_with_device(&in_t[i], net->stages[stage_idx].input_mems[i],
                                net->input_dtypes[i], net->stages[stage_idx].input_shapes[i]);
    }
    for (int i = 0; i < net->output_num; i++) {
        bmrt_tensor_with_device(&out_t[i], net->stages[stage_idx].output_mems[i],
                                net->output_dtypes[i], net->stages[stage_idx].output_shapes[i]);
    }
    bool ok =
        bmrt_launch_tensor_ex(p_bmrt, net->name, in_t.data(), net->input_num,
                              out_t.data(), net->output_num, true, false);
    if (!ok) {
        throw std::runtime_error(std::string("[QwenEngine] launch failed: ") + net->name);
    }
}

void QwenEngine::net_launch_decode(int idx, int kv_offset, bm_device_mem_t& input_mem,
                                   const int* position_id,
                                   std::vector<uint16_t>& attention_mask)
{
    auto& net      = net_blocks_cache[idx];
    auto& in1_mem  = net_blocks_cache[idx]->stages[0].input_mems[1];
    auto& in2_mem  = net_blocks_cache[idx]->stages[0].input_mems[2];
    auto& in3_mem  = net_blocks_cache[idx]->stages[0].input_mems[3];
    auto& in4_mem  = net_blocks_cache[idx]->stages[0].input_mems[4];
    auto& out0_mem = net_blocks_cache[idx]->stages[0].output_mems[0];

    std::vector<bm_tensor_t> in_t(5);
    std::vector<bm_tensor_t> out_t(3);

    bmrt_tensor_with_device(&in_t[0], input_mem, net->input_dtypes[0],
                            net->stages[0].input_shapes[0]);
    if (idx == 0) {
        bm_memcpy_s2d(bm_handle, in1_mem, (void*)position_id);
        bm_memcpy_s2d(bm_handle, in2_mem, (void*)attention_mask.data());
        bmrt_tensor_with_device(&in_t[1], in1_mem, net->input_dtypes[1],
                                net->stages[0].input_shapes[1]);
        bmrt_tensor_with_device(&in_t[2], in2_mem, net->input_dtypes[2],
                                net->stages[0].input_shapes[2]);
    } else {
        // 共享缓冲已在 init 校验一致
        bmrt_tensor_with_device(&in_t[1], net_blocks_cache[0]->stages[0].input_mems[1],
                                net->input_dtypes[1], net->stages[0].input_shapes[1]);
        bmrt_tensor_with_device(&in_t[2], net_blocks_cache[0]->stages[0].input_mems[2],
                                net->input_dtypes[2], net->stages[0].input_shapes[2]);
    }
    bmrt_tensor_with_device(&in_t[3], in3_mem, net->input_dtypes[3],
                            net->stages[0].input_shapes[3]);
    bmrt_tensor_with_device(&in_t[4], in4_mem, net->input_dtypes[4],
                            net->stages[0].input_shapes[4]);

    bmrt_tensor_with_device(&out_t[0], out0_mem, net->output_dtypes[0],
                            net->stages[0].output_shapes[0]);
    auto k_mem =
        bm_mem_from_device(past_key[idx].u.device.device_addr + kv_offset, kv_bytes);
    auto v_mem =
        bm_mem_from_device(past_value[idx].u.device.device_addr + kv_offset, kv_bytes);
    bmrt_tensor_with_device(&out_t[1], k_mem, net->output_dtypes[1],
                            net->stages[0].output_shapes[1]);
    bmrt_tensor_with_device(&out_t[2], v_mem, net->output_dtypes[2],
                            net->stages[0].output_shapes[2]);

    bool ok =
        bmrt_launch_tensor_ex(p_bmrt, net->name, in_t.data(), in_t.size(),
                              out_t.data(), out_t.size(), true, false);
    if (!ok) {
        throw std::runtime_error(std::string("[QwenEngine] launch failed: ") + net->name);
    }
}

int QwenEngine::forward_first(std::vector<int>& tokens)
{
    // H1：固定 shape 缓冲按 SEQLEN 分配，超长输入直接拒绝（Release 下也生效）
    if (tokens.size() >= (size_t)SEQLEN || tokens.empty()) {
        throw std::runtime_error("[QwenEngine] invalid input length: " +
                                 std::to_string(tokens.size()) +
                                 " (SEQLEN=" + std::to_string(SEQLEN) + ")");
    }

    std::vector<int>      position_id(MAX_INPUT_LENGTH, 0);
    std::vector<uint16_t> attention_mask(MAX_INPUT_LENGTH * MAX_INPUT_LENGTH, mask_value);
    std::fill(visited_tokens.begin(), visited_tokens.end(), 0);
    std::copy(tokens.begin(), tokens.end(), visited_tokens.data());

    token_length = static_cast<int>(tokens.size());

    for (int i = 0; i < token_length; i++) {
        position_id[i] = i;
    }
    for (int i = 0; i < token_length; i++) {
        for (int j = 0; j <= i; j++) {
            attention_mask[i * MAX_INPUT_LENGTH + j] = 0;
        }
    }

    auto in_mem  = net_embed->stages[0].input_mems[0];
    auto out_mem = net_embed->stages[0].output_mems[0];
    empty_mem(in_mem);
    bm_memcpy_s2d_partial(bm_handle, in_mem, (void*)tokens.data(),
                          token_length * sizeof(int));
    net_launch(net_embed);
    d2d(dev_buffer, out_mem, 0, static_cast<int>(bm_mem_get_device_size(out_mem)));
    out_mem = dev_buffer;

    empty_net(net_blocks[0]);
    for (int idx = 0; idx < NUM_LAYERS; idx++) {
        auto& in0 = net_blocks[idx]->stages[0].input_mems[0];
        auto& in1 = net_blocks[idx]->stages[0].input_mems[1];
        auto& in2 = net_blocks[idx]->stages[0].input_mems[2];
        if (!is_same_addr || idx == 0) {
            d2d(in0, out_mem, 0, token_length * hidden_bytes);
        }
        if (idx == 0) {
            bm_memcpy_s2d(bm_handle, in1, (void*)position_id.data());
            bm_memcpy_s2d(bm_handle, in2, (void*)attention_mask.data());
        }
        net_launch(net_blocks[idx]);
        out_mem = net_blocks[idx]->stages[0].output_mems[0];
        d2d(past_key[idx], net_blocks[idx]->stages[0].output_mems[1], 0,
            token_length * kv_bytes);
        d2d(past_value[idx], net_blocks[idx]->stages[0].output_mems[2], 0,
            token_length * kv_bytes);
    }

    auto& lm_in  = net_lm->stages[0].input_mems[0];
    auto& lm_out = net_lm->stages[0].output_mems[0];
    bm_memcpy_d2d_byte(bm_handle, lm_in, 0, out_mem,
                       (token_length - 1) * hidden_bytes, hidden_bytes);
    net_launch(net_lm);

    // lm_head 输出 top-k 标量 token_id（lmhead_with_topk）或完整 logits 首元素（读 4 字节），
    // CPU 读结果前必须 sync
    bm_thread_sync(bm_handle);
    int token = 0;
    bm_memcpy_d2s(bm_handle, &token, lm_out);

    visited_tokens[token_length] = token;
    token_length++;
    history_length = token_length;
    return token;
}

int QwenEngine::forward_next()
{
    // L4：forward_next 在 forward_first 之前调用会读到 -1 下标
    if (token_length == 0) {
        throw std::runtime_error("[QwenEngine] forward_next called before forward_first");
    }
    if (token_length >= SEQLEN) {
        throw std::runtime_error("[QwenEngine] sequence length exhausted: " +
                                 std::to_string(SEQLEN));
    }
    int cur_token = visited_tokens[token_length - 1];

    std::vector<uint16_t> attention_mask(SEQLEN + 1, 0);
    for (int i = history_length - 1; i < SEQLEN; i++) {
        attention_mask[i] = mask_value;
    }
    int32_t position_id = history_length - 1;

    auto in_mem  = net_embed_cache->stages[0].input_mems[0];
    auto out_mem = net_embed_cache->stages[0].output_mems[0];
    bm_memcpy_s2d(bm_handle, in_mem, (void*)&cur_token);
    net_launch(net_embed_cache);

    int token_offset = (token_length - 1) * kv_bytes;
    for (int idx = 0; idx < NUM_LAYERS; idx++) {
        net_launch_decode(idx, token_offset, out_mem, &position_id, attention_mask);
        out_mem = net_blocks_cache[idx]->stages[0].output_mems[0];
    }

    auto& lm_in  = net_lm->stages[0].input_mems[0];
    auto& lm_out = net_lm->stages[0].output_mems[0];
    d2d(lm_in, out_mem);
    net_launch(net_lm);

    bm_thread_sync(bm_handle);
    int token = 0;
    bm_memcpy_d2s(bm_handle, &token, lm_out);

    visited_tokens[token_length] = token;
    token_length++;
    history_length++;
    return token;
}
