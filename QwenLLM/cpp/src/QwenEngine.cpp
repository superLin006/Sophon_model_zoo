#include "QwenEngine.h"

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <numeric>
#include <random>
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

    const char** net_names = nullptr;
    int          num_nets  = bmrt_get_network_number(p_bmrt);
    bmrt_get_network_names(p_bmrt, &net_names);

    net_greedy_head = nullptr;
    net_sample_head = nullptr;
    int num_blocks  = num_nets - 3; // embed, embed_cache, lm_head

    if (is_exist("greedy_head", net_names, num_nets)) {
        net_greedy_head = bmrt_get_network_info(p_bmrt, "greedy_head");
        num_blocks--;
    }
    if (is_exist("sample_head", net_names, num_nets)) {
        net_sample_head = bmrt_get_network_info(p_bmrt, "sample_head");
        num_blocks--;
    }

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

    MAX_INPUT_LENGTH   = net_embed->stages[0].input_shapes[0].dims[1];
    SEQLEN             = net_blocks_cache[0]->stages[0].input_shapes[3].dims[1];
    support_prefill_kv = net_blocks[0]->input_num == 5;
    history_length     = 0;

    if (support_prefill_kv) {
        PREFILL_KV_LENGTH = net_blocks[0]->stages[0].input_shapes[3].dims[1];
    }
}

void QwenEngine::init(const std::vector<int>& devices, const std::string& model_path)
{
    for (auto d : devices) {
        bm_handle_t h;
        bm_status_t s = bm_dev_request(&h, d);
        assert(BM_SUCCESS == s);
        handles.push_back(h);
    }
    bm_handle = handles[0];

#ifdef SOC_TARGET
    p_bmrt = bmrt_create(handles[0]);
#else
    p_bmrt = bmrt_create_ex(handles.data(), static_cast<int>(handles.size()));
#endif
    assert(p_bmrt != nullptr);
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
    bm_malloc_device_byte(bm_handle, &dev_buffer, buffer_size);

    past_key.resize(NUM_LAYERS);
    past_value.resize(NUM_LAYERS);
    is_dynamic = net_blocks[0]->is_dynamic;

    for (int i = 0; i < NUM_LAYERS; i++) {
        past_key[i]   = net_blocks_cache[i]->stages[0].input_mems[3];
        past_value[i] = net_blocks_cache[i]->stages[0].input_mems[4];
        empty_mem(past_key[i]);
        empty_mem(past_value[i]);
    }

    is_same_addr =
        (net_blocks[0]->stages[0].input_mems[0].u.device.device_addr ==
         net_blocks[0]->stages[0].output_mems[0].u.device.device_addr);
}

void QwenEngine::deinit()
{
    if (p_bmrt) {
        bm_free_device(bm_handle, dev_buffer);
        bmrt_destroy(p_bmrt);
        p_bmrt = nullptr;
    }
    for (auto h : handles) {
        bm_dev_free(h);
    }
    handles.clear();
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
    assert(ok);
}

void QwenEngine::net_launch_dyn(const bm_net_info_t* net, int real_len, int stage_idx)
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
    in_t[0].shape.dims[1] = real_len;
    in_t[1].shape.dims[1] = real_len;
    in_t[2].shape.dims[2] = real_len;
    in_t[2].shape.dims[3] = real_len;

    bool ok =
        bmrt_launch_tensor_ex(p_bmrt, net->name, in_t.data(), net->input_num,
                              out_t.data(), net->output_num, true, false);
    assert(ok);
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
    assert(ok);
}

int QwenEngine::greedy_search(bm_device_mem_t& logits_mem)
{
    auto& in_mem  = net_greedy_head->stages[0].input_mems[0];
    auto& out_mem = net_greedy_head->stages[0].output_mems[0];
    d2d(in_mem, logits_mem, 0, static_cast<int>(bm_mem_get_device_size(logits_mem)));
    net_launch(net_greedy_head);
    int token = 0;
    bm_memcpy_d2s(bm_handle, &token, out_mem);
    return token;
}

int QwenEngine::penalty_sample(bm_device_mem_t& logits_mem)
{
    auto& in0  = net_sample_head->stages[0].input_mems[0];
    auto& in1  = net_sample_head->stages[0].input_mems[1];
    auto& in2  = net_sample_head->stages[0].input_mems[2];
    auto& in3  = net_sample_head->stages[0].input_mems[3];
    auto& in4  = net_sample_head->stages[0].input_mems[4];
    auto& in5  = net_sample_head->stages[0].input_mems[5];
    auto& out0 = net_sample_head->stages[0].output_mems[0];
    auto& out1 = net_sample_head->stages[0].output_mems[1];

    bm_memcpy_s2d(bm_handle, in1, (void*)visited_tokens.data());
    bm_memcpy_s2d(bm_handle, in2, (void*)&penalty);
    bm_memcpy_s2d(bm_handle, in3, (void*)&temperature);
    bm_memcpy_s2d(bm_handle, in4, (void*)&top_k);
    bm_memcpy_s2d(bm_handle, in5, (void*)&top_p);

    d2d(in0, logits_mem, 0, static_cast<int>(bm_mem_get_device_size(logits_mem)));
    net_launch(net_sample_head);

    int                candidate_num = top_k;
    std::vector<float> probs(candidate_num);
    std::vector<int>   tokens(candidate_num);
    bm_memcpy_d2s_partial_offset(bm_handle, probs.data(), out0,
                                 top_k * sizeof(float), 0);
    bm_memcpy_d2s_partial_offset(bm_handle, tokens.data(), out1,
                                 top_k * sizeof(float), 0);
    std::discrete_distribution<> dist(probs.begin(), probs.end());
    return tokens[dist(sgen)];
}

int QwenEngine::forward_first(std::vector<int>& tokens)
{
    if (support_prefill_kv) {
        return forward_first_with_kv(tokens);
    }

    std::vector<int>      position_id(MAX_INPUT_LENGTH, 0);
    std::vector<uint16_t> attention_mask(MAX_INPUT_LENGTH * MAX_INPUT_LENGTH, mask_value);
    std::fill(visited_tokens.begin(), visited_tokens.end(), 0);
    std::copy(tokens.begin(), tokens.end(), visited_tokens.data());

    token_length = static_cast<int>(tokens.size());

    for (int i = 0; i < token_length; i++) {
        position_id[i] = i;
    }
    if (is_dynamic) {
        for (int i = 0; i < token_length; i++) {
            for (int j = 0; j <= i; j++) {
                attention_mask[i * token_length + j] = 0;
            }
        }
    } else {
        for (int i = 0; i < token_length; i++) {
            for (int j = 0; j <= i; j++) {
                attention_mask[i * MAX_INPUT_LENGTH + j] = 0;
            }
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
        if (is_dynamic) {
            net_launch_dyn(net_blocks[idx], token_length);
        } else {
            net_launch(net_blocks[idx]);
        }
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

    int token = 0;
    if (lmhead_with_topk) {
        bm_memcpy_d2s(bm_handle, &token, lm_out);
    } else if (!net_greedy_head && !net_sample_head) {
        bm_memcpy_d2s(bm_handle, &token, lm_out);
    } else if (generation_mode == "greedy") {
        token = greedy_search(lm_out);
    } else {
        token = penalty_sample(lm_out);
    }

    visited_tokens[token_length] = token;
    token_length++;
    history_length = token_length;
    return token;
}

int QwenEngine::forward_first_with_kv(std::vector<int>& inputs)
{
    int max_kv_length = MAX_INPUT_LENGTH + PREFILL_KV_LENGTH;
    std::vector<int>      position_id(MAX_INPUT_LENGTH, 0);
    std::copy(inputs.begin(), inputs.end(), visited_tokens.data());

    int old_length = history_length;
    token_length   = static_cast<int>(inputs.size());
    history_length += token_length;

    std::vector<uint16_t> attention_mask(MAX_INPUT_LENGTH * max_kv_length, mask_value);
    assert(history_length < SEQLEN);
    assert(old_length <= PREFILL_KV_LENGTH);

    for (int i = 0; i < token_length; i++) {
        for (int j = 0; j < old_length; j++) {
            attention_mask[i * max_kv_length + j] = 0;
        }
        for (int j = 0; j <= i; j++) {
            attention_mask[i * max_kv_length + j + PREFILL_KV_LENGTH] = 0;
        }
    }
    for (int i = 0; i < token_length; i++) {
        position_id[i] = i + old_length;
    }

    auto in_mem  = net_embed->stages[0].input_mems[0];
    auto out_mem = net_embed->stages[0].output_mems[0];
    empty_mem(in_mem);
    bm_memcpy_s2d_partial(bm_handle, in_mem, (void*)inputs.data(),
                          token_length * sizeof(int));
    net_launch(net_embed);
    d2d(dev_buffer, out_mem, 0, static_cast<int>(bm_mem_get_device_size(out_mem)));
    out_mem = dev_buffer;

    empty_net(net_blocks[0]);
    for (int idx = 0; idx < NUM_LAYERS; idx++) {
        auto& in0 = net_blocks[idx]->stages[0].input_mems[0];
        auto& in1 = net_blocks[idx]->stages[0].input_mems[1];
        auto& in2 = net_blocks[idx]->stages[0].input_mems[2];
        auto& in3 = net_blocks[idx]->stages[0].input_mems[3];
        auto& in4 = net_blocks[idx]->stages[0].input_mems[4];

        if (!is_same_addr || idx == 0) {
            d2d(in0, out_mem);
        }
        if (old_length > 0) {
            bm_memcpy_d2d_byte(bm_handle, in3, 0, past_key[idx], 0,
                               kv_bytes * old_length);
            bm_memcpy_d2d_byte(bm_handle, in4, 0, past_value[idx], 0,
                               kv_bytes * old_length);
        } else if (idx == 0) {
            empty_mem(in3);
            empty_mem(in4);
        }
        bm_memcpy_s2d(bm_handle, in1, (void*)position_id.data());
        bm_memcpy_s2d(bm_handle, in2, (void*)attention_mask.data());
        net_launch(net_blocks[idx]);
        out_mem = net_blocks[idx]->stages[0].output_mems[0];

        auto& out1 = net_blocks[idx]->stages[0].output_mems[1];
        auto& out2 = net_blocks[idx]->stages[0].output_mems[2];
        bm_memcpy_d2d_byte(bm_handle, past_key[idx], old_length * kv_bytes,
                           out1, 0, kv_bytes * token_length);
        bm_memcpy_d2d_byte(bm_handle, past_value[idx], old_length * kv_bytes,
                           out2, 0, kv_bytes * token_length);
    }

    auto& lm_in  = net_lm->stages[0].input_mems[0];
    auto& lm_out = net_lm->stages[0].output_mems[0];
    bm_memcpy_d2d_byte(bm_handle, lm_in, 0, out_mem,
                       (token_length - 1) * hidden_bytes, hidden_bytes);
    net_launch(net_lm);

    int token = 0;
    if (!net_greedy_head && !net_sample_head) {
        bm_memcpy_d2s(bm_handle, &token, lm_out);
    } else if (generation_mode == "greedy") {
        token = greedy_search(lm_out);
    } else {
        token = penalty_sample(lm_out);
    }

    visited_tokens[token_length] = token;
    token_length++;
    history_length++;
    return token;
}

int QwenEngine::forward_next()
{
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

    int token = 0;
    if (lmhead_with_topk) {
        bm_memcpy_d2s(bm_handle, &token, lm_out);
    } else if (!net_greedy_head && !net_sample_head) {
        bm_memcpy_d2s(bm_handle, &token, lm_out);
    } else if (generation_mode == "greedy") {
        token = greedy_search(lm_out);
    } else {
        token = penalty_sample(lm_out);
    }

    visited_tokens[token_length] = token;
    token_length++;
    history_length++;
    return token;
}

void QwenEngine::clear_kv()
{
    if (!support_prefill_kv)
        return;
    for (int i = 0; i < NUM_LAYERS; i++) {
        empty_mem(past_key[i]);
        empty_mem(past_value[i]);
    }
    history_length = 0;
}
