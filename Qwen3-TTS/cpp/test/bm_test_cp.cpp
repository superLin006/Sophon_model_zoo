// 板卡单测：加载 f32cache cp bmodel 的 cp_block_cache_<l>，用 Python dump 的真实输入
// 跑单层，输出 hidden/新KV 供本机与 ONNX/PyTorch 对比。
// 用法: bm_test_cp <bmodel> <net_name> <in_dir> <out_dir>
#include <bmruntime_interface.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static std::vector<char> read_file(const std::string& p) {
    FILE* f = fopen(p.c_str(), "rb");
    if (!f) { fprintf(stderr, "read fail %s\n", p.c_str()); exit(1); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<char> v(n); if (n) fread(v.data(), 1, n, f);
    fclose(f); return v;
}

static void write_file(const std::string& p, const void* data, size_t n) {
    FILE* f = fopen(p.c_str(), "wb");
    if (!f) { fprintf(stderr, "write fail %s\n", p.c_str()); exit(1); }
    fwrite(data, 1, n, f); fclose(f);
}

int main(int argc, char** argv) {
    // 可选前置加载（模拟 C++ 多 bmodel 环境）：bm_test_cp <bmodel> <net> <in> <out> [extra_bmodel...]
    // chain 模式: bm_test_cp <bmodel> chain <in_dir> <out_dir>
    if (argc == 5 && std::string(argv[2]) == "chain") {
        bm_handle_t handle;
        if (bm_dev_request(&handle, 0) != BM_SUCCESS) { fprintf(stderr, "dev fail\n"); return 1; }
        void* rt = bmrt_create(handle);
        if (!rt || !bmrt_load_bmodel(rt, argv[1])) { fprintf(stderr, "load fail\n"); return 1; }
        std::vector<char> pos_raw = read_file(std::string(argv[3]) + "/position_ids.bin");
        std::vector<int32_t> pos32(pos_raw.size() / 8);
        for (size_t j = 0; j < pos32.size(); j++) pos32[j] = (int32_t)((int64_t*)pos_raw.data())[j];
        std::vector<char> mask_raw = read_file(std::string(argv[3]) + "/attention_mask.bin");
        std::vector<char> hid = read_file(std::string(argv[3]) + "/input_states.bin");
        for (int l = 0; l < 5; l++) {
            std::string nm = "cp_block_" + std::to_string(l);
            const bm_net_info_t* net = bmrt_get_network_info(rt, nm.c_str());
            if (!net) { fprintf(stderr, "net %s not found\n", nm.c_str()); return 1; }
            std::vector<bm_tensor_t> ins(net->input_num), outs(net->output_num);
            for (int i = 0; i < net->input_num; i++) {
                bmrt_tensor(&ins[i], rt, net->input_dtypes[i], net->stages[0].input_shapes[i]);
                std::string in = net->input_names[i];
                if (in == "input_states") bm_memcpy_s2d(handle, ins[i].device_mem, hid.data());
                else if (in == "position_ids") bm_memcpy_s2d(handle, ins[i].device_mem, pos32.data());
                else if (in == "attention_mask") bm_memcpy_s2d(handle, ins[i].device_mem, mask_raw.data());
            }
            for (int i = 0; i < net->output_num; i++)
                bmrt_tensor(&outs[i], rt, net->output_dtypes[i], net->stages[0].output_shapes[i]);
            bool ok = bmrt_launch_tensor_ex(rt, net->name, ins.data(), ins.size(), outs.data(), outs.size(), true, false);
            if (ok) bm_thread_sync(handle);
            fprintf(stderr, "L%d launch %s\n", l, ok ? "OK" : "FAIL");
            if (!ok) return 1;
            // out0 hidden → 下一层; out1/out2 KV → 文件
            size_t hb = bm_mem_get_device_size(outs[0].device_mem);
            hid.resize(hb);
            bm_memcpy_d2s(handle, hid.data(), outs[0].device_mem);
            size_t kb = bm_mem_get_device_size(outs[1].device_mem);
            std::vector<char> kbuf(kb), vbuf(kb);
            bm_memcpy_d2s(handle, kbuf.data(), outs[1].device_mem);
            bm_memcpy_d2s(handle, vbuf.data(), outs[2].device_mem);
            write_file(std::string(argv[4]) + "/k" + std::to_string(l) + ".bin", kbuf.data(), kb);
            write_file(std::string(argv[4]) + "/v" + std::to_string(l) + ".bin", vbuf.data(), kb);
            for (int i = 0; i < net->input_num; i++) bm_free_device(handle, ins[i].device_mem);
            for (int i = 0; i < net->output_num; i++) bm_free_device(handle, outs[i].device_mem);
        }
        // 最后一层 hidden
        write_file(std::string(argv[4]) + "/hidden_final.bin", hid.data(), hid.size());
        bmrt_destroy(rt); bm_dev_free(handle);
        return 0;
    }
    if (argc < 5) { fprintf(stderr, "usage: %s <bmodel> <net|chain> <in_dir> <out_dir>\n", argv[0]); return 1; }
    bm_handle_t handle;
    if (bm_dev_request(&handle, 0) != BM_SUCCESS) { fprintf(stderr, "dev fail\n"); return 1; }
    void* rt = bmrt_create(handle);
    if (!rt || !bmrt_load_bmodel(rt, argv[1])) { fprintf(stderr, "load fail\n"); return 1; }
    // 额外加载（模拟 C++ 多 bmodel 环境，不执行）
    for (int a = 5; a < argc; a++) {
        void* extra_rt = bmrt_create(handle);
        if (!extra_rt || !bmrt_load_bmodel(extra_rt, argv[a]))
            fprintf(stderr, "extra load fail: %s\n", argv[a]);
        else
            fprintf(stderr, "extra loaded: %s\n", argv[a]);
    }
    const bm_net_info_t* net = bmrt_get_network_info(rt, argv[2]);
    if (!net) { fprintf(stderr, "net %s not found\n", argv[2]); return 1; }

    // 输入文件名按 net 输入名匹配：input_states/position_ids/attention_mask/history_k/history_v.bin
    // position_ids bin 为 int64，bmodel 输入 int32 → 转
    std::vector<bm_tensor_t> ins(net->input_num), outs(net->output_num);
    for (int i = 0; i < net->input_num; i++) {
        bmrt_tensor(&ins[i], rt, net->input_dtypes[i], net->stages[0].input_shapes[i]);
        size_t bytes = bm_mem_get_device_size(ins[i].device_mem);
        std::string nm = net->input_names[i];
        std::vector<char> raw = read_file(std::string(argv[3]) + "/" + nm + ".bin");
        if (nm == "position_ids") {
            std::vector<int32_t> pos32(raw.size() / 8);
            for (size_t j = 0; j < pos32.size(); j++) pos32[j] = (int32_t)((int64_t*)raw.data())[j];
            bm_memcpy_s2d(handle, ins[i].device_mem, pos32.data());
        } else {
            bm_memcpy_s2d(handle, ins[i].device_mem, raw.data());
        }
        fprintf(stderr, "in %s @%llx sz=%u (host %zu)\n", nm.c_str(),
                ins[i].device_mem.u.device.device_addr, bm_mem_get_device_size(ins[i].device_mem), raw.size());
    }
    for (int i = 0; i < net->output_num; i++)
        bmrt_tensor(&outs[i], rt, net->output_dtypes[i], net->stages[0].output_shapes[i]);

    bool ok = bmrt_launch_tensor_ex(rt, net->name, ins.data(), ins.size(), outs.data(), outs.size(), true, false);
    if (ok) bm_thread_sync(handle);
    fprintf(stderr, "launch %s\n", ok ? "OK" : "FAIL");
    if (!ok) return 1;

    for (int i = 0; i < net->output_num; i++) {
        size_t bytes = bm_mem_get_device_size(outs[i].device_mem);
        std::vector<char> buf(bytes);
        bm_memcpy_d2s(handle, buf.data(), outs[i].device_mem);
        std::string onm = net->output_names[i];
        std::string base = std::string(argv[4]) + "/out" + std::to_string(i) + "_" + onm;
        // 简化文件名
        base = std::string(argv[4]) + "/out" + std::to_string(i);
        write_file(base + ".bin", buf.data(), bytes);
        fprintf(stderr, "out%d %s bytes=%zu\n", i, onm.c_str(), bytes);
    }
    for (int i = 0; i < net->input_num; i++) bm_free_device(handle, ins[i].device_mem);
    for (int i = 0; i < net->output_num; i++) bm_free_device(handle, outs[i].device_mem);
    bmrt_destroy(rt); bm_dev_free(handle);
    return 0;
}
