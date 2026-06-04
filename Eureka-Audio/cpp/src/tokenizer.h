#pragma once
// Qwen3 tokenizer for Eureka-Audio C++ inference
// 使用 tiktoken BPE，与 Python 侧 tokenizer.json 完全对齐
// 仅需要：
//   1. 将系统提示词 + 音频占位符模板 → token_ids（用于拼 prefix_embeds）
//   2. 将生成的 token_ids → 文本（解码输出）

#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>

namespace eureka {

class Qwen3Tokenizer {
public:
    Qwen3Tokenizer() = default;

    // 从模型目录加载 tokenizer.json
    bool load(const std::string& model_dir);

    // 编码（不添加特殊 token）
    std::vector<int> encode(const std::string& text) const;

    // 解码（跳过特殊 token）
    std::string decode(const std::vector<int>& ids) const;

    // 特殊 token ID
    int eos_token_id()  const { return eos_id_; }
    int bos_token_id()  const { return bos_id_; }
    int pad_token_id()  const { return pad_id_; }

    // 应用 Qwen3 对话模板（无 thinking 模式），返回完整 token_id 序列
    // messages: [{role, content}, ...]
    // 末尾自动添加 <|im_start|>assistant\n
    std::vector<int> apply_chat_template(
        const std::vector<std::pair<std::string,std::string>>& messages) const;

private:
    // BPE 词表：token 字符串 → id
    std::unordered_map<std::string, int> vocab_;
    // id → token 字符串
    std::vector<std::string> id_to_token_;
    // BPE 合并规则（rank → pair）
    std::vector<std::pair<std::string,std::string>> merges_;

    int eos_id_ = 151645;   // <|im_end|>
    int bos_id_ = 151643;   // <|im_start|>
    int pad_id_ = 151643;

    // 内部 BPE 编码
    std::vector<int> bpe_encode(const std::string& text) const;

    // 简单 UTF-8 字节编码（类 tiktoken）
    static std::string bytes_to_unicode_str(unsigned char c);
    bool loaded_ = false;
};

}  // namespace eureka
