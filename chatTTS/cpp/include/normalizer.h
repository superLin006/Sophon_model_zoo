#pragma once
#include <string>
#include <unordered_map>

// Text normalizer: homophone replacement + ChatTTS [tag] preservation
class Normalizer {
public:
    explicit Normalizer(const std::string& homophones_map_path);

    // Replace mispronounced chars, preserve [tags] unchanged
    std::string normalize(const std::string& text) const;

private:
    std::unordered_map<uint32_t, uint32_t> map_;  // UTF-32 cp → replacement cp

    static std::string process_segment(const std::string& seg,
                                       const std::unordered_map<uint32_t, uint32_t>& map);
    static uint32_t decode_utf8(const char* s, int& out_bytes);
    static void     encode_utf8(uint32_t cp, std::string& out);
};
