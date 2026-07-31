#include "manifest.h"
#include <cstdlib>
#include <fstream>
#include <limits>
#include <regex>
#include <sstream>

namespace {

std::string ReadFile(const std::string& path) {
    std::ifstream f(path.c_str());
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// Extract the text between matching '{' and '}' starting at position pos.
std::string ExtractBlock(const std::string& s, size_t pos, char open,
                         char close) {
    pos = s.find(open, pos);
    if (pos == std::string::npos) return {};
    int depth = 0;
    for (size_t i = pos; i < s.size(); ++i) {
        if (s[i] == open) ++depth;
        else if (s[i] == close && --depth == 0)
            return s.substr(pos, i - pos + 1);
    }
    return {};
}

std::string FieldStr(const std::string& json, const char* key,
                     const std::string& default_val = "") {
    std::regex re(std::string("\"") + key + "\"\\s*:\\s*\"([^\"]*)\"");
    std::smatch m;
    return std::regex_search(json, m, re) ? m[1].str() : default_val;
}

int FieldInt(const std::string& json, const char* key, int default_val = 0) {
    std::regex re(std::string("\"") + key + "\"\\s*:\\s*(-?[0-9]+)");
    std::smatch m;
    return std::regex_search(json, m, re)
               ? std::atoi(m[1].str().c_str())
               : default_val;
}

double FieldNum(const std::string& json, const char* key, double default_val) {
    std::regex re(
        std::string("\"") + key +
        "\"\\s*:\\s*(-?[0-9]+(?:\\.[0-9]+)?)");
    std::smatch m;
    return std::regex_search(json, m, re)
               ? std::atof(m[1].str().c_str())
               : default_val;
}

std::vector<int64_t> ParseShape(const std::string& json) {
    size_t pos = json.find("\"shape\"");
    std::string arr = ExtractBlock(json, pos, '[', ']');
    std::vector<int64_t> v;
    std::regex re("-?[0-9]+");
    auto begin = std::sregex_iterator(arr.begin(), arr.end(), re);
    auto end = std::sregex_iterator();
    for (auto it = begin; it != end; ++it)
        v.push_back(std::atoll(it->str().c_str()));
    return v;
}

void ParseTensors(const std::string& section, const char* direction,
                  std::vector<TensorSpec>* out) {
    size_t pos = section.find(std::string("\"") + direction + "\"");
    std::string arr = ExtractBlock(section, pos, '[', ']');
    for (size_t i = 0; i < arr.size();) {
        size_t b = arr.find('{', i);
        if (b == std::string::npos) break;
        std::string obj = ExtractBlock(arr, b, '{', '}');

        TensorSpec t;
        t.name = FieldStr(obj, "name");
        t.index = FieldInt(obj, "index", static_cast<int>(out->size()));
        t.dtype = FieldStr(obj, "dtype");
        t.runtime_dtype = FieldStr(obj, "runtime_dtype", t.dtype);
        t.shape = ParseShape(obj);

        // Validate.
        bool ok = !t.name.empty() && !t.dtype.empty() && !t.shape.empty();
        size_t bytes = 1;
        for (auto d : t.shape) {
            if (d <= 0 || bytes > std::numeric_limits<size_t>::max() /
                                    static_cast<size_t>(d)) {
                ok = false;
                break;
            }
            bytes *= static_cast<size_t>(d);
        }
        if (t.dtype != "float32" && t.dtype != "float16" &&
            t.dtype != "int32" && t.dtype != "int64" &&
            t.dtype != "int8" && t.dtype != "uint8")
            ok = false;
        for (const auto& existing : *out)
            if (existing.name == t.name || existing.index == t.index) ok = false;
        if (!ok) {
            out->clear();
            return;
        }
        t.bytes = bytes;
        out->push_back(t);
        i = b + obj.size();
    }
}

}  // namespace

bool LoadManifest(const std::string& path, Manifest* manifest,
                  std::string* error) {
    *manifest = Manifest();
    std::string json = ReadFile(path);
    if (json.empty()) {
        if (error) *error = "cannot read manifest";
        return false;
    }

    // Parse each network's tensor specs.
    size_t pos = json.find("\"networks\"");
    std::string nets = ExtractBlock(json, pos, '{', '}');
    for (const char* name : {"encoder", "decoder", "joiner"}) {
        size_t x = nets.find(std::string("\"") + name + "\"");
        std::string section = ExtractBlock(nets, x, '{', '}');
        NetworkSpec net;
        net.name = name;
        ParseTensors(section, "inputs", &net.inputs);
        ParseTensors(section, "outputs", &net.outputs);
        if (net.inputs.empty() || net.outputs.empty()) {
            if (error) *error = "invalid network manifest";
            return false;
        }
        // Indices must be unique and contiguous starting from 0.
        for (const auto& vec : {net.inputs, net.outputs}) {
            for (size_t i = 0; i < vec.size(); ++i) {
                if (vec[i].index != static_cast<int>(i)) {
                    if (error)
                        *error =
                            "manifest indices must be unique and contiguous";
                    return false;
                }
            }
        }
        manifest->networks.push_back(net);
    }

    // Extract protocol constants.
    manifest->sample_rate = FieldInt(json, "sample_rate", 16000);
    manifest->n_mels = FieldInt(json, "n_mels", 80);
    manifest->segment = FieldInt(json, "segment", 103);
    manifest->offset = FieldInt(json, "offset", 96);
    manifest->tail_seconds =
        static_cast<float>(FieldNum(json, "tail_padding_seconds", 1.03));
    manifest->blank = FieldInt(json, "blank_id", 0);
    manifest->unk = FieldInt(json, "unk_id", 2);
    manifest->context = FieldInt(json, "context_size", 2);
    manifest->vocab_size = FieldInt(json, "vocab_size", 6254);

    // Validate fixed protocol constants.
    if (manifest->sample_rate != 16000 || manifest->n_mels != 80 ||
        manifest->segment != 103 || manifest->offset != 96 ||
        manifest->context != 2 || manifest->vocab_size != 6254) {
        if (error) *error = "manifest frontend/decoding protocol mismatch";
        return false;
    }

    // Final per-tensor validation.
    for (const auto& net : manifest->networks) {
        for (const auto& t : net.inputs) {
            if (t.index < 0 || t.shape.empty()) return false;
            for (auto d : t.shape)
                if (d <= 0) {
                    if (error) *error = "manifest shape must be positive";
                    return false;
                }
        }
        for (const auto& t : net.outputs) {
            for (auto d : t.shape) {
                if (d <= 0) {
                    if (error) *error = "manifest shape must be positive";
                    return false;
                }
            }
        }
    }
    return true;
}
