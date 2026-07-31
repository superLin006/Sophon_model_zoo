#include "manifest.h"
#include "audio_frontend.h"
#include "tokenizer.h"
#include "zipformer.h"

#include <cassert>
#include <fstream>
#include <iostream>

// Test-only runtime that records call counts and produces deterministic
// logits so we can verify the greedy loop.
struct ControlRuntime : FakeRuntime {
    int joins = 0;
    bool Joiner(const std::vector<float>&, const std::vector<float>&,
                std::vector<float>* out, std::string*) override {
        ++joins;
        out->assign(6254, 0.0f);
        out->at(joins % 2 ? 7 : 0) = 1.0f;  // alternate best=7, best=0
        return true;
    }
};

int main() {
    Manifest m;
    std::string e;
    if (!LoadManifest(ZIPFORMER_TEST_MANIFEST, &m, &e)) {
        std::cerr << e << "\n";
        return 1;
    }

    // Manifest must describe 3 networks with 36 encoder I/Os.
    assert(m.networks.size() == 3);
    assert(m.networks[0].inputs.size() == 36);
    assert(m.networks[0].outputs.size() == 36);

    // Streaming chunk splitting.
    auto c = zipformer::MakeFeatureChunks(
        std::vector<float>(689 * 80), 689, 80, 103, 96);
    assert(c.size() == 8 && c[0].size() == 103 * 80);
    assert(zipformer::MakeFeatureChunks(
               std::vector<float>(664 * 80), 664, 80, 103, 96)
               .size() == 7);

    // Tokenizer.
    std::ofstream tf("tokens_test.txt");
    tf << "blank 0\n▁hello 1\nworld 2\n";
    tf.close();
    Tokenizer t;
    assert(t.Load("tokens_test.txt", &e));
    assert(t.Decode({1, 2, 0}, 0, 2) == "hello");

    // Greedy decode: 1 chunk of 103 frames → 24 encoder output frames.
    // ControlRuntime alternates argmax between 7 and 0:
    //   non-blank tokens every other frame → 12 tokens accepted.
    //   Decoder called once initially + 12 times on token accept = 13.
    //   Joiner called 24 times (once per frame).
    ControlRuntime rt;
    Zipformer z;
    assert(z.Init(ZIPFORMER_TEST_MANIFEST, &rt, &e));
    std::vector<int> ids;
    assert(z.Greedy({std::vector<float>(16480)}, &ids, &e));
    assert(rt.decoder_calls == 13 && rt.joins == 24);

    std::remove("tokens_test.txt");
    std::cout << "host tests passed\n";
    return 0;
}
