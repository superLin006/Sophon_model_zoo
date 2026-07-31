#pragma once
#include "zipformer.h"
#ifdef ZIPFORMER_WITH_BM
#include "bmruntime_interface.h"
#include <vector>
class BmRuntime final : public Runtime {
 public:
  BmRuntime(); ~BmRuntime();
  bool Load(const std::string&, const std::string&, const std::string&, std::string*);
  bool Validate(const Manifest&, std::string*) override;
  bool Reset(std::string*) override;
  bool Encoder(const std::vector<float>&, std::vector<float>*, std::string*) override;
  bool Decoder(const int64_t[2], std::vector<float>*, std::string*) override;
  bool Joiner(const std::vector<float>&, const std::vector<float>&, std::vector<float>*, std::string*) override;
 private:
  struct Buffer { bm_device_mem_t mem{}; size_t bytes=0; bool owned=false; };
  bm_handle_t handle_=nullptr; void* runtime_[3]{}; const bm_net_info_t* info_[3]{};
  std::string graph_[3]; Manifest manifest_; bool validated_=false;
  Buffer enc_x_, enc_out_, dec_in_, dec_out_, join_enc_, join_dec_, join_out_;
  Buffer state_[2][35]; int state_count_=0, state_slot_=0;
  int state_in_[36]{}, state_out_[36]{};
  static bool dtype(const std::string&, bm_data_type_t*); static bool shape(const TensorSpec&, bm_shape_t*);
  static size_t elements(const TensorSpec&); static size_t type_bytes(const std::string&);
  void release(Buffer*); void cleanup(); bool alloc(Buffer*, size_t, std::string*);
  bool alloc_buffers(std::string*); bool launch(int, std::vector<bm_tensor_t>*, std::vector<bm_tensor_t>*, std::string*);
  const NetworkSpec* net(const char*) const;
};
#endif
