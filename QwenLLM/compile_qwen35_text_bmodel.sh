#!/bin/bash
# 将 Qwen3.5-0.8B 的纯文本部分编译为 bmodel（跳过视觉编码器）
#
# 背景：
#   Qwen3.5 是原生 VL 模型（head_dim=256, partial_rotary_factor=0.25）
#   docker v1.28.1-20260429 已有 ChunkGatedDeltaRule 的 BM1684X 降级实现。
#
# 已确认约束：
#   - ChunkGatedDeltaRule 只支持 dynamic codegen（--dynamic 必须）
#   - W4/W8BF16 A16MatMul 在 BM1684X 上要求 N≥某对齐值，Qwen3.5 DeltaNet
#     门控投影 N=16 不满足 → 只能用 --quantize bf16
#   - 最终 bmodel: ~1.5GB BF16 dynamic
#
# 需要三个补丁（均已幂等）：
#   Patch1: Qwen3_5Converter.__init__ 支持 max_pixels=0 → 纯文本路径
#   Patch2: llm_convert.py 绕过 AutoConfig（transformers 4.51.1 不识别 qwen3_5）
#   Patch3: rotary_dim + mrope methods 支持 partial_rotary_factor<1
#
# 用法: bash compile_qwen35_text_bmodel.sh
# 预计耗时: ~30-60 min

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WEIGHTS_DIR="${SCRIPT_DIR}/Qwen3.5-0.8B"
OUTPUT_DIR="${SCRIPT_DIR}/qwen3/qwen35_0.8b_text_seq2048"

if [ ! -f "${WEIGHTS_DIR}/config.json" ]; then
    echo "[ERROR] 权重目录不存在: ${WEIGHTS_DIR}"
    exit 1
fi

echo "[INFO] Qwen3.5-0.8B 文本模式编译 (跳过 ViT)"
docker start sophon-tpumlir

# ── Patch 1: Qwen3_5Converter ──────────────────────────────────────────────
cat > /tmp/qwen35_patch1.py << 'PYEOF'
conv = '/usr/local/lib/python3.10/dist-packages/tpu_mlir/python/llm/Qwen3_5Converter.py'
src = open(conv).read()
if 'max_pixels=0: text-only mode' in src:
    print('[PATCH1] already applied'); exit(0)

OLD = ('        if self.max_pixels == 0 or self.max_pixels % (32 * 32) != 0:\n'
       '            raise RuntimeError(\n'
       '                f"max_pixels values must be multiples of 32*32 and non-zero: {args.max_pixels}")\n'
       '        self.do_vit = True\n'
       '        self.dynamic = True  # force dynamic\n'
       '        self.rmsnorm_type = WeightType.ZEROCENTERED_RMSNORM\n'
       '        # vision config\n'
       '        self.init_vconfig()\n'
       '        self.vit_path = "model.visual"\n'
       '\n'
       '        # extern compiles\n'
       '        self.extern_block_weights = {"mrope_interleave_idx": self.get_mrope_index()}')

NEW = ('        # max_pixels=0: text-only mode, skip ViT (BM1684X supports ChunkGatedDeltaRule)\n'
       '        if self.max_pixels == 0:\n'
       '            self.do_vit = False\n'
       '            self.rmsnorm_type = WeightType.ZEROCENTERED_RMSNORM\n'
       '            self.extern_block_weights = {"mrope_interleave_idx": self.get_mrope_index()}\n'
       '            return\n'
       '        if self.max_pixels % (32 * 32) != 0:\n'
       '            raise RuntimeError(\n'
       '                f"max_pixels values must be multiples of 32*32 and non-zero: {args.max_pixels}")\n'
       '        self.do_vit = True\n'
       '        self.dynamic = True  # force dynamic\n'
       '        self.rmsnorm_type = WeightType.ZEROCENTERED_RMSNORM\n'
       '        # vision config\n'
       '        self.init_vconfig()\n'
       '        self.vit_path = "model.visual"\n'
       '\n'
       '        # extern compiles\n'
       '        self.extern_block_weights = {"mrope_interleave_idx": self.get_mrope_index()}')

if OLD not in src:
    print('[PATCH1 ERROR] target not found'); exit(1)
open(conv, 'w').write(src.replace(OLD, NEW, 1))
print('[PATCH1] Qwen3_5Converter: text-only path added')
PYEOF

docker cp /tmp/qwen35_patch1.py sophon-tpumlir:/tmp/qwen35_patch1.py
docker exec sophon-tpumlir python3 /tmp/qwen35_patch1.py

# ── Patch 2: llm_convert.py ────────────────────────────────────────────────
cat > /tmp/qwen35_patch2.py << 'PYEOF'
lc = '/usr/local/lib/python3.10/dist-packages/tpu_mlir/python/tools/llm_convert.py'
src = open(lc).read()

# Already correctly patched?
if ('qwen3_5 requires transformers' in src and
        src.count('import json as _json') == 1 and
        'IndentationError' not in src):
    print('[PATCH2] already applied correctly'); exit(0)

# Remove any broken previous patch first, restore to original anchor
BROKEN_START = '    # qwen3_5 requires transformers'
CLEAN_ANCHOR  = '\n    if config.model_type in ['
start = src.find(BROKEN_START)
end   = src.find(CLEAN_ANCHOR, max(0, start))
if start != -1 and end != -1:
    src = src[:start] + '    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)' + src[end:]
    print('[PATCH2] removed previous broken patch')

OLD = '    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)'
NEW = ('    # qwen3_5 requires transformers>=4.57 (needs PyTorch 2.2+); load config from JSON directly\n'
       '    import json as _json, os as _os\n'
       '    from types import SimpleNamespace as _NS\n'
       '    def _to_ns(obj):\n'
       '        if isinstance(obj, dict): return _NS(**{k: _to_ns(v) for k, v in obj.items()})\n'
       '        if isinstance(obj, list): return [_to_ns(i) for i in obj]\n'
       '        return obj\n'
       '    _raw = _json.load(open(_os.path.join(args.model_path, "config.json")))\n'
       '    if _raw.get("model_type") == "qwen3_5":\n'
       '        config = _to_ns(_raw)\n'
       '    else:\n'
       '        config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)')

if src.count(OLD) != 1:
    print(f'[PATCH2 ERROR] found {src.count(OLD)} occurrences of anchor'); exit(1)
open(lc, 'w').write(src.replace(OLD, NEW, 1))
import py_compile
try:
    py_compile.compile(lc, doraise=True)
    print('[PATCH2] llm_convert.py patched and syntax OK')
except py_compile.PyCompileError as e:
    print(f'[PATCH2 SYNTAX ERROR] {e}'); exit(1)
PYEOF

docker cp /tmp/qwen35_patch2.py sophon-tpumlir:/tmp/qwen35_patch2.py
docker exec sophon-tpumlir python3 /tmp/qwen35_patch2.py

# ── Patch 3: rotary_dim for partial_rotary_factor ─────────────────────────
cat > /tmp/qwen35_patch3.py << 'PYEOF'
import py_compile

conv = '/usr/local/lib/python3.10/dist-packages/tpu_mlir/python/llm/Qwen3_5Converter.py'
src = open(conv).read()

if 'rotary_dim = int(self.head_dim' in src:
    print('[PATCH3] already applied'); exit(0)

OLD1 = ('        if self.max_pixels == 0:\n'
        '            self.do_vit = False\n'
        '            self.rmsnorm_type = WeightType.ZEROCENTERED_RMSNORM\n'
        '            self.extern_block_weights = {"mrope_interleave_idx": self.get_mrope_index()}\n'
        '            return\n')
NEW1 = ('        if self.max_pixels == 0:\n'
        '            self.do_vit = False\n'
        '            self.rmsnorm_type = WeightType.ZEROCENTERED_RMSNORM\n'
        '            _prf = getattr(self.llm_config, "partial_rotary_factor", 1.0)\n'
        '            self.rotary_dim = int(self.head_dim * _prf)  # 64 for Qwen3.5-0.8B text\n'
        '            _rp = getattr(self.llm_config, "rope_parameters", None)\n'
        '            self.mrope_section = getattr(_rp, "mrope_section", [11, 11, 10])\n'
        '            self.extern_block_weights = {"mrope_interleave_idx": self.get_mrope_index()}\n'
        '            return\n')
if src.count(OLD1) != 1:
    print(f'[PATCH3 ERROR] __init__ anchor not found'); exit(1)
src = src.replace(OLD1, NEW1, 1)

OLD2 = ('        assert (cos.shape[-1] == self.head_dim)\n'
        '        assert (sin.shape[-1] == self.head_dim)\n'
        '        # half\n'
        '        cos = cos[:, :, :self.head_dim // 2]\n'
        '        sin = sin[:, :, :self.head_dim // 2]\n'
        '        return cos.numpy(), sin.numpy()  #[seq, 1, 64]\n')
NEW2 = ('        _prf = getattr(self.llm_config, "partial_rotary_factor", 1.0)\n'
        '        _rdim = int(self.head_dim * _prf)  # e.g. 64 for Qwen3.5 (factor=0.25, head_dim=256)\n'
        '        assert (cos.shape[-1] == _rdim), f"cos.shape[-1]={cos.shape[-1]} != rotary_dim={_rdim}"\n'
        '        assert (sin.shape[-1] == _rdim), f"sin.shape[-1]={sin.shape[-1]} != rotary_dim={_rdim}"\n'
        '        # half\n'
        '        cos = cos[:, :, :_rdim // 2]\n'
        '        sin = sin[:, :, :_rdim // 2]\n'
        '        return cos.numpy(), sin.numpy()  #[seq, 1, rotary_dim//2]\n')
if src.count(OLD2) != 1:
    print(f'[PATCH3 ERROR] rotary_embedding assert anchor not found'); exit(1)
src = src.replace(OLD2, NEW2, 1)

OLD3 = ('        freqs = np.arange(0, 3 * self.head_dim // 2,\n'
        '                          dtype=np.int32).reshape(3, 1, self.head_dim // 2)\n'
        '        freqs_t = self.apply_interleaved_mrope(freqs)  # [1, 64]\n'
        '        freqs_t = np.tile(freqs_t, (1, 2))  # [1, 128]\n'
        '        return freqs_t.astype(np.float32)\n')
NEW3 = ('        _rdim = getattr(self, "rotary_dim", self.head_dim)\n'
        '        _rhalf = _rdim // 2\n'
        '        freqs = np.arange(0, 3 * _rhalf, dtype=np.int32).reshape(3, 1, _rhalf)\n'
        '        freqs_t = self.apply_interleaved_mrope(freqs)  # [1, _rhalf]\n'
        '        freqs_t = np.tile(freqs_t, (1, 2))  # [1, _rdim]\n'
        '        return freqs_t.astype(np.float32)\n')
if src.count(OLD3) != 1:
    print(f'[PATCH3 ERROR] get_mrope_index anchor not found'); exit(1)
src = src.replace(OLD3, NEW3, 1)

OLD4 = ('    def mrope_batch(self, mlir_gen, in_op, name: str):\n'
        '        dim = in_op.type.shape[-1]\n'
        '        weight_op = mlir_gen.create_weight_op(name + ".weight",\n'
        '                                              [self.seq_length, self.head_dim // 2])\n'
        '        in_op = top.GatherOp(mlir_gen.get_tensor_type([self.batch, 3, dim, self.head_dim // 2]),\n'
        '                             weight_op,\n'
        '                             in_op,\n'
        '                             axis=0,\n'
        '                             loc=self.get_loc(name, mlir_gen),\n'
        '                             ip=mlir_gen.insert_point).output\n'
        '        new_op = top.PermuteOp(mlir_gen.get_tensor_type([3, self.head_dim // 2, self.batch, dim]),\n'
        '                               in_op,\n'
        '                               order=[1, 3, 0, 2],\n'
        '                               loc=self.get_loc(name + ".permute", mlir_gen),\n'
        '                               ip=mlir_gen.insert_point).output\n'
        '        new_op = top.ReshapeOp(mlir_gen.get_tensor_type([3 * self.head_dim // 2, self.batch, dim]),\n'
        '                               new_op,\n'
        '                               shape=[3 * self.head_dim // 2, self.batch, -1],\n'
        '                               loc=self.get_loc(name + ".reshape", mlir_gen),\n'
        '                               ip=mlir_gen.insert_point).output\n'
        '        weight_op = mlir_gen.create_weight_op("mrope_interleave_idx", [1, self.head_dim])\n'
        '        new_op = top.GatherOp(mlir_gen.get_tensor_type([1, self.head_dim, self.batch, dim]),\n'
        '                              new_op,\n'
        '                              weight_op,\n'
        '                              axis=0,\n'
        '                              loc=self.get_loc(name + ".gather", mlir_gen),\n'
        '                              ip=mlir_gen.insert_point).output\n'
        '        new_op = top.PermuteOp(mlir_gen.get_tensor_type([self.batch, dim, 1, self.head_dim]),\n'
        '                               new_op,\n'
        '                               order=[2, 3, 0, 1],\n'
        '                               loc=self.get_loc(name + ".permute2", mlir_gen),\n'
        '                               ip=mlir_gen.insert_point).output\n'
        '        return new_op\n')
NEW4 = ('    def mrope_batch(self, mlir_gen, in_op, name: str):\n'
        '        _rdim = getattr(self, "rotary_dim", self.head_dim)\n'
        '        _rhalf = _rdim // 2\n'
        '        dim = in_op.type.shape[-1]\n'
        '        weight_op = mlir_gen.create_weight_op(name + ".weight",\n'
        '                                              [self.seq_length, _rhalf])\n'
        '        in_op = top.GatherOp(mlir_gen.get_tensor_type([self.batch, 3, dim, _rhalf]),\n'
        '                             weight_op,\n'
        '                             in_op,\n'
        '                             axis=0,\n'
        '                             loc=self.get_loc(name, mlir_gen),\n'
        '                             ip=mlir_gen.insert_point).output\n'
        '        new_op = top.PermuteOp(mlir_gen.get_tensor_type([3, _rhalf, self.batch, dim]),\n'
        '                               in_op,\n'
        '                               order=[1, 3, 0, 2],\n'
        '                               loc=self.get_loc(name + ".permute", mlir_gen),\n'
        '                               ip=mlir_gen.insert_point).output\n'
        '        new_op = top.ReshapeOp(mlir_gen.get_tensor_type([3 * _rhalf, self.batch, dim]),\n'
        '                               new_op,\n'
        '                               shape=[3 * _rhalf, self.batch, -1],\n'
        '                               loc=self.get_loc(name + ".reshape", mlir_gen),\n'
        '                               ip=mlir_gen.insert_point).output\n'
        '        weight_op = mlir_gen.create_weight_op("mrope_interleave_idx", [1, _rdim])\n'
        '        new_op = top.GatherOp(mlir_gen.get_tensor_type([1, _rdim, self.batch, dim]),\n'
        '                              new_op,\n'
        '                              weight_op,\n'
        '                              axis=0,\n'
        '                              loc=self.get_loc(name + ".gather", mlir_gen),\n'
        '                              ip=mlir_gen.insert_point).output\n'
        '        new_op = top.PermuteOp(mlir_gen.get_tensor_type([self.batch, dim, 1, _rdim]),\n'
        '                               new_op,\n'
        '                               order=[2, 3, 0, 1],\n'
        '                               loc=self.get_loc(name + ".permute2", mlir_gen),\n'
        '                               ip=mlir_gen.insert_point).output\n'
        '        return new_op\n')
if src.count(OLD4) != 1:
    print(f'[PATCH3 ERROR] mrope_batch anchor not found'); exit(1)
src = src.replace(OLD4, NEW4, 1)

OLD5 = ('    def mrope(self, mlir_gen, in_op, name: str):\n'
        '        dim = in_op.type.shape[-1]\n'
        '        weight_op = mlir_gen.create_weight_op(name + ".weight",\n'
        '                                              [self.seq_length, 1, self.head_dim // 2])\n'
        '        in_op = top.GatherOp(mlir_gen.get_tensor_type([3, dim, 1, self.head_dim // 2]),\n'
        '                             weight_op,\n'
        '                             in_op,\n'
        '                             axis=0,\n'
        '                             loc=self.get_loc(name, mlir_gen),\n'
        '                             ip=mlir_gen.insert_point).output\n'
        '        new_op = top.PermuteOp(mlir_gen.get_tensor_type([3, self.head_dim // 2, 1, dim]),\n'
        '                               in_op,\n'
        '                               order=[0, 3, 2, 1],\n'
        '                               loc=self.get_loc(name + ".permute", mlir_gen),\n'
        '                               ip=mlir_gen.insert_point).output\n'
        '        new_op = top.ReshapeOp(mlir_gen.get_tensor_type([3 * self.head_dim // 2, 1, dim]),\n'
        '                               new_op,\n'
        '                               shape=[3 * self.head_dim // 2, 1, -1],\n'
        '                               loc=self.get_loc(name + ".reshape", mlir_gen),\n'
        '                               ip=mlir_gen.insert_point).output\n'
        '        weight_op = mlir_gen.create_weight_op("mrope_interleave_idx", [1, self.head_dim])\n'
        '        new_op = top.GatherOp(mlir_gen.get_tensor_type([1, self.head_dim, 1, dim]),\n'
        '                              new_op,\n'
        '                              weight_op,\n'
        '                              axis=0,\n'
        '                              loc=self.get_loc(name + ".gather", mlir_gen),\n'
        '                              ip=mlir_gen.insert_point).output\n'
        '        new_op = top.PermuteOp(mlir_gen.get_tensor_type([1, dim, 1, self.head_dim]),\n'
        '                               new_op,\n'
        '                               order=[0, 3, 2, 1],\n'
        '                               loc=self.get_loc(name + ".permute2", mlir_gen),\n'
        '                               ip=mlir_gen.insert_point).output\n'
        '        return new_op\n')
NEW5 = ('    def mrope(self, mlir_gen, in_op, name: str):\n'
        '        _rdim = getattr(self, "rotary_dim", self.head_dim)\n'
        '        _rhalf = _rdim // 2\n'
        '        dim = in_op.type.shape[-1]\n'
        '        weight_op = mlir_gen.create_weight_op(name + ".weight",\n'
        '                                              [self.seq_length, 1, _rhalf])\n'
        '        in_op = top.GatherOp(mlir_gen.get_tensor_type([3, dim, 1, _rhalf]),\n'
        '                             weight_op,\n'
        '                             in_op,\n'
        '                             axis=0,\n'
        '                             loc=self.get_loc(name, mlir_gen),\n'
        '                             ip=mlir_gen.insert_point).output\n'
        '        new_op = top.PermuteOp(mlir_gen.get_tensor_type([3, _rhalf, 1, dim]),\n'
        '                               in_op,\n'
        '                               order=[0, 3, 2, 1],\n'
        '                               loc=self.get_loc(name + ".permute", mlir_gen),\n'
        '                               ip=mlir_gen.insert_point).output\n'
        '        new_op = top.ReshapeOp(mlir_gen.get_tensor_type([3 * _rhalf, 1, dim]),\n'
        '                               new_op,\n'
        '                               shape=[3 * _rhalf, 1, -1],\n'
        '                               loc=self.get_loc(name + ".reshape", mlir_gen),\n'
        '                               ip=mlir_gen.insert_point).output\n'
        '        weight_op = mlir_gen.create_weight_op("mrope_interleave_idx", [1, _rdim])\n'
        '        new_op = top.GatherOp(mlir_gen.get_tensor_type([1, _rdim, 1, dim]),\n'
        '                              new_op,\n'
        '                              weight_op,\n'
        '                              axis=0,\n'
        '                              loc=self.get_loc(name + ".gather", mlir_gen),\n'
        '                              ip=mlir_gen.insert_point).output\n'
        '        new_op = top.PermuteOp(mlir_gen.get_tensor_type([1, dim, 1, _rdim]),\n'
        '                               new_op,\n'
        '                               order=[0, 3, 2, 1],\n'
        '                               loc=self.get_loc(name + ".permute2", mlir_gen),\n'
        '                               ip=mlir_gen.insert_point).output\n'
        '        return new_op\n')
if src.count(OLD5) != 1:
    print(f'[PATCH3 ERROR] mrope anchor not found'); exit(1)
src = src.replace(OLD5, NEW5, 1)

open(conv, 'w').write(src)
try:
    py_compile.compile(conv, doraise=True)
    print('[PATCH3] Qwen3_5Converter: rotary_dim support added, syntax OK')
except py_compile.PyCompileError as e:
    print(f'[PATCH3 SYNTAX ERROR] {e}'); exit(1)
PYEOF

docker cp /tmp/qwen35_patch3.py sophon-tpumlir:/tmp/qwen35_patch3.py
docker exec sophon-tpumlir python3 /tmp/qwen35_patch3.py

# ── Compile ────────────────────────────────────────────────────────────────
docker exec sophon-tpumlir bash -c "
    set -e
    pip3 install 'transformers==4.51.1' -q 2>/dev/null || true
    echo '[docker] TPU-MLIR:' && llm_convert.py --version
    echo '[docker] 开始编译 Qwen3.5-0.8B 纯文本 (BF16 dynamic)...'
    # Note: W4BF16 fails due to BM1684X A16MatMul constraint (N=16 < min) for DeltaNet gate projections
    # BF16+dynamic works: ChunkGatedDeltaRule only supports dynamic mode, no quantization constraint
    llm_convert.py \
        -m /workspace/QwenLLM/Qwen3.5-0.8B \
        -s 2048 \
        --quantize bf16 \
        -c bm1684x \
        --max_pixels 0 \
        --dynamic \
        --out_dir /workspace/QwenLLM/qwen3/qwen35_0.8b_text_seq2048
    echo '[docker] 完成:'
    ls -lh /workspace/QwenLLM/qwen3/qwen35_0.8b_text_seq2048/*.bmodel 2>/dev/null || \
    ls /workspace/QwenLLM/qwen3/qwen35_0.8b_text_seq2048/
"

echo "[INFO] 本地 bmodel:"
ls -lh "${OUTPUT_DIR}"/*.bmodel 2>/dev/null || ls "${OUTPUT_DIR}/"
echo ""
echo "[INFO] 下一步: bash ${SCRIPT_DIR}/deploy_to_board.sh"
