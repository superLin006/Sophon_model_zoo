# BM1684X 板卡部署规范 v1.0

---

## 流程概览

```
交叉编译（sophon-cross-build Docker）→ scp 上传 → 板卡运行 → RTF 统计
```

---

## 部署目录结构（板卡上）

```
/home/<user>/{model}/
├── {model}_bm1684          # 可执行文件
├── models/
│   ├── {model}_F32.bmodel
│   ├── {model}_F16.bmodel
│   └── *.txt / *.npy       # 资产文件（vocab、filters 等）
└── test_data/              # 测试音频/图片（可复用其他模型的）
```

---

## 执行步骤

### Step 1: 交叉编译

```bash
# 从仓库根目录执行
bash {model}/cpp/build.sh
# 产物: {model}/cpp/build/{model}_bm1684
```

### Step 2: 上传文件

```bash
BOARD_IP=<ip>
BOARD_USER=<user>
BOARD_PASS=<password>
BOARD_PATH=/home/<user>/{model}

# 创建目录
sshpass -p "${BOARD_PASS}" ssh ${BOARD_USER}@${BOARD_IP} \
    "mkdir -p ${BOARD_PATH}/models"

# 上传二进制（每次重新编译后）
sshpass -p "${BOARD_PASS}" scp \
    {model}/cpp/build/{model}_bm1684 \
    ${BOARD_USER}@${BOARD_IP}:${BOARD_PATH}/

# 上传 bmodel（首次或更新 bmodel 时）
sshpass -p "${BOARD_PASS}" scp \
    {model}/models/BM1684X/{model}_F32.bmodel \
    {model}/models/BM1684X/{model}_F16.bmodel \
    ${BOARD_USER}@${BOARD_IP}:${BOARD_PATH}/models/

# 上传资产文件
sshpass -p "${BOARD_PASS}" scp \
    {model}/models/BM1684X/*.txt \
    ${BOARD_USER}@${BOARD_IP}:${BOARD_PATH}/models/ 2>/dev/null || true
```

### Step 3: 验证上传

```bash
sshpass -p "${BOARD_PASS}" ssh ${BOARD_USER}@${BOARD_IP} "ls -lh ${BOARD_PATH}/"
```

### Step 4: 运行测试

```bash
sshpass -p "${BOARD_PASS}" ssh ${BOARD_USER}@${BOARD_IP} "
cd ${BOARD_PATH}
echo '=== F32 ==='
./{model}_bm1684 models/ /path/to/test.wav F32
echo ''
echo '=== F16 ==='
./{model}_bm1684 models/ /path/to/test.wav F16
"
```

---

## RTF 统计口径

> **只计特征提取 + TPU 推理，不含模型加载**（实际部署时模型预加载到内存）

程序输出格式（必须遵循）：
```
[Timing] audio=<x>ms  feat=<x>ms  infer=<x>ms  total=<x>ms  RTF=<x>
```

---

## 常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| `GLIBC_2.34 not found` | 编译器版本过高 | 确认使用 sophon-cross-build（Ubuntu 20.04 gcc 9.4） |
| `libbmrt.so not found` | rpath 未设置 | CMakeLists.txt 加 `-Wl,-rpath,/opt/sophon/libsophon-0.5.1/lib` |
| `runtime arch[BM1684] != bmodel arch[BM1684X]` | gen_bmodel.sh 中 chip 写错 | 确认 `--chip bm1684x` |
| 推理结果与 Python 不一致 | 预处理逻辑不一致 | 用 save_debug + npy 对比定位差异 |

---

## 性能输出参考（已验证案例）

| 模型 | 精度 | 特征提取 | TPU 推理 | 合计 | RTF |
|------|------|---------|---------|------|-----|
| Whisper Base | F16 | - | - | ~200ms | - |
| SenseVoice Small | F32 | ~34ms | ~155ms | ~189ms | 0.034 |
| SenseVoice Small | F16 | ~34ms | ~20ms | ~54ms | 0.0095 |

---

**版本**: v1.0
