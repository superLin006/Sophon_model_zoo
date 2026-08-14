# Sophon BM1684X Subagent 系统

## Subagent 列表

| Subagent | 版本 | 职责 | 输出文件 |
|---------|------|------|---------|
| **project-initializer** | v1.2 | 项目初始化、环境配置、PyTorch baseline 测试 | `{model}/.context/baseline.md` |
| **operator-analyst** | v1.2 | ONNX 算子兼容性分析、修改方案（参考官方源码改造） | `{model}/.context/operator_analysis.md` |
| **python-converter** | v1.2 | ONNX 导出、bmodel 转换、**板卡 sail Python 推理验证** | `{model}/.context/bmodel_info.md` |
| **cpp-implementer** | v1.2 | C++ 推理实现（复用第三方库）、交叉编译、板卡部署、RTF 测试 | 可执行程序 + 测试结果 |
| **performance-optimizer** | v1.1 | 分模块计时定位瓶颈、KV/批处理/编译参数优化、量化实验 | `{model}/.context/perf_log.md` |

## 设计原则

1. **每个 subagent 自闭环**：做完 → 验证 → 有问题自修 → 确认后返回
2. **做一步验一步**：每个 Step 有明确验证标准和失败修复策略
3. **Python 验证先行**：bmodel 先过板卡 sail 端到端验证，才进 C++——C++ 出错时可排除 bmodel 嫌疑
4. **复用不造轮子**：C++ 优先复用 `1_third_party/`；Python 导出参考官方源码仓库改造
5. **精度优先于性能**：性能优化每步复验精度，退化回退
6. **精简文档**：每个 subagent 只生成必要的 context 文件

## 标准流程

```
1. project-initializer   → baseline.md
2. operator-analyst      → operator_analysis.md
3. python-converter      → bmodel + 板卡 sail 验证通过 + bmodel_info.md
4. cpp-implementer       → 可执行程序 + 板卡测试（含 RTF）
5. performance-optimizer → perf_log.md（优化 + 精度复验）
```

每步完成后主 Agent 向用户报告结果，用户确认后再进行下一步。

## 主 Agent 注入方式

启动 subagent 时注入：项目信息（模型路径/目标精度/官方源码仓库）、**前序 `.context/*.md` 内容**、板卡信息（IP/用户/密码）、用户要求。公共资源路径见下方"配套资源"。

## Context 传递

`{model}/.context/`：`baseline.md → operator_analysis.md → bmodel_info.md → perf_log.md`，主 Agent 逐级读取传递。

## 配套资源（公共，各 subagent 直接使用）

| 资源 | 路径 |
|------|------|
| 知识库 | `.claude/doc/sophon_bm1684_knowledge_base.md` |
| 算子列表 | `.claude/doc/sophon_tpumlir_operators.md` |
| 输出规范 | `.claude/standards/bmodel_output_management.md` |
| 部署规范 | `.claude/standards/board_deploy_workflow.md` |
| SOC SDK | `0_Toolkits/soc-sdk-sp4/` |
| 第三方库（tokenizers-cpp 等） | `1_third_party/` |
| TPU-MLIR Docker | `sophon/tpuc_dev:v3.4-tpumlir-1.28.1`（`3_docker/run_docker.sh`） |
| 交叉编译 Docker | `sophon-cross-build` |
| 参考项目 ASR | `whisper/`、`sensevoice/` |
| sail 上板参考 | `chatTTS/python/ChatTTS/npuengine.py` |
| 性能优化实战 | `Qwen3-TTS/.context/cp_debug_log.md` |

## 版本历史

- v1.2 (2026-08-14): 全系统瘦身——删除代码模板（改为指向参考文件）、公共资源统一到 README、context 模板压缩为字段清单；内容聚焦"判断标准 + 踩坑经验"
- v1.1 (2026-08-14): 新增 performance-optimizer；python-converter 增加板卡 sail 验证步骤；cpp-implementer 强调复用第三方库；工具链更新；网络改造参考官方源码
- v1.0 (2026-05-12): 初始版本，面向 Sophon BM1684X 工具链设计
