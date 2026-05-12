# Sophon BM1684X Subagent 系统

## Subagent 列表

| Subagent | 版本 | 职责 | 输出文件 |
|---------|------|------|---------|
| **project-initializer** | v1.0 | 项目初始化、环境配置、PyTorch baseline 测试 | `{model}/.context/baseline.md` |
| **operator-analyst** | v1.0 | ONNX 算子兼容性分析、具体修改方案 | `{model}/.context/operator_analysis.md` |
| **python-converter** | v1.0 | ONNX 导出、bmodel 转换（F32+F16）、精度验证 | `{model}/.context/bmodel_info.md` |
| **cpp-implementer** | v1.0 | C++ 推理实现、交叉编译、板卡部署、RTF 测试 | 最终可执行程序 + 测试结果 |

## 设计原则

1. **每个 subagent 自闭环**：做完自己的事 → 自己验证 → 有问题自己修 → 确认无误后返回
2. **做一步验一步**：每个 Step 都有明确的验证标准和失败修复策略
3. **算子分析独立**：operator-analyst 独立存在，避免转换失败时回溯代价过大
4. **Context 传递**：通过 `.context/` 目录的 md 文件在 subagent 间传递信息
5. **精简文档**：每个 subagent 只生成必要的 context 文件，不产生冗余文档

## 标准流程

```
1. project-initializer
   输入: 模型路径、Conda 环境
   输出: .context/baseline.md（PyTorch 基线结果 + 输入输出 shape）

2. operator-analyst
   输入: baseline.md、ONNX 模型
   输出: .context/operator_analysis.md（兼容性报告 + 修改方案）

3. python-converter
   输入: operator_analysis.md、SDK 路径、目标精度
   输出: bmodel（F32+F16）、.context/bmodel_info.md

4. cpp-implementer
   输入: bmodel_info.md、Python debug 输出、板卡信息
   输出: 可执行程序 + 板卡测试结果（含 RTF）
```

每步完成后主 Agent 向用户报告结果，用户确认后再进行下一步。

## 信息注入机制

### 用户提示词中通常包含的信息

| 信息类型 | 示例 | 接收的 subagent |
|---------|------|---------------|
| 模型路径 | `whisper/` | 全部 |
| 模型信息 | "Whisper base，带 KV cache" | 全部 |
| Conda 环境 | `sophon-export` | project-initializer, python-converter |
| 目标精度 | `F32 + F16` | python-converter, cpp-implementer |
| 板卡信息 | IP / 用户名 / 密码 | cpp-implementer |
| 参考项目 | `Sophon_model_zoo/whisper/` | operator-analyst, cpp-implementer |
| 特殊要求 | "固定输入 shape"、"KV cache 分步导出" | operator-analyst, python-converter |

### 主 Agent 注入方式

```
启动 {subagent_name}，注入以下信息：

## 项目信息
- 模型: <模型名>（<简要描述>）
- 仓库路径: /home/xh/itc_project/Sophon_model_zoo/<model>/
- 目标精度: F32 + F16

## 环境
- Conda 环境: sophon-export
- TPU-MLIR Docker: sophgo/tpuc_dev:latest
- 交叉编译 Docker: sophon-cross-build

## 资源路径
- SOC SDK: Sophon_model_zoo/0_Toolkits/soc-sdk-sp4/
- tpu_mlir whl: Sophon_model_zoo/0_Toolkits/tpu_mlir*.whl
- 第三方库: Sophon_model_zoo/1_third_party/
- 参考项目: Sophon_model_zoo/whisper/ 或 sensevoice/

## 板卡信息（cpp-implementer 用）
- IP: <ip>  用户: <user>  密码: <pwd>
- 部署路径: /home/<user>/<model>/

## 前序 Context
{读取 .context/*.md 文件内容}

## 你的任务
{从 subagent 模板中复制任务描述}
```

## Context 传递机制

### 目录结构
```
{model}/.context/
├── baseline.md           # project-initializer 生成
├── operator_analysis.md  # operator-analyst 生成
└── bmodel_info.md        # python-converter 生成
```

### 传递流程
```
用户上下文 → 主 Agent 注入 → subagent 执行 → 生成 .context/md → 主 Agent 读取并传递给下个 subagent
```

## 用户提示词模板

```
<模型名> 移植到 Sophon BM1684X 推理测试

查看 Sophon_model_zoo/.claude/subagents/README.md 根据 README.md 完成任务

1. 上下文

目标：将 <模型名> 移植到 BM1684X，通过 scp 部署到板卡完成测试

历史与计划：分三个大步骤
- 第一步（Python 端）：PyTorch → ONNX → bmodel（F32+F16）
  转换脚本：export_onnx.py、gen_bmodel.sh
  测试脚本：test_pytorch.py（baseline）、test_onnx.py
- 第二步（C++ 端）：实现 BMRuntime 推理程序
  参考：Sophon_model_zoo/whisper/cpp/ 或 sensevoice/cpp/
- 第三步（部署）：sophon-cross-build 交叉编译 → scp → 板卡测试 RTF

2. 当前意图
先完成 Python 端工作，有问题分析解决
Python 端完成后逐一检查输出，没有问题后再给出下一步任务

3. 资源

| 资源 | 路径 |
|------|------|
| SOC SDK | Sophon_model_zoo/0_Toolkits/soc-sdk-sp4/ |
| tpu_mlir whl | Sophon_model_zoo/0_Toolkits/tpu_mlir*.whl |
| 第三方库 | Sophon_model_zoo/1_third_party/ |
| 知识库 | Sophon_model_zoo/.claude/doc/sophon_bm1684_knowledge_base.md |
| 算子列表 | Sophon_model_zoo/.claude/doc/sophon_tpumlir_operators.md |
| 参考项目（Whisper） | Sophon_model_zoo/whisper/ |
| 参考项目（SenseVoice） | Sophon_model_zoo/sensevoice/ |

板卡信息：IP: <ip>  用户: <user>  密码: <pwd>
模型路径: <path>

4. 注意事项
- ONNX 导出：dummy tensor 必须用列表推导式创建（KV cache 场景）
- 算子不支持时：先确认算子列表，再考虑用等价算子替换，保持语义不变
- bmodel 转换：复杂图（多输入输出）加 --disable_layer_group
- RTF 统计：只计特征提取 + TPU 推理，不含模型加载

先让我看看你的计划
```

## 配套资源

- 算子列表: `Sophon_model_zoo/.claude/doc/sophon_tpumlir_operators.md`
- 知识库: `Sophon_model_zoo/.claude/doc/sophon_bm1684_knowledge_base.md`
- 输出规范: `Sophon_model_zoo/.claude/standards/bmodel_output_management.md`
- 部署规范: `Sophon_model_zoo/.claude/standards/board_deploy_workflow.md`
- 参考项目（Whisper）: `Sophon_model_zoo/whisper/`
- 参考项目（SenseVoice）: `Sophon_model_zoo/sensevoice/`

## 版本历史

- v1.0 (2026-05-12): 初始版本，基于 MTK subagent 系统改造，适配 Sophon BM1684X 工具链
