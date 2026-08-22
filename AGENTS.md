# Standalone Transcriber 项目协作约定

## 工作范围

本仓库是独立的 FireRedASR2S 转录服务。直接在当前分支工作，不创建 worktree；除非用户明确要求，
不要委派给子 agent。

## 生产架构

- 生产固定使用物理 GPU0（`cuda:0`）。
- 生产默认常驻两个 FireRed AED worker 和一个独立的 FireRedPunc owner。
- AED worker-0 同时承载 FireRedVAD helper；worker-1 只加载 AED；Punc owner 只加载 Punc。
- FireRedVAD 在功能上属于说话人分离阶段。它因为官方依赖和生命周期隔离而暂时托管在
  `.venv-asr` 子进程中，不能把它描述成 AED/Punc 的功能阶段。
- 普通流程先运行 3D-Speaker 说话人分离。超过 10 秒的 speaker 区间使用全局 FireRedVAD 细分，
  再与 speaker 区间求交，最后进入 AED/Punc。
- AED 使用 memory-first 调度：预热后的 GPU 空闲显存减去安全余量，决定每个 batch 的具体片段；
  生产 batch 数量不是固定配置。完整计划必须在第一个 AED 请求前确定。
- 生产默认 `use_half = true`。当前 RTX 3090 上，固定版本的 FireRed 实现会使用
  `torch.bfloat16`；`false` 才是 FP32 回退，不能默认改回 FP32。
- overlap 默认关闭。启用时，先关闭常驻 FireRed/3D-Speaker，再运行 pyannote/ClearVoice，完成后
  在下一个排队作业前重新加载常驻模型。
- 模型加载和显式预热不计入作业的端到端耗时。

## 诊断级别

- 默认级别是 `basic`：记录低开销的阶段耗时、计数和调度摘要。
- `resources` 额外启用只读 GPU/RSS 采样，记录 service、speaker、两个 AED、Punc 和 overlap
  owner 的资源峰值。
- `deep` 在 `resources` 基础上写出完整 scheduler 和逐 batch JSONL trace。
- `off` 只保留最小成功/失败摘要；不要删除 `diagnostics.json` 文件本身。
- CLI 的 `--diagnostics` 优先级高于配置文件；旧的 `--profile-resources` 是 `resources` 的兼容别名。
- 资源采样只能查询 GPU 和本项目登记的 PID，绝不向其他项目的进程发送信号或进行管理。

## 安全约束

- 绝不能停止、暂停、迁移、重启、详细检查或以其他方式管理其他项目的进程。
- 可以只读使用 `nvidia-smi`；所有显存采样必须过滤到本项目登记的 PID。
- GPU 工作只能在物理 GPU0 上进行。不能用 `CUDA_VISIBLE_DEVICES` 把其他 GPU 重映射为逻辑
  `cuda:0`。
- `artifacts/release_validation_corpus` 必须保持 Git 忽略。生产输出放在专用输出目录，不能写回语料目录。
- 执行删除或覆盖前，必须确认精确目标，并优先使用可恢复操作。

## 输出和中间文件

- 正式输出目录包含 `transcript.txt`、`transcript.json`、`diagnostics.json`。
- 深度诊断额外写入 `diagnostics/scheduler.json` 和 `diagnostics/batch-events.jsonl`，不能把完整
  batch 计划塞进普通 `diagnostics.json`。
- `--keep-intermediate` 只用于调试和验收，不应作为生产默认设置。

## 验证

迭代时先运行聚焦测试，完成前运行完整测试和格式检查：

```bash
.venv/bin/python -m pytest -q tests/unit/test_config.py tests/unit/test_firered_client.py
.venv/bin/python -m pytest -q
git diff --check
```

GPU 实测必须排除模型加载和显式预热，记录实际精度（`torch.float32`、`torch.bfloat16` 或
`torch.float16`），报告保存在 `outputs/measurements/`。没有新鲜命令输出时，不能声称基准结果。

## 提交

- 使用仓库已有的 Git 用户身份。
- 测试通过后提交聚焦变更。
- 不要重写、丢弃或覆盖工作区中与当前任务无关的用户修改。
