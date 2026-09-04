# 作业计时与 AED 缓存清理设计

## 目标

生产服务只把首次模型加载排除在作业端到端计时之外。首个作业的代表性预热、正常转录、输出准备以及作业末尾的 CUDA cache 清理都计时；后续作业不重复预热，但每个作业都清理两个 AED worker 的 PyTorch caching allocator。

## 生命周期与计时边界

`HotModelHost.start()` 只并行加载 FireRed owner 与 3D-Speaker owner，并读取模型清单。首个排队作业在端到端计时和资源采样开始后调用一次 `HotModelHost.warmup()`；服务生命周期内后续作业复用这个已预热状态。服务关闭或 overlap 流程卸载 owner 后，预热状态重置，新 owner 的下一项正常作业再次执行计时内预热。

端到端边界为：首次模型加载完成后开始计时，依次包括资源采样启动、首任务预热、输入探测与规范化、说话人分离/FireRedVAD、AED/Punc、对齐、结果组装和两个 AED worker 的缓存清理。两个 worker 均确认清理完成后才停止端到端计时并导出 diagnostics。`end_to_end_after_load` 是正式字段；`end_to_end_after_warmup` 暂时保留为同值兼容别名，并在 runtime 中标记计时口径。

## Worker 协议与清理语义

FireRed worker 新增 `cleanup_cuda_cache` action。每个 worker 顺序执行 `gc.collect()`、`torch.cuda.synchronize()`、`torch.cuda.empty_cache()`、再次 `torch.cuda.synchronize()`，并返回耗时、GC 回收计数以及 allocated/reserved 的清理前后值。该 action 不接收音频路径，也不改变模型权重或成本模型。

`ParallelFireRedWorkerClient.cleanup_cuda_cache()` 只对两个 AED worker 并行发请求，不清理独立 Punc owner。返回值保留每个 worker 的结果以及宿主侧 wall time；作业 diagnostics 记录为 `runtime.cache_cleanup` 和 `stage_seconds.cache_cleanup`。任一清理请求失败时，本作业失败；如果主流程已失败，则仍尽力清理，但保留原始错误。

## 诊断与资源采样

首任务 `runtime.warmup` 标记 `performed=true`、`excluded_from_job_timing=false`；后续作业记录 `performed=false` 和 `reason=already_performed`。deep/resources 的 sampler 覆盖预热和清理，所以资源峰值包含作业内的真实峰值，deep 时序也能看到清理后的低水位。`off` 仍只输出最小摘要，不移除 `diagnostics.json`。

## 验证

单元测试按 worker、client、双 worker owner、service 生命周期顺序做红绿验证。完成后运行完整 pytest 与 `git diff --check`。GPU0 实测依次运行单个 `刘洋洋.wav`、同一常驻服务连续三个相同任务、deep AED 显存时序、0.60/0.70/0.80/0.90/0.95 预算矩阵。报告写入 `outputs/measurements/`，不操作或详细检查任何其他项目进程。
