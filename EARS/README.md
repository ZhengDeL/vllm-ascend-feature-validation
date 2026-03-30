# EARS (EARS - Entropy-Adaptive Rejection Sampling) 验证操作方案

## 一、概述

**EARS (Entropy-Adaptive Rejection Sampling)** 是针对 vllm-ascend 投机推理（Speculative Decoding）中拒绝采样环节的优化。当目标模型对某个位置的预测不确定性较高时（max_prob 较低），EARS 会动态放松拒绝阈值，从而提高 draft token 的接受率，提升推理吞吐量。

**核心原理**: 通过环境变量 `VLLM_EARS_TOLERANCE` 控制容忍度（0.0~1.0），当值 > 0 时，对均匀采样概率施加基于不确定性的偏移，使得在模型不确定时更容易接受 draft token。

**补丁来源**: https://github.com/sunchendd/vllm-ascend/commit/6d2d7cea38e83349cc624460b24a0b01a02baba3

---

## 二、环境信息

| 项目 | 详情 |
|------|------|
| 机器 IP | 7.6.50.x |
| 容器 | lzd-lzd_0.18.0-test |
| 芯片 | Ascend 910B4 × 8 |
| vllm-ascend版本 | v0.18.0 |
| vllm-ascend路径 | /vllm-workspace/vllm-ascend/ |
| 模型 | /data/GLM-5-w4a8-mtp-QuaRot/（GLM-5 MoE, w4a8量化, MTP投机推理） |
| Python | 3.11.14 |

---

## 三、修改文件及详细对比

### 文件1: `vllm_ascend/envs.py`

**文件路径**: `/vllm-workspace/vllm-ascend/vllm_ascend/envs.py`  
**修改位置**: 第 110~114 行（在环境变量字典末尾 `}` 之前插入）

#### 修改前（第 109~110 行）:
```python
    ),                                                                      # 第 109 行
}                                                                           # 第 110 行（原）
```

#### 修改后（第 109~115 行）:
```python
    ),                                                                      # 第 109 行
    # EARS (Entropy-Adaptive Rejection Sampling) tolerance for speculative decoding.
    # When > 0, dynamically relaxes rejection threshold based on target model uncertainty.
    # Higher values accept more draft tokens but may reduce quality.
    # Only effective for random sampling (not greedy). Range: [0.0, 1.0]. Default: 0.0 (disabled).
    "VLLM_EARS_TOLERANCE": lambda: float(os.getenv("VLLM_EARS_TOLERANCE", "0.0")),
}                                                                           # 第 115 行
```

#### 变更说明:
在 `environment_variables` 字典中新增 `VLLM_EARS_TOLERANCE` 条目。该变量通过 `os.getenv` 读取环境变量，默认值为 `0.0`（禁用 EARS）。

---

### 文件2: `vllm_ascend/sample/rejection_sampler.py`

**文件路径**: `/vllm-workspace/vllm-ascend/vllm_ascend/sample/rejection_sampler.py`  
**修改位置**: 第 182~190 行（在 `generate_uniform_probs()` 调用之后、`sample_recovered_tokens()` 调用之前插入）

#### 修改前（第 175~182 行）:
```python
    uniform_probs = generate_uniform_probs(                                 # 第 175 行
        num_tokens,
        num_draft_tokens,
        sampling_metadata.generators,
        device,
    )

    # Sample recovered tokens for each position.                            # 第 182 行（原）
```

#### 修改后（第 175~193 行）:
```python
    uniform_probs = generate_uniform_probs(                                 # 第 175 行
        num_tokens,
        num_draft_tokens,
        sampling_metadata.generators,
        device,
    )

    # EARS: Entropy-Adaptive Rejection Sampling                             # 第 182 行（新增）
    # Dynamically relax rejection threshold based on target model uncertainty.
    # When the model is uncertain (low max prob), tolerance is higher → easier acceptance.
    ears_tolerance = envs_ascend.VLLM_EARS_TOLERANCE                        # 第 185 行
    if ears_tolerance > 0:                                                  # 第 186 行
        max_target_probs = target_probs.max(dim=-1).values  # [num_tokens]  # 第 187 行
        uncertainties = 1.0 - max_target_probs                              # 第 188 行
        tolerance = ears_tolerance * uncertainties                          # 第 189 行
        uniform_probs = (uniform_probs - tolerance).clamp(min=0.01)         # 第 190 行

    # Sample recovered tokens for each position.                            # 第 192 行
```

#### 变更说明:
在随机采样拒绝判断之前插入 EARS 逻辑：
1. 读取 `VLLM_EARS_TOLERANCE` 配置值
2. 当 tolerance > 0 时：
   - 计算目标模型的最大概率 `max_target_probs`
   - 计算不确定性 `uncertainties = 1.0 - max_target_probs`
   - 计算动态容忍度 `tolerance = ears_tolerance * uncertainties`
   - 从 `uniform_probs` 减去该容忍度（下限 clamp 到 0.01 防止全接受）
3. 效果：当目标模型不确定（max_prob 低）时，容忍度更大，uniform_probs 被减小更多，更容易满足接受条件

---

## 四、启动配置

### vllm 启动脚本 `run_vllm_dars.sh`:

```bash
#!/bin/bash
# EARS 相关配置
export VLLM_EARS_TOLERANCE=0.3

# NPU 相关优化
export HCCL_OP_EXPANSION_MODE="AIV"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE=200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_BALANCE_SCHEDULING=1

vllm serve /data/GLM-5-w4a8-mtp-QuaRot/ \
    --host 0.0.0.0 \
    --port 9991 \
    --data-parallel-size 1 \
    --tensor-parallel-size 8 \
    --enable-expert-parallel \
    --seed 1024 \
    --served-model-name glm-5 \
    --max-num-seqs 2 \
    --max-model-len 32768 \
    --max-num-batched-tokens 4096 \
    --trust-remote-code \
    --gpu-memory-utilization 0.95 \
    --quantization ascend \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --async-scheduling \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --additional-config '{"enable_npugraph_ex": true,"fuse_muls_add":true,"multistream_overlap_shared_expert":true}' \
    --speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp"}'
```

**关键参数说明**:
- `VLLM_EARS_TOLERANCE=0.3`: EARS 容忍度，值越大接受率越高但质量可能下降
- `--speculative-config`: 启用 MTP 投机推理，3个投机token
- `--quantization ascend`: 使用昇腾量化
- `--tensor-parallel-size 8`: 8卡张量并行
- `--enable-expert-parallel`: 启用专家并行（MoE模型）

---

## 五、测试方法

### 安装 evalscope:
```bash
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple
pip config set install.trusted-host mirrors.aliyun.com
pip install evalscope[perf]
```

### 运行 benchmark:
```bash
evalscope perf \
    --url "http://localhost:9991/v1/chat/completions" \
    --parallel 1 \
    --model glm-5 \
    --number 20 \
    --api openai \
    --dataset openqa \
    --temperature 0.9 \
    --stream
```

**说明**: openqa 数据集来自 ModelScope HC3-Chinese（真实中文问答），使用连贯上下文测试 acceptance rate 更有代表性。

---

## 六、验证结果

| 指标 | 值 |
|------|-----|
| **成功率** | **100% (20/20)** |
| **总生成token数** | 40,072 tokens |
| **总测试时间** | 934.02 秒 |
| **平均输出速率** | 42.90 tokens/s |
| **平均延迟** | 46.701 秒 |
| **P99延迟** | 49.463 秒 |
| **平均TTFT (首token时间)** | 0.913 秒 |
| **P99 TTFT** | 5.183 秒 |
| **平均TPOT (每token时间)** | 0.023 秒 |
| **P99 TPOT** | 0.024 秒 |
| **请求吞吐量** | 0.02 req/s |
| **平均输入token/请求** | 24.9 |
| **平均输出token/请求** | 2043.7 |

**结论**: EARS 修复在 Ascend 910B4 + GLM-5-w4a8-mtp 模型上验证通过，所有请求均成功完成，未出现 NPU 内核崩溃。

---

## 七、一键脚本使用方法

```bash
# 在容器内执行一键部署脚本：
bash apply_ears_patch.sh
```

脚本将自动：
1. 备份原始文件
2. 应用 EARS 补丁到 envs.py 和 rejection_sampler.py
3. 生成 vllm 启动脚本
4. 安装 evalscope
5. 提示启动 vllm 和运行 benchmark
