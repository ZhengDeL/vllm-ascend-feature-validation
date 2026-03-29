#!/bin/bash
# ============================================================
# EARS (Entropy-Adaptive Rejection Sampling) 一键部署脚本
# 适用于 vllm-ascend v0.18.0 容器环境
# ============================================================

set -e

# ==================== 配置区域 ====================
VLLM_ASCEND_DIR="/vllm-workspace/vllm-ascend"
ENVS_FILE="${VLLM_ASCEND_DIR}/vllm_ascend/envs.py"
SAMPLER_FILE="${VLLM_ASCEND_DIR}/vllm_ascend/sample/rejection_sampler.py"
WORK_DIR="/data/lzd"
EARS_TOLERANCE="0.3"
MODEL_PATH="/data/GLM-5-w4a8-mtp-QuaRot/"
SERVER_NAME="glm-5"
PORT="9991"
# ==================================================

echo "============================================"
echo "  EARS 一键部署脚本"
echo "============================================"
echo ""

# ---------- Step 1: 备份原始文件 ----------
echo "[Step 1/5] 备份原始文件..."
BACKUP_DIR="${WORK_DIR}/ears_backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${BACKUP_DIR}"
cp "${ENVS_FILE}" "${BACKUP_DIR}/envs.py.bak"
cp "${SAMPLER_FILE}" "${BACKUP_DIR}/rejection_sampler.py.bak"
echo "  备份已保存到: ${BACKUP_DIR}"
echo ""

# ---------- Step 2: 补丁 envs.py ----------
echo "[Step 2/5] 补丁 envs.py ..."

# 检查是否已经打过补丁
if grep -q "VLLM_EARS_TOLERANCE" "${ENVS_FILE}"; then
    echo "  [跳过] envs.py 已包含 VLLM_EARS_TOLERANCE 配置"
else
    # 在 environment_variables 字典末尾的 } 之前插入 EARS 配置
    # 查找最后一个 ")" 后紧跟 "}" 的位置（字典末尾）
    python3 << 'PATCH_ENVS'
import re

filepath = "${ENVS_FILE}"
with open(filepath, 'r') as f:
    content = f.read()

# 找到字典结束的 } （在 # end-env-vars-definition 之前）
ears_entry = '''    # EARS (Entropy-Adaptive Rejection Sampling) tolerance for speculative decoding.
    # When > 0, dynamically relaxes rejection threshold based on target model uncertainty.
    # Higher values accept more draft tokens but may reduce quality.
    # Only effective for random sampling (not greedy). Range: [0.0, 1.0]. Default: 0.0 (disabled).
    "VLLM_EARS_TOLERANCE": lambda: float(os.getenv("VLLM_EARS_TOLERANCE", "0.0")),
'''

# 在最后一个 } 之前（即 end-env-vars-definition 注释前的 }）插入
pattern = r'(\n)(})\s*\n(\s*#\s*end-env-vars-definition)'
replacement = r'\1' + ears_entry + r'\2\n\3'
new_content = re.sub(pattern, replacement, content)

if new_content == content:
    # 备用方案：在字典末尾 } 之前插入
    # 找到 "}: 后面是 end-env-vars" 的模式
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if line.strip() == '}' and i + 1 < len(lines) and 'end-env-vars' in lines[i + 1]:
            lines.insert(i, ears_entry.rstrip())
            break
    new_content = '\n'.join(lines)

with open(filepath, 'w') as f:
    f.write(new_content)

print("  envs.py 补丁已应用")
PATCH_ENVS
fi
echo ""

# ---------- Step 3: 补丁 rejection_sampler.py ----------
echo "[Step 3/5] 补丁 rejection_sampler.py ..."

if grep -q "EARS: Entropy-Adaptive Rejection Sampling" "${SAMPLER_FILE}"; then
    echo "  [跳过] rejection_sampler.py 已包含 EARS 逻辑"
else
    python3 << 'PATCH_SAMPLER'
filepath = "${SAMPLER_FILE}"
with open(filepath, 'r') as f:
    content = f.read()

# 在 generate_uniform_probs() 调用后、sample_recovered_tokens() 调用前插入 EARS 逻辑
ears_code = '''
    # EARS: Entropy-Adaptive Rejection Sampling
    # Dynamically relax rejection threshold based on target model uncertainty.
    # When the model is uncertain (low max prob), tolerance is higher → easier acceptance.
    ears_tolerance = envs_ascend.VLLM_EARS_TOLERANCE
    if ears_tolerance > 0:
        max_target_probs = target_probs.max(dim=-1).values  # [num_tokens]
        uncertainties = 1.0 - max_target_probs
        tolerance = ears_tolerance * uncertainties
        uniform_probs = (uniform_probs - tolerance).clamp(min=0.01)
'''

# 定位插入点：在 "# Sample recovered tokens for each position." 注释之前
target_comment = "    # Sample recovered tokens for each position."
if target_comment in content:
    content = content.replace(
        target_comment,
        ears_code + "\n" + target_comment
    )
    with open(filepath, 'w') as f:
        f.write(content)
    print("  rejection_sampler.py 补丁已应用")
else:
    print("  [错误] 未找到插入点，请手动应用补丁")
    exit(1)
PATCH_SAMPLER
fi
echo ""

# ---------- Step 4: 生成 vllm 启动脚本 ----------
echo "[Step 4/5] 生成 vllm 启动脚本..."

cat > "${WORK_DIR}/run_vllm_dars.sh" << VLLM_SCRIPT
#!/bin/bash
# EARS 启动脚本 - VLLM_EARS_TOLERANCE=${EARS_TOLERANCE}
export VLLM_EARS_TOLERANCE=${EARS_TOLERANCE}

export HCCL_OP_EXPANSION_MODE="AIV"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE=200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_BALANCE_SCHEDULING=1

vllm serve ${MODEL_PATH} \\
    --host 0.0.0.0 \\
    --port ${PORT} \\
    --data-parallel-size 1 \\
    --tensor-parallel-size 8 \\
    --enable-expert-parallel \\
    --seed 1024 \\
    --served-model-name ${SERVER_NAME} \\
    --max-num-seqs 2 \\
    --max-model-len 32768 \\
    --max-num-batched-tokens 4096 \\
    --trust-remote-code \\
    --gpu-memory-utilization 0.95 \\
    --quantization ascend \\
    --enable-chunked-prefill \\
    --enable-prefix-caching \\
    --async-scheduling \\
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \\
    --additional-config '{"enable_npugraph_ex": true,"fuse_muls_add":true,"multistream_overlap_shared_expert":true}' \\
    --speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp"}'
VLLM_SCRIPT
chmod +x "${WORK_DIR}/run_vllm_dars.sh"
echo "  启动脚本已生成: ${WORK_DIR}/run_vllm_dars.sh"
echo ""

# ---------- Step 5: 安装 evalscope ----------
echo "[Step 5/5] 安装 evalscope..."
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple 2>/dev/null || true
pip config set install.trusted-host mirrors.aliyun.com 2>/dev/null || true
pip install evalscope[perf] -q 2>/dev/null && echo "  evalscope 已安装" || echo "  evalscope 安装失败，请手动安装"
echo ""

# ---------- 完成 ----------
echo "============================================"
echo "  EARS 部署完成！"
echo "============================================"
echo ""
echo "后续步骤："
echo ""
echo "1. 启动 vllm 服务（约15-20分钟加载）："
echo "   mkdir -p ${WORK_DIR}/logs"
echo "   nohup bash ${WORK_DIR}/run_vllm_dars.sh > ${WORK_DIR}/logs/vllm_dars.log 2>&1 &"
echo ""
echo "2. 等待服务就绪后验证："
echo "   curl -s http://localhost:${PORT}/v1/models | python3 -m json.tool"
echo ""
echo "3. 运行 benchmark 测试："
echo "   evalscope perf \\"
echo "     --url \"http://localhost:${PORT}/v1/chat/completions\" \\"
echo "     --parallel 1 \\"
echo "     --model ${SERVER_NAME} \\"
echo "     --number 20 \\"
echo "     --api openai \\"
echo "     --dataset openqa \\"
echo "     --temperature 0.9 \\"
echo "     --stream"
echo ""
echo "备份文件位置: ${BACKUP_DIR}"
echo ""
