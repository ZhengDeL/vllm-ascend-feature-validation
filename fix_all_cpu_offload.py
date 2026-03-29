#!/usr/bin/env python3
"""
vLLM CPUOffloadingConnector 一键修复脚本
========================================
修复 cpu_offload_connector.py 和 cpu_kv_cache_manager.py 中的 4 处兼容性问题。

用法:
    # 执行修复
    python3 fix_all_cpu_offload.py

    # 验证修复（对比原始备份）
    python3 fix_all_cpu_offload.py --verify

说明:
    - 自动备份原始文件（如果尚未备份）
    - 基于原始备份文件进行修复（幂等，可重复执行）
    - --verify 模式会校验每处修复是否正确到位
"""

import os
import sys
import re
import shutil
import difflib
from datetime import datetime

# ============================================================
# 配置
# ============================================================
BASE_DIR = "/vllm-workspace/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/cpu_offload"

CONNECTOR_FILE = os.path.join(BASE_DIR, "cpu_offload_connector.py")
CONNECTOR_BACKUP = os.path.join(BASE_DIR, "cpu_offload_connector.py.bak.2026-03-29-132550")

KV_MANAGER_FILE = os.path.join(BASE_DIR, "cpu_kv_cache_manager.py")

# ============================================================
# 修复函数
# ============================================================

def fix_connector(source_code: str) -> str:
    """
    对 cpu_offload_connector.py 进行 3 处修复:
      1. __init__ 中添加 self.current_layer = -1
      2. register_kv_caches 开头添加 self.current_layer = -1
      3. wait_for_layer_load 中添加 hasattr 兜底
    """
    lines = source_code.split('\n')
    result = []
    i = 0

    # 标记是否在 CPUOffloadingConnectorWorker 类内部
    in_worker_class = False
    in_init = False
    in_register = False
    in_wait = False

    # 修复计数
    fix_count = 0

    while i < len(lines):
        line = lines[i]

        # 检测进入 CPUOffloadingConnectorWorker 类
        if re.match(r'^class CPUOffloadingConnectorWorker:', line):
            in_worker_class = True
            result.append(line)
            i += 1
            continue

        # 在 Worker 类内，检测顶级 class 定义（退出 Worker 类）
        if in_worker_class and re.match(r'^class \w+', line) and 'CPUOffloadingConnectorWorker' not in line:
            in_worker_class = False

        # 在 Worker 类内，检测顶级函数定义（退出 Worker 类）
        if in_worker_class and re.match(r'^def ', line):
            in_worker_class = False

        if in_worker_class:
            # ---- 修复 1: __init__ 中添加 self.current_layer = -1 ----
            # 找到 self.done_sending_count 那一行，在其后插入
            if 'self.done_sending_count' in line and 'defaultdict(int)' in line:
                result.append(line)
                i += 1
                # 检查下一行是否已经是 self.current_layer
                if i < len(lines) and 'self.current_layer' in lines[i]:
                    # 已修复，跳过
                    result.append(lines[i])
                    i += 1
                else:
                    # 插入修复
                    indent = '        '  # 8 spaces
                    result.append('')
                    result.append(indent + 'self.current_layer = -1')
                    fix_count += 1
                    print(f"  [修复1] __init__: 添加 self.current_layer = -1")
                continue

            # ---- 修复 2: register_kv_caches 开头添加 self.current_layer = -1 ----
            if re.match(r'    def register_kv_caches\(self', line):
                result.append(line)
                i += 1
                # 检查下一行是否已经是 self.current_layer
                if i < len(lines) and 'self.current_layer' in lines[i]:
                    # 已修复
                    result.append(lines[i])
                    i += 1
                else:
                    indent = '        '  # 8 spaces
                    result.append(indent + 'self.current_layer = -1')
                    fix_count += 1
                    print(f"  [修复2] register_kv_caches: 添加 self.current_layer = -1")
                continue

            # ---- 修复 3: wait_for_layer_load 中添加 hasattr 兜底 ----
            if re.match(r'    def wait_for_layer_load\(self', line):
                result.append(line)
                i += 1
                # 收集函数体直到 self.current_layer += 1
                temp_lines = []
                found_increment = False
                while i < len(lines):
                    cur = lines[i]
                    # 如果已有 hasattr 保护（之前的简单 return 版本），跳过它
                    if "hasattr(self, 'current_layer')" in cur or 'hasattr(self, "current_layer")' in cur:
                        # 检查: 如果下一行是 return，说明是旧的简单修复，需要替换
                        if i + 1 < len(lines) and lines[i + 1].strip() == 'return':
                            # 旧修复，跳过这两行
                            i += 2
                            continue
                        else:
                            # 可能已经是正确的修复
                            temp_lines.append(cur)
                            i += 1
                            continue

                    if 'self.current_layer += 1' in cur:
                        found_increment = True
                        # 检查前面是否已有 hasattr 兜底
                        has_guard = any('hasattr' in t and 'current_layer' in t for t in temp_lines)
                        if not has_guard:
                            indent = '        '  # 8 spaces
                            temp_lines.append(indent + 'if not hasattr(self, "current_layer"):')
                            temp_lines.append(indent + '    self.current_layer = -1')
                            fix_count += 1
                            print(f"  [修复3] wait_for_layer_load: 添加 hasattr 兜底")
                        temp_lines.append(cur)
                        i += 1
                        break
                    temp_lines.append(cur)
                    i += 1

                result.extend(temp_lines)
                continue

        result.append(line)
        i += 1

    if fix_count > 0:
        print(f"  cpu_offload_connector.py: 共应用 {fix_count} 处修复")
    else:
        print(f"  cpu_offload_connector.py: 所有修复已到位，无需修改")

    return '\n'.join(result)


def fix_kv_manager(source_code: str) -> str:
    """
    对 cpu_kv_cache_manager.py 进行 1 处修复:
      在 get_manager_for_kv_cache_spec() 调用中添加 enable_caching=True
    """
    if 'enable_caching=True' in source_code:
        print(f"  cpu_kv_cache_manager.py: enable_caching=True 已存在，无需修改")
        return source_code

    # 找到 kv_cache_group_id=0, 后面添加 enable_caching=True,
    old = """            kv_cache_group_id=0,
        )"""
    new = """            kv_cache_group_id=0,
            enable_caching=True,
        )"""

    if old in source_code:
        source_code = source_code.replace(old, new)
        print(f"  [修复4] cpu_kv_cache_manager.py: 添加 enable_caching=True")
    else:
        print(f"  [警告] cpu_kv_cache_manager.py: 未找到预期的代码模式，请手动检查")

    return source_code


# ============================================================
# 验证函数
# ============================================================

def verify_connector(code: str) -> list:
    """验证 connector 文件的 4 处修复是否到位"""
    issues = []

    # 先提取 CPUOffloadingConnectorWorker 类的代码块
    # 从 "class CPUOffloadingConnectorWorker:" 开始，到下一个顶级 class/def 或文件结尾
    worker_match = re.search(
        r'(class CPUOffloadingConnectorWorker:.*?)(?=\nclass |\ndef |\Z)',
        code, re.DOTALL
    )
    if not worker_match:
        issues.append("[全局] 警告: 未找到 CPUOffloadingConnectorWorker 类")
        return issues
    worker_code = worker_match.group(1)

    # 检查 1: __init__ 中有 self.current_layer = -1
    init_match = re.search(
        r'def __init__\(self.*?\n(.*?)(?=\n    def )',
        worker_code, re.DOTALL
    )
    if init_match:
        init_body = init_match.group(1)
        if 'self.current_layer = -1' not in init_body:
            issues.append("[修复1] 缺失: __init__ 中未找到 self.current_layer = -1")
        else:
            print("  [修复1] ✅ __init__ 中 self.current_layer = -1 已存在")
    else:
        issues.append("[修复1] 警告: 未找到 CPUOffloadingConnectorWorker.__init__")

    # 检查 2: register_kv_caches 开头有 self.current_layer = -1
    reg_match = re.search(
        r'def register_kv_caches\(self.*?\n(.*?)(?=\n    def )',
        worker_code, re.DOTALL
    )
    if reg_match:
        reg_body = reg_match.group(1)
        # 确保在函数体的前 5 行内
        first_lines = reg_body.split('\n')[:5]
        found = any('self.current_layer = -1' in line for line in first_lines)
        if not found:
            issues.append("[修复2] 缺失: register_kv_caches 开头未找到 self.current_layer = -1")
        else:
            print("  [修复2] ✅ register_kv_caches 开头 self.current_layer = -1 已存在")
    else:
        issues.append("[修复2] 警告: 未找到 register_kv_caches 函数")

    # 检查 3: wait_for_layer_load 中有 hasattr 兜底
    wait_match = re.search(
        r'def wait_for_layer_load\(self.*?\n(.*?)(?=\n    def )',
        worker_code, re.DOTALL
    )
    if wait_match:
        wait_body = wait_match.group(1)
        has_guard = 'hasattr(self, "current_layer")' in wait_body or "hasattr(self, 'current_layer')" in wait_body
        has_increment = 'self.current_layer += 1' in wait_body
        if not has_guard:
            issues.append("[修复3] 缺失: wait_for_layer_load 中未找到 hasattr 兜底")
        elif not has_increment:
            issues.append("[修复3] 缺失: wait_for_layer_load 中未找到 self.current_layer += 1")
        else:
            # 验证 hasattr 在 += 1 之前
            guard_pos = wait_body.find('hasattr')
            incr_pos = wait_body.find('self.current_layer += 1')
            if guard_pos < incr_pos:
                print("  [修复3] ✅ wait_for_layer_load 中 hasattr 兜底已正确放置")
            else:
                issues.append("[修复3] 顺序错误: hasattr 应在 current_layer += 1 之前")
    else:
        issues.append("[修复3] 警告: 未找到 wait_for_layer_load 函数")

    return issues


def verify_kv_manager(code: str) -> list:
    """验证 kv_manager 文件的修复是否到位"""
    issues = []

    if 'enable_caching=True' in code:
        print("  [修复4] ✅ cpu_kv_cache_manager.py 中 enable_caching=True 已存在")
    else:
        issues.append("[修复4] 缺失: get_manager_for_kv_cache_spec 调用中缺少 enable_caching=True")

    return issues


def show_diff(original: str, modified: str, filename: str):
    """显示两个文件内容的 diff"""
    original_lines = original.splitlines(keepends=True)
    modified_lines = modified.splitlines(keepends=True)

    diff = difflib.unified_diff(
        original_lines,
        modified_lines,
        fromfile=f'{filename} (原始)',
        tofile=f'{filename} (修复后)',
        lineterm=''
    )
    diff_text = ''.join(diff)
    if diff_text:
        print(f"\n{'='*60}")
        print(f"  DIFF: {filename}")
        print(f"{'='*60}")
        print(diff_text)
    else:
        print(f"\n  {filename}: 无差异")


# ============================================================
# 主流程
# ============================================================

def main():
    verify_mode = '--verify' in sys.argv

    if verify_mode:
        print("=" * 60)
        print("  验证模式：检查修复是否正确到位")
        print("=" * 60)

        # 读取当前文件
        if not os.path.exists(CONNECTOR_FILE):
            print(f"  错误: 找不到 {CONNECTOR_FILE}")
            sys.exit(1)

        with open(CONNECTOR_FILE, 'r') as f:
            connector_code = f.read()

        with open(KV_MANAGER_FILE, 'r') as f:
            kv_manager_code = f.read()

        print("\n--- 验证 cpu_offload_connector.py ---")
        issues1 = verify_connector(connector_code)

        print("\n--- 验证 cpu_kv_cache_manager.py ---")
        issues2 = verify_kv_manager(kv_manager_code)

        all_issues = issues1 + issues2

        # 如果有原始备份，显示 diff
        if os.path.exists(CONNECTOR_BACKUP):
            with open(CONNECTOR_BACKUP, 'r') as f:
                original_code = f.read()
            show_diff(original_code, connector_code, "cpu_offload_connector.py")

        print("\n" + "=" * 60)
        if all_issues:
            print("  ❌ 验证失败！以下修复未到位：")
            for issue in all_issues:
                print(f"    - {issue}")
            sys.exit(1)
        else:
            print("  ✅ 全部 4 处修复验证通过！")
            print("=" * 60)
            sys.exit(0)

    # ---- 修复模式 ----
    print("=" * 60)
    print("  vLLM CPUOffloadingConnector 一键修复")
    print("=" * 60)

    # 检查文件存在
    for f in [CONNECTOR_FILE, KV_MANAGER_FILE]:
        if not os.path.exists(f):
            print(f"  错误: 找不到 {f}")
            sys.exit(1)

    # 备份（如果原始备份不存在）
    if not os.path.exists(CONNECTOR_BACKUP):
        ts = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        backup = CONNECTOR_FILE + f".bak.{ts}"
        shutil.copy2(CONNECTOR_FILE, backup)
        print(f"  已备份: {backup}")
    else:
        print(f"  原始备份已存在: {CONNECTOR_BACKUP}")

    kv_backup = KV_MANAGER_FILE + ".bak"
    if not os.path.exists(kv_backup):
        shutil.copy2(KV_MANAGER_FILE, kv_backup)
        print(f"  已备份: {kv_backup}")

    # 从原始备份读取并修复 connector
    print("\n--- 修复 cpu_offload_connector.py ---")
    if os.path.exists(CONNECTOR_BACKUP):
        with open(CONNECTOR_BACKUP, 'r') as f:
            original_connector = f.read()
        print(f"  基于原始备份进行修复（幂等）")
    else:
        with open(CONNECTOR_FILE, 'r') as f:
            original_connector = f.read()

    fixed_connector = fix_connector(original_connector)

    with open(CONNECTOR_FILE, 'w') as f:
        f.write(fixed_connector)
    print(f"  已写入: {CONNECTOR_FILE}")

    # 修复 kv_manager
    print("\n--- 修复 cpu_kv_cache_manager.py ---")
    with open(KV_MANAGER_FILE, 'r') as f:
        kv_manager_code = f.read()

    fixed_kv_manager = fix_kv_manager(kv_manager_code)

    with open(KV_MANAGER_FILE, 'w') as f:
        f.write(fixed_kv_manager)
    print(f"  已写入: {KV_MANAGER_FILE}")

    # 自动验证
    print("\n--- 自动验证 ---")
    issues1 = verify_connector(fixed_connector)
    issues2 = verify_kv_manager(fixed_kv_manager)
    all_issues = issues1 + issues2

    # 显示 diff
    show_diff(original_connector, fixed_connector, "cpu_offload_connector.py")

    print("\n" + "=" * 60)
    if all_issues:
        print("  ⚠️ 修复后仍有问题：")
        for issue in all_issues:
            print(f"    - {issue}")
        sys.exit(1)
    else:
        print("  ✅ 全部 4 处修复已成功应用并验证通过！")
        print("=" * 60)


if __name__ == "__main__":
    main()
