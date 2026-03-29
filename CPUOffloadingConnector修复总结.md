# vLLM CPUOffloadingConnector 代码修复总结

> **环境信息**
> - 服务器：`7.6.50.28`，用户 `root`
> - 容器：`lzd-lzd_0.18.0-test`
> - vLLM 版本：`v0.18.0` + `vllm-ascend`（华为昇腾 NPU 910B4-1 × 8）
> - 模型：`/data/GLM-5-w4a8-mtp-QuaRot/`（8 路 TP，MTP 推测解码）
> - 修复日期：2026-03-29

---

## 一、问题背景

执行 vLLM 推理脚本 `vllm-glm5-CPUOffloadingConnector.sh` 时，依次遇到 **4 个运行时错误**，均因 vLLM v0.18.0 的 CPUOffloadingConnector 插件代码与主框架接口不兼容导致。

---

## 二、涉及文件

容器内基础路径：

```
/vllm-workspace/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/cpu_offload/
```

| 文件 | 说明 |
|------|------|
| `cpu_offload_connector.py` | CPUOffloadingConnector 主逻辑（Worker + Scheduler） |
| `cpu_kv_cache_manager.py` | CPU 端 KV Cache 管理器 |
| `cpu_offload_connector.py.bak.2026-03-29-132550` | **原始备份**（未修改） |

---

## 三、修复详情（共 4 处修改）

### 修复 1：`__init__` 初始化 `current_layer`

| 项目 | 内容 |
|------|------|
| **文件** | `cpu_offload_connector.py` |
| **类** | `CPUOffloadingConnectorWorker` |
| **方法** | `__init__`（原始第 228 行） |
| **关联错误** | `AttributeError: 'CPUOffloadingConnectorWorker' object has no attribute 'current_layer'` |

**原始代码（第 247 行附近）：**

```python
        self.done_sending_count: defaultdict[str, int] = defaultdict(int)

        # start metadata server to init cpu_kv_cache_manager ...
```

**修复后：**

```python
        self.done_sending_count: defaultdict[str, int] = defaultdict(int)

        self.current_layer = -1

        # start metadata server to init cpu_kv_cache_manager ...
```

> **说明**：初始值设为 `-1`，因为 `wait_for_layer_load()` 会先 `+= 1`，这样第一次调用后 `current_layer` 变为 `0`，正确对应第 0 层。

---

### 修复 2：`register_kv_caches` 重置 `current_layer`

| 项目 | 内容 |
|------|------|
| **文件** | `cpu_offload_connector.py` |
| **类** | `CPUOffloadingConnectorWorker` |
| **方法** | `register_kv_caches`（原始第 293 行） |
| **关联错误** | 每轮新请求时 `current_layer` 残留旧值 |

**原始代码：**

```python
    def register_kv_caches(self, kv_caches: dict[str, Sequence[torch.Tensor]]):
        self.gpu_kv_caches = kv_caches
```

**修复后：**

```python
    def register_kv_caches(self, kv_caches: dict[str, Sequence[torch.Tensor]]):
        self.current_layer = -1
        self.gpu_kv_caches = kv_caches
```

> **说明**：每次重新注册 KV Cache 时把层号归位，避免跨请求状态残留。

---

### 修复 3：`wait_for_layer_load` 加兜底保护 + 正确递增逻辑

| 项目 | 内容 |
|------|------|
| **文件** | `cpu_offload_connector.py` |
| **类** | `CPUOffloadingConnectorWorker` |
| **方法** | `wait_for_layer_load`（原始第 316 行） |
| **关联错误** | `AttributeError: 'CPUOffloadingConnectorWorker' object has no attribute 'current_layer'` |

**原始代码：**

```python
    def wait_for_layer_load(self) -> None:
        # TODO: Replace with `torch.npu.current_stream().wait_stream(self.load_stream)` after fixing the bug.
        self.load_stream.synchronize()
        self.current_layer += 1
        self.load_kv_layer(self.current_layer)
```

**修复后：**

```python
    def wait_for_layer_load(self) -> None:
        # TODO: Replace with `torch.npu.current_stream().wait_stream(self.load_stream)` after fixing the bug.
        self.load_stream.synchronize()
        if not hasattr(self, "current_layer"):
            self.current_layer = -1
        self.current_layer += 1
        self.load_kv_layer(self.current_layer)
```

> **说明**：`hasattr` 兜底是保险丝——即使某些路径没走到 `__init__` 或 `register_kv_caches`，也不会直接报 `AttributeError`。

---

### 修复 4：`CPUKVCacheManager` 补充 `enable_caching` 参数

| 项目 | 内容 |
|------|------|
| **文件** | `cpu_kv_cache_manager.py` |
| **类** | `CPUKVCacheManager` |
| **方法** | `__init__`（第 69 行） |
| **关联错误** | `TypeError: SingleTypeKVCacheManager.__init__() missing 1 required positional argument: 'enable_caching'` |

**原始代码：**

```python
        self.single_type_manager = get_manager_for_kv_cache_spec(
            kv_cache_spec=kv_cache_spec,
            block_pool=self.block_pool,
            kv_cache_group_id=0,
        )
```

**修复后：**

```python
        self.single_type_manager = get_manager_for_kv_cache_spec(
            kv_cache_spec=kv_cache_spec,
            block_pool=self.block_pool,
            kv_cache_group_id=0,
            enable_caching=True,
        )
```

> **说明**：vLLM v0.18.0 中 `SingleTypeKVCacheManager.__init__()` 新增了必填参数 `enable_caching: bool`。`get_manager_for_kv_cache_spec()` 通过 `**kwargs` 透传参数，因此必须在调用处显式传入。设为 `True` 以启用 KV Cache 前缀缓存。

---

## 四、错误触发时序

| 阶段 | 修复编号 | 错误信息 |
|------|---------|---------|
| Graph Capture → `_dummy_run` → attention hook | 修复 1/3 | `AttributeError: ... 'current_layer'` |
| Graph Capture → `_dummy_run` → attention hook | 修复 3（`load_kv_layer` 中 `gpu_kv_caches_load_iter` 保护） | `AttributeError: ... 'gpu_kv_caches_load_iter'` |
| Scheduler Init → `MetadataServer.post_init()` | 修复 4 | `TypeError: ... missing 'enable_caching'` |

> **注意**：Graph Capture 阶段 `start_load_kv()` 不会被调用（因此 `current_layer` 和 `gpu_kv_caches_load_iter` 不会被正常赋值），但 attention 层的 hook 会触发 `wait_for_layer_load()` 和 `load_kv_layer()`。

---

## 五、傻瓜式操作步骤

### 方法一：一键修复脚本（推荐）

```bash
# 1. 把修复脚本上传到服务器
scp fix_all_cpu_offload.py root@7.6.50.28:/data/lzd/fix_all_cpu_offload.py

# 2. 在容器内执行
ssh root@7.6.50.28 "docker exec lzd-lzd_0.18.0-test python3 /data/lzd/fix_all_cpu_offload.py"

# 3. 验证修复结果
ssh root@7.6.50.28 "docker exec lzd-lzd_0.18.0-test python3 /data/lzd/fix_all_cpu_offload.py --verify"
```

### 方法二：手动修改（vi 方式）

#### 第 0 步：进入目录并备份

```bash
cd /vllm-workspace/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/cpu_offload
cp cpu_offload_connector.py cpu_offload_connector.py.bak.$(date +%F-%H%M%S)
cp cpu_kv_cache_manager.py cpu_kv_cache_manager.py.bak.$(date +%F-%H%M%S)
```

#### 第 1 步：修改 cpu_offload_connector.py

```bash
vi cpu_offload_connector.py
```

**改动 A**：在 `CPUOffloadingConnectorWorker.__init__` 里，找到 `self.done_sending_count` 那一行，在其后面加一行：

```python
        self.current_layer = -1
```

**改动 B**：在 `register_kv_caches` 函数体第一行加：

```python
        self.current_layer = -1
```

**改动 C**：在 `wait_for_layer_load` 函数体里，`self.current_layer += 1` 那行**前面**加两行：

```python
        if not hasattr(self, "current_layer"):
            self.current_layer = -1
```

保存退出 `:wq`

#### 第 2 步：修改 cpu_kv_cache_manager.py

```bash
vi cpu_kv_cache_manager.py
```

找到 `get_manager_for_kv_cache_spec(` 调用（约第 69 行），在 `kv_cache_group_id=0,` 后面加一行：

```python
            enable_caching=True,
```

保存退出 `:wq`

---

## 六、验证方法

一键修复脚本自带 `--verify` 模式，会自动对比原始备份和修复后的文件，检查所有改动是否正确到位。也可手动验证：

```bash
# 用 diff 对比原始和修复后的文件
diff cpu_offload_connector.py.bak.2026-03-29-132550 cpu_offload_connector.py
```

预期 diff 输出应该恰好包含上述 4 处改动，不多不少。

---

## 七、修复后运行结果

修复后执行 vLLM 脚本，服务成功启动：

```
INFO 03-29 21:54:00 [api_server.py:580] Starting vLLM server on http://0.0.0.0:9991
INFO:     Application startup complete.
```

8 个 NPU Worker 全部正常运行，每个占用约 49724 MB HBM。
