# Eidolon Agent（Brain）侧性能优化建议

> 来源：`eidolon_channel` 团队
> 触发：channel ↔ agent gRPC 集成端到端验证完成后的实测观察
> 性质：**给 brain 团队的优化建议**，channel 侧已经做了能做的部分（[commit history](https://git.aimanthor.com/jarvis/eidolon/eidolon_channel)：`30160b6` → `9f46fdb`），下列项需要 brain 侧动手才能根治

---

## 背景

`eidolon_channel`（LiveKit 语音前端）已经通过 `EidolonAgent.Chat` gRPC 接入 `eidolon_agent`，
真实跑通了 STT → brain → TTS 全链路。我们对 channel 这一段做了完整的可靠性 / 可观测性 /
性能优化。但**实测发现真正的延迟大头不在 channel 这段**，channel 整段占总 TTFD < 10 ms，
**0.6 ~ 2 秒的热路径首字延迟和 42 秒的冷启动延迟，都在 brain 端**。

本文档列出从 channel 侧观察到的 6 个 brain 侧瓶颈和具体优化建议。每一项标注了"为什么
channel 上不能根治"，避免将来双方互相推。

### 实测数据（基线）

| 场景 | 首字延迟 (TTFD) | 总耗时 | 备注 |
| --- | --- | --- | --- |
| 新 `(tenant, user)` 第一通 | **42440 ms** | 42544 ms | "冷启动"，brain 端 `CompanionAgent` 懒加载 |
| 同 user 第二通（任意 conversation_id） | 684 ms | 833 ms | 热路径 |
| 同 user 第三通 | 2007 ms | 2153 ms | 热路径但波动大 |
| 同 user 多轮（单 adapter，共享 bidi 流） | 641-1205 ms | 765-1225 ms | 热路径，stream 复用 |

环境：channel worker 和 eidolon_agent 同机部署（supervisord），gRPC over loopback
`127.0.0.1:45051`，TLS off。LLM 后端走 brain → litellm 网关 → 上游模型。

> ⚠️ 实测后续多次出现 `litellm 502 Bad Gateway`，重试到 max_retry 后失败（4 次）。这是
> brain → 上游 LLM 的间歇性问题，与 channel 无关，但**影响真实用户体验**，单独列在 B-5。

---

## B-1. **CompanionAgent 懒加载 → 首通 42 秒冷启动**（最高优先级，最大单点延迟）

### 现象

`AgentRegistry` 按 `(tenant_id, user_id)` 懒创建 `CompanionAgent` 实例。新 user 第一次
`Chat()` 进来，brain 端要现场做这些事，串行：

- 加载 persona instance（YAML 或从 template 复制生成）
- 初始化 `HistoryManager`（SQLite 查询 / 建行 / 启动 sliding window）
- 订阅 NATS topic（多个 KV bucket 首次访问，需要 fetch bucket info + 建 consumer）
- 启动 `SignalBus` 的 ring buffer
- 第一次调上游 LLM（litellm 路由 + 上游模型可能 cold start）

整段串行下来 **~42 秒**。第二次以后同 user 任意 conversation_id 都走热路径 600 ms - 2 秒。

### 影响

用户第一次和陪伴智能体说话 —— 也就是最重要的"第一印象"那一通对话 —— 要等 42 秒才听到回应。
按当前部署模型（一个 worker 绑一个 user），意味着**每次部署、每次重启、每个新 user，至少有一个用户体验是
42 秒延迟**。

### 为什么 channel 不能根治

`CompanionAgent` 实例的状态完全在 brain 进程里。channel 没有任何方式预先触发它的创建，
除非主动调一次 fake `Chat()` —— 这是 workaround，不是修复。

### 推荐措施

按 ROI 排序：

#### 1. **进程启动时批量预热近期活跃 user**（最高 ROI）

brain 启动时查 `devices` 表（按 `last_active_at` 降序取最近 N 天活跃的 user），对每个
batch 调一次内部 `_create_companion_agent()`。**完全不调上游 LLM，只完成本地状态初始化**
（persona、history、NATS 订阅、SignalBus）。

伪代码：
```python
async def on_brain_startup():
    recent_users = await db.query(
        "SELECT tenant_id, user_id FROM devices "
        "WHERE last_active_at > NOW() - INTERVAL '7 days' "
        "LIMIT 100"
    )
    await asyncio.gather(*[
        agent_registry.preload(t, u) for t, u in recent_users
    ])
    logger.info("brain preloaded %d CompanionAgents", len(recent_users))
```

预期收益：activate 用户在 brain 重启后第一通对话**直接热路径**，省 42 秒。

#### 2. **拆分懒加载阶段并行化**

当前 5 件事是**串行**的（persona load → history init → NATS subscribe → SignalBus → LLM warm）。
其中 persona / history / NATS 三件事**没有依赖关系**，可以 `asyncio.gather` 并行：

```python
async def _create_companion_agent(tenant_id, user_id):
    persona_task = load_persona_instance(user_id)
    history_task = init_history_manager(user_id)
    nats_task = subscribe_user_topics(user_id)
    persona, history, _ = await asyncio.gather(persona_task, history_task, nats_task)
    return CompanionAgent(persona, history, ...)
```

预期收益：把懒加载从 42 秒压到 ~10-15 秒（取决于 max 三者）。

#### 3. **LLM 上游 connection / model warm**

brain 进程启动时对 litellm 发一个 `model warmup` 空 prompt 触发：
- DNS / TLS / TCP / HTTP 连接池建立
- 如果上游是会动态加载模型的服务（vLLM、Triton 等），让它把模型加载到 GPU

预期收益：消除 first-call 的连接 + 模型加载延迟，**这一项可能占 42 秒里的 10-20 秒**。

### Channel 侧 workaround（已记录，未实施）

我们在 channel 计划里**故意没做**这个 workaround，把球留给 brain 团队：

> channel worker 的 `_prewarm` 同步 hook 里 `asyncio.run` 一次 fake chat，target 是
> 本 worker 绑定的那个 user（user_id 从 JWT payload 解出）。把 42 秒摊到 worker 启动时间。

这是**治标**。如果 brain 团队按上面 1+2+3 修了，channel 不需要做这个 workaround。
**强烈倾向于让 brain 修**，因为：
- brain 修一次，所有 channel / web / IoT 等所有 caller 都受益
- channel 修是补丁，每个 caller 都得重复
- brain 内部更清楚冷启动每一步的时长，能做更精细的 profile

---

## B-2. **Token rotation RPC 缺失**

### 现象

`scripts/provision_eidolon_token.py` 取号默认 `ttl=30 天`。device_token 是签好的 JWT，
30 天过期那一刻 brain 直接返回 `UNAUTHENTICATED`。channel worker 没有任何机制提前感知 +
主动续期。**第 31 天某时刻所有对话突然全部失败**，health probe（gRPC 长连）仍然显示绿色，
排查极其困难。

### 为什么 channel 不能根治

token 续期协议必须 brain 定义。channel 自己不能签发新 token。

### 推荐措施

admin HTTP 暴露一个 rotation 端点：

```
POST /api/admin/devices/{device_id}/rotate
Authorization: Bearer <current_token>   # 当前 token（即使快过期也仍有效）
Body: {}

Response 200:
{
  "device_id": "...",
  "device_token": "<new JWT>",
  "expires_at": "<ISO timestamp>"
}

Response 401:
  token 已经过期超过 grace window（比如 7 天）→ 需要重新走 pairing
```

逻辑：
- 接受 Bearer auth（包括 token 已快过期但仍在有效期内）
- 验证当前 token 的 device_id 匹配 URL 里的 device_id
- 用同一个 `jwt_secret` 重新 `sign_device_token()`，复用原 payload（tenant_id / user_id /
  template_id），延长 `exp` 到当前时间 +30 天
- 在 NATS KV `DEVICE_REVOCATIONS` 检查是否被吊销

### Channel 侧后续动作（需要 brain 这个 RPC 落地后才能做）

参见 channel 仓库 [docs/todo.md](./todo.md)：worker 启动时检查 JWT exp，剩余 < 7 天时主动
调这个 rotation 端点换新 token，写回 `.env`。

---

## B-3. **`eidolon.proto` 没有工具注册字段**

### 现象

LiveKit 框架允许给 LLM 注册工具（function call tools，如 `EmitEvent`、`LookupCalendar`
等）。当前 `eidolon.proto` 的 `StartTurn` 没有 tool 字段，channel 这边收到 LiveKit 注册的
工具只能**直接吞掉**：

```python
# eidolon/livekit/agent/eidolon_agent_rpc/grpc_llm.py
def chat(self, ..., tools: list[Tool] | None = None, ...):
    if tools:
        logger.debug("tools are not forwarded over eidolon.agent.v1")
```

### 影响

LiveKit 生态里有大量 LLM 工具（function calling、JSON schema 工具描述等）开发都做不了。
channel 这一侧的 `LlmStage` 抽象支持工具，brain 这边收不到 → 整链路工具能力为零。

### 为什么 channel 不能根治

协议改动必须 brain 一起接 proto。

### 推荐措施

`eidolon.proto` 的 `StartTurn` 加：

```proto
message StartTurn {
    string turn_id = 1;
    string conversation_id = 2;
    string text = 3;
    google.protobuf.Struct realtime = 4;
    google.protobuf.Struct metadata = 5;
    repeated ToolDescriptor channel_tools = 6;  // 新增：通道提供的工具
}

message ToolDescriptor {
    string name = 1;
    string description = 2;
    string json_schema = 3;     // JSON Schema 描述参数
}
```

brain `TurnEngine` 在 dispatch 阶段把 `channel_tools` 合并到 brain 自带工具集，给 LLM
做工具选择。channel 通过现有的 `TurnEvent.TOOL_CALL` 收回调，执行后通过新的 `ChatRequest.ToolResult`
帧回填：

```proto
message ChatRequest {
    oneof payload {
        StartTurn start = 1;
        CancelTurn cancel = 2;
        PushSignalInline signal = 3;
        ToolResult tool_result = 4;  // 新增：channel 工具执行结果回填
    }
}

message ToolResult {
    string turn_id = 1;
    string tool_call_id = 2;       // 对应 TOOL_CALL 事件里的 id
    string result_json = 3;
    string error = 4;              // 非空表示工具执行失败
}
```

---

## B-4. **Memory recall 延迟超预算？**

### 现象

`eidolon_agent` 内部文档说 memory recall 预算 ≤ 250 ms（带 200 ms timeout）。但 channel
这边实测第三轮 TTFD **2007 ms**，远超 brain plan 里写的"first DELTA < 100 ms (test
assertion, no memory)"基准。

不能确定是不是 memory recall 慢，但波动模式（同 user 同 conversation 多轮，第二轮 684 ms
第三轮 2007 ms）很像是某些路径上 memory recall 在轮次间被异步触发，下一轮被它阻塞。

### 为什么 channel 不能根治

memory 是 brain → eidolon-memory 服务的内部调用，channel 看不见。

### 推荐措施

#### 1. 加分段 timestamp 日志

在 `TurnEngine` 的每个阶段（guardrail / triage / context compile / memory recall / LLM
TTFT / output guardrail）打 timestamp，让我们能直接看 P50 / P95 / P99 分布：

```python
logger.info(
    "[TurnEngine] turn=%s timings_ms="
    "guardrail=%d triage=%d compile=%d memory=%d llm_ttft=%d total=%d",
    turn_id, t_guardrail, t_triage, t_compile, t_memory, t_llm_ttft, t_total,
)
```

#### 2. memory recall 如真普遍超预算，改非阻塞

**方案 A**：调大 timeout 到 500-800 ms，接受 memory recall 影响 TTFD（语义质量优先）。

**方案 B**（更有趣）：把 memory recall 改成**非阻塞**。

- TurnEngine 同时启动 memory recall 和 LLM 推理
- LLM 先用基础 context 开始生成 DELTA
- memory recall 完成后通过新的 `TurnEvent.MEMORY_HIT` 事件追加 context
- 后续 turns 享受到 recall 结果，本轮 first DELTA 不被它阻塞

这要求 brain plan 里允许"first DELTA 在 memory recall 完成前发出"，需要权衡。

---

## B-5. **LLM TTFT 高且波动大（包括 litellm 502）**

### 现象

热路径 TTFD 600 ms - 2 秒，波动大。同时**实测多次出现** `litellm 502 Bad Gateway`，
重试到 `max_retry=3` 后彻底失败（4 次尝试）。channel 这边的报错堆栈：

```
livekit.agents._exceptions.APIConnectionError: failed to generate LLM completion after 4 attempts
```

### 为什么 channel 不能根治

LLM 推理是 brain → litellm 网关 → 上游模型，channel 完全无干预空间。

### 推荐措施

#### 1. 看 litellm 的 connection 池配置

是不是每次都重新 DNS / TLS / TCP 到上游？换成长连池 + keepalive，跟我们 channel 这边
对 brain 做的事一样（参见 channel commit `eb652ac`）。

#### 2. 上游路由稳定性

如果上游是负载均衡到多个 model 实例的，每次路由到不同实例 → 每次都是 cold start。
绑定单个 user 到固定上游 instance（sticky routing），让上游 KV cache 累计起来。

#### 3. system prompt 优化

如果 system prompt 含完整人格描述（几 K token），每次都要 prefill 一遍，TTFT 大头就在
这里。措施：

- 用 prompt cache（OpenAI / Anthropic / 国产 LLM 大多支持）
- 精简 system prompt
- 把"长期人格描述"挪到 fine-tune 模型权重里

#### 4. litellm 502 的根因

502 通常意味着 litellm 后面的上游不可达 / 超时。建议：

- litellm 加重试 + 多上游 fallback
- brain 这一侧打详细 litellm 日志（status code、上游 latency、error response）
- 上游 model 服务的健康检查 + 自愈

#### 5. Triage 阶段慢路径

brain plan 里写 triage < 1 ms。如果 triage 模型本身也调 LLM（哪怕是小模型），实测可能远不止 1 ms。
建议把 triage 改成纯规则 / embedding cosine 之类的无 LLM 路径。

---

## B-6. **NATS JetStream KV 首次访问慢**

### 现象

B-1 提到的 42 秒冷启动里，有一部分是 NATS KV bucket 首次访问的代价。brain 用 NATS KV 存
cache / session context / rate limits / feature flags / 配置 等多个 bucket。第一次访问每个
bucket 时 NATS 客户端要：

- fetch bucket info
- 订阅 ACK subject
- 建 stream consumer

每个 bucket 第一次访问几百毫秒到 1-2 秒不等，累加起来可观。

### 为什么 channel 不能根治

NATS 客户端在 brain 进程里。

### 推荐措施

brain 进程启动时**预热所有 KV bucket handle**：

```python
async def on_brain_startup():
    bucket_names = ["cache", "session_context", "rate_limits", "feature_flags", ...]
    await asyncio.gather(*[
        nats_kv.get_or_create_bucket(name) for name in bucket_names
    ])
    logger.info("brain warmed %d NATS KV buckets", len(bucket_names))
```

完成后所有 bucket handle 在内存里，后续访问直接命中本地缓存，毫秒级。

---

## 总结表

| 编号 | 痛点 | 实测延迟代价 | 推荐措施 | 工作量估计 | Channel 能否根治 |
| --- | --- | --- | --- | --- | --- |
| **B-1** | CompanionAgent 懒加载 | **42 秒首通** | 启动预热活跃 user + 懒加载并行化 + LLM warm | 2-3 天 | ❌ |
| **B-2** | Token rotation RPC 缺失 | 第 31 天突然失败 | 加 admin rotation 端点 | 1 天 | ❌ |
| **B-3** | Proto 工具字段缺失 | 功能阻塞（工具调用不可用） | 加 `StartTurn.channel_tools` + `ToolResult` | 半天 + 协调 | ❌ |
| **B-4** | Memory recall 可能超预算 | TTFD 抖动 2 秒 | 分段日志 + 非阻塞 recall | 1-2 天 | ❌ |
| **B-5** | LLM TTFT 高 + litellm 502 | TTFD 600-2000 ms + 整链路 fail | litellm 长连 + sticky 路由 + prompt cache + 502 自愈 | 1-3 天 | ❌ |
| **B-6** | NATS KV 首访慢 | 首通延迟一部分 | 启动预热 bucket handles | 半天 | ❌ |

---

## 建议优先级

按用户感知 × ROI 排：

1. 🔥🔥🔥 **B-1**（42 秒冷启） — 最大单点延迟，做了立竿见影
2. 🔥🔥🔥 **B-5**（litellm 502） — 当前生产链路实测在挂，影响功能可用性
3. 🔥🔥 **B-6**（NATS KV 预热） — 附带在 B-1 一起做，工作量小
4. 🔥🔥 **B-4**（memory recall） — 先加日志看分布，再决定怎么修
5. 🔥 **B-3**（工具字段） — 功能扩展，等下一波产品需求
6. 🔥 **B-2**（token rotation） — 第 31 天之前修就行，当前不紧急

---

## 联系 / 协作

`eidolon_channel` 这边已经在协议契约上预留了所有需要的扩展位（typed event handlers、
TLS 模式、UDS 支持等）。一旦 brain 团队上述任何一项落地，channel 端基本零工作量配合：

- B-2 落地 → channel 仓库 [docs/todo.md](./todo.md) 里"JWT expiry detection"对应实施
- B-3 落地 → channel 仓库的 LiveKit 工具 forwarding 可以 unblock
- B-4 加分段日志后 → channel 可以把这些日志通过 `TurnEvent.STATE` 暴露给运维 dashboard

有疑问请联系 channel 团队。

---

**附：本文档涉及的 channel 侧实测脚本**

实测代码就是从 channel 仓库的 [scripts/provision_eidolon_token.py](../scripts/provision_eidolon_token.py)
取号后，直接调 `EidolonAgentGrpcLlm.chat()` 跑多轮，测量首字延迟。任何 brain 团队成员都
可以自己复现，需要协助请直接联系 channel 团队。
