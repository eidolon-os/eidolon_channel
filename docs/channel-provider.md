# Hub Channel Provider v1

`eidolon-channel-provider` 是 `eidolon_channel` 自有的正式控制面进程。它只负责把 Hub 已
批准、已绑定 Owner 的 Device 映射为有时效的 LiveKit room/token；不负责 Wi-Fi 配网、
人工审批、Companion 选择或 Pi 部署。Agent worker 与 Provider 是两个独立进程。

## 运行与配置

默认配置文件是 `config/channel-provider.yaml`，可用
`EIDOLON_CHANNEL_PROVIDER_SETTINGS_YAML` 指向另一份严格 YAML。默认监听
`127.0.0.1:8767`，Hub 的 `channel_provider.contract_url` 应为
`http://127.0.0.1:8767/v1`。8090 属于 eidolond，不是此协议入口。

秘密只能从进程环境或 `EIDOLON_CHANNEL_PROVIDER_ENV_FILE` 指向的 dotenv 进入：

- `EIDOLON_CHANNEL_PROVIDER_TOKEN`：至少 32 bytes。Hub 侧把同一个随机值放在
  `EIDOLON_HUB_CHANNEL_PROVIDER_TOKEN`；Provider 不接受 query credential。
- `LIVEKIT_API_KEY`、`LIVEKIT_API_SECRET`：只供 Provider 管理 room 和签发 JWT，绝不进入
  Hub handoff。
- `EIDOLON_STATE_ROOT`：Provider 数据库默认位于
  `$EIDOLON_STATE_ROOT/channel/provider.sqlite3`；`traces.root` 默认位于同级的
  `$EIDOLON_LOG_ROOT/channel/traces`（Agent worker 写、Provider 只读）——落在 log root
  而不是 state root，因为产品 Host 上 worker unit 是 `ProtectSystem=strict`，它在
  logs/runtime/cache 三个角色下持有 `eidolon/channel`，而 state 是 `eidolon/voiceprints`；
  `ProtectSystem` 只挡写不挡读，所以 Provider 读得到一个它自己创建不了的目录。
- `EIDOLON_LIVEKIT_CLIENT_URL`：写入设备 binding 的可达 `wss://` origin。非 loopback 的
  `ws://` 会被配置校验拒绝。

LiveKit 管理 API 默认使用进程本机的 `http://127.0.0.1:7880`。Provider 数据目录、数据库
权限分别收紧到 `0700`、`0600`，SQLite 使用 WAL、`synchronous=FULL` 和
`secure_delete=ON`。

启动：

```bash
eidolon-channel-provider
```

`GET /health` 不需要 bearer token，成功返回：

```json
{"status":"ok","service":"eidolon-channel-provider","contract_version":"v1"}
```

它同时探测 Provider SQLite 和 LiveKit 管理 API；任一不可用返回 503。它不宣称 Hub
onboarding、设备可达的 LiveKit `wss://` origin 或 ESP TLS 端到端可用。

## 认证与授权边界

除 `GET /health` 外每个接口都要求精确的 `Authorization: Bearer <token>`；写接口
（`provision`、`revoke`、`current`、`sessions/open`、`sessions/close`）还要求
`Content-Type: application/json`。只读接口（`device-channels/presence`、
`session-traces`）是 GET，不带请求体。Provider 只信任通过 bearer 认证的调用方：

1. 设备 Enrollment 后进入 `pending-approval`，不携带 Owner 认领 secret，也不要求屏幕。
2. 持有 `hub-admin` 权限的管理员在 Hub 管理面选择 `owner_id` 并人工批准；普通 Owner 或
   `device-manager` 不能批准尚未绑定 Owner 的 pending 设备。
3. Provider 直接校验并持久化 SDK canonical `DeviceRef` 五元组，再把独立的业务 `owner_id`
   写入 LiveKit participant metadata。Provider 不重新实现 Hub 管理面授权，也不接受 Mobile 直连。
4. Companion 不在 Channel Provider grant 中绑定。Device 可以先建立 Owner-scoped data
   connection；后续 Companion interaction 必须继续经过 Kernel/System Data runtime authority。

因此 bearer token 泄漏等价于伪造 Hub channel authority，必须按服务密钥存放和轮换；
Provider 的 loopback 默认监听是第二层隔离。

## 只读接口

两个 GET，都不改任何状态，凭据与写接口同一个 bearer。

```text
GET /v1/device-channels/presence
GET /v1/session-traces?owner_id=&companion_id=&since=&limit=
GET /v1/session-traces/{session_id}?kinds=
```

`presence` 回答"哪些 Body 正在自己的 channel 上"——这是本 Host 唯一的 presence 权威。

`session-traces` 提供 Agent worker 写下的**每会话链路追踪**（从进房到离开的时序、每跳
耗时、轮次的 phase/milestone/终态与决策证据）。worker 与 Provider 是同一个 Channel
authority，所以由 Provider 服务这些文件不跨越任何边界；worker 决定是否记录
（channel `settings.yaml` 的 `observability.session_trace_path`，默认关），Provider
只决定去哪里找（`traces.root`）。root 不存在时 `recording:false` + 空列表，不是错误。

**这是唯一一份读实现。** Admin 用已有的 generic proxy
`/api/services/channel-provider/v1/session-traces` 透传，无需改动；CLI
（`scripts/report_session_trace.py`）走同一个 HTTP 契约；将来给 Mobile 看，由 Local API
加一层薄投影，不重写解析。

`limit` 上限 200、默认 50；`since` 取 ISO-8601；`kinds` 用逗号分隔
（`session_open,session_mark,session_close,turn_progress,turn_final,event`）。
非法的 `limit`/`since` 返回 422 而不是被忽略 —— 被丢掉的过滤条件会让调用方把未过滤的
答案当成过滤后的答案。查询按记录内的 `session_id` 匹配，不按文件名：文件名经过脱敏，
而 `conversation_id` 允许 `.` 与 `:`。

## Provision wire contract

Hub 调用 `POST /v1/device-channels/provision`。请求只能使用 SDK 发布的 canonical
`DeviceRef(device_instance_id, owner_domain_id, owner_domain_generation, claim_generation,
trust_epoch)`，Manifest 不进入 DeviceRef。

```json
{
  "operation": "channel.provision-device",
  "operation_id": "<idempotency key>",
  "device_ref": {
    "device_instance_id": "<device instance id>",
    "owner_domain_id": "<owner domain id>",
    "owner_domain_generation": 1,
    "claim_generation": 1,
    "trust_epoch": 1
  },
  "device": {
    "owner_id": "<business owner id in owner_ namespace>",
    "display_name": "<display name>",
    "device_kind": "<hardware/product kind>",
    "manifest": {
      "schema_version": 1,
      "title": "<title>",
      "properties": [],
      "actions": [],
      "events": [],
      "media": []
    },
    "manifest_revision": "<stable manifest revision>"
  }
}
```

输入拒绝未知字段、重复 JSON key、错误类型和超过 256 KiB 的 body。成功响应：

```json
{
  "operation": "channel.provisioned-device",
  "operation_id": "<same idempotency key>",
  "device_ref": {"...": "<same canonical DeviceRef>"},
  "manifest_revision": "<same revision>",
  "channels": [{
    "channel_id": "<stable opaque channel id>",
    "purpose": "livekit-device-session",
    "kinds": ["reliable-data", "realtime-data", "audio"],
    "binding_format": "application/vnd.eidolon.livekit-device+json;v=1",
    "issued_at_ms": 0,
    "expires_at_ms": 0,
    "opaque_binding": "<base64>"
  }]
}
```

`issued_at_ms`、`expires_at_ms` 是 Unix epoch milliseconds。Hub 只校验关联关系、时效并将
opaque binding 交给设备，不解析或持久化其中的 LiveKit secret。

base64 解码后的 binding v2 是单一 session channel：

```json
{
  "schema_version": 2,
  "session": {
    "server_url": "wss://<device-reachable-livekit-origin>",
    "token": "<short-lived LiveKit JWT>",
    "identity": "<device_id>",
    "room_name": "<stable voice room>"
  },
  "audio": {"sample_rate": 16000, "channels": 1}
}
```

默认 JWT TTL 为 1800 秒。token metadata 包含 `kind=device`、`device_id`、`owner_id`、
`device_kind` 和 `interaction_mode`（适用时）—— 都是 channel 存续期内恒为真的事实。grant 只能
加入固定 room；publish source 按 Manifest 限定为 microphone/camera，禁止创建 room 或修改受信
metadata（`can_update_own_metadata=false`）。

token 里**没有** `session_intent`：它描述的是某一次会话而不是这条 channel，且它决定该会话被
允许做什么（见下节）。写进一条设备握在手里数小时的凭据，既与生命周期不符，也等于把授权交给
被授权方。

## 幂等、重试与恢复

- room 名和 `channel_id` 由 `SHA-256(owner_domain_id, device_instance_id)` 稳定派生，不含 Owner
  credential。
- 幂等 scope 是完整 DeviceRef + operation kind + operation id；同 scope 同 payload 返回第一次
  逐 byte 结果，跨进程重启仍成立；同 scope/ID 不同 payload 返回 `IDEMPOTENCY_CONFLICT`。
- credential 到期后 operation 明确进入 `expired/credential_expired`，不再占 active 唯一锁。
  刷新必须用新 idempotency key 和 `channel.refresh-device`；原 provision replay 不延长 TTL。
- 更高 owner-domain/claim/trust generation 到达时，旧 active/expired operation 进入
  `fenced/generation_advanced`；旧代请求返回 `STALE_GENERATION`，不能关闭或覆盖新代。
- `INVALID_TRANSITION`、`UNAUTHENTICATED`、`FORBIDDEN`、`PROVIDER_UNAVAILABLE` 保持独立
  RFC 9457 problem code/category/retryable，不能统一折叠为 409/503。

正式部署仍只有一个 Provider writer；SQLite `BEGIN IMMEDIATE`、五元 scope primary key 和 active
partial unique index 还保证进程竞态/重启不会提交两个 active operation。

## Session wire contract 与 session_intent 的授权边界

一条 channel 上开始/结束一次会话有两条路，它们的**区别只在于谁有资格说出 `session_intent`**：

```json
POST /v1/device-channels/sessions/open
{
  "operation": "channel.open-session",
  "device_ref": {"...": "<complete canonical DeviceRef>"},
  "conversation_id": "<per-conversation id>",
  "session_intent": "user_initiated | presence_initiated | proactive_initiated"
}
```

`session_intent` 可选，缺省 `user_initiated`；`sessions/close` 不接受该字段（结束一次会话没有
意图可言，带上就是 drift，返回 422）。非法取值一律 422 拒绝，**不做降级**：
`normalize_session_intent` 是给不可信 wire value 用的失败安全语义，而这里调用方已通过 bearer
认证，把拼错的 presence 唤醒静默变成普通会话，会让编排方以为自己拿到了并不存在的 Owner lease。
成功响应回显 `session_intent`，调用方据此分辨"被授予"和"被降级"。

另一条路是设备自己在 channel 上发 `session.open` 数据包。它**没有**这个字段，而且不会有：
`ServingRequest` 只携带 `action` 与 `conversation_id`，Provider 在 sink 处直接写死
`user_initiated`。设备请求被听见这件事本身就是 user-initiated 的定义；另外两种意图描述的是
"别人替它决定的唤醒"，尤其 `presence_initiated` 会换来一个由外部治理的可续期 Owner lease ——
能自报意图的 Body 就是在给自己提权。

Provider 把意图写进 **LiveKit agent dispatch metadata**（与 `conversation_id`、`output_plan`
同一条总线），Channel worker 从 `ctx.job.metadata` 读取。选这条总线的理由与
`output_plan` 相同：它是房间里唯一"按会话生成、且参与者写不了"的通道。同一个
`conversation_id` 用不同意图再次 open，会替换而不是复用既有 dispatch —— 否则 agent 会继续按上
一次的规则运行。

worker 侧对缺失/无法识别的意图降级为 `user_initiated`（与控制契约的严格拒绝相反）：能走到这里
的"沉默"只可能来自早于该字段的 Provider，而沉默的安全读法是普通会话。

`generate_token()`（HTTP API / web client 路径）不涉及意图，也不需要涉及：它签发的 token 通过
room config 触发自动 dispatch，dispatch metadata 里没有意图，worker 于是落到
`user_initiated`。这条路径 by construction 就是 user-only。

## Revoke wire contract

Hub 调用 `POST /v1/device-channels/revoke`：

```json
{
  "operation": "channel.revoke-device",
  "operation_id": "<stable revocation request id>",
  "device_ref": {"...": "<complete canonical DeviceRef>"},
  "reason": "<bounded reason>"
}
```

成功返回：

```json
{
  "operation": "channel.revoked-device",
  "operation_id": "<same revocation request id>",
  "device_ref": {"...": "<same canonical DeviceRef>"}
}
```

Provider 先停止监听并删除当前 channel，再原子标记同一 DeviceRef credential 已 revoke、清空
adapter handle/expiry，并记录可重放 revoke response。未知 DeviceRef 也可记录目标状态成功；旧代
revoke 返回 `STALE_GENERATION`，绝不能作用于新代 channel。

LiveKit JWT 没有单 token 主动撤销 API。删除 room 会断开当前参与者；grant 明确禁止
`roomCreate`，旧 token 还受原 TTL 上限约束。部署仍应把 LiveKit room auto-create 策略纳入审计，
不能把数据库清空等同于密码学即时失效。

## 真机 TLS 硬门禁

Channel Provider 完成不等于 ESP handoff 已经可用，至少存在两段独立 TLS origin：

1. **ESP → Hub descriptor/enrollment/handoff。** 当前 ESP `HttpClient` 的 EspSsl 强制使用公开
   CA bundle。`https://eidolon-hub.local` 指向没有证书/TLS listener 的 Pi 时必定失败；mDNS
   只能发现 endpoint，不能建立身份或信任。`hub.eidolon.live` 当前指向 Vercel，也不能假装是
   Pi LAN origin。正式方案只能是：
   - 经已认证 provisioning 安全通道向设备写入 Hub endpoint 和绑定的 CA/SPKI trust material；
     或
   - 使用设备公开 CA bundle 可验证、SAN 与 endpoint 匹配且 LAN 可达的公有证书。
2. **ESP → LiveKit。** `active/control.server_url` 同样必须是设备可达的 `wss://` origin，并有
   ESP 能验证的证书或通过同一认证 provisioning 安装的明确信任材料。

不得关闭证书校验、接受任意自签证书或把 mDNS TXT 当作信任根。在上述 Hub TLS gate 完成前，
Provider 可以独立通过合同与 LiveKit 控制面测试，但不能据此报告 Box3 已完成真实 handoff。


### Device Manifest 更新与现有 Channel

同一 DeviceRef 的新 Manifest 通过 `channel.refresh-device` 更新既有 Channel，
不重复 provision，也不改变 Claim。refresh 在凭据过期或 Authority 提交的
manifest_revision 与当前 active 凭据不同，或 adapter 观察到绑定依赖的运行环境已变化时成立。
未过期、声明相同且运行环境仍匹配的 refresh 仍拒绝。刷新原子地 fence 旧 active/expired 操作，保留同一传输资源，
重放旧操作不能复活旧凭据，撤销后的刷新仍拒绝。

Hub 先读 current，再用当前 operation_id、到期时间及目标声明标识派生 refresh
的幂等键。A→B→A 因而是两次沿当前 Channel 前进的刷新，不会重放首次 A 的操作。
设备在模式变化后用新 binding 重新入房，让新的声明进入 participant metadata。


### 运行环境变化与路由候选

`channel.current-device` 的响应可携带 `refresh_required: true`。省略等价于 false。
这是 adapter 对当前绑定运行时有效性的判断，不改写 binding 的凭据到期时间，
也不在查询过程中创建资源。Hub 收到该标记后沿上述幂等 refresh 流程前进。
Provider 的事务只接受仍与被观察记录的 operation_id、adapter 和 handle 相同的
active 记录作为提前刷新的依据；外部请求不能自行声明运行环境失效。

LiveKit 本地动态地址策略每次从活跃网卡获取候选，默认出口仅参与排序，没有默认
路由仍可以提供局域网服务。显式远端 URL 保持配置策略；策略变更同样使旧绑定失效。
私有 handle 保存签发时的候选用于比较，旧版未保存候选的 handle 会经过一次刷新。

v2 session 保留 `server_url`。可选 `server_urls` 是同一房间和凭据的有序信令地址
列表，非空、无重复，首项必须等于 `server_url`。省略时仅使用 `server_url`。
该字段由 SDK `DF-LIVEKIT-SESSION-BINDING-001` 的 routing 正反例定义；Hub 不解析
opaque payload。客户端在自身连接生命周期中尝试候选，信令地址不能替代 ICE 候选
或作为 Host 身份证明。Hub 与 Provider 需一起更新，多候选能力需更新客户端。
