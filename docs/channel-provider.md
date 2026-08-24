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
  `$EIDOLON_STATE_ROOT/channel/provider.sqlite3`。
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

两个写接口都要求精确的 `Authorization: Bearer <token>` 和
`Content-Type: application/json`。Provider 只信任通过 bearer 认证的 Hub：

1. 设备 Enrollment 后进入 `pending-approval`，不携带 Owner 认领 secret，也不要求屏幕。
2. 持有 `hub-admin` 权限的管理员在 Hub 管理面选择 `owner_id` 并人工批准；普通 Owner 或
   `device-manager` 不能批准尚未绑定 Owner 的 pending 设备。
3. Provider 直接校验并持久化 SDK canonical `DeviceRef` 五元组，再把独立的业务 `owner_id`
   写入 LiveKit participant metadata。Provider 不重新实现 Hub 管理面授权，也不接受 Mobile 直连。
4. Companion 不在 Channel Provider grant 中绑定。Device 可以先建立 Owner-scoped data
   connection；后续 Companion interaction 必须继续经过 Kernel/System Data runtime authority。

因此 bearer token 泄漏等价于伪造 Hub channel authority，必须按服务密钥存放和轮换；
Provider 的 loopback 默认监听是第二层隔离。

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
`device_kind`、`interaction_mode`（适用时）和 `session_intent=user_initiated`。grant 只能加入固定
room；publish source 按 Manifest 限定为 microphone/camera，禁止创建 room 或修改受信 metadata。

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
