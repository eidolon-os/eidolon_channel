# LiveKit Agent Server 架构分析

> 生成时间: 2026-04-24
> 代码路径: `eidolon/livekit/`

---

## 1. 整体架构图

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           Eidolon Agent Server (eidolon_daemon)                  │
│                                                                                 │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                         server.py (main entry point)                    │   │
│  │                                                                             │   │
│  │   AgentConfig ──loads──> .env + env vars                                   │   │
│  │                                                                             │   │
│  │   _build_llm() ──> livekit.plugins.openai.LLM (OpenAI-compatible)           │   │
│  │   _build_stt() ──> SttStage + BailianFunASRSTT (百练 FunASR)              │   │
│  │   _build_tts() ──> TtsStage + SenseTimeTTS (商汤 SenseAudio)              │   │
│  │   _build_vad() ──> FireredPvadVAD.load() (FireRed pVAD)                   │   │
│  │                                                                             │   │
│  │   AgentServer(ws_url, api_key, api_secret, host, port)                      │   │
│  │        │                                                                    │   │
│  │        └───> server.rtc_session(_on_session)  注册会话回调                   │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                       │                                         │
│                                       ▼                                         │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                    LiveKit Agent Framework (livekit.agents)                 │   │
│  │                                                                             │   │
│  │   AgentServer ────────────────────────────> LiveKit Cloud/Self-Hosted      │   │
│  │        │                                        Server (端口 7880)          │   │
│  │        │                                                                    │   │
│  │        └── _on_session() 触发 ──spawn worker──> 每个 Room 分配一个 worker     │   │
│  │                                           (macOS spawn 模式会重新 import)  │   │
│  │                                                                             │   │
│  │   在 worker 内:                                                            │   │
│  │   _on_session(ctx: JobContext)                                              │   │
│  │        │                                                                    │   │
│  │        ▼                                                                  │   │
│  │   run_agent(ctx, cfg)                                                      │   │
│  │        │                                                                    │   │
│  │        ├── SharedStageFactory(VoiceConfig)                                  │   │
│  │        │        ├── stt_stage (BailianFunASRSTT)                             │   │
│  │        │        ├── llm (livekit.plugins.openai.LLM)                         │   │
│  │        │        ├── tts_stage (SenseTimeTTS)                                 │   │
│  │        │        └── vad (FireredPvadVAD)                                     │   │
│  │        │                                                                    │   │
│  │        ├── AgentSession(                                                    │   │
│  │        │        stt=stt, llm=llm, tts=tts,                                  │   │
│  │        │        vad=vad,                                                    │   │
│  │        │        turn_detection=ChineseModel()                                │   │
│  │        │   )                                                               │   │
│  │        │        │                                                           │   │
│  │        │        ├── RoomIO(session, room)                                    │   │
│  │        │        │        │                                                  │   │
│  │        │        │        └───> room.local_publish_audio()  发布音频到房间     │   │
│  │        │        │                                                           │   │
│  │        │        └── session.start(agent, room, ...)  开始会话                 │   │
│  │        │                                                                    │   │
│  │        ├── StreamingPipeline (AGENT_MODE=streaming, 默认)                     │   │
│  │        │    或                                                               │   │
│  │        └── BatchPipeline   (AGENT_MODE=batch)                                │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 目录结构

```
eidolon/livekit/
├── agent/
│   ├── server.py           # 主入口: AgentServer 启动、配置加载、插件构建
│   ├── factory.py         # SharedStageFactory: 统一创建 stt/llm/tts/vad 实例
│   ├── streaming.py        # StreamingPipeline: 实时流式音频处理
│   ├── batch.py           # BatchPipeline: 批量音频 blob 处理
│   ├── __init__.py
│   └── pipeline/
│       ├── pipeline.py    # VoicePipeline: 统一编排器 (包含 streaming + manual 两种模式)
│       ├── stt.py         # SttStage: 封装 BailianFunASRSTT
│       ├── tts.py         # TtsStage: 封装 SenseTimeTTS
│       ├── vad.py         # FireredVadStage + VadStage: 封装 FireredPvadVAD
│       ├── llm.py         # LivekitLlmStage: 封装 livekit.plugins.openai.LLM
│       ├── types.py       # PipelineMode, PipelineState, PipelineCallbacks 等类型定义
│       ├── canceller.py   # PipelineCanceller: 跨 stage 协调打断
│       └── __init__.py
└── plugins/
    ├── __init__.py
    ├── stt/
    │   └── bailian/
    │       ├── stt.py           # BailianFunASRSTT (实现 livekit.agents.stt.STT)
    │       ├── speech_stream.py # BailianFunASRSpeechStream (流式 WebSocket)
    │       ├── connection_manager.py  # BailianConnectionManager (WebSocket 管理)
    │       ├── registry.py
    │       ├── config.py         # BailianSTTConfig
    │       ├── models.py        # FunASR JSON 消息解析
    │       └── test_*.py
    ├── tts/
    │   └── sensetime/
    │       ├── tts.py           # SenseTimeTTS + SenseTimeSynthesizeStream
    │       ├── tts_client.py    # SenseTimeTTSClient (WebSocket 客户端)
    │       ├── protocol.py      # WebSocket 协议常量
    │       ├── config.py        # SenseTimeTTSConfig
    │       └── test_*.py
    ├── vad/
    │   └── firered/
    │       ├── vad.py           # VAD + VADStream (实现 livekit.agents.vad.VAD)
    │       ├── processor.py     # PvadProcessor (ONNX 推理, 共享单例)
    │       ├── config.py        # FireredPvadConfig
    │       └── test_*.py
    └── eot/
        ├── __init__.py
        ├── _plugin.py          # EidolonEOTPlugin (LiveKit 插件注册)
        ├── config.py           # EidolonEOTConfig
        ├── log.py
        ├── version.py
        ├── models/
        │   ├── base.py         # EidolonEOTModel (实现 LiveKit _TurnDetector 协议)
        │   ├── chinese.py       # ChineseModel (中文 EOT)
        │   └── multilingual.py  # MultilingualModel (多语言 EOT)
        ├── impl/
        │   ├── eot_manager.py          # EotManager (ONNX 推理单例)
        │   ├── context_enhanced_eot.py # ContextEnhancedEot (上下文增强)
        │   ├── turn_end_policy.py      # TurnEndPolicy (动态静默阈值)
        │   ├── state.py                # TurnDetectionStateManager (VAD/ASR 状态)
        │   ├── eot_policy.py          # PolicyChain (切句决策规则)
        │   └── eot_backend.py
        ├── firered/
        │   └── eot.py          # FireRedChatEOT (keyword fallback EOT)
        └── test_eot_plugin.py
```

---

## 3. 两种 Pipeline 模式对比

| | StreamingPipeline | BatchPipeline |
|---|---|---|
| 触发条件 | `AGENT_MODE=streaming` (默认) | `AGENT_MODE=batch` |
| 音频处理 | 实时流式音频 (VAD 驱动) | 客户端上传完整音频 blob |
| 打断能力 | 支持 (用户可随时打断 Agent 回复) | 不支持 |
| EOT 检测 | ChineseModel (EidolonEOTModel) | 无 |
| 编排方式 | `AgentSession` 全权管理音频流 | 手动顺序: STT→LLM→TTS |
| WebSocket | STT/TTS 各自保持长连接 | STT 每次新建连接 |
| 适用场景 | 实时对话、语音助手 | 异步音频处理、消息回复 |

### Streaming 模式数据流

```
VAD 检测语音开始 → STT 实时转写 → LLM 流式生成 → ChineseModel EOT 检测 → TTS 流式合成 → Room 发布音频
```

### Batch 模式数据流

```
用户上传音频 blob → track_subscribed → 等待 track_ended
  → STT.recognize() (一次性 WebSocket)
  → LLM.chat() (OpenAI API)
  → TTS.synthesize() (流式 WebSocket)
  → room.local_publish_audio() → 发布到 Room
```

---

## 4. Plugin 架构详解

### 4.1 STT Plugin (Bailian FunASR)

```
BailianFunASRSTT (stt.py) ──实现 livekit.agents.stt.STT 接口
     │
     ├── stream() ──> BailianFunASRSpeechStream (speech_stream.py)
     │        │
     │        ├── AudioByteStream (accumulate audio chunks → 100ms 帧化)
     │        ├── BailianConnectionManager (WebSocket 连接管理)
     │        │        │
     │        │        └───> WebSocket: wss://dashscope.aliyuncs.com/api-ws/v1/inference
     │        │
     │        ├── send_loop: push_frame() → audio_bytes → WebSocket.send()
     │        └── recv_loop: WebSocket.recv() → FunASR JSON → SpeechEvent
     │                │
     │                ├── INTERIM_TRANSCRIPT → 实时中间结果
     │                └── FINAL_TRANSCRIPT  → 最终识别文字
     │
     └── _recognize_impl() ──> BailianConnectionManager (一次性 WebSocket)
                                  用于 BatchPipeline 的整段音频识别
```

**关键配置:**
- `model`: `fun-asr-realtime-2026-02-28`
- `api_url`: `wss://dashscope.aliyuncs.com/api-ws/v1/inference`
- `language`: 默认 `"zh"`
- `itn`: 逆文本规范化 (数字转阿拉伯数字, 默认开启)

---

### 4.2 TTS Plugin (SenseTime SenseAudio)

```
SenseTimeTTS (tts.py) ──实现 livekit.agents.tts.TTS 接口
     │
     ├── synthesize(text) ──> 内部调用 stream()
     │
     └── stream() ──> SenseTimeSynthesizeStream
              │
              ├── AudioByteStream (60ms PCM framing)
              ├── SenseTimeTTSClient (WebSocket 连接)
              │        │
              │        └───> WebSocket: wss://api.senseaudio.cn/ws/v1/t2a_v2
              │
              ├── send_loop: _input_ch (token text) → send_task_continue()
              └── recv_loop: task_continue 音频块 → output_emitter.push()
                      │
                      ├── 接收 hex PCM 音频数据
                      ├── 丢弃 leading silence (全零片段)
                      ├── 帧化为 60ms 块
                      └── output_emitter.push(pcm_bytes) → AudioFrame
```

**关键配置:**
- `model`: `SenseAudio-TTS-1.0`
- `voice`: `female_0033_a`
- `sample_rate`: `16000`
- `speed`: `1.0`

---

### 4.3 VAD Plugin (FireRed pVAD)

```
FireredPvadVAD.load() (vad.py) ──实现 livekit.agents.vad.VAD 接口
     │
     ├── VAD.load() ──> PvadProcessor (共享 ONNX session, lazy load)
     │
     └── stream() ──> VADStream
              │
              ├── PvadProcessor (ONNX 推理, 单例共享)
              ├── SpeakerEmbExtractor (说话人自适应, update_speaker())
              │
              ├── ThreadPoolExecutor (max_workers=1)
              ├── ExpFilter (alpha=0.5, 平滑概率曲线)
              │
              └── 状态机:
                  WAITING ──speech_prob >= 0.5 且持续 >= 100ms──> SPEAKING
                    ↑            │
                    └────── silence >= 400ms ──┘

              VADEvent 触发:
                START_OF_SPEECH ──> AgentSession 唤醒 STT
                END_OF_SPEECH  ──> AgentSession 触发 LLM

              关键参数:
                activation_threshold: 0.5 (语音概率阈值)
                min_speech_duration: 0.1s (最小语音时长)
                min_silence_duration: 0.4s (最小静音时长)
                prefix_padding_duration: 0.5s (语音前缀补白)
                max_buffered_speech: 60s (最大缓冲语音)
```

**设计亮点:** ONNX session 是**类级别共享单例** (`VAD._processor`)，多个 VADStream 实例共享同一个模型权重，只有流状态缓冲区是 per-stream 的。

---

### 4.4 EOT Plugin (End-of-Turn 检测)

```
ChineseModel (models/chinese.py) ──继承 EidolonEOTModel
     │
     └── EidolonEOTModel (models/base.py)
              │
              ├── EotManager (impl/eot_manager.py) ── ONNX 推理单例
              ├── ContextEnhancedEot ── 上下文增强打分
              ├── TurnEndPolicy ── 动态静默阈值
              ├── TurnDetectionStateManager ── VAD/ASR 状态
              └── PolicyChain ── 切句决策规则

实现 LiveKit _TurnDetector 协议:
  predict_end_of_turn(chat_ctx) → score [0.0, 1.0]
  unlikely_threshold(language)  → 阈值
  supports_language(language)  → True

分数含义:
  1.0 = 用户非常可能继续说话 (agent 不应打断)
  0.0 = 用户非常可能已经说完 (agent 可以开始回复)
```

**EOT 决策机制:**

```
用户说完 → VAD 触发 END_OF_SPEECH
  → STT 返回 FINAL_TRANSCRIPT
    → EOT Model 分析用户文本
      → predict_end_of_turn() → 分数 >= threshold
        → AgentSession 开始 LLM 生成回复
          → 每个 LLM token 都经过 EOT 检测
            → 分数 <= unlikely_threshold (默认 0.08)
              → TTS 开始合成并发布音频
```

**备选 FireRedChatEOT (keyword fallback):**

```
end_punct (。！？!」』"'") → 0.05 (句号感叹号问号 = 很可能是结尾)
mid_punct (，；:)：  → 0.8  (逗号分号 = 还在句子中间)
no_punct            → 0.95 (无标点 = 还没结束)
问号: 短文本 → 0.7, 长文本 → 0.2
```

---

## 5. 启动与连接时序图

```
┌──────────────┐     ┌────────────────────────┐     ┌──────────────────┐     ┌────────────────────────────┐
│   运维人员    │     │   Eidolon Agent Server  │     │   LiveKit Server  │     │   客户端 (Web/App/SDK)       │
└──────┬───────┘     └───────────┬──────────────┘     └────────┬─────────┘     └──────────────┬───────────────┘
       │                         │                             │                          │
       │ python -m eidolon.livekit.agent.server        │                          │
       │────────────────────────>│                             │                          │
       │                         │                             │                          │
       │                         │  1. AgentConfig.from_env()  │                          │
       │                         │     加载 .env + env vars    │                          │
       │                         │                             │                          │
       │                         │  2. _build_llm/stt/tts/vad │                          │
       │                         │     创建各 stage 实例       │                          │
       │                         │                             │                          │
       │                         │  3. AgentServer(ws_url, ...)│                          │
       │                         │───────────────────────────>│  WebSocket 连接 (7880)    │
       │                         │                             │                          │
       │                         │  4. server.rtc_session(     │  注册 _on_session 回调    │
       │                         │      _on_session)          │                          │
       │                         │                             │                          │
       │                         │  5. asyncio.run(server.run)│  进入事件循环             │
       │                         │     [保持连接]              │  [等待 job 调度]         │
       │                         │                             │                          │
       │                         │                             │       客户端加入 Room      │
       │                         │                             │<─────────────────────────│
       │                         │                             │                          │
       │                         │                             │  6. on_job_started 事件   │
       │                         │<────────────────────────────│                          │
       │                         │                             │                          │
       │                         │  7. _on_session(ctx)        │                          │
       │                         │     (在 worker 进程中)       │                          │
       │                         │                             │                          │
       │                         │  8. AgentConfig.from_env()   │                          │
       │                         │     (spawn 模式下重新加载)   │                          │
       │                         │                             │                          │
       │                         │  9. run_agent(ctx, cfg)      │                          │
       │                         │                             │                          │
       │                         │  10. SharedStageFactory      │                          │
       │                         │      (创建 stt/llm/tts/vad) │                          │
       │                         │                             │                          │
       │                         │  11. StreamingPipeline /     │                          │
       │                         │      BatchPipeline          │                          │
       │                         │                             │                          │
       │                         │  12. AgentSession(stt,llm,  │                          │
       │                         │      tts,vad,              │                          │
       │                         │      turn_detection=...)    │                          │
       │                         │                             │                          │
       │                         │  13. RoomIO(session, room) │                          │
       │                         │      await room_io.start()  │                          │
       │                         │                             │                          │
       │                         │  14. session.start(agent,  │                          │
       │                         │      room, ...)            │                          │
       │                         │                             │                          │
       │                         │     Agent 作为 participant  │                          │
       │                         │     加入 Room              │                          │
       │                         │───────────────────────────>│  Agent join Room           │
       │                         │                             │                          │
       │                         │  15. while room.is_connected│  [开始音频事件循环]       │
       │                         │     await asyncio.sleep(1) │                          │
       │                         │                             │                          │
       │                         │                             │       客户端开始说话        │
       │                         │                             │<─────────────────────────│
       │                         │                             │                          │
       │                         │  16. 音频流进入 Room        │                          │
       │                         │<────────────────────────────│  audio_frames             │
       │                         │                             │                          │
       │                         │  17. VAD stream 检测语音    │                          │
       │                         │     START_OF_SPEECH         │                          │
       │                         │                             │                          │
       │                         │  18. STT stream 实时转写    │                          │
       │                         │     INTERIM_TRANSCRIPT       │                          │
       │                         │───────────────────────────>│  transcription (可选)       │
       │                         │                             │                          │
       │                         │  19. VAD 检测语音结束       │                          │
       │                         │     END_OF_SPEECH           │                          │
       │                         │                             │                          │
       │                         │  20. STT stream 返回        │                          │
       │                         │     FINAL_TRANSCRIPT        │                          │
       │                         │                             │                          │
       │                         │  21. LLM.chat(chat_ctx)     │                          │
       │                         │     流式生成回复 token       │                          │
       │                         │                             │                          │
       │                         │  22. ChineseModel           │                          │
       │                         │     predict_end_of_turn()   │                          │
       │                         │     控制 TTS 开始/结束       │                          │
       │                         │                             │                          │
       │                         │  23. TTS stream 流式合成    │                          │
       │                         │     AudioFrame PCM          │                          │
       │                         │                             │                          │
       │                         │  24. room.local_publish_   │                          │
       │                         │      audio(frame)           │                          │
       │                         │───────────────────────────>│  Agent 音频流发布         │
       │                         │                             │                          │
       │                         │  [循环等待下一个用户语音]    │                          │
       │                         │     while room.is_connected│                          │
       │                         │                             │                          │
```

---

## 6. Streaming 模式完整音频流链路

```
┌──────────────────────────────────────────────────────────────────────────┐
│                     实时音频流 (StreamingPipeline)                          │
│                                                                          │
│  客户端麦克风 ──PCM 16kHz──> LiveKit Room ──audio_frames──> Agent Server   │
│                                       │                                   │
│                                       ▼                                   │
│                              ┌───────────────┐                          │
│                              │     VAD       │ FireRed pVAD               │
│                              │ FireredPvadVAD.stream()                  │
│                              │               │                           │
│                              │  状态机:      │                           │
│                              │  WAITING ──────── speech_prob >= 0.5      │
│                              │     │              且持续 >= 100ms        │
│                              │     │                                       │
│                              │     └───────────── silence >= 400ms ────  │
│                              └───────┬───────────────┘                   │
│                                      │                                   │
│                     START_OF_SPEECH / END_OF_SPEECH 事件                  │
│                                      │                                   │
│                                      ▼                                   │
│                              ┌───────────────┐                           │
│                              │     STT       │ BailianFunASRSTT.stream()  │
│                              │ BailianFunASR │                           │
│                              │               │                           │
│                              │  WebSocket:   │                           │
│                              │  dashscope    │                           │
│                              │  .aliyuncs.com│                           │
│                              └───────┬───────┘                           │
│                                      │                                   │
│              INTERIM_TRANSCRIPT ──── + ─── FINAL_TRANSCRIPT               │
│                      (实时中间文字)          (最终识别结果)                  │
│                                      │                                   │
│                                      ▼                                   │
│                              ┌───────────────┐                           │
│                              │     LLM       │ livekit.plugins.openai.LLM │
│                              │  OpenAI-style │                           │
│                              │               │                           │
│                              │  流式 token 输出│                          │
│                              │  chat_ctx 上下文│                          │
│                              └───────┬───────┘                           │
│                                      │                                   │
│                              每个 token 都经过:                           │
│                                      │                                   │
│                                      ▼                                   │
│                              ┌───────────────┐                           │
│                              │  ChineseModel │ EidolonEOTModel           │
│                              │  (EOT 检测)   │                           │
│                              │               │                           │
│                              │ predict_end_of_turn()                     │
│                              │  分数 >= unlikely_threshold → TTS 开始    │
│                              │  用户新语音 → should_interrupt() → 打断   │
│                              └───────┬───────┘                           │
│                                      │                                   │
│                                      ▼                                   │
│                              ┌───────────────┐                           │
│                              │     TTS       │ SenseTimeTTS.stream()      │
│                              │  SenseAudio   │                           │
│                              │               │                           │
│                              │  WebSocket:   │                           │
│                              │  api.senseaudio│                          │
│                              │  .cn/ws/v1    │                           │
│                              │  /t2a_v2      │                           │
│                              └───────┬───────┘                           │
│                                      │                                   │
│                              AudioFrame PCM 16kHz                         │
│                                      │                                   │
│                                      ▼                                   │
│                              ┌───────────────┐                           │
│                              │  RoomIO       │                           │
│                              │               │                           │
│                              │ room.local_   │                           │
│                              │ publish_audio │                           │
│                              └───────┬───────┘                           │
│                                      │                                   │
│                                      ▼                                   │
│                              LiveKit Room ──audio_frames──> 所有参与者    │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 7. BatchPipeline 音频处理流程

```
用户 (上传音频 blob)
    │
    ▼
LiveKit Room ──track_subscribed 事件──> BatchPipeline
    │
    ▼
等待 track_ended (最多 30s 超时)
    │
    ▼
收集所有 audio_frames → _frames_to_pcm_blob() → PCM bytes
    │
    ▼
Step 1: STT
    BailianFunASRSTT._recognize_impl()
        │
        ├── BailianConnectionManager.connect()
        │       │
        │       └───> WebSocket: wss://dashscope.aliyuncs.com/api-ws/v1/inference
        │
        ├── conn.send_audio(audio_bytes)
        ├── conn.finish()
        └── 等待 server 返回 RESULT_GENERATED
                    │
                    ▼
            all_sentences[] → text = "".join(s.text)
                    │
                    ▼
            FINAL_TRANSCRIPT → transcript
    │
    ▼
Step 2: LLM
    LivekitLlmStage.chat(LlmInput(text=transcript))
        │
        ├── _chat_ctx.add_message(user)
        ├── llm.chat(chat_ctx=_chat_ctx, tools=fnc_ctx)
        │       │
        │       └───> OpenAI-compatible API (cfg.llm_base_url)
        │
        └── stream → full_text → LlmOutput(text=full_text)
    │
    ▼
Step 3: TTS
    TtsStage.synthesize(response_text)
        │
        ├── _tts.synthesize(text) → SenseTimeSynthesizeStream
        │       │
        │       ├── client.connect()
        │       │       │
        │       │       └───> WebSocket: wss://api.senseaudio.cn/ws/v1/t2a_v2
        │       │
        │       ├── send_task_continue(tokens)
        │       └── recv_loop: task_continue → hex PCM → AudioFrame
        │
        └── async for frame in synthesize():
                │
                └── room.local_publish_audio(frame)
                    │
                    ▼
            发布到 LiveKit Room
```

---

## 8. 关键设计决策

### 8.1 macOS multiprocessing spawn 兼容性

在 macOS 上，`AgentServer` 使用 `spawn` 模式创建 worker 子进程。`spawn` 会重新 `import` 整个模块，导致模块级全局变量被重置。

```python
# server.py
_agent_config: AgentConfig | None = None  # 主进程中设置

async def _on_session(ctx):
    cfg = _agent_config
    if cfg is None:
        # spawn 模式下重新加载 (因为 _agent_config 在子进程中为 None)
        cfg = AgentConfig.from_env()
    await run_agent(ctx, cfg)
```

### 8.2 Plugin 预注册 (避免 multiprocessing 冲突)

```python
# server.py 在 main() 之前执行
try:
    from livekit.plugins import openai as _  # noqa: F401
except ImportError:
    logger.warning("livekit-plugins-openai not installed, LLM support unavailable")
```

在主进程 import LiveKit 插件可以避免子进程中的 import 竞争条件。

### 8.3 插件即 LiveKit 接口实现

所有 STT/TTS/VAD/EOT 插件都**直接实现 LiveKit 的标准接口** (`livekit.agents.stt.STT`, `livekit.agents.tts.TTS`, `livekit.agents.vad.VAD`, `_TurnDetector`)，因此可以直接传入 `AgentSession`，无需额外适配层。

### 8.4 VAD 单例共享

`FireredPvadVAD` 的 ONNX session 是**类级别单例** (`VAD._processor`)，所有 VADStream 实例共享同一个模型权重，只有 per-stream 的状态缓冲区 (mel_buffer, gru_buffer) 是独立的。

### 8.5 VAD speaker 自适应

`VADStream.update_speaker(audio_16k_samples)` 允许在第一次用户说话后，用 1-5 秒的音频更新说话人嵌入向量，从而提高对该特定说话人的 VAD 准确率。

### 8.6 配置优先级

```
环境变量 > .env 文件 > 代码默认值
```

`AgentConfig.from_env()` 先读取 `.env` 文件，再通过 `os.environ.get()` 让环境变量覆盖。

---

## 9. 配置项说明

**单一真相源**：所有 env 变量、默认值与注释见 [`deploy/livekit-channel.env.template`](../../../deploy/livekit-channel.env.template)。下面只列核心运行时分组，详细 plugin-specific 配置请直接读 template。

| 分组 | 关键变量 | 说明 |
|---|---|---|
| LiveKit 连接 | `LIVEKIT_URL` / `LIVEKIT_API_KEY` / `LIVEKIT_API_SECRET` | 连接 LiveKit Server 的鉴权三元组 |
| Agent 监听 | `AGENT_HOST` / `AGENT_PORT` / `AGENT_MODE` | Agent server 绑定地址与流水线模式（`streaming` \| `batch`） |
| Agent 行为 | `AGENT_INSTRUCTIONS` / `AGENT_WELCOME_MESSAGE` / `AGENT_FALSE_INTERRUPTION_TIMEOUT` / `AGENT_AUDIO_SAMPLE_RATE` | 系统提示、欢迎语、误打断恢复阈值、链路采样率 |
| Provider 选择 | `STT_PROVIDER` / `TTS_PROVIDER` / `VAD_PROVIDER` | 选择哪一家插件——`bailian` \| `sensetime` / `firered_pvad` \| `firered` \| `silero` \| `none` |
| LLM (OpenAI 兼容) | `OPENAI_LLM_BASE_URL` / `OPENAI_LLM_MODEL` / `OPENAI_LLM_API_KEY` | LLM provider 的 endpoint、模型名与密钥 |
| Remote Agent (可选) | `REMOTE_AGENT_RPC_TARGET` / `REMOTE_AGENT_RPC_LOCALE` | 非空时走 gRPC `RemoteAgent.Session`，绕过上面的 OpenAI 兼容 LLM |
| Plugin-specific | `BAILIAN_STT_*` / `BAILIAN_TTS_*` / `SENSETIME_STT_*` / `SENSETIME_TTS_*` | 各 provider 的 URL、密钥、采样率、池大小、聚合阈值等——详见 template |
| 模型路径覆盖 | `EIDOLON_EOT_MODEL_DIR` / `EIDOLON_FIRERED_PVAD_MODEL_DIR` / `EIDOLON_EOT_DEBUG_LOG` | 留空使用插件自带 bundled 模型 |

**命名约定**：`<PROVIDER>_<STAGE>_<FIELD>` 用于 plugin 配置；`<STAGE>_PROVIDER` 选择激活的 provider；`LIVEKIT_*` / `AGENT_*` 用于平台与 agent 身份。所有 env 文件值都可被 shell 环境变量覆盖（优先级更高）。

---

## 10. 启动命令

日常本地开发推荐用仓库里的启动脚本（会设置 `EIDOLON_CHANNEL_LIVEKIT_ENV`、`EIDOLON_ENV=dev`、`PYTHONPATH`），详见 [`deploy/README.md`](../../../deploy/README.md)：

```bash
./deploy/run_livekit_channel.sh
```

也可在项目根手动激活虚拟环境后直接跑模块；**必须**设置 `EIDOLON_CHANNEL_LIVEKIT_ENV` 指向已存在的 env 文件，否则 `AgentConfig.from_env()` 会抛 `ValueError`（未设置或路径非文件均失败）：

```bash
cd eidolon_daemon && source .venv/bin/activate
export EIDOLON_CHANNEL_LIVEKIT_ENV=/path/to/deploy/.livekit-channel.env
python -m eidolon.livekit.agent.server
```

配置加载顺序见上文「8.6 配置优先级」；仅当设置了 `EIDOLON_CHANNEL_LIVEKIT_ENV` 且文件存在时，才通过 `load_dotenv` 合并该文件；否则只依赖已有 `os.environ` 与代码默认值。
