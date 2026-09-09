# 本机 STT 为什么一句都不出：断句没有主人（2026-09-09）

本文记录 `local_asr` 在 opi5max 上的真机缺陷定位。写下来的主要理由不是缺陷本身——它已经修了——而是**定位它的过程连续三次被自己的判据骗**。三条错误结论都来自同一个动作：拿云端 provider 的轮次当成本机 provider 的证据。

结论标注了来源：**【日志】**=板上日志或时间线记录；**【代码】**=读 livekit-agents / 本仓代码得出；**【实测】**=本机跑出来的；**【假设】**=仍未证实。

## 结论

**`local_asr` 在真机会话里从来没有产出过一次转写。** 不是"会话中途永久失效"。

那次会话（板上 18:33:58 起）的**第一轮**就没有转写：三轮用户说话分别在 18:34:09 / 18:34:24 / 18:34:39，每轮都以 `speech_stopped_without_transcript_deadline` 收场、设备讲兜底话；而服务端的 60 秒拒绝发生在 18:34:38。**前两轮早于那次拒绝**。【日志】

根因是**没有人给它断句**。框架不发、服务不判、插件在等——三方都以为是别人的活。

## 判据先纠三处

定位这个缺陷时被当成「已确证」的前提里，有三条是错的。

### ① "更早一次会话的正常轮次证明插件断句是工作的"——不成立

`turn-timeline.jsonl` 里 t≈8573–8698（板上约 12:18）那 9 条记录 `stt_first_audio_sent_ms` 14–73 ms、转写准确，看起来证明了断句机制工作。但它们的 `stt_stream.provider` **全部是 `bailian`**（model `fun-asr-realtime-2026-02-28`）。【日志】

`local_asr` 在那份时间线里一条成功记录都没有。**看时间线，先看 `stt_stream.provider`。**

### ② "`stt_first_audio_sent_ms = null` 说明一帧音频都没送给 STT"——是仪表缺口

`session/provider_events.py` 的 `observe_stt_turn_audio()` 只在 STT 插件暴露 `observe_next_audio_for_turn` 时才打点，否则直接 `return`。**全仓只有 `plugins/stt/bailian` 实现了它。**【代码】

所以 `local_asr` 的 `stt_first_audio_sent_at` 与整个 `stt_stream` 属性**恒为 null**。失败轮次连 `stt_turn_audio_observer_installed` 这个属性都没有，而 9 条 bailian 记录都有 `= True`——这个差异本身就是 provider 的指纹。【日志】

`null` 在这里的意思是"没人打点"，不是"没送音频"。

### ③ "服务端 60 秒拒绝导致了那三轮失效"——是同一根因的后果，不是那三轮的原因

前两轮早于拒绝（见上）。拒绝让失效在会话内变得**不可逆**，但失效从第一轮就开始了。

顺带一条也被撤回的推论：曾用"真机 10 个成功轮次首个 interim 144–701 ms"来证明 `_drain` 的接收是有效的——那 10 组同样是 bailian 的数。`asyncio.wait_for(coro, timeout=0)` 的真实语义见下。

## ① 那一句 60 秒是怎么攒出来的

### 框架对声明 `streaming=True` 的 STT 全程不发 `_FlushSentinel`

三段代码合起来决定了这件事：【代码】

- **音频是全量推送，没有 VAD 门控。** `voice/audio_recognition.py:739` 的 `_push_audio` 无条件把每一帧推进 `_stt_pipeline.audio_ch`，不在任何"是否在说话"的判断里。
- **`_FlushSentinel` 只由 `RecognizeStream.flush()`（`stt/stt.py:579`）和 `end_input()`（:583）推入。** 框架里的调用方只有 `StreamAdapter`（**仅用于非流式 STT**）、`FallbackAdapter`、`MultiSpeakerAdapter` 和 AMD detector。
- **默认 `stt_node`（`voice/agent.py:505-531`）只调 `push_frame`**，一次 flush 都不调。而 `LocalAsrSTT` 声明 `streaming=True`，所以不会被包进 `StreamAdapter`。

插件那一侧（`plugins/stt/local_asr/speech_stream.py`）在**第一帧音频**上开 utterance，只在 `_FlushSentinel` 上收句。**开的条件永远成立，收的条件永远不成立** → 一个会话 = 一个 utterance，静音全算在里面。

这一条不是纯推导：新增的集成测试用框架自己的 `Agent.default.stt_node` + `_STTPipeline` 驱动 10 秒房间音频（两段说话夹一段静音），实测 `utterances_ended == 0`。【实测】

插件文件头当时写的是"One utterance per VAD flush"，`bailian/speech_stream.py` 的文件头也写着"框架会在 turn 结束时调 `end_input()`"。**这个误解是全仓共有的**，bailian 不受伤只是因为它的 final 来自云端自己的端点检测，从不依赖那个 flush。

### 为什么 40 秒就撞上 60 秒：采样率没声明

`RecognizeStream` 是这条路上唯一做重采样的地方，而它只在被告知 recognizer 需要什么时才做（`stt/stt.py:555-568` 的 `_needed_sr`）。`LocalAsrSpeechStream` 调 `super().__init__()` 时**没传 `sample_rate=`**；`bailian`(:143) 和 `sensetime`(:128) 都传了。【代码】

而 `AudioInputOptions.sample_rate` 默认 **24000**（`voice/room_io/types.py:59`），本仓未覆盖。于是 24 kHz PCM 原样送进一个按 16 kHz 记账的服务：

```
1,920,000 B ÷ (24000 Hz × 2 B) = 40.00 s        预测
18:33:58.443 connected → 18:34:38.497 rejected  = 40.05 s   实测
```

【日志】+【代码】。这条对上之后，"音频从连接建立起就在连续流入、且全在同一个 utterance 里"两件事同时得到证实——它是整轮定位里最干净的一条因果。

插件里 `AudioByteStream(sample_rate=16000, …)` 只按字节重新分帧、不做转换，所以这个错配是静默的。副作用是模型听到的语音快 1.5 倍。

### interim 也一样到不了

`_drain` 当时用 `asyncio.wait_for(socket.receive(), timeout=0)`。零超时不是"取走已到达的"：`wait_for` 把协程排上去，发现没 done 就**取消**它。本机实测——消息已经在队列里，仍然 `TimeoutError`，队列长度不变。【实测】

所以本机这条在真实会话里既没有 final（无 flush）也没有 interim（drain 不可达），一个字都出不来。这与三轮全部走兜底话完全一致。

## ② 为什么只有重启才恢复

**作用域是这一次 AgentSession，不是整个 worker 进程。** 但"重连能自愈"同样是误判——因为它在任何会话里都不工作。

链条（与板上 traceback 逐帧对应）：【日志】+【代码】

1. 插件的 `_emit_error` 无条件 **raise**。它自己的 `recoverable=` 参数当时**完全惰性**——`_await_final` 里那个 `recoverable=True` 后面的 `return` 是死代码，同样杀掉流。
2. 抛的是**裸 `RuntimeError`**。`RecognizeStream._main_task` 只对 `APIError` 重试（`stt/stt.py:477`），`RuntimeError` 落到 `except Exception` 原样抛出（:502-504）。
3. `_STTPipeline._stt_pump` **只在 `except APIError` 时重建流**（`audio_recognition.py:211-224`，注释写明其他异常 "propagates and stops the pump"）→ pump 任务死亡 → done-callback 关闭 `event_ch`（:174）。
4. `_STTPipeline` 按会话创建（:827，`is_closing=self._session._is_closing`），所以死的是会话。LiveKit 每个 job 一个子进程，进程级污染不存在。

板上原话：`18:34:48,549 ERROR livekit.agents Error in _stt_pump`，栈是 `_stt_pump`(206) → `stt_node`(528) → `_main_task`(503) → `speech_stream.py _emit_error` → `RuntimeError`。

SDK 契约把 `ERROR_UTTERANCE_TOO_LONG` 注释成 "Not retryable **as sent**"——说的是**这一句**（那些词没了，重发无意义），不是这条流。插件把两件事混成了一件。`bailian` 反而 `import APIError` 用对了。

## 两处顺带发现

修的过程中撞出两处原本没预料到的，都不是从现象推出来的，是修第一版之后测试不过才暴露的。

### 插件覆盖了框架的 `_emit_error`

那个名字属于 `RecognizeStream`，契约是"发一个 error 事件然后返回"，而 `_main_task` 在自己的**重试路径上也会调它**（`stt/stt.py:487`）。用一个会抛异常的版本覆盖它，导致：【代码】+【实测】

- 框架的重试被劫持——`_emit_error` 抛出，重试永远不发生；
- 框架的 `error` 事件**从来没发出过**，任何监听 session 错误的东西都听不到 STT 失败；
- 板上 18:34:48 那**两条一模一样**的 `local_asr stream error` 就是这个从外面看的样子（一条我们自己调的，一条框架调的）。

改名成 `_fail`，`_emit_error` 恢复继承。改完框架的退避重试立刻正常工作。

### "读绑在发上"只修一半是不够的

把读挪到专职任务之后，*处理*读到的东西仍然只发生在发送循环（`_drain`）里。音频一停（静音、或框架不再推帧），服务端的告别就躺在 inbox 里没人看——直到有人再开口。

现在 `_run` 让读者和发送者赛跑：**读者先结束意味着 socket 先没了，那本身就是消息。** 新增的端到端用例（说太久触顶 → 下一句仍被转写）是靠这个才通的。

## 修了什么

`eidolon_channel`，每个提交去掉恰好一个 xfail：

| 提交 | 内容 |
| --- | --- |
| `127a16b` | 四条 xfail 钉住缺陷，由框架驱动 |
| `6a43cab` | 声明 `sample_rate`，24 kHz 不再当 16 kHz |
| `80df1a6` | 专职读者收 socket，interim 真的到得了 |
| `6765455` | 抛 `APIError`，一句被拒不再断掉整条流 |
| `9f89611` | channel 在 VAD 停止说话时调插件 `end_utterance()`；不再覆盖 `_emit_error`；读者/发送者赛跑 |

断句的归属定在**客户端**：channel 的 VAD 是这个系统里唯一的 VAD，所以只有它能说"那是一句话"。调用方式沿用本仓已有的鸭子类型先例（`observe_next_audio_for_turn`）——不需要的插件不实现，`_close_stt_utterance()` 返回 False 不作声。云端 provider 因此完全不受影响。

配套的契约措辞由另一个 session 落在另外两个仓：`eidolon_models` 的 `4eb4d77`（`/v1/info` 改成 `utterance_boundary_owner: "client"`）和 `eidolon_sdk` 的 `8eadcc1`（契约里写明"谁结束一个 utterance"）。

## 留下的测试为什么这样写

先前那次"local_asr 真机跑通"是拿 wav 驱动插件、**自己发的 flush**——验的是假设，不是框架行为。一个自己供边界的测试，测的是从来没坏的那一半。

`tests/stt/local_asr/test_local_asr_session_integration.py` 因此把插件交给框架驱动：被测节点是 livekit-agents 自己的 `Agent.default.stt_node`，迭代者是它自己的 `_STTPipeline`，测试只决定麦克风决定的事——**哪些帧到、以什么采样率到**，从不发 flush。要发边界的地方，调的是 `StreamingPipeline._close_stt_utterance()` 真正调的那个方法。

其中一条是特意留下的**表征测试**：`test_the_framework_alone_never_closes_an_utterance` 断言框架自己关掉 0 个 utterance。如果将来某版 livekit-agents 开始给流式插件发 flush，它会失败——那正是它的用途。

测试里的假服务按 **utterance** 计字节，与 `service.py` 里 `_audio_bytes` 挂在 utterance 对象上一致。第一版按连接累计，那是个失真的判据；不要让失真的判据去验被测的东西。

## 状态

**未部署。** 板子是共享资源。

全量 `eidolon/livekit/tests`：**1723 通过、6 失败、7 跳过、51 deselected、17 xfailed**（779 s）。6 条失败全部是 `scenarios/test_overlap_settlement.py::test_final_and_silence_resume_same_speech` 的参数化用例，本次修改之前就在失败，与本文无关。

真机验收不能拿"听到欢迎语"当通过——那次失效的会话里欢迎语是正常播出的。每一轮要关联：**音频 → ASR final → 话轮提交 → LLM → TTS → 平板播放**。
