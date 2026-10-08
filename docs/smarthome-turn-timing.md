# Smart-home turn timing

The home transcript handler logs one turn_id across handler_started, agent elapsed_ms, result_delivery elapsed_ms and final delivered elapsed_ms. Agent time includes the HTTP request and its server work; delivery is the Provider request after the Agent response. Agent additionally records per-session FIFO queue and primary/understanding/execution time.

These clocks use monotonic time. Handler start is not ASR final time or actual speech end. LiveKit/VAD finalization and scheduling can delay handler entry and remain outside handler_elapsed. Provider HTTP success is not a Korvo receipt/render ACK. Do not label these values microphone-to-display latency or actual appliance actuation latency.

The transport continues to forward the original transcript and authoritative session scope. No semantic rules, model calls or device execution were moved into Channel. Mock HTTP tests check scope, error handling and stage logs. No board or live device tests were performed.
