# FireRedChat-turn-detector

EOU 语义端句模型，与标点分数融合使用。推理仅依赖：

- `chinese_best_model_q8.onnx` / `multilingual_best_model_q8.onnx`（二选一或保留两个）
- `tokenizer/` 目录

由 `scripts/download_firered_chat_turn_detector.py` 下载后已删除 ModelScope 缓存、示例脚本等多余文件。
