# 3D-Speaker CAM++ zh-cn 16k common

This directory is the local Git LFS managed model bundle for speaker
verification.

Baseline model:

- Provider: 3D-Speaker
- ModelScope id: `iic/speech_campplus_sv_zh-cn_16k-common`
- Source revision: `624dcb0f2e40153d7bbbbbcb02cb9f3518f225c3`
- Task: speaker verification and speaker embedding enrollment
- Sample rate: 16000 Hz
- Channels: mono
- Input format: PCM WAV for enrollment samples; provider may convert to tensors

Bundled runtime artifacts:

- `campplus_cn_common.bin`
- `config.yaml`
- `configuration.json`

Runtime ownership:

- `eidolon_admin` uses this model for enrollment. It turns the recorded WAV
  samples into a persistent voiceprint profile bound to a user.
- `eidolon_channel` uses this model for verification. It compares turn audio
  against the user's stored voiceprint profile.
- The model files should exist in one repository-managed bundle, not as one
  copy under admin and another copy under channel.

Operational rules:

- Do not download model weights at runtime in admin or channel.
- Put downloaded model artifacts under this directory and let `.gitattributes`
  route large files to Git LFS.
- Keep transient ModelScope or torch cache files out of this directory.
- If deployment needs a different location, point both services at that same
  directory with `EIDOLON_3DSPEAKER_MODEL_DIR`.

Sources:

- 3D-Speaker: https://github.com/modelscope/3D-Speaker
- ModelScope model: https://modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common
