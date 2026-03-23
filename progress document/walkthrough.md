# WhisperCPP 迁移 — 变更总结 (Walkthrough)

## 变更概览

将 OpenLRC 的 ASR 后端从 `faster-whisper` (Python/CUDA) 替换为 [whisper.cpp](file:///Users/fuzuorui/Documents/agent_play/openlrc_mac/whisper.cpp/src/whisper.cpp) CLI (Metal/跨平台)。共涉及 **2 个新增文件** + **6 个修改文件**。

---

## 文件变更列表

### 新增文件

| 文件 | 用途 |
|------|------|
| [whisper_types.py](file:///Users/fuzuorui/Documents/agent_play/openlrc_mac/openlrc/whisper_types.py) | [Segment](file:///Users/fuzuorui/Documents/agent_play/openlrc_mac/openlrc/whisper_types.py#37-69) / [Word](file:///Users/fuzuorui/Documents/agent_play/openlrc_mac/openlrc/whisper_types.py#20-35) dataclass，替代 `faster_whisper.transcribe` 的 NamedTuple |
| [whisper_backend.py](file:///Users/fuzuorui/Documents/agent_play/openlrc_mac/openlrc/whisper_backend.py) | CLI subprocess 桥接层：双管道 IO + stderr 进度解析线程 |

### 修改文件

| 文件 | 变更内容 |
|------|----------|
| [transcribe.py](file:///Users/fuzuorui/Documents/agent_play/openlrc_mac/openlrc/transcribe.py) | **完整重写**：移除 faster-whisper import，新增 [map_cli_json_to_segments()](file:///Users/fuzuorui/Documents/agent_play/openlrc_mac/openlrc/transcribe.py#67-167) 适配器，[Transcriber](file:///Users/fuzuorui/Documents/agent_play/openlrc_mac/openlrc/transcribe.py#169-511) 改用 [WhisperCLIBackend](file:///Users/fuzuorui/Documents/agent_play/openlrc_mac/openlrc/whisper_backend.py#48-181) |
| [config.py](file:///Users/fuzuorui/Documents/agent_play/openlrc_mac/openlrc/config.py) | [TranscriptionConfig](file:///Users/fuzuorui/Documents/agent_play/openlrc_mac/openlrc/config.py#10-38) 新增 `cli_path`, `vad_model` 字段 |
| [defaults.py](file:///Users/fuzuorui/Documents/agent_play/openlrc_mac/openlrc/defaults.py) | 新增 `default_whisper_cpp_options` dict |
| [openlrc.py](file:///Users/fuzuorui/Documents/agent_play/openlrc_mac/openlrc/openlrc.py) | TYPE_CHECKING import 改为 `whisper_types`；[transcriber](file:///Users/fuzuorui/Documents/agent_play/openlrc_mac/openlrc/openlrc.py#169-184) property 传入 `cli_path`/`vad_model` |
| [test_transcribe.py](file:///Users/fuzuorui/Documents/agent_play/openlrc_mac/tests/test_transcribe.py) | 重写：import 更新 + 新增 [TestMapCliJsonToSegments](file:///Users/fuzuorui/Documents/agent_play/openlrc_mac/tests/test_transcribe.py#82-201)（8 个测试用例）+ [TestParseTimestampStr](file:///Users/fuzuorui/Documents/agent_play/openlrc_mac/tests/test_transcribe.py#203-217)（4 个） |
| [test_openlrc.py](file:///Users/fuzuorui/Documents/agent_play/openlrc_mac/tests/test_openlrc.py) | import 从 `faster_whisper` 改为 `whisper_types`；移除 `BatchedInferencePipeline` mock |

---

## 关键设计决策

1. **offset 时间单位**：whisper.cpp JSON 中 `offsets.from/to` 为毫秒（源码 `cli.cpp L691: t0*10`），转换公式 `sec = offset / 1000.0`
2. **管道输出**：使用 `-ojf -of -` 将 JSON 输出至 stdout，Python 从 `proc.stdout.read()` 一次性读取
3. **进度回调**：`--no-prints` + `-pp` 组合，前者压制日志但不影响 progress callback（源码确认 `fprintf(stderr)` 独立于 `whisper_log_set`）
4. **空 segment 过滤**：adapter 层过滤掉无有效 words 的 segment，防止 [sentence_split](file:///Users/fuzuorui/Documents/agent_play/openlrc_mac/openlrc/transcribe.py#298-511) 的 assert 断言失败

## 待验证（环境就绪后）

```bash
# 1. 安装依赖
pip install -e .

# 2. 单元测试（不需要 whisper-cli 二进制）
python -m pytest tests/test_transcribe.py -v -k "TestMapCliJsonToSegments or TestParseTimestampStr"

# 3. 集成测试（需要 whisper-cli + 模型文件）
python -m pytest tests/test_transcribe.py tests/test_openlrc.py -v

# 4. 端到端测试
python -c "
from openlrc import LRCer, TranscriptionConfig
lrcer = LRCer(transcription=TranscriptionConfig(
    whisper_model='path/to/ggml-large-v3-turbo.bin',
    cli_path='whisper-cli',
))
lrcer.run('tests/data/test_audio.wav', target_lang='en', skip_trans=True)
"
```
