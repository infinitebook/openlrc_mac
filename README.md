# OpenLRC Mac

OpenLRC turns audio or video into subtitle files. It can transcribe speech,
split the transcript into subtitle-friendly segments, and optionally translate
or polish the result with LLMs. It is useful for making `.lrc` lyrics-style
subtitles or `.srt` video subtitles from local media files.

This repository is a macOS-focused fork of
[zh-plus/openlrc](https://github.com/zh-plus/openlrc).

The current goal is to make OpenLRC work well with local
[whisper.cpp](https://github.com/ggml-org/whisper.cpp) transcription on macOS,
including Metal acceleration and a future `.app` packaging path. It is still a
development fork rather than a polished end-user application.

## What Is Different From Upstream

This fork keeps OpenLRC's subtitle, translation, optimization, and preprocessing
pipeline, but replaces the transcription path with local `whisper.cpp`.
`whisper.cpp` is managed as a pinned submodule, and `scripts/setup_whisper_cpp.py`
builds `whisper-cli` and downloads the default GGML models.

## Current Status

Working:

- Build `whisper.cpp` locally with CMake Release.
- Resolve `whisper-cli` from the submodule build, app bundle resources, explicit
  config, environment variables, or `PATH`.
- Download and use GGML whisper models from the setup script.
- Transcribe audio and video locally with `whisper.cpp`.
- Generate `.lrc` or `.srt` subtitles through the existing OpenLRC pipeline.
- Translate subtitles through the existing LLM backends.

Still in progress:

- macOS app packaging.
- User-facing model management UI.
- Better defaults for larger local models.
- Cleanup of upstream documentation and old GUI references.

## Requirements

- macOS, tested on Apple Silicon.
- Python `>=3.10,<3.13`.
- [uv](https://github.com/astral-sh/uv).
- [ffmpeg](https://ffmpeg.org/download.html) on `PATH`.
- CMake and Xcode Command Line Tools for building `whisper.cpp`.

## Quick Start

Clone with submodules:

```shell
git clone --recurse-submodules https://github.com/infinitebook/openlrc_mac.git
cd openlrc_mac
```

If you already cloned without submodules:

```shell
git submodule update --init --recursive
```

Install Python dependencies:

```shell
uv sync
```

Build `whisper.cpp` and download the default models:

```shell
uv run python scripts/setup_whisper_cpp.py
```

The setup script builds:

```text
vendor/whisper.cpp/build/bin/whisper-cli
```

It downloads models to:

```text
~/Library/Application Support/OpenLRC/models/
```

Default files:

```text
ggml-base.bin
ggml-silero-v6.2.0.bin
```

## Basic Usage

Transcribe only:

```python
from openlrc import LRCer

lrcer = LRCer()
lrcer.transcribe("video.mp4", src_lang="en")
```

Generate subtitles without translation:

```python
from openlrc import LRCer

lrcer = LRCer()
lrcer.run("video.mp4", src_lang="en", target_lang="en", skip_trans=True)
```

Translate subtitles:

```python
from openlrc import LRCer

lrcer = LRCer()
lrcer.run("video.mp4", src_lang="en", target_lang="zh-cn")
```

For translation, set the API key for the model provider you use. Common options:

```shell
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
export GOOGLE_API_KEY="..."
export OPENROUTER_API_KEY="..."
```

## Switching Models And Paths

The default model is `base`. To download another whisper.cpp model:

```shell
uv run python scripts/setup_whisper_cpp.py --skip-build --model small
```

Then select it in Python:

```python
from openlrc import LRCer, TranscriptionConfig

lrcer = LRCer(
    transcription=TranscriptionConfig(
        whisper_model="small",
    )
)
```

Common model names include `tiny`, `base`, `small`, `medium`, and
`large-v3-turbo`. You can also pass a GGML filename or an absolute path:

```python
TranscriptionConfig(whisper_model="ggml-small.bin")
TranscriptionConfig(whisper_model="/path/to/ggml-small.bin")
```

OpenLRC resolves `whisper-cli` and model files automatically from explicit
config, environment variables, app bundle resources, the local submodule build,
and standard model directories. Useful overrides:

```shell
export OPENLRC_WHISPER_CLI="/path/to/whisper-cli"
export OPENLRC_WHISPER_MODEL="/path/to/ggml-base.bin"
export OPENLRC_WHISPER_VAD_MODEL="/path/to/ggml-silero-v6.2.0.bin"
export OPENLRC_WHISPER_MODEL_DIR="$HOME/Library/Application Support/OpenLRC/models"
```

To disable native VAD:

```python
TranscriptionConfig(vad_model="")
```

## Development

Run the unit test suite:

```shell
uv run --with pytest python -m pytest -q
```

Run a real local transcription smoke test after setup:

```shell
uv run python -c "from openlrc import LRCer; print(LRCer().transcribe('tests/data/test_audio.wav', src_lang='en'))"
```

Useful checks:

```shell
uv run ruff check openlrc/ tests/
uv run ruff format --check openlrc/ tests/
uv run pyright openlrc/
```

Manual test files can be kept under `manual tests/`; that directory is ignored
by Git.

## Submodule Notes

This repository does not vendor the full `whisper.cpp` source as ordinary files.
It stores a submodule pointer to a specific upstream commit.

Update an existing checkout:

```shell
git submodule update --init --recursive
```

Move the pinned `whisper.cpp` version intentionally:

```shell
cd vendor/whisper.cpp
git fetch
git checkout <commit-or-tag>
cd ../..
git add vendor/whisper.cpp
```

## Packaging Direction

The resolver already checks for `whisper-cli` inside a macOS app bundle before
falling back to the development submodule build. A future app can bundle the
binary and manage model downloads without changing the transcription backend.

For now, this repository should be treated as a local development fork.

## Upstream And Credits

This fork is based on [zh-plus/openlrc](https://github.com/zh-plus/openlrc).
Most of the subtitle pipeline, translation flow, and project structure come from
that upstream project.

This fork currently focuses on the macOS `whisper.cpp` path. If you need the
general-purpose OpenLRC package and documentation, use upstream OpenLRC.

Key related projects:

- [zh-plus/openlrc](https://github.com/zh-plus/openlrc)
- [ggml-org/whisper.cpp](https://github.com/ggml-org/whisper.cpp)
- [openai/openai-python](https://github.com/openai/openai-python)
- [anthropics/anthropic-sdk-python](https://github.com/anthropics/anthropic-sdk-python)

## License

This fork follows the upstream OpenLRC license. See [LICENSE](LICENSE).
