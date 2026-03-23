#  Copyright (C) 2025. Hao Zheng
#  All rights reserved.

from dataclasses import dataclass
from pathlib import Path

from openlrc.models import ModelConfig


@dataclass
class TranscriptionConfig:
    """
    Configuration for the transcription stage.

    Args:
        whisper_model: Path to whisper GGML model file, or model name.
            Default: ``large-v3``
        cli_path: Path to the whisper-cli executable. Default: ``whisper-cli``
        vad_model: Path to the Silero VAD model for whisper.cpp.
            Empty string disables native VAD. Default: ``""``
        compute_type: Computation type (legacy, kept for compatibility).
            Default: ``float16``
        device: Device for computation (legacy, kept for compatibility).
            Default: ``cuda``
        asr_options: Parameters for whisper model.
        vad_options: Parameters for VAD model (legacy, kept for compatibility).
        preprocess_options: Options for audio preprocessing.
    """

    whisper_model: str = "whisper.cpp/models/ggml-base.bin"
    cli_path: str = "whisper.cpp/build/bin/whisper-cli"
    vad_model: str = "whisper.cpp/models/ggml-silero-v6.2.0.bin"
    compute_type: str = "float16"
    device: str = "cuda"
    asr_options: dict | None = None
    vad_options: dict | None = None
    preprocess_options: dict | None = None


@dataclass
class TranslationConfig:
    """
    Configuration for the translation stage.

    Args:
        chatbot_model: The chatbot model to use. Can be a string like
            ``'gpt-4.1-nano'`` or ``'provider:model-name'``, or a ``ModelConfig``
            instance. Default: ``gpt-4.1-nano``
        fee_limit: Maximum fee per translation call in USD. Default: ``0.8``
        consumer_thread: Number of parallel translation threads. Default: ``4``
        proxy: Proxy for API requests. e.g. ``'http://127.0.0.1:7890'``
        base_url_config: Base URL dict for OpenAI & Anthropic.
        glossary: Dictionary or path mapping source words to translations.
        retry_model: Fallback model for translation retries.
        is_force_glossary_used: Force glossary usage in context. Default: ``False``
    """

    chatbot_model: str | ModelConfig = "gpt-4.1-nano"
    fee_limit: float = 0.8
    consumer_thread: int = 4
    proxy: str | None = None
    base_url_config: dict | None = None
    glossary: dict | str | Path | None = None
    retry_model: str | ModelConfig | None = None
    is_force_glossary_used: bool = False
