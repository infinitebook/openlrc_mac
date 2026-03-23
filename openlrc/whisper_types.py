#  Copyright (C) 2025. Hao Zheng
#  All rights reserved.

"""
替代 faster_whisper 的 Segment / Word 数据类型。

Design Decision:
  使用 @dataclass 而非 namedtuple，原因：
  1. faster_whisper.Segment 本身是 NamedTuple，但 sentence_split 中的
     seg_from_words() 使用位置参数重新构造 Segment，@dataclass 同样支持
  2. @dataclass 支持属性修改，更灵活
  3. 比 namedtuple 易于 IDE 索引和 type hint
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Word:
    """词级时间戳单元，对齐 faster_whisper.transcribe.Word 的字段签名。

    Attributes:
        start: 词开始时间（秒）。
        end: 词结束时间（秒）。
        word: 词文本内容。
        probability: 词的置信概率。
    """

    start: float
    end: float
    word: str
    probability: float


@dataclass
class Segment:
    """段级转录单元，对齐 faster_whisper.transcribe.Segment 的字段签名。

    字段顺序必须与 faster_whisper.Segment (NamedTuple) 一致，
    因为 sentence_split.seg_from_words() 使用位置参数构造。

    Attributes:
        id: 段 ID。
        seek: Seek 偏移。
        start: 段开始时间（秒）。
        end: 段结束时间（秒）。
        text: 段文本内容。
        tokens: 原始 token ID 列表。
        avg_logprob: 平均对数概率。
        compression_ratio: 压缩率。
        no_speech_prob: 非语音概率。
        words: 词级时间戳列表。
        temperature: 采样温度。
    """

    id: int
    seek: int
    start: float
    end: float
    text: str
    tokens: list
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float
    words: list[Word] | None
    temperature: float
