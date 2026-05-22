#  Copyright (C) 2025. Hao Zheng
#  All rights reserved.
import abc
import json
import re

from json_repair import repair_json
from langcodes import Language as Langcode
from lingua import Language as LinguaLanguage
from lingua import LanguageDetectorBuilder

from openlrc.defaults import supported_languages_lingua
from openlrc.logger import logger

_LINGUA_LANGUAGES = [getattr(LinguaLanguage, name) for name in supported_languages_lingua]

ORIGINAL_PREFIX = "Original>"
TRANSLATION_PREFIX = "Translation>"
PROOFREAD_PREFIX = "Proofread>"

POTENTIAL_PREFIX_COMBOS = [
    [ORIGINAL_PREFIX, TRANSLATION_PREFIX],
    ["原文>", "翻译>"],
    ["原文>", "译文>"],
    ["原文>", "翻譯>"],
    ["原文>", "譯文>"],
    ["Original>", "Translation>"],
    ["Original>", "Traducción>"],
]


class BaseValidator(abc.ABC):
    @abc.abstractmethod
    def validate(self, user_input, generated_content):
        raise NotImplementedError()


class ChunkedTranslateValidator(BaseValidator):
    def __init__(self, target_lang):
        self.lan_detector = LanguageDetectorBuilder.from_languages(*_LINGUA_LANGUAGES).build()
        self.target_lang = target_lang

    def _extract_translation(self, content: str) -> list[str]:
        for potential_ori_prefix, potential_trans_prefix in POTENTIAL_PREFIX_COMBOS:
            translation = re.findall(f"{potential_trans_prefix}\n*(.*?)(?:#\\d+|<summary>|\\n*$)", content, re.DOTALL)
            if translation:
                return translation
        return []

    def _is_translation_in_target_language(self, translation: list[str]) -> bool:
        if len(translation) >= 3:
            chunk_size = len(translation) // 3
            translation_chunks = [translation[i : i + chunk_size] for i in range(0, len(translation), chunk_size)]
            if len(translation_chunks) > 3:
                translation_chunks[-2].extend(translation_chunks[-1])
                translation_chunks.pop()

            translated_langs = [self.lan_detector.detect_language_of(" ".join(chunk)) for chunk in translation_chunks]
            translated_langs = [lang.name.lower() for lang in translated_langs if lang]

            if not translated_langs:
                return True

            translated_lang = max(set(translated_langs), key=translated_langs.count)
        else:
            detected_lang = self.lan_detector.detect_language_of(" ".join(translation))
            if not detected_lang:
                return True
            translated_lang = detected_lang.name.lower()

        target_lang = Langcode.get(self.target_lang).language_name().lower()
        if translated_lang != target_lang:
            logger.warning(f"Translated language is {translated_lang}, not {target_lang}.")
            return False

        return True

    def validate(self, user_input, generated_content):
        if not generated_content:
            logger.warning("Empty or None response content.")
            return False

        summary = re.search(r"<summary>(.*)</summary>", generated_content)
        scene = re.search(r"<scene>(.*)</scene>", generated_content)

        original = re.findall(ORIGINAL_PREFIX + r"\n(.*?)\n" + TRANSLATION_PREFIX, user_input, re.DOTALL)
        if not original:
            logger.error("Fail to extract original text.")
            return False

        translation = self._extract_translation(generated_content)
        if not translation:
            logger.warning("Fail to extract translation.")
            logger.debug(f"Content: {generated_content}")
            return False

        if len(original) != len(translation):
            logger.warning(
                f"Fail to ensure length consistent: original is {len(original)}, translation is {len(translation)}"
            )
            logger.debug(f"original: {original}")
            logger.debug(f"translation: {translation}")
            return False

        if not self._is_translation_in_target_language(translation):
            return False

        if not summary or not summary.group(1):
            logger.warning("Fail to extract summary.")
        if not scene or not scene.group(1):
            logger.warning("Fail to extract scene.")

        return True


class AtomicTranslateValidator(BaseValidator):
    def __init__(self, target_lang):
        self.lan_detector = LanguageDetectorBuilder.from_languages(*_LINGUA_LANGUAGES).build()
        self.target_lang = target_lang

    def validate(self, user_input, generated_content):
        if not generated_content:
            logger.warning("Empty or None response content.")
            return False

        detected_lang = self.lan_detector.detect_language_of(generated_content)
        if not detected_lang:
            return True

        translated_lang = detected_lang.name.lower()
        target_lang = Langcode.get(self.target_lang).language_name().lower()
        if translated_lang != target_lang:
            logger.warning(f'Translated text: "{generated_content}" is {translated_lang}, not {target_lang}.')
            return False

        return True


class ProofreaderValidator(BaseValidator):
    def validate(self, user_input, generated_content):
        if not generated_content:
            logger.warning("Empty or None response content.")
            return False

        original = re.findall(ORIGINAL_PREFIX + r"\n(.*?)\n" + TRANSLATION_PREFIX, user_input, re.DOTALL)
        if not original:
            logger.error("Fail to extract original text.")
            return False

        localized = re.findall(PROOFREAD_PREFIX + r"\s*(.*)", generated_content, re.MULTILINE)

        if not localized:
            logger.warning("Fail to extract translation.")
            logger.debug(f"Content: {generated_content}")
            return False

        if len(original) != len(localized):
            logger.warning(
                f"Fail to ensure length consistent: original is {len(original)}, translation is {len(localized)}"
            )
            logger.debug(f"original: {original}")
            logger.debug(f"translation: {localized}")
            return False

        return True


class ContextReviewerValidateValidator(BaseValidator):
    def validate(self, user_input: str, generated_content: str) -> bool:
        """
        Validate the generated content based on user input.

        Args:
            user_input (str): The user input to compare against.
            generated_content (str): The content generated for validation.

        Returns:
            bool: True if validation passes, False otherwise.
        """
        if not generated_content:
            logger.warning("Empty or None response content.")
            return False

        if re.search(r"\b(?:true|false)\b", generated_content, re.IGNORECASE):
            return True
        else:
            logger.warning(f"Context reviewer validation failed: {generated_content}")

        return False


class TranslationEvaluatorValidator(BaseValidator):
    def validate(self, user_input, generated_content):
        repaired_content = repair_json(generated_content)
        # verify json format
        if not repaired_content:
            try:
                json.loads(generated_content)
            except json.JSONDecodeError:
                logger.warning(f"Fail to parse json: {generated_content}")
                return False

        return True


class LeanTranslateValidator(BaseValidator):
    """Validator for LeanTranslator output.

    Parses ``#<id>`` anchors from the model response and checks that at
    least ``min_match_ratio`` of the expected line IDs are present.
    """

    # Matches a line that is exactly ``#<digits>`` (with optional whitespace).
    _ANCHOR_RE = re.compile(r"^\#(\d+)\s*$", re.MULTILINE)

    def __init__(self, expected_ids: list[int], min_match_ratio: float = 0.8):
        self.expected_ids = set(expected_ids)
        self.min_match_ratio = min_match_ratio

    def validate(self, user_input: str, generated_content: str) -> bool:
        if not generated_content:
            logger.warning("Empty or None response content.")
            return False

        parsed = self.parse_anchored_translations(generated_content)
        if not parsed:
            logger.warning("No anchored translations found in response.")
            return False

        matched = self.expected_ids & parsed.keys()
        ratio = len(matched) / len(self.expected_ids) if self.expected_ids else 0.0
        if ratio < self.min_match_ratio:
            logger.warning(
                f"Anchor match ratio {ratio:.0%} ({len(matched)}/{len(self.expected_ids)})"
                f" below threshold {self.min_match_ratio:.0%}."
            )
            return False

        return True

    @classmethod
    def parse_anchored_translations(cls, content: str) -> dict[int, str]:
        """Parse model output into ``{line_id: translation}`` mapping.

        Splits *content* on ``#<digits>`` anchors.  Text between two
        consecutive anchors (stripped, lines joined with a space) is the
        translation for the preceding anchor.
        """
        parts = cls._ANCHOR_RE.split(content)
        # parts layout: [preamble, id1, text1, id2, text2, ...]
        result: dict[int, str] = {}
        # Start from index 1 (skip preamble), step by 2.
        for i in range(1, len(parts) - 1, 2):
            line_id = int(parts[i])
            raw_text = parts[i + 1].strip()
            # Join multi-line translations into a single line.
            translation = " ".join(line.strip() for line in raw_text.splitlines() if line.strip())
            if translation:
                result[line_id] = translation

        return result
