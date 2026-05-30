#  Copyright (C) 2026. Hao Zheng
#  All rights reserved.

import unittest
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("litellm")

from openlrc.chatbot import LiteLLMBot, route_chatbot
from openlrc.exceptions import ChatBotException


def _make_response(content="translated text", finish_reason="stop", prompt_tokens=10, completion_tokens=20):
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    resp.choices[0].finish_reason = finish_reason
    resp.usage = MagicMock()
    resp.usage.prompt_tokens = prompt_tokens
    resp.usage.completion_tokens = completion_tokens
    resp.usage.total_tokens = prompt_tokens + completion_tokens
    return resp


def _bot(model="anthropic/claude-sonnet-4-6", **kw):
    bot = LiteLLMBot(model_name=model, **kw)
    bot.api_fees = [0]
    return bot


class TestLiteLLMBot(unittest.TestCase):
    @patch("litellm.completion", return_value=_make_response())
    def test_create_chat_dispatches_to_litellm(self, mock_completion):
        bot = _bot()
        messages = [{"role": "system", "content": "Translate to French."}, {"role": "user", "content": "Hello world"}]
        response = bot._create_chat(messages)
        mock_completion.assert_called_once()
        kw = mock_completion.call_args.kwargs
        self.assertEqual(kw["model"], "anthropic/claude-sonnet-4-6")
        self.assertTrue(kw["drop_params"])
        self.assertEqual(bot.get_content(response), "translated text")

    @patch("litellm.completion", return_value=_make_response())
    def test_params_forwarded(self, mock_completion):
        bot = _bot(model="openai/gpt-4o", temperature=0.3)
        bot._create_chat([{"role": "user", "content": "test"}], temperature=0.5)
        kw = mock_completion.call_args.kwargs
        self.assertEqual(kw["temperature"], 0.5)

    @patch("litellm.completion", return_value=_make_response())
    def test_top_p_default_when_not_set(self, mock_completion):
        bot = _bot(temperature=0.5)
        bot._create_chat([{"role": "user", "content": "test"}])
        kw = mock_completion.call_args.kwargs
        self.assertEqual(kw["top_p"], 1.0)

    @patch("litellm.completion", return_value=_make_response())
    def test_top_p_included_when_temperature_default(self, mock_completion):
        bot = _bot(model="openai/gpt-4o", temperature=1.0, top_p=0.8)
        bot._create_chat([{"role": "user", "content": "test"}])
        kw = mock_completion.call_args.kwargs
        self.assertEqual(kw["top_p"], 0.8)

    @patch("litellm.completion", return_value=_make_response())
    def test_fee_tracking(self, mock_completion):
        bot = _bot(model="openai/gpt-4o")
        bot._create_chat([{"role": "user", "content": "test"}])
        self.assertGreater(bot.api_fees[-1], 0)

    @patch("litellm.completion", return_value=_make_response(content=None))
    def test_null_content_returns_empty(self, mock_completion):
        bot = _bot(model="openai/gpt-4o")
        response = bot._create_chat([{"role": "user", "content": "test"}])
        self.assertEqual(bot.get_content(response), "")

    @patch("litellm.completion", return_value=_make_response())
    def test_stop_sequences_forwarded(self, mock_completion):
        bot = _bot(model="openai/gpt-4o")
        bot._create_chat([{"role": "user", "content": "test"}], stop_sequences=["END"])
        kw = mock_completion.call_args.kwargs
        self.assertEqual(kw["stop"], ["END"])

    @patch("litellm.completion", return_value=_make_response())
    def test_api_key_forwarded(self, mock_completion):
        bot = _bot(model="openai/gpt-4o", api_key="sk-test-123")
        bot._create_chat([{"role": "user", "content": "test"}])
        kw = mock_completion.call_args.kwargs
        self.assertEqual(kw["api_key"], "sk-test-123")

    @patch("litellm.completion")
    def test_auth_error_not_retried(self, mock_completion):
        import litellm as _litellm

        mock_completion.side_effect = _litellm.AuthenticationError(
            message="invalid key", llm_provider="openai", model="openai/gpt-4o"
        )
        bot = _bot(model="openai/gpt-4o")
        bot.retry = 3
        with self.assertRaises(ChatBotException):
            bot._create_chat([{"role": "user", "content": "test"}])
        self.assertEqual(mock_completion.call_count, 1)

    def test_route_chatbot_litellm(self):
        cls, model = route_chatbot("litellm:openai/gpt-4o")
        self.assertIs(cls, LiteLLMBot)
        self.assertEqual(model, "openai/gpt-4o")

    @patch("litellm.completion", return_value=_make_response())
    def test_extra_body_forwarded(self, mock_completion):
        bot = _bot(model="openai/gpt-4o", extra_body={"seed": 42})
        bot._create_chat([{"role": "user", "content": "test"}])
        kw = mock_completion.call_args.kwargs
        self.assertEqual(kw["seed"], 42)


if __name__ == "__main__":
    unittest.main()
