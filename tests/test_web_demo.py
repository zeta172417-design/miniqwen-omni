import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from scripts.web_demo_omni import (
    Runtime,
    auth_from_environment,
    build_demo,
    decode_audio,
    file_path,
    first_image,
    frames_to_mimi,
    normalize_audio,
    prepare_audio,
)


class WebDemoTests(unittest.TestCase):
    def test_decoded_audio_is_finite_nonempty_24khz(self):
        class FakeMimi(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(()))

            def decode(self, codes):
                self.last_codes = codes
                samples = torch.linspace(-0.5, 0.5, 2400).reshape(1, 1, -1)
                return SimpleNamespace(audio_values=samples)

        mimi = FakeMimi()
        runtime = Runtime(None, None, mimi, None, {}, "cpu", torch.float32, 30)
        sample_rate, samples = decode_audio(runtime, [[index] * 8 for index in range(10)])
        self.assertEqual(sample_rate, 24000)
        self.assertEqual(samples.dtype, np.float32)
        self.assertEqual(samples.shape, (2400,))
        self.assertTrue(np.isfinite(samples).all())
        self.assertGreater(samples.size / sample_rate, 0)

    def test_auth_requires_password(self):
        args = SimpleNamespace(no_auth=False, share=False, host="0.0.0.0")
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "MINIQWEN_WEB_PASSWORD"):
                auth_from_environment(args)

    def test_auth_uses_constant_time_credentials(self):
        args = SimpleNamespace(no_auth=False, share=False, host="0.0.0.0")
        env = {"MINIQWEN_WEB_USERNAME": "friend", "MINIQWEN_WEB_PASSWORD": "safe-pass-123"}
        with patch.dict(os.environ, env, clear=True):
            auth = auth_from_environment(args)
        self.assertTrue(auth("friend", "safe-pass-123"))
        self.assertFalse(auth("friend", "wrong-pass"))
        self.assertFalse(auth("stranger", "safe-pass-123"))

    def test_no_auth_cannot_be_public(self):
        for args in (
            SimpleNamespace(no_auth=True, share=False, host="0.0.0.0"),
            SimpleNamespace(no_auth=True, share=True, host="127.0.0.1"),
        ):
            with self.assertRaisesRegex(ValueError, "仅允许本机"):
                auth_from_environment(args)

    def test_audio_normalization_and_mimi_layout(self):
        stereo = np.array([[32767, -32768], [0, 0]], dtype=np.int16)
        mono = normalize_audio(stereo, 16000)
        self.assertEqual(mono.dtype, np.float32)
        self.assertEqual(mono.shape, (2,))
        self.assertLessEqual(float(np.abs(mono).max()), 1.0)
        codes = frames_to_mimi([[index] * 8 for index in range(3)])
        self.assertEqual(tuple(codes.shape), (1, 8, 3))

    def test_audio_without_display_asr_has_no_thread_task(self):
        class Processor:
            def __call__(self, *args, **kwargs):
                return SimpleNamespace(
                    input_features=torch.zeros(1, 4, 8),
                    attention_mask=torch.ones(1, 4, dtype=torch.long),
                )

        runtime = Runtime(
            model=SimpleNamespace(audio_processor=Processor()),
            tokenizer=None,
            mimi_model=None,
            asr_model=None,
            voices={},
            device="cpu",
            dtype=torch.float32,
            max_audio_seconds=30,
        )
        mel, lengths, token_len, asr_task = prepare_audio(
            runtime, (16000, np.zeros(1600, dtype=np.float32))
        )
        self.assertEqual(tuple(mel.shape), (1, 4, 8))
        self.assertEqual(lengths.item(), 4)
        self.assertEqual(token_len, 4)
        self.assertIsNone(asr_task)

    def test_multimodal_file_values(self):
        self.assertEqual(file_path({"path": "/tmp/a.jpg"}), "/tmp/a.jpg")
        self.assertEqual(first_image({"files": [{"path": "/tmp/a.jpg"}]}), "/tmp/a.jpg")
        self.assertIsNone(first_image({"files": [{"path": "/tmp/a.txt"}]}))

    def test_ui_builds_without_loading_a_model(self):
        runtime = Runtime(None, None, None, None, {}, "cpu", None, 30)
        args = SimpleNamespace(open_thinking=False, max_text_chars=4000)
        demo = build_demo(runtime, args)
        self.assertGreater(len(demo.blocks), 10)
        api_names = {
            dependency.get("api_name")
            for dependency in demo.get_config_file()["dependencies"]
        }
        self.assertIn("chat", api_names)
        self.assertIn("chat_audio", api_names)
        send_audio = [
            component for component in demo.get_config_file()["components"]
            if component.get("type") == "button"
            and component.get("props", {}).get("value") == "发送这段语音"
        ]
        self.assertEqual(len(send_audio), 1)
        self.assertTrue(send_audio[0]["props"]["interactive"])


if __name__ == "__main__":
    unittest.main()
