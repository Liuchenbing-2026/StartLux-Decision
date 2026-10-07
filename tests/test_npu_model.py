"""CPU contracts and NPU recurrence regression; no checkpoint required."""

import importlib.util
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from startlux_decision.npu_model import NPUChunkRule, NPUDecision


class NPUContractTests(unittest.TestCase):
    def test_ragged_batch_reads_last_real_token_in_original_order(self):
        engine = object.__new__(NPUDecision)
        engine.device = torch.device("cpu")
        engine.tok = SimpleNamespace(pad_token_id=0)
        engine.max_length = 8192
        engine.letter_rows = torch.tensor([[1.0], [2.0], [3.0]])
        rows = [{"options": [1, 2]}, {"options": [1, 2, 3]}]

        def forward(input_ids, attention_mask, use_cache):
            self.assertEqual(input_ids.tolist(), [[10, 20, 0], [30, 40, 50]])
            self.assertEqual(attention_mask.tolist(), [[1, 1, 0], [1, 1, 1]])
            self.assertFalse(use_cache)
            return SimpleNamespace(
                last_hidden_state=input_ids.unsqueeze(-1).to(torch.bfloat16)
            )

        engine.body = forward
        with patch(
            "startlux_decision.npu_model.jevfmt.render_ids",
            side_effect=[([10, 20], None), ([30, 40, 50], None)],
        ):
            logits, tokens = engine._logits(rows)
        self.assertEqual(tokens, 5)
        self.assertEqual(
            [v.tolist() for v in logits], [[20.0, 40.0], [50.0, 100.0, 150.0]]
        )
        self.assertTrue(all(v.dtype == torch.float32 for v in logits))

    def test_stateless_layout_and_state_reinitialization(self):
        q = torch.ones(2, 5, 2, 128, dtype=torch.bfloat16)

        def op(query, key, value, **kwargs):
            self.assertEqual(query.shape, (10, 2, 128))
            self.assertEqual(kwargs["actual_seq_lengths"].tolist(), [5, 5])
            self.assertEqual(kwargs["initial_state"].count_nonzero(), 0)
            kwargs["initial_state"].fill_(1)
            return value, kwargs["initial_state"]

        with patch.dict(
            sys.modules, {"torch_npu": SimpleNamespace(npu_chunk_gated_delta_rule=op)}
        ):
            for _ in range(2):
                actual, state = NPUChunkRule()(q, q, q, q[..., 0], q[..., 0])
                torch.testing.assert_close(actual, q, atol=0, rtol=0)
                self.assertIsNone(state)
            with self.assertRaises(ValueError):
                NPUChunkRule()(q, q, q, q[..., 0], q[..., 0], initial_state=q)


@unittest.skipUnless(importlib.util.find_spec("torch_npu"), "requires torch-npu")
class NPUKernelTests(unittest.TestCase):
    def test_chunk_rule_against_hf_and_repeat(self):
        import torch_npu  # noqa: F401
        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            torch_chunk_gated_delta_rule,
        )

        if not torch.npu.is_available():
            self.skipTest("No assigned NPU")
        torch.manual_seed(42)
        for batch, length in [
            (1, 1),
            (1, 63),
            (1, 64),
            (1, 65),
            (1, 257),
            (2, 65),
            (8, 65),
        ]:
            with self.subTest(batch=batch, length=length):
                q, k, v = [
                    torch.randn(
                        batch, length, 2, 128, device="npu", dtype=torch.bfloat16
                    )
                    for _ in range(3)
                ]
                g = -torch.rand(batch, length, 2, device="npu")
                beta = torch.rand(batch, length, 2, device="npu", dtype=torch.bfloat16)
                expected, _ = torch_chunk_gated_delta_rule(
                    q, k, v, g, beta, use_qk_l2norm_in_kernel=True
                )
                actual, _ = NPUChunkRule()(
                    q, k, v, g, beta, use_qk_l2norm_in_kernel=True
                )
                repeated, _ = NPUChunkRule()(
                    q, k, v, g, beta, use_qk_l2norm_in_kernel=True
                )
                torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-2)
                torch.testing.assert_close(actual, repeated, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
