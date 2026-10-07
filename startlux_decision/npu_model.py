"""Standalone Ascend text decisions with native torch-npu GDN prefill."""

import json
from pathlib import Path

import torch

from . import jevfmt
from .model import StartLuxDecision


class NPUChunkRule:
    """Stateless BF16 sequence-major prefill; no cache or decode state."""

    def __call__(
        self,
        query,
        key,
        value,
        g,
        beta,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=False,
        **kwargs,
    ):
        import torch_npu

        if initial_state is not None or output_final_state:
            raise ValueError("NPU decisions support stateless prefill only")
        if (
            query.dtype != torch.bfloat16
            or query.shape[-1] != 128
            or value.shape[-1] != 128
        ):
            raise ValueError(
                "NPU GDN requires BF16 with 128-dimensional query/value heads"
            )
        if use_qk_l2norm_in_kernel:
            from transformers.models.qwen3_5.modeling_qwen3_5 import l2norm

            query, key = l2norm(query), l2norm(key)
        batch, length = query.shape[:2]
        state = value.new_zeros(batch, value.shape[2], value.shape[-1], query.shape[-1])
        lengths = torch.full((batch,), length, dtype=torch.int32, device=query.device)
        output, _ = torch_npu.npu_chunk_gated_delta_rule(
            query.flatten(0, 1).contiguous(),
            key.flatten(0, 1).contiguous(),
            value.flatten(0, 1).contiguous(),
            beta=beta.flatten(0, 1).contiguous(),
            g=g.flatten(0, 1).float().contiguous(),
            initial_state=state,
            actual_seq_lengths=lengths,
            scale=query.shape[-1] ** -0.5,
        )
        return output.reshape(batch, length, value.shape[2], value.shape[-1]), None


class NPUDecision(StartLuxDecision):
    """Keep the official prompt and probability protocol; replace model execution.

    The reference implementation is text-only, eager, BF16 and stateless. Optional
    ``device_map="auto"`` partitions layers across explicitly visible NPUs; this
    is layer placement, not tensor parallel inference.
    """

    def __init__(self, path, device="npu:0", max_length=8192, device_map=None):
        import torch_npu  # noqa: F401
        import transformers

        if not torch.npu.is_available() or torch.device(device).type != "npu":
            raise ValueError("An available Ascend NPU device is required")
        if not 0 < max_length <= 8192:
            raise ValueError(
                "This NPU backend supports --max-length between 1 and 8192"
            )
        torch.npu.set_device(device)
        config = json.loads((Path(path) / "decision_config.json").read_text())
        self.tok = transformers.AutoTokenizer.from_pretrained(path)
        self.letters = jevfmt.check_tokenizer(self.tok)
        if self.letters != config["letter_token_ids"]:
            raise ValueError("Tokenizer and decision configuration letter IDs disagree")
        self.temperature = {
            kind: float(t) for kind, t in config["temperature_by_type"].items()
        }
        if any(not 0 < t < float("inf") for t in self.temperature.values()):
            raise ValueError("Temperatures must be finite and positive")
        wide = config["wide_choice"]
        self.group, self.keep, self.residual = (
            wide["group"],
            wide["keep"],
            wide["residual"],
        )
        hf_config = transformers.AutoConfig.from_pretrained(path)
        kwargs = (
            {"experts_implementation": "eager"}
            if getattr(hf_config.get_text_config(), "num_experts", 0)
            else {}
        )
        model = (
            getattr(transformers, hf_config.architectures[0])
            .from_pretrained(
                path,
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
                device_map=device_map or {"": device},
                **kwargs,
            )
            .eval()
        )
        self.body = getattr(model.model, "language_model", model.model)
        weight = model.get_output_embeddings().weight
        self.letter_rows = weight[self.letters].float().detach().clone()
        self.device = self.body.embed_tokens.weight.device
        self.gdn_layers = 0
        for module in self.body.modules():
            if hasattr(module, "chunk_gated_delta_rule"):
                module.chunk_gated_delta_rule = NPUChunkRule()
                self.gdn_layers += 1
        if not self.gdn_layers:
            raise ValueError("No supported Qwen3.5 GDN layers found")
        self.max_length = max_length
        self._media = self._kept = self.vision = None
        self.vision_note = "NPU text-only backend"
        self.graphs = {}
        self.fast_kernels = True

    @torch.inference_mode()
    def _logits(self, rows):
        encoded = [
            jevfmt.render_ids(row, self.tok, max_length=self.max_length)[0]
            for row in rows
        ]
        if not encoded:
            return [], 0
        length = max(map(len, encoded))
        ids = torch.full(
            (len(encoded), length),
            self.tok.pad_token_id,
            dtype=torch.long,
            device=self.device,
        )
        mask = torch.zeros_like(ids)
        for i, tokens in enumerate(encoded):
            ids[i, : len(tokens)] = torch.tensor(
                tokens, dtype=torch.long, device=self.device
            )
            mask[i, : len(tokens)] = 1
        hidden = self.body(
            input_ids=ids, attention_mask=mask, use_cache=False
        ).last_hidden_state
        last = torch.tensor(
            [len(tokens) - 1 for tokens in encoded], device=hidden.device
        )
        selected = hidden[torch.arange(len(encoded), device=hidden.device), last]
        logits = selected.to(self.letter_rows.device).float() @ self.letter_rows.T
        logits = logits.cpu()
        if not torch.isfinite(logits).all():
            raise RuntimeError("Non-finite NPU decision logits")
        return [z[: len(row["options"])] for row, z in zip(rows, logits)], sum(
            map(len, encoded)
        )

    def decide(self, state, questions, images=None):
        if images:
            raise ValueError("The NPU backend currently supports text only")
        return super().decide(state, questions)
