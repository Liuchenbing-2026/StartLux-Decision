# Ascend NPU text inference

This standalone backend uses PyTorch, Transformers and torch-npu. It reuses the
official prompt renderer, question/option order, wide-choice reduction and
checkpoint temperatures. It does not depend on vLLM or its Ascend plugin.

The Qwen3.5 decoder runs stateless BF16 prefill. Native
`torch_npu.npu_chunk_gated_delta_rule` executes the GDN recurrence; convolution
and dense attention use PyTorch operators. The last real token of each question
is projected against the A–Z output-head rows in FP32. Inputs are right-padded
with an explicit mask. No text tokens are generated.

## Installation

Start a dedicated container from this tested base (no image export is needed):

```text
quay.io/ascend/vllm-ascend@sha256:2c8aac4281e56953764a7fa60773cec734342d32d9d6d76c0c8f3a8660a64b85
```

The image supplies Python 3.12.13, PyTorch 2.10.0+cpu, torch-npu 2.10.0.post4 and
Transformers 5.14.1. Its preinstalled inference packages are not imported by this
backend. Mount the matching Ascend driver and your assigned NPU devices according
to your host's container configuration. Keep checkpoints read-only and results
in a separate writable directory. Run from this repository root:

```bash
python -m venv --system-site-packages .venv-npu
.venv-npu/bin/python -m pip install -r requirements-npu.txt
```

Only install CUDA FLA/causal-conv1d dependencies for the CUDA backend; they are
not dependencies of this backend. Accelerate is required for checkpoint placement.
The additional tested versions are accelerate 1.15.0 and pyarrow 25.0.1.

## Start 4B

```bash
export ASCEND_RT_VISIBLE_DEVICES=0  # one assigned physical NPU
export OMP_NUM_THREADS=4
.venv-npu/bin/python -m startlux_decision.server \
  --backend npu --device npu:0 --model /models/StartLux-Decision-4B \
  --max-length 8192 --no-images --port 8090
curl -f http://127.0.0.1:8090/health
```

The listener defaults to loopback. `--backend npu` is explicit and does not change
CUDA or MLX defaults. Use the existing `/v1/systemone` request/response protocol.
For the larger MoE checkpoint, `--device-map auto` can place decoder layers across
multiple explicitly visible NPUs. This is experimental layer placement, not tensor
parallelism; it is not covered by the 4B acceptance results.

## Correctness and evaluation

```bash
.venv-npu/bin/python -m unittest discover -s tests -p test_npu_model.py -v
bash eval/fetch_benchmarks.sh
.venv-npu/bin/python eval/suites.py predict \
  --endpoint http://127.0.0.1:8090 --out /outputs/npu-seven
.venv-npu/bin/python eval/suites.py score /outputs/npu-seven \
  --json /outputs/npu-seven-metrics.json
```

The tests check ragged last-token readout, FP32 candidate projection and state
isolation. The device test compares seven GDN shapes (including chunk boundaries
and multiple rows) against the Transformers reference with `atol=0.002`,
`rtol=0.02`; repeated calls must match exactly. These operator checks do not
establish whole-model accuracy equivalence. Seven-suite evaluation is separate.
The CPU contracts and all seven real-device recurrence cases passed. The 4B
service (one NPU) and 35B-A3B service (two-NPU automatic layer placement) passed
startup with a three-field request. Full seven-suite evaluation is in progress;
these smoke checks do not establish model acceptance.

Performance results are not published in this change. A reproducible optional
latency command is:

```bash
.venv-npu/bin/python eval/latency.py http://127.0.0.1:8090/v1/systemone 200
```

## Scope

This implementation is text-only, eager and stateless, with a maximum of 8,192
input tokens per question. Images, graph replay and shared-prefix caches are not
implemented. Full seven-suite model acceptance and the 35B placement test must be
reported separately; startup or operator tests alone are insufficient.
