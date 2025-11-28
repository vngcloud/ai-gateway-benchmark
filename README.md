# AI Gateway Benchmark

A lightweight benchmark tool for measuring latency metrics of OpenAI-compatible API endpoints.

## Features

- Measures **TTFT** (Time to First Token), **ITL** (Inter-Token Latency), and **E2E latency**
- Supports OpenAI-compatible `/v1/chat/completions` endpoint
- Concurrent request support (up to 10 for light benchmarking)
- Loads prompts from JSON/JSONL dataset files
- Outputs detailed statistics: mean, median, std, P50, P90, P99

## Installation

```bash
# Create virtual environment
python -m venv venv

# Activate (Linux/macOS)
source venv/bin/activate

# Activate (Windows)
# venv\Scripts\activate

# Install dependencies
pip install aiohttp numpy tqdm
```

## Usage

### Basic

```bash
python benchmark.py \
    --dataset sample_dataset.jsonl \
    --base-url http://localhost:8000 \
    --model your-model-name
```

### With Options

```bash
python benchmark.py \
    --dataset sample_dataset.jsonl \
    --base-url http://localhost:8000 \
    --model qwen2.5-7b \
    --concurrency 5 \
    --max-tokens 256 \
    --temperature 0.0 \
    --output results.json
```

### Benchmark direct call to Google Gemini API

```bash
python benchmark.py \
    --dataset sample_dataset.jsonl \
    --base-url https://generativelanguage.googleapis.com/v1beta/openai \
    --model gemini-2.5-flash-lite \
    --api-key <API KEY>
```
API Key is google api key get from console of google

### Benchmark call google gemini via AIPlatform

```bash
python benchmark.py \
    --dataset sample_dataset.jsonl \
    --base-url https://maas-llm-aiplatform-hcm.api.vngcloud.vn \
    --model gemini/gemini-2.5-flash-lite \
    --api-key <API KEY>
```

Note: API key from aiplatform console https://aiplatform.console.vngcloud.vn/keys

### Benchmark call google gemini via AIGW point to AIPlatform

```bash
python benchmark.py \
    --dataset sample_dataset.jsonl \
    --base-url https://user-<userid>.ai-gateway.vngcloud.vn/gemini/gemini-2.5-flash-lite \
    --model gemini-2.5-flash-lite \
    --api-key <API KEY>
```

Note: API key from AI Gateway console

## Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--dataset` | (required) | Path to dataset file (.json or .jsonl) |
| `--base-url` | `http://localhost:8000` | Base URL of the API server |
| `--model` | (required) | Model name for requests |
| `--api-key` | `$OPENAI_API_KEY` | API key for authentication |
| `--max-tokens` | `256` | Maximum tokens to generate |
| `--temperature` | `0.0` | Sampling temperature |
| `--concurrency` | `1` | Concurrent requests (max 10) |
| `--num-prompts` | all | Limit number of prompts from dataset |
| `--output` | None | Save results to JSON file |
| `--disable-tqdm` | False | Disable progress bar |

## Dataset Format

### JSONL (recommended)

```jsonl
{"prompt": "What is the capital of France?", "max_tokens": 100}
{"prompt": "Explain machine learning.", "max_tokens": 200}
{"prompt": "Write a poem about the ocean.", "max_tokens": 150}
```

### JSON

```json
[
  {"prompt": "What is the capital of France?", "max_tokens": 100},
  {"prompt": "Explain machine learning.", "max_tokens": 200}
]
```

Supported field names for prompts: `prompt`, `question`, `text`, `input`

## Output Metrics

| Metric | Description |
|--------|-------------|
| **TTFT** | Time to First Token - latency until first token is received |
| **ITL** | Inter-Token Latency - time between consecutive tokens |
| **E2E** | End-to-End latency - total request duration |

For each metric, the tool reports:
- Mean, Median, Std
- P50, P90, P99 percentiles

## Example Output

```
============================================================
                    Benchmark Results
============================================================

---------------------------Requests------------------------
  Completed:              10
  Failed:                 0
  Duration:               15.23 s
  Requests/sec:           0.66
  Output tokens/sec:      98.45

-----------------Time to First Token (TTFT)----------------
  Mean:                   125.45 ms
  Median:                 118.20 ms
  Std:                    32.10 ms
  P50:                    118.20 ms
  P90:                    165.30 ms
  P99:                    198.50 ms

------------------Inter-Token Latency (ITL)----------------
  Mean:                   15.23 ms
  Median:                 14.10 ms
  Std:                    5.20 ms
  P50:                    14.10 ms
  P90:                    22.30 ms
  P99:                    28.90 ms

---------------------End-to-End Latency--------------------
  Mean:                   1523.45 ms
  Median:                 1456.20 ms
============================================================
```

## JSON Output Format

When using `--output results.json`:

```json
{
  "config": {
    "dataset": "sample_dataset.jsonl",
    "model": "qwen2.5-7b",
    "base_url": "http://localhost:8000",
    "max_tokens": 256,
    "temperature": 0.0,
    "concurrency": 5,
    "num_prompts": 10
  },
  "metrics": {
    "completed": 10,
    "failed": 0,
    "duration_s": 15.23,
    "requests_per_second": 0.66,
    "tokens_per_second": 98.45,
    "ttft": {
      "mean_ms": 125.45,
      "median_ms": 118.20,
      "std_ms": 32.10,
      "p50_ms": 118.20,
      "p90_ms": 165.30,
      "p99_ms": 198.50
    },
    "itl": {
      "mean_ms": 15.23,
      "median_ms": 14.10,
      "std_ms": 5.20,
      "p50_ms": 14.10,
      "p90_ms": 22.30,
      "p99_ms": 28.90
    },
    "e2e": {
      "mean_ms": 1523.45,
      "median_ms": 1456.20
    }
  }
}
```
