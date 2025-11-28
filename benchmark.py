#!/usr/bin/env python3
"""
Simple benchmark script for OpenAI-compatible API endpoints.
Measures TTFT (Time to First Token), ITL (Inter-Token Latency), and other metrics.

Usage:
    python benchmark.py --dataset dataset.jsonl --base-url http://localhost:8000 --model your-model
"""

import argparse
import asyncio
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import aiohttp
import numpy as np
from tqdm.asyncio import tqdm


@dataclass
class RequestInput:
    """Input for a single benchmark request."""
    prompt: str
    expected_output_len: int = 256
    prompt_len: int = 0


@dataclass
class RequestOutput:
    """Output metrics for a single request."""
    success: bool = False
    generated_text: str = ""
    latency: float = 0.0
    ttft: float = 0.0  # Time to First Token
    itl: list[float] = field(default_factory=list)  # Inter-token latencies
    output_tokens: int = 0
    prompt_len: int = 0
    error: str = ""


@dataclass
class BenchmarkMetrics:
    """Aggregated benchmark metrics."""
    completed: int = 0
    failed: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0

    # TTFT metrics (ms)
    mean_ttft_ms: float = 0.0
    median_ttft_ms: float = 0.0
    std_ttft_ms: float = 0.0
    p50_ttft_ms: float = 0.0
    p90_ttft_ms: float = 0.0
    p99_ttft_ms: float = 0.0

    # ITL metrics (ms)
    mean_itl_ms: float = 0.0
    median_itl_ms: float = 0.0
    std_itl_ms: float = 0.0
    p50_itl_ms: float = 0.0
    p90_itl_ms: float = 0.0
    p99_itl_ms: float = 0.0

    # E2E latency (ms)
    mean_e2e_ms: float = 0.0
    median_e2e_ms: float = 0.0

    # Throughput
    duration_s: float = 0.0
    requests_per_second: float = 0.0
    tokens_per_second: float = 0.0


def load_dataset(dataset_path: str) -> list[RequestInput]:
    """Load prompts from a dataset file (JSON or JSONL)."""
    path = Path(dataset_path)
    requests = []

    if path.suffix == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    prompt = data.get("prompt") or data.get("question") or data.get("text") or data.get("input")
                    if prompt:
                        requests.append(RequestInput(
                            prompt=prompt,
                            expected_output_len=data.get("max_tokens", 256)
                        ))
    elif path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Handle both list and dict with "data" key
            items = data if isinstance(data, list) else data.get("data", [data])
            for item in items:
                prompt = item.get("prompt") or item.get("question") or item.get("text") or item.get("input")
                if prompt:
                    requests.append(RequestInput(
                        prompt=prompt,
                        expected_output_len=item.get("max_tokens", 256)
                    ))
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}. Use .json or .jsonl")

    if not requests:
        raise ValueError(f"No valid prompts found in {dataset_path}")

    return requests


class StreamHandler:
    """Handles SSE streaming responses."""

    def __init__(self):
        self.buffer = ""

    def add_chunk(self, chunk_bytes: bytes) -> list[str]:
        """Add a chunk and return complete messages."""
        self.buffer += chunk_bytes.decode("utf-8")
        messages = []

        while "\n\n" in self.buffer:
            message, self.buffer = self.buffer.split("\n\n", 1)
            message = message.strip()
            if message:
                messages.append(message)

        # Check for complete single-line message
        if self.buffer.startswith("data: "):
            content = self.buffer.removeprefix("data: ").strip()
            if content == "[DONE]":
                messages.append(self.buffer.strip())
                self.buffer = ""
            elif content:
                try:
                    json.loads(content)
                    messages.append(self.buffer.strip())
                    self.buffer = ""
                except json.JSONDecodeError:
                    pass

        return messages


async def send_request(
    request: RequestInput,
    session: aiohttp.ClientSession,
    api_url: str,
    model: str,
    api_key: str | None,
    max_tokens: int,
    temperature: float,
    pbar: tqdm | None = None,
) -> RequestOutput:
    """Send a single request and measure metrics."""
    output = RequestOutput()
    output.prompt_len = len(request.prompt.split())  # Rough estimate

    # Build payload for chat completions API
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": request.prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    generated_text = ""
    st = time.perf_counter()
    most_recent_timestamp = st
    first_token_received = False

    try:
        async with session.post(api_url, json=payload, headers=headers) as response:
            if response.status == 200:
                handler = StreamHandler()

                async for chunk_bytes in response.content.iter_any():
                    chunk_bytes = chunk_bytes.strip()
                    if not chunk_bytes:
                        continue

                    messages = handler.add_chunk(chunk_bytes)
                    for message in messages:
                        if message.startswith(":"):
                            continue

                        chunk = message.removeprefix("data: ")
                        if chunk == "[DONE]":
                            continue

                        try:
                            data = json.loads(chunk)
                        except json.JSONDecodeError:
                            continue

                        timestamp = time.perf_counter()

                        # Extract content from chat completions response
                        content = None
                        if choices := data.get("choices"):
                            content = choices[0].get("delta", {}).get("content")

                        if content is not None:
                            if not first_token_received:
                                first_token_received = True
                                output.ttft = timestamp - st
                            else:
                                output.itl.append(timestamp - most_recent_timestamp)

                            most_recent_timestamp = timestamp
                            generated_text += content

                        # Get token count from usage
                        if usage := data.get("usage"):
                            output.output_tokens = usage.get("completion_tokens", 0)

                output.generated_text = generated_text
                output.success = first_token_received
                output.latency = most_recent_timestamp - st

                if not first_token_received:
                    output.error = "No tokens received"
            else:
                output.error = f"HTTP {response.status}: {response.reason}"
                output.success = False
    except Exception:
        output.success = False
        output.error = traceback.format_exc()

    if pbar:
        pbar.update(1)

    return output


def calculate_metrics(outputs: list[RequestOutput], duration: float) -> BenchmarkMetrics:
    """Calculate aggregated metrics from request outputs."""
    metrics = BenchmarkMetrics()

    ttfts = []
    itls = []
    e2els = []

    for output in outputs:
        if output.success:
            metrics.completed += 1
            metrics.total_output_tokens += output.output_tokens or len(output.generated_text.split())
            ttfts.append(output.ttft)
            itls.extend(output.itl)
            e2els.append(output.latency)
        else:
            metrics.failed += 1

    metrics.duration_s = duration

    if metrics.completed > 0:
        metrics.requests_per_second = metrics.completed / duration
        metrics.tokens_per_second = metrics.total_output_tokens / duration

    # TTFT metrics
    if ttfts:
        ttfts_ms = [t * 1000 for t in ttfts]
        metrics.mean_ttft_ms = float(np.mean(ttfts_ms))
        metrics.median_ttft_ms = float(np.median(ttfts_ms))
        metrics.std_ttft_ms = float(np.std(ttfts_ms))
        metrics.p50_ttft_ms = float(np.percentile(ttfts_ms, 50))
        metrics.p90_ttft_ms = float(np.percentile(ttfts_ms, 90))
        metrics.p99_ttft_ms = float(np.percentile(ttfts_ms, 99))

    # ITL metrics
    if itls:
        itls_ms = [t * 1000 for t in itls]
        metrics.mean_itl_ms = float(np.mean(itls_ms))
        metrics.median_itl_ms = float(np.median(itls_ms))
        metrics.std_itl_ms = float(np.std(itls_ms))
        metrics.p50_itl_ms = float(np.percentile(itls_ms, 50))
        metrics.p90_itl_ms = float(np.percentile(itls_ms, 90))
        metrics.p99_itl_ms = float(np.percentile(itls_ms, 99))

    # E2E latency
    if e2els:
        e2els_ms = [t * 1000 for t in e2els]
        metrics.mean_e2e_ms = float(np.mean(e2els_ms))
        metrics.median_e2e_ms = float(np.median(e2els_ms))

    return metrics


async def verify_endpoint(
    api_url: str,
    model: str,
    api_key: str | None,
) -> tuple[bool, str]:
    """Verify the API endpoint is accessible and working."""
    # Use non-streaming for verification (more reliable)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Say OK"}],
        "max_tokens": 10,
        "temperature": 0.0,
        "stream": False,
    }

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    timeout = aiohttp.ClientTimeout(total=30)

    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(api_url, json=payload, headers=headers) as response:
                if response.status == 401:
                    return False, "Authentication failed. Check your API key."
                elif response.status == 404:
                    return False, f"Endpoint not found: {api_url}"
                elif response.status == 400:
                    body = await response.text()
                    if "model" in body.lower():
                        return False, f"Invalid model name: {model}"
                    return False, f"Bad request: {body[:200]}"
                elif response.status != 200:
                    body = await response.text()
                    return False, f"HTTP {response.status}: {body[:200]}"

                # Check JSON response
                data = await response.json()
                if data.get("choices"):
                    return True, "OK"
                return False, f"Unexpected response: {str(data)[:100]}"

    except aiohttp.ClientConnectorError as e:
        return False, f"Connection failed: {e}. Check if the server is running."
    except asyncio.TimeoutError:
        return False, "Connection timeout. Server may be slow or unreachable."
    except Exception as e:
        return False, f"Verification failed: {str(e)}"


async def run_benchmark(
    requests: list[RequestInput],
    api_url: str,
    model: str,
    api_key: str | None,
    max_tokens: int,
    temperature: float,
    concurrency: int,
    disable_tqdm: bool,
) -> tuple[list[RequestOutput], float]:
    """Run the benchmark with specified concurrency."""

    connector = aiohttp.TCPConnector(
        limit=concurrency,
        limit_per_host=concurrency,
    )

    timeout = aiohttp.ClientTimeout(total=600)  # 10 min timeout

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        semaphore = asyncio.Semaphore(concurrency)
        pbar = None if disable_tqdm else tqdm(total=len(requests), desc="Benchmarking")

        async def limited_request(req: RequestInput) -> RequestOutput:
            async with semaphore:
                return await send_request(
                    request=req,
                    session=session,
                    api_url=api_url,
                    model=model,
                    api_key=api_key,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    pbar=pbar,
                )

        start_time = time.perf_counter()
        tasks = [asyncio.create_task(limited_request(req)) for req in requests]
        outputs = await asyncio.gather(*tasks)
        duration = time.perf_counter() - start_time

        if pbar:
            pbar.close()

        return outputs, duration


def print_results(metrics: BenchmarkMetrics):
    """Print benchmark results."""
    print("\n" + "=" * 60)
    print(" Benchmark Results ".center(60, "="))
    print("=" * 60)

    print(f"\n{'Requests':-^60}")
    print(f"  Completed:              {metrics.completed}")
    print(f"  Failed:                 {metrics.failed}")
    print(f"  Duration:               {metrics.duration_s:.2f} s")
    print(f"  Requests/sec:           {metrics.requests_per_second:.2f}")
    print(f"  Output tokens/sec:      {metrics.tokens_per_second:.2f}")

    print(f"\n{'Time to First Token (TTFT)':-^60}")
    print(f"  Mean:                   {metrics.mean_ttft_ms:.2f} ms")
    print(f"  Median:                 {metrics.median_ttft_ms:.2f} ms")
    print(f"  Std:                    {metrics.std_ttft_ms:.2f} ms")
    print(f"  P50:                    {metrics.p50_ttft_ms:.2f} ms")
    print(f"  P90:                    {metrics.p90_ttft_ms:.2f} ms")
    print(f"  P99:                    {metrics.p99_ttft_ms:.2f} ms")

    print(f"\n{'Inter-Token Latency (ITL)':-^60}")
    print(f"  Mean:                   {metrics.mean_itl_ms:.2f} ms")
    print(f"  Median:                 {metrics.median_itl_ms:.2f} ms")
    print(f"  Std:                    {metrics.std_itl_ms:.2f} ms")
    print(f"  P50:                    {metrics.p50_itl_ms:.2f} ms")
    print(f"  P90:                    {metrics.p90_itl_ms:.2f} ms")
    print(f"  P99:                    {metrics.p99_itl_ms:.2f} ms")

    print(f"\n{'End-to-End Latency':-^60}")
    print(f"  Mean:                   {metrics.mean_e2e_ms:.2f} ms")
    print(f"  Median:                 {metrics.median_e2e_ms:.2f} ms")

    print("=" * 60)


def save_results(metrics: BenchmarkMetrics, output_file: str, args: argparse.Namespace):
    """Save results to JSON file."""
    result = {
        "config": {
            "dataset": args.dataset,
            "model": args.model,
            "base_url": args.base_url,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "concurrency": args.concurrency,
            "num_prompts": args.num_prompts,
        },
        "metrics": {
            "completed": metrics.completed,
            "failed": metrics.failed,
            "duration_s": metrics.duration_s,
            "requests_per_second": metrics.requests_per_second,
            "tokens_per_second": metrics.tokens_per_second,
            "ttft": {
                "mean_ms": metrics.mean_ttft_ms,
                "median_ms": metrics.median_ttft_ms,
                "std_ms": metrics.std_ttft_ms,
                "p50_ms": metrics.p50_ttft_ms,
                "p90_ms": metrics.p90_ttft_ms,
                "p99_ms": metrics.p99_ttft_ms,
            },
            "itl": {
                "mean_ms": metrics.mean_itl_ms,
                "median_ms": metrics.median_itl_ms,
                "std_ms": metrics.std_itl_ms,
                "p50_ms": metrics.p50_itl_ms,
                "p90_ms": metrics.p90_itl_ms,
                "p99_ms": metrics.p99_itl_ms,
            },
            "e2e": {
                "mean_ms": metrics.mean_e2e_ms,
                "median_ms": metrics.median_e2e_ms,
            },
        },
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"\nResults saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark OpenAI-compatible API endpoints",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to dataset file (.json or .jsonl) containing prompts",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:8000",
        help="Base URL of the API server",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name to use for requests",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key (or set OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate per request",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of concurrent requests (max 10 for non-stress testing)",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=None,
        help="Number of prompts to use from dataset (default: all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file to save results",
    )
    parser.add_argument(
        "--disable-tqdm",
        action="store_true",
        help="Disable progress bar",
    )

    args = parser.parse_args()

    # Validate concurrency
    if args.concurrency > 10:
        print("Warning: Concurrency > 10. This is meant for light benchmarking, not stress testing.")
        print("Capping concurrency at 10.")
        args.concurrency = 10

    # Get API key
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")

    # Load dataset
    print(f"Loading dataset from: {args.dataset}")
    requests = load_dataset(args.dataset)
    print(f"Loaded {len(requests)} prompts")

    # Limit prompts if specified
    if args.num_prompts:
        requests = requests[:args.num_prompts]
        print(f"Using {len(requests)} prompts")

    # Build API URL
    api_url = f"{args.base_url.rstrip('/')}/v1/chat/completions"

    print(f"\nBenchmark Configuration:")
    print(f"  API URL:      {api_url}")
    print(f"  Model:        {args.model}")
    print(f"  Max tokens:   {args.max_tokens}")
    print(f"  Temperature:  {args.temperature}")
    print(f"  Concurrency:  {args.concurrency}")
    print(f"  Prompts:      {len(requests)}")

    # Verify endpoint
    print("\nVerifying endpoint...")
    ok, msg = asyncio.run(verify_endpoint(api_url, args.model, api_key))
    if not ok:
        print(f"FAILED: {msg}")
        return 1
    print("OK")

    # Run benchmark
    print("\nStarting benchmark...")
    outputs, duration = asyncio.run(
        run_benchmark(
            requests=requests,
            api_url=api_url,
            model=args.model,
            api_key=api_key,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            concurrency=args.concurrency,
            disable_tqdm=args.disable_tqdm,
        )
    )

    # Calculate and print metrics
    metrics = calculate_metrics(outputs, duration)
    print_results(metrics)

    # Save results if requested
    if args.output:
        save_results(metrics, args.output, args)

    # Print any errors
    errors = [o.error for o in outputs if o.error and not o.success]
    if errors:
        print(f"\nErrors ({len(errors)} requests failed):")
        for i, err in enumerate(errors[:5]):
            print(f"  {i+1}. {err[:200]}...")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more errors")

    return 0 if metrics.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
