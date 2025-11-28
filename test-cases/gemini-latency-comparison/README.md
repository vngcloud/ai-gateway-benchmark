# Gemini API Latency Comparison: Direct vs AIPlatform vs AI Gateway

This benchmark compares the latency performance of calling Google Gemini API through three different methods:

1. **Direct Google Gemini API** - Calling Google's API directly
2. **VNGCloud AIPlatform** - Calling Gemini via VNGCloud AIPlatform (Model as a Service)
3. **VNGCloud AI Gateway** - Calling Gemini via VNGCloud AI Gateway (pointing to AIPlatform)

## Test Configuration

| Parameter | Value |
|-----------|-------|
| Model | `gemini-2.5-flash-lite` |
| Dataset | ShareGPT_Vicuna_unfiltered (1000 prompts) |
| Max Tokens | 512 |
| Temperature | 0.0 |
| Concurrency | 5 |
| Total Requests | 1000 |

## Results Summary

### Overall Performance

| Metric | Direct Gemini | AIPlatform | AI Gateway |
|--------|--------------|------------|------------|
| Completed | 1000 | 1000 | 1000 |
| Failed | 0 | 0 | 0 |
| Duration (s) | 617.47 | 627.44 | 624.11 |
| Requests/sec | 1.62 | 1.59 | 1.60 |
| Tokens/sec | 695.22 | 684.54 | 686.95 |

### Time to First Token (TTFT)

| Metric | Direct Gemini | AIPlatform | AI Gateway |
|--------|--------------|------------|------------|
| Mean | 599.66 ms | 617.91 ms | 599.29 ms |
| Median | 594.96 ms | 613.19 ms | 594.15 ms |
| Std | 74.47 ms | 59.91 ms | 60.93 ms |
| P50 | 594.96 ms | 613.19 ms | 594.15 ms |
| P90 | 663.24 ms | 699.03 ms | 670.16 ms |
| P99 | 816.47 ms | 793.71 ms | 810.63 ms |

### Inter-Token Latency (ITL)

| Metric | Direct Gemini | AIPlatform | AI Gateway |
|--------|--------------|------------|------------|
| Mean | 241.93 ms | 244.80 ms | 273.68 ms |
| Median | 269.42 ms | 273.93 ms | 294.13 ms |
| Std | 158.88 ms | 158.63 ms | 144.76 ms |
| P50 | 269.42 ms | 273.93 ms | 294.13 ms |
| P90 | 431.33 ms | 433.72 ms | 446.42 ms |
| P99 | 538.20 ms | 537.23 ms | 561.39 ms |

### End-to-End Latency (E2E)

| Metric | Direct Gemini | AIPlatform | AI Gateway |
|--------|--------------|------------|------------|
| Mean | 3072.62 ms | 3121.45 ms | 3107.57 ms |
| Median | 3420.06 ms | 3470.96 ms | 3464.71 ms |

## Latency Overhead Analysis

Comparing to Direct Gemini API as baseline:

### AIPlatform Overhead
| Metric | Overhead |
|--------|----------|
| TTFT Mean | +18.25 ms (+3.04%) |
| ITL Mean | +2.87 ms (+1.19%) |
| E2E Mean | +48.83 ms (+1.59%) |

### AI Gateway Overhead
| Metric | Overhead |
|--------|----------|
| TTFT Mean | -0.37 ms (-0.06%) |
| ITL Mean | +31.75 ms (+13.12%) |
| E2E Mean | +34.95 ms (+1.14%) |

## Conclusions

1. **TTFT (Time to First Token)**:
   - AI Gateway performs almost identically to Direct Gemini (-0.06%)
   - AIPlatform adds ~18ms overhead (+3.04%)

2. **ITL (Inter-Token Latency)**:
   - AIPlatform has minimal overhead (+1.19%)
   - AI Gateway shows higher ITL overhead (+13.12%)

3. **E2E (End-to-End Latency)**:
   - Both AIPlatform (+1.59%) and AI Gateway (+1.14%) have minimal overhead
   - AI Gateway actually performs slightly better than AIPlatform for E2E

4. **Overall Throughput**:
   - All three methods achieve similar throughput (~685-695 tokens/sec)
   - Direct Gemini: 695.22 tokens/sec
   - AIPlatform: 684.54 tokens/sec
   - AI Gateway: 686.95 tokens/sec

**Summary**: The latency overhead introduced by VNGCloud AIPlatform and AI Gateway is minimal for most use cases. End-to-end latency overhead is approximately 1-2%, which should be negligible for production workloads.

## Test Environment

- Date: 2025-11-28
- Region: Vietnam (VNGCloud)
- Client: Standard internet connection

## Raw Results

See the following JSON files for detailed metrics:
- `results_direct_gemini.json` - Direct Google Gemini API results
- `results_aiplatform.json` - VNGCloud AIPlatform results
- `results_aigw.json` - VNGCloud AI Gateway results
