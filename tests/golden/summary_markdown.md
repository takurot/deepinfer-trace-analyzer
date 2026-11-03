**trace** `TMP/t.json`, **device** 0

## prefill: 1 step, step time mean 100.000 us, max 100.000 us

| category | exclusive us/step | share | inclusive us/step |
|---|--:|--:|--:|
| comm.nccl | 20.000 | 20.0% | 30.000 |
| gemm.deepgemm | 30.000 | 30.0% | 30.000 |
| memcpy | 5.000 | 5.0% | 5.000 |
| idle | 45.000 | 45.0% |  |

overlap 10.000 us, bubble 0.000 us, other 0.0%, unmatched 100.0%, spill n/a

## decode: 2 steps, step time mean 60.000 us, max 70.000 us

| category | exclusive us/step | share | inclusive us/step |
|---|--:|--:|--:|
| other | 20.000 | 33.3% | 20.000 |
| idle | 40.000 | 66.7% |  |

overlap 0.000 us, bubble 0.000 us, other 33.3%, unmatched 100.0%, spill n/a
