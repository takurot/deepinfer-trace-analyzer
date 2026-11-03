# 機能仕様書: deepinfer-trace-analyzer

- 版: 0.1（ドラフト）
- 対象: DeepInfer `main` @ `5537415` が出力する torch chrome trace
- 以下で `file:line` と書く箇所は、すべて DeepInfer リポジトリ内のパスを指す。

## 1. 目的

設定を変えたときに、「なぜ速く（遅く）なったか」をカーネルのカテゴリの単位で説明できるようにする。

現状は、trace を Perfetto などで目で見るしかない。集計や比較をする道具はない。

- `examples/_trace.py` は、prefill と decode の trace を連結・整形するだけである。
- `benchmark.py` と `sweep/` が出すのは、スカラーの指標だけである。

## 2. 入力となる trace

| 生成元 | 中身 |
|---|---|
| `examples/benchmark.py`（`PROFILE=1`） | full-chunk の prefill 2 step と、定常状態の decode 3 step を、`export_profile_trace`（`_trace.py:143-183`）で 1 本につないだもの |
| `deepinfer/engine/profile_server.py`（`POST /profile`） | driver worker 1 枚の、連続した N step（既定 20、最大 50） |

本ツールが前提とする trace の性質:

- 形式は chrome trace JSON（`traceEvents` 配列）。`ts` と `dur` は文字列のこともある（`_trace.py:19-31`）。そのため Decimal で読む。
- CPU 側のイベントの `pid` はプロセス番号、GPU 側のイベントの `pid` は device index（`_trace.py:10`）。
- CPU と GPU のイベントは、`args.correlation` と `args["External id"]` で対応する（`_trace.py:14`）。
- step の開始点は、`name == "deepinfer:schedule"` のイベントの `ts`。この名前は `timer.py:60` の `record_function(f"deepinfer:{name}")` が、`llm_engine.py:881` の `cpu_timer.record("schedule")` から記録する。

## 3. 機能要件

### FR-1 読み込みと正規化

- `.json` と `.json.gz` に対応する。
- 対象は `ph == "X"` のイベントだけで、メタデータ（`ph == "M"`）は名前の解決にだけ使う。
- GPU のイベントは、`cat` が `kernel`、`gpu_memcpy`、`gpu_memset` のものと定義する。
- 数百 MB の trace でも動くように、ストリーミングでパースする（ijson など）。

### FR-2 step への分割

- step k の区間は `[schedule_ts[k], schedule_ts[k+1])` とする。最後の step は、最後の GPU イベントの終了時刻までとする。
- prefill と decode の判別は、次のいずれかで行う。
  1. `--prefill-steps N` で先頭の N step を prefill とみなす。benchmark の trace の既定は 2。
  2. 区間内の CPU イベント名 `launch_forward_N{reqs}_B{tokens}`（`model_runner.py:322`）から、batch のトークン数と要求数を読み、`tokens / reqs` が大きいものを prefill とする。
- 連結済みの trace では、prefill と decode の境界で時刻がシフトされている（`_trace.py` の `_rebase`）。境界をまたぐ区間は集計から外す。

### FR-3 カーネルの分類

- 正規表現の規則を上から順に当て、最初に当たったカテゴリに入れる。
- 規則は内蔵の既定を持ち、`--rules rules.yaml` で上書き・追加できる。
- どの規則にも当たらないものは `other` に入れる。`--show-unclassified` を付けると、`other` に入った時間の上位カーネル名を出し、規則を育てる材料にする。

既定のカテゴリ:

| カテゴリ | 既定のパターン（暫定） |
|---|---|
| `comm.nccl` | `nccl` |
| `comm.deepep` | `deep_ep\|dispatch\|combine\|notify` |
| `attn.flashmla` | `flash_mla\|flashmla\|sparse_attn` |
| `gemm.deepgemm` | `deep_gemm\|fp8_gemm\|m_grouped\|gemm` |
| `tilelang` | `tilelang\|tl_` |
| `sampler` | `sample\|topk\|softmax` |
| `memcpy` | `cat` が `gpu_memcpy` または `gpu_memset` |
| `other` | 上記のどれにも当たらないもの |

**注意:** 実際のカーネルのシンボル名は、GPU 実機の trace でまだ確認していない。この表は暫定で、最初の実機 trace をもとに必ず改訂する（受け入れ基準 2）。

### FR-4 GPU の空き時間（bubble）

- step の区間ごとに、device pid ごとの GPU イベント区間の和集合を取る。区間の長さからその長さを引いた値を `idle` とする。
- 複数のストリームで重なっている時間は、和集合なので二重に数えない。カテゴリ別の時間は重なりを含む合計（`busy_sum`）として、和集合（`busy_union`）とは分けて出す。
- 参考として、CPU 側の `deepinfer:bubble` 区間（`llm_engine.py:991`）の合計時間も並べて出す。

### FR-5 launch の遅延（v0.1 では参考値）

- correlation ID で CPU 側の `cudaLaunchKernel` などと GPU 側のカーネルを結び、`gpu.ts - cpu.ts` の p50 と p99 を出す。
- CUDA Graph で再生されたカーネルは、launch と 1 対 1 に対応しない。そのため、対応が取れたものだけを集計し、取れた割合を併記する。

### FR-6 `summary` コマンド

```
deepinfer-trace-analyzer summary TRACE [--prefill-steps N] [--rules FILE] [--format table|markdown|json]
```

phase（prefill / decode）ごとに、次の値を出す。

- step 数
- step 時間の平均と最大
- カテゴリごとの時間（1 step あたりの平均）と構成比
- `idle` の割合
- `other` の割合

### FR-7 `diff` コマンド

```
deepinfer-trace-analyzer diff BEFORE AFTER [--prefill-steps N] [--format ...]
```

- phase ごと・カテゴリごとに、BEFORE と AFTER の値、差分（μs/step）、構成比の差分（pt）を出す。
- 差分の絶対値が大きい順に並べる。
- step 時間の差分の内訳（各カテゴリの寄与の和と `idle` の差分）が、step 時間そのものの差分とどれだけ食い違うかを「説明できない残差」として出す。
- 2 つの trace で device 数や phase ごとの step 数が違う場合は、警告を出す。どちらも 1 step あたりの値に正規化してから比べる。

### FR-8 JSON 出力

`--format json` のときは、上記のすべての数値を機械で読める形で出す。CI でしきい値をチェックする用途は v0.2 以降で検討する。

## 4. 非機能要件

- 依存は Python 3.12 以上の標準ライブラリと、ストリーミング JSON パーサのみ。DeepInfer、torch、GPU は不要。
- 500 MB の trace を 60 秒以内で処理し、メモリ使用量は 2 GB 以内に収める（目標値）。
- 出力する時間の単位は μs。trace の `ts` の精度（小数）は保ったまま計算する。

## 5. 受け入れ基準

1. 手で作った合成 trace（schedule 2 回、2 ストリームで重なるカーネル、correlation の対応ありとなし）で、step の分割・和集合・idle・カテゴリ別の時間が手計算と一致する。これは GPU なしで CI として回す。
2. 実機で取った benchmark の trace（decode、EP8）で、`other` の割合が 10% 以下になるまで既定の規則を改訂する。
3. 同じ trace 同士で `diff` を取ると、すべての差分が 0 になる。
4. `examples/_trace.py` の `_step_starts` と同じ trace から、同じ step の開始点の列が得られる。

## 6. 未決事項とリスク

- カーネル名は推測の段階で、実機の trace による補正が前提である。
- profile server の trace は driver worker 1 枚分しかない（`profile_server.py` 冒頭のコメント）。そのため、ランク間の偏りは本ツールでは見えない。
- Ascend（torch_npu）の trace は、イベントの形式やカーネル名が異なる可能性がある。v0.1 では、壊れずに `other` へ入れることだけを保証する。
- 連結済みの trace で、境界をまたいだ後も correlation ID の対応が保たれているかは、`_trace.py` の実装上は保たれる設計になっている。ただし実機で確かめる必要がある。
