# 機能仕様書: deepinfer-trace-analyzer

- 版: 0.2（ドラフト）
- 対象: DeepInfer `main` @ `5537415` が出力する torch chrome trace
- 以下で `file:line` と書く箇所は、すべて DeepInfer リポジトリ内のパスを指す。

## 1. 目的

設定を変えたときに、「なぜ速く（遅く）なったか」をカーネルのカテゴリの単位で説明できるようにする。

現状は、trace を Perfetto などで目で見るしかない。集計や比較をする道具はない。

- `examples/_trace.py` は、prefill と decode の trace を連結・整形するだけである。
- `benchmark.py` と `sweep/` が出すのは、スカラーの指標だけである。

プロファイラを有効にすると step 時間そのものが膨らむ。そのため本ツールの主な出力は、絶対値よりも**構成比と差分**とする。

## 2. 用語

| 用語 | 定義 |
|---|---|
| CPU イベント | `ph == "X"` かつ GPU イベントでないもの |
| GPU イベント | `ph == "X"` かつ `cat` が `kernel`、`gpu_memcpy`、`gpu_memset` のいずれか。`gpu_user_annotation` は区間の注釈なので含めない |
| launch | GPU イベントと `args.correlation` が一致する CPU イベント（`cudaLaunchKernel`、`cudaGraphLaunch`、`cudaMemcpyAsync` など） |
| step 窓 | step k の CPU 側の区間 `[schedule_ts[k], schedule_ts[k+1])`（§FR-2） |
| segment | 時間軸が連続している trace の部分。benchmark の trace は prefill と decode の 2 segment、profile server の trace は 1 segment |
| exclusive 時間 | GPU の稼働時間（和集合）を、各時刻にカテゴリを 1 つだけ割り当てて分配した時間（§FR-4） |
| inclusive 時間 | カテゴリごとの GPU イベントの `dur` の単純な和。ストリーム間の重なりを含む |

## 3. 入力となる trace

| 生成元 | 中身 |
|---|---|
| `examples/benchmark.py`（`PROFILE=1`、既定で有効） | full-chunk の prefill 2 step（`benchmark.py:105`）と、定常状態の decode 3 step（`benchmark.py:110`）を、`export_profile_trace`（`_trace.py:143-186`）で 1 本につないだもの。decode は 5 step 採ったうちの後ろ 3 step だけを残す（`benchmark.py:515`、`_trace.py:165`） |
| `deepinfer/engine/profile_server.py`（`POST /profile`） | driver worker 1 枚の、連続した N step（既定 20、最大 50、`profile_server.py:46-47`）。`export_chrome_trace` の生の出力で、`_trace.py` の整形は通らない |

本ツールが前提とする trace の性質:

- 形式は chrome trace JSON。トップレベルは `{"traceEvents": [...]}` だが、torch_npu ではイベントの配列そのものになる（`_trace.py:65-67`）。両方を受け付ける。
- `ts` と `dur` は数値のことも文字列のこともある（`_trace.py:18-33`）。どちらも Decimal として読む。
- `args.correlation` と `args["External id"]` は、torch_npu では文字列のことがある（`_trace.py:73-75`）。整数に揃えて扱う。
- CPU 側のイベントの `pid` はプロセス番号、GPU 側のイベントの `pid` は device index である（`_trace.py:12-13`）。`pid` が `"Spans"`、`"Traces"`、`""` のトラックはプロファイラ自身のもので、無視する（`_trace.py:15`、`:86`）。
- CPU と GPU のイベントは、`args.correlation` で対応する（`_trace.py:14`）。CUDA Graph の再生では、1 つの `cudaGraphLaunch` に複数のカーネルが同じ correlation で対応する。
- step の開始点は、`name == "deepinfer:schedule"` のイベントの `ts` である。この名前は `deepinfer/timer.py:60` の `record_function(f"deepinfer:{record_name or name}")` が、`llm_engine.py:881` の `cpu_timer.record("schedule")` から記録する。
- 各 step の CPU 側には `deepinfer:launch_forward_N{reqs}_B{tokens}` が記録される（`model_runner.py:322`）。
- **重要:** `DEEPINFER_ZERO_OVERHEAD_SCHEDULING=1` のとき、decode では step k の後処理が step k+1 に回される（`worker.py:183-185`）。このため step k の GPU カーネルは、時刻としては step k+1 の窓に入ることがある。GPU イベントを時刻だけで step 窓に振り分けると、帰属を誤る。
- profile server の trace は、`LLMEngine.step` の冒頭（schedule の前）でプロファイラを開始・停止する（`llm_engine.py:878-879`、`profile_server.py:256-269`）。そのため schedule は N 個入り、最後の step は停止時点で打ち切られる。プロファイラ起動直後の step は、起動コストで遅くなる（`benchmark.py:515` のコメント）。

## 4. 機能要件

### FR-1 読み込みと正規化

- `.json` と `.json.gz` に対応する。
- 対象は `ph == "X"` のイベントだけとする。メタデータ（`ph == "M"`）は名前の解決にだけ使う。カウンタ（`ph == "C"`）、フロー（`cat == "ac2g"`）などは読み飛ばす。
- 数百 MB の trace でも動くように、ストリーミングでパースする（ijson の C バックエンド）。イベントは必要なフィールド（`ph`、`cat`、`name`、`pid`、`tid`、`ts`、`dur`、`args.correlation`）だけを保持する。
- `deepinfer:schedule` が 1 つもない trace は、エラーで終了する（§FR-9）。

### FR-2 step への分割と phase の判定

**step 窓**

- step k の窓は `[schedule_ts[k], schedule_ts[k+1])` とする。segment の最後の step は、その segment の最後のイベントの終了時刻までとする。
- benchmark の trace では、`_rebase` によって decode segment の先頭が prefill segment の終端にぴったり接続される（`_trace.py:176`）。そのため、segment の境界をまたぐ step はない。境界は「先頭から `--prefill-steps` 個目の schedule の次」として扱い、境界の直前の step はそこで打ち切る。境界の前後の時間差は人工的に 0 になっているので、step 間の隙間として数えない。

**GPU イベントの step への帰属**

- GPU イベントは、launch の `ts` が入っている step 窓に帰属させる。これで、zero-overhead scheduling のもとでも、カーネルは自分を launch した step に数えられる。
- launch が見つからない GPU イベントは、自身の `ts` が入っている step 窓に帰属させる。こうしたイベントの割合（時間ベース）を `unmatched_ratio` として出す。
- どの step にも入らない GPU イベント（最初の schedule より前に launch されたものなど）は捨て、その件数を出す。

**phase の判定**（優先順）

1. `--prefill-steps N` を指定したときは、先頭の N step を prefill、残りを decode とする。benchmark の trace の既定は 2 で、trace 名から判別できないので明示的に指定する。
2. 指定がないときは、step 窓の中の最初の `deepinfer:launch_forward_N{reqs}_B{tokens}` を読み、`tokens / reqs` が `--prefill-threshold`（既定 16）以上なら prefill、未満なら decode とする。decode では MTP/NextN を使っても `tokens / reqs` は小さい値にとどまる。
3. `launch_forward` が見つからない step は `unknown` とし、summary では独立した phase として出す。

**除外**

- `--skip-steps K` で先頭の K step を集計から外す。profile server の trace では、プロファイラ起動直後の step を外すために使う。既定は 0。

### FR-3 カーネルの分類

- 規則は `(category, pattern)` の列で、上から順に当て、最初に当たったカテゴリに入れる。パターンは Python の正規表現で、`re.search`、大文字小文字を区別しない、で照合する。
- 照合の対象は `name`（torch が出力するデマングル済みの名前）とする。`cat` が `gpu_memcpy` か `gpu_memset` のイベントは、名前で照合せずに `memcpy` に入れる。
- 規則の上書きは、`--rules rules.toml` で行う（`tomllib` は標準ライブラリなので、依存を増やさない）。ユーザーの規則は既定の規則の前に評価される。既定の規則を使わないときは `--no-default-rules` を付ける。
- どの規則にも当たらないものは `other` に入れる。`--show-unclassified [K]` を付けると、`other` に入った exclusive 時間の上位 K 個（既定 20）のカーネル名を出し、規則を育てる材料にする。

`rules.toml` の形:

```toml
[[rule]]
category = "gemm.deepgemm"
pattern = "deep_gemm|fp8_gemm"
```

既定の規則（評価順）:

| 順 | カテゴリ | 既定のパターン（暫定） |
|---|---|---|
| 1 | `comm.nccl` | `nccl` |
| 2 | `comm.deepep` | `deep_ep\|deepep` |
| 3 | `attn.flashmla` | `flash_mla\|flashmla\|sparse_attn` |
| 4 | `gemm.deepgemm` | `deep_gemm\|fp8_gemm\|m_grouped` |
| 5 | `gemm.other` | `gemm\|cutlass\|cublas` |
| 6 | `tilelang` | `tilelang` |
| 7 | `sampler` | `sampl\|topk\|top_k` |
| – | `memcpy` | `cat` が `gpu_memcpy` または `gpu_memset`（名前の照合より前に決まる） |
| – | `other` | 上記のどれにも当たらないもの |

**注意:** 実際のカーネルのシンボル名は、GPU 実機の trace でまだ確認していない。この表は暫定で、最初の実機 trace をもとに必ず改訂する（受け入れ基準 2）。0.1 版にあった `dispatch|combine|notify`、`tl_`、`softmax`、単独の `gemm` は、無関係のカーネルを誤って拾いやすいので外した。

### FR-4 時間の分解（exclusive 時間と idle）

step ごと、device ごとに、次のように時間を分ける。

- `step_time` = step 窓の長さ。
- `busy_union` = その step に帰属した GPU イベントの区間の和集合のうち、step 窓と重なる部分の長さ。
- `idle` = `step_time - busy_union`。
- **exclusive 時間:** `busy_union` を時刻ごとに分け、その時刻に動いているイベントのうち、優先度が最も高いカテゴリに割り当てる。優先度は、計算系（`gemm.*`、`attn.*`、`tilelang`、`sampler`、`other`）> `memcpy` > 通信系（`comm.*`）の順とする。同じ階層の中では、規則表の順に従う。
  - これにより、通信の exclusive 時間は「計算に隠れずに見えている通信（exposed comm）」になる。
  - 恒等式 `Σ exclusive(c) + idle = step_time` が step ごとに厳密に成り立つ。
- **inclusive 時間:** カテゴリごとの `dur` の和（重なりを含む）。exclusive 時間と並べて出し、`overlap = Σ inclusive - busy_union` を、重なりの量として出す。
- 参考として、CPU 側の `deepinfer:bubble` 区間（`llm_engine.py:991`）の合計時間も、step 窓の単位で並べて出す。zero-overhead scheduling のもとでは、これは前の step の後処理の時間である。

device が複数ある trace では、device ごとに計算し、出力にも device を残す。summary の既定の表示は device 0 とする。

### FR-5 launch の遅延（v0.1 では参考値）

- correlation ID ごとに、`(その correlation の GPU イベントの最も早い ts) - (launch の ts)` を求め、p50 と p99 を出す。CUDA Graph の再生では、1 つの `cudaGraphLaunch` に対して、最初のカーネルまでの遅延を 1 件として数える。
- launch の名前（`cudaLaunchKernel`、`cudaGraphLaunch` など）ごとに分けて出す。
- launch と対応が取れた GPU イベントの割合を併記する。

### FR-6 `summary` コマンド

```
deepinfer-trace-analyzer summary TRACE [--prefill-steps N] [--prefill-threshold R] [--skip-steps K]
                                       [--rules FILE] [--no-default-rules] [--show-unclassified [K]]
                                       [--device D] [--format table|markdown|json]
```

phase（prefill / decode / unknown）ごとに、次の値を出す。

- step 数
- `step_time` の平均と最大
- カテゴリごとの exclusive 時間（1 step あたりの平均）と構成比
- カテゴリごとの inclusive 時間（1 step あたりの平均）
- `idle` と `overlap` の 1 step あたりの平均と構成比
- `other` の構成比（exclusive 時間ベース）
- `unmatched_ratio`
- `deepinfer:bubble` の 1 step あたりの平均

### FR-7 `diff` コマンド

```
deepinfer-trace-analyzer diff BEFORE AFTER [--prefill-steps N] [--prefill-threshold R] [--skip-steps K]
                                           [--rules FILE] [--no-default-rules] [--device D] [--format ...]
```

- 2 つの trace に同じ分類規則と同じオプションを適用する。
- phase ごと・カテゴリごとに、BEFORE と AFTER の 1 step あたりの exclusive 時間、差分（μs/step）、構成比の差分（pt）を出す。差分の絶対値が大きい順に並べる。
- §FR-4 の恒等式により、`Σ Δexclusive(c) + Δidle = Δstep_time` が成り立つ。表の最後に、この分解の行（step 時間の差分と、その内訳の上位）を出す。
- inclusive 時間の差分と `overlap` の差分も並べる。これで、「通信そのものが減った」のか「通信が計算に隠れるようになった」のかを見分けられる。
- 次の場合は警告を出す。どちらも 1 step あたりの値に正規化してから比べる。
  - 2 つの trace で device 数や phase ごとの step 数が違う。
  - どちらかの `unmatched_ratio` が 5% を超える。
  - どちらかの `other` の構成比が 10% を超える（分類規則が実態に合っていない）。

### FR-8 JSON 出力

`--format json` のときは、上記のすべての数値を機械で読める形で出す。

- トップレベルに `schema_version`（この版では `1`）、入力ファイル名、適用したオプションと規則を含める。
- 時間は μs の数値（Decimal を文字列にしたもの）で出し、精度を落とさない。
- step ごとの値（窓、phase、カテゴリ別 exclusive / inclusive 時間、idle）も含める。summary の表は、これを集計したものにすぎない。

CI でしきい値をチェックする用途は v0.2 以降で検討する。

### FR-9 エラーと終了コード

| 終了コード | 条件 |
|---|---|
| 0 | 成功（警告があっても 0） |
| 1 | 入力が読めない（ファイルがない、JSON として壊れている、トップレベルの形式が不明） |
| 2 | 入力は読めたが解析できない（`deepinfer:schedule` がない、`--prefill-steps` が step 数より大きい、規則ファイルが不正） |

警告とエラーは標準エラーに出し、標準出力は結果だけにする（JSON をパイプで渡せるようにするため）。

## 5. 非機能要件

- 依存は、Python 3.12 以上の標準ライブラリと ijson のみ。DeepInfer、torch、GPU は不要。
- profile server の最大の trace（約 4.9 MB/step × 50 step ≈ 250 MB、`profile_server.py:45`）を、60 秒以内、メモリ 2 GB 以内で処理する（目標値。ijson の C バックエンドを前提とする）。
- 出力する時間の単位は μs。trace の `ts` の精度（小数）は保ったまま、Decimal で計算する。

## 6. 受け入れ基準

1. 手で作った合成 trace で、次の値が手計算と一致する。これは GPU なしで CI として回す。
   - schedule が 2 回以上あり、2 ストリームで重なるカーネル（計算と通信）を含む。
   - correlation の対応があるものとないものを含む。
   - step k に launch されたカーネルが、時刻としては step k+1 の窓で実行される（zero-overhead scheduling の模擬）。
   - 確かめる値: step の分割、GPU イベントの step への帰属、`busy_union`、`idle`、カテゴリ別の exclusive / inclusive 時間、`overlap`。
   - すべての step で `Σ exclusive + idle = step_time` が成り立つ。
2. 実機で取った benchmark の trace（decode、EP8）で、`other` の構成比が 10% 以下になるまで既定の規則を改訂する。
3. 同じ trace 同士で `diff` を取ると、すべての差分が 0 になる。
4. `examples/_trace.py` の `_step_starts` と同じ trace から、同じ step の開始点の列が得られる。
5. トップレベルが配列で、`ts`、`dur`、correlation ID が文字列の trace（torch_npu の形式）を、壊れずに読める。

## 7. 設計判断

| 判断 | 採らなかった案 | 理由 |
|---|---|---|
| GPU イベントは launch の時刻で step に帰属させる | GPU イベント自身の時刻で step 窓に振り分ける（0.1 版） | zero-overhead scheduling では GPU の実行が次の step の窓にずれ込む（`worker.py:183`）。時刻で振り分けると、step ごとのカテゴリ構成が 1 step ずれて混ざる |
| exclusive 時間（優先度による分配）を主な指標にする | inclusive 時間（`dur` の単純な和）を主にし、食い違いを「説明できない残差」として出す（0.1 版） | inclusive 時間では、重なりがあると差分の和が step 時間の差分と一致せず、残差を説明できない。exclusive 時間なら分解が厳密に成り立ち、通信は「隠れずに見えている時間」として読める |
| 規則ファイルは TOML | YAML | `tomllib` は標準ライブラリにあり、依存を増やさない |
| 境界をまたぐ step を除外しない | 境界をまたぐ区間を除外する（0.1 版） | `_rebase` で境界はぴったり接続されるので、またぐ step はない。除外の規則を入れると、prefill の最後の step（2 step 中の 1 つ）を誤って落とす恐れがあった |

## 8. 未決事項とリスク

- カーネル名は推測の段階で、実機の trace による補正が前提である。
- CUPTI が、CUDA Graph の中のカーネルに `cudaGraphLaunch` と同じ correlation ID を付けるかどうかは、torch と CUDA のバージョンに依存する。付かない場合、zero-overhead scheduling のもとで GPU イベントの帰属がずれる。実機の trace の `unmatched_ratio` で確かめる。
- profile server の trace は driver worker 1 枚分しかない（`profile_server.py:7-8`）。そのため、ランク間の偏りは本ツールでは見えない。
- profile server はプロファイラを停止した時点で trace を閉じる。最後の step のうち、停止時点でまだ GPU に積まれていたカーネルが trace に入るかどうかは未確認である。必要なら最後の step も `--skip-steps` と同様に外す選択肢を検討する。
- Ascend（torch_npu）の trace は、GPU イベントの `cat` やカーネル名が異なる可能性がある。v0.1 では、壊れずに読めること（受け入れ基準 5）だけを保証する。GPU イベントとして認識できない場合、すべての step が idle 100% になるので、その場合は警告を出す。
- 連結済みの trace で、境界をまたいだ後も correlation ID の対応が保たれているかは、`_trace.py` の実装上は保たれる設計になっている（`_trace.py:176` で ID をまとめてずらす）。ただし実機で確かめる必要がある。
