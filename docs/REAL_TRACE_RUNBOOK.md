# 実機の trace で検証する手順

Issue #11 の作業手順です。GPU 実機（NVIDIA SM100、EP8）で取った trace を、このツールに通して、分類規則を直し、SPEC §8 の未確認事項に答えを出します。DeepInfer の `5537415` のコードを読んで書いた手順で、実機ではまだ動かしていません。

## 1. 取る trace

| 名前 | 目的 | 取り方 |
|---|---|---|
| A. benchmark、zero-overhead scheduling 有効 | 受け入れ基準 2（`other` ≤ 10%）、`spill`、CUDA Graph の correlation | `DEEPINFER_ZERO_OVERHEAD_SCHEDULING=1 PROFILE=1` で `examples/benchmark.py` を EP8 で実行 |
| B. benchmark、zero-overhead scheduling 無効 | A との比較（`spill` が 0 に近いはず） | A と同じ設定で `DEEPINFER_ZERO_OVERHEAD_SCHEDULING=0` を**明示的に**付ける（未設定のときの既定は engine の種類で決まる: `deepinfer/engine/arg_utils.py:50-55`） |
| C. profile server | 最後の step の扱い、`--skip-steps` の確認 | 下記 |

**A と B（benchmark）**

- `PROFILE` は既定で `1` です（`benchmark.py:68`）。prefill 2 step と decode 3 step をつないだ trace が、`--tracing_path`（既定 `tracing/`）に `<名前>.json` で出ます。書くのは `dp_rank == 0` だけです。
- 起動方法は、お使いの環境で DeepInfer のベンチマークを動かしているやり方に合わせてください。`examples/sweep/run_sweep.sh:72` が `python -u examples/benchmark.py` を直接呼ぶ例です（スイープは `PROFILE=0` なので、取るときは外す）。EP の大きさは `WORLD_SIZE` から決まります。
- decode の trace が欲しいので、`--decode_num_requests` と `--context_length` は、見たい設定にしてください。

**C（profile server）**

- サーバー側に `DEEPINFER_PROFILE_HTTP_ENABLE=1` を付けて起動します。ポートは `DEEPINFER_PROFILE_HTTP_PORT_BASE`（既定 6070）に driver の GPU id を足した値です。
- 負荷をかけている最中に、次を送ります。

```bash
curl -XPOST http://<ip>:<port>/profile -H 'Content-Type: application/json' \
     -d '{"num_steps": 20, "output_dir": "/tmp/prof"}'
```

- 採れるのは driver worker 1 枚分で、`num_steps` は 1〜50（既定 20）です。ファイル名は `profile-<ip>-gpu<id>-rank<n>-<時刻>.json` です。

## 2. 取るときに控えること

結果の読み方が、これで変わります。trace と一緒に書いておいてください。

- DeepInfer のコミット、torch と CUDA のバージョン、GPU の型番と枚数。
- 実行したコマンド全部と環境変数（特に `DEEPINFER_ZERO_OVERHEAD_SCHEDULING`、`DEEPINFER_USE_NEXTN_DECODE`）。

## 3. 持ち帰ってからの手順

```bash
uv sync

# 1. step の開始点が DeepInfer の _trace.py と同じか（受け入れ基準 4）
uv run python scripts/compare_step_starts.py /path/to/DeepInfer A.json

# 2. 合否の一覧と、分類されなかったカーネル
uv run python scripts/real_trace_report.py A.json --prefill-steps 2

# 3. 全体の表
uv run deepinfer-trace-analyzer summary A.json --prefill-steps 2 --show-unclassified 30
```

`real_trace_report.py` は、`other` が 10% を超えると `FAIL`（終了コード 1）、それ以外の気になる点を `WARN` と `INFO` で出します。末尾に、分類されなかったカーネルが長い順に並びます。

## 4. 分類規則を直す

1. 分類されなかったカーネルの名前を見て、`rules.toml` に規則を足して試します（書式は SPEC の FR-3）。通信は `comm.` で始まるカテゴリ名にします。そうしないと計算として扱われ、通信が見えなくなります。

```toml
[[rule]]
category = "attn.flashmla"
pattern = "実際のカーネル名の一部"
```

```bash
uv run python scripts/real_trace_report.py A.json --prefill-steps 2 --rules rules.toml
```

2. `other` が 10% 以下になったら、`src/deepinfer_trace_analyzer/rules.py` の `DEFAULT_RULES` に同じ規則を移します。そのとき、実際のカーネル名を例に、`tests/test_rules.py` にテストを足します。
3. SPEC の FR-3 の表を、同じ内容に直します。

## 5. SPEC §8 の項目を確かめる

| §8 の項目 | 見る場所 | どうなっていれば |
|---|---|---|
| CUDA Graph のカーネルに `cudaGraphLaunch` と同じ correlation が付くか | `real_trace_report` の `cudaGraphLaunch seen`、`unmatched share, decode` | 起動が見えて、`unmatched` が小さい。付かないなら `unmatched` が大きい。結果を SPEC に書き、`spill` の見方を直す |
| zero-overhead scheduling で `spill` が出るか | `spill share, decode`（A と B を比べる） | A で 0 より大きく、B で 0 に近い |
| 連結した trace で、境界の後も correlation が保たれるか | `unmatched share, prefill` と `unmatched share, decode` を別々に | どちらも小さい |
| profile server の最後の step | C の `last step length` | 他の step の 0.8〜1.2 倍に収まる。外れるなら、最後の step を `--skip-steps` と同じように外す選択肢を SPEC で決める |
| カーネル名 | 末尾の分類されなかったカーネルの一覧 | 規則を足して、`other` ≤ 10% |

## 6. 終わりの条件（Issue #11）

- A（decode）で、`real_trace_report.py` に `FAIL` がない。
- 改訂した規則が `DEFAULT_RULES`、テスト、SPEC の FR-3 に入っている。
- 上の表の各項目に、確かめた結果を SPEC の §8 に書いた。
- A、B、C の trace と、2 で控えた環境の情報を、Issue #11 に添える（trace が大きければ置き場所のリンクで構いません）。
- 実機の trace でも `docs/PERFORMANCE.md` の手順で時間とメモリを測り、数字を追記する。
