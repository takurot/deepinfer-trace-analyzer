# deepinfer-trace-analyzer

[DeepInfer](https://github.com/deepseek-ai/DeepInfer) が出力する torch profiler の chrome trace を、step 単位・カーネルのカテゴリ単位で集計する CLI。設定を変える前と後の 2 つの trace の差分も出せる。

> **ステータス: v0.1 開発中。** `summary` と `diff` は動作します。実機の trace による分類規則の確認は未完了です（[docs/SPEC.md](docs/SPEC.md) の §8、Issue #11）。機能仕様は [docs/SPEC.md](docs/SPEC.md)、実機の trace で検証する手順は [docs/REAL_TRACE_RUNBOOK.md](docs/REAL_TRACE_RUNBOOK.md)、性能は [docs/PERFORMANCE.md](docs/PERFORMANCE.md) を参照。

## 何をするか

- `examples/benchmark.py`（`PROFILE=1`）や、profile server（`POST /profile`）が出力した trace を読む。
- エンジンが毎 step 記録する `deepinfer:schedule` の区間で step を区切る。
- GPU カーネルを、GEMM（DeepGEMM）/ Attention（FlashMLA）/ MoE 通信（DeepEP）/ NCCL / TileLang / Sampler / other に分類して集計する。
- step ごとの GPU 空き時間（bubble）を算出する。
- 2 つの trace の差分を、カテゴリ別の表で出す。

プロファイラが入ると step 時間は膨らむ（DeepInfer の README によると最大約 50%）。そのため本ツールは、**絶対値より構成比と差分**を見るためのものとして設計する。

## 使い方

```bash
uv sync
uv run deepinfer-trace-analyzer summary trace.json --prefill-steps 2
uv run deepinfer-trace-analyzer diff before.json after.json --prefill-steps 2 --format markdown
```

- `benchmark.py` の trace は prefill 2 step と decode 3 step をつないだものなので、`--prefill-steps 2` を付けます。profile server の trace は、`--skip-steps 1` でプロファイラ起動直後の step を外すのを勧めます。
- 分類されなかったカーネルは `--show-unclassified` で見られます。規則は TOML で足せます（`--rules rules.toml`、書式は SPEC の FR-3）。
- 結果は標準出力、警告は標準エラーに出ます。`--format json` は機械で読める形です。
- 終了コードは、成功が 0、入力が読めないのが 1、読めたが解析できないのが 2 です。

DeepInfer 本体にも GPU にも依存しない、純粋な Python の JSON 処理として実装する。

## ライセンス

[MIT](LICENSE)
