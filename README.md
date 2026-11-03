# deepinfer-trace-analyzer

[DeepInfer](https://github.com/deepseek-ai/DeepInfer) が出力する torch profiler の chrome trace を、step 単位・カーネルのカテゴリ単位で集計する CLI。設定を変える前と後の 2 つの trace の差分も出せる。

> **ステータス: 仕様策定段階（未実装）。** 機能仕様は [docs/SPEC.md](docs/SPEC.md) を参照。

## 何をするか

- `examples/benchmark.py`（`PROFILE=1`）や、profile server（`POST /profile`）が出力した trace を読む。
- エンジンが毎 step 記録する `deepinfer:schedule` の区間で step を区切る。
- GPU カーネルを、GEMM（DeepGEMM）/ Attention（FlashMLA）/ MoE 通信（DeepEP）/ NCCL / TileLang / Sampler / other に分類して集計する。
- step ごとの GPU 空き時間（bubble）を算出する。
- 2 つの trace の差分を、カテゴリ別の表で出す。

プロファイラが入ると step 時間は膨らむ（DeepInfer の README によると最大約 50%）。そのため本ツールは、**絶対値より構成比と差分**を見るためのものとして設計する。

## 使い方（予定）

```bash
deepinfer-trace-analyzer summary trace.json
deepinfer-trace-analyzer diff before.json after.json --format markdown
```

DeepInfer 本体にも GPU にも依存しない、純粋な Python の JSON 処理として実装する。

## ライセンス

未定。
