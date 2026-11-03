# 性能の確認

SPEC §5 の目標は「約 250 MB の trace を 60 秒以内、メモリ 2 GB 以内」です。

## 手順

```bash
uv run python scripts/make_synthetic_trace.py /tmp/big.json --steps 50 --blocks-per-step 970
uv run python scripts/bench_summary.py /tmp/big.json --runs 5
```

- `make_synthetic_trace.py` は、torch の形式に似せた trace をストリームで書きます。1 ブロック（30 μs）を繰り返す形なので、各 step の正解が閉じた式で分かります（ファイル先頭の docstring）。1 step あたり約 5 MB で、DeepInfer の実測（約 4.9 MB/step）に合わせています。
- `bench_summary.py` は、`summary` を新しいプロセスで N 回動かし、時間とピークメモリの中央値と範囲を出します。出力が正解と違えば失敗します。時間だけを測って、仕事をしていない実行を数えることがありません。
- 小さい規模での生成器と解析器の一致は `tests/test_synthetic_trace.py` が確かめます。

## 結果

| 項目 | 値 |
|---|---|
| trace | 252 MB、873,151 イベント（GPU イベント 291,000）、50 step |
| 時間 | 中央値 6.9 秒（範囲 6.5〜7.0 秒、5 回） |
| ピークメモリ | 中央値 318 MB（範囲 317〜320 MB） |
| 出力の検証 | 5 回とも、50 step の `step_time`、exclusive、`idle` が正解と一致 |
| 目標との差 | 時間は 60 秒の約 1/9、メモリは 2 GB の約 1/6 |

測定環境は Apple Silicon の macOS（8 コア）、Python 3.12.14、ijson 3.6.0（C バックエンド `yajl2_c`）です。測定中の負荷平均は 4〜5 でした。

## 何が時間を決めているか

プロファイル（報告した実行とは別）では、`load_trace` が CPU 時間の約 75% を占めます。ijson のパースと、イベントごとの辞書の作成です。sweep と集計は約 25% です。読み込みは入力の大きさに比例するので、目標を満たす限り、時間の表現（`Decimal`）を整数に替える必要はありません。

## 限界

- 合成 trace です。実機の trace は、カーネル名が長く、`args` が大きく、ストリームが多いことがあります。1 step あたりのバイト数は合わせましたが、イベントの数や重なり方は実機と違います。Issue #11 で実機の trace が手に入ったら、同じ手順で測り直してください。
- 1 device の trace だけを測りました。`summary` は 1 つの device を分析するので、他の device のイベントは読み込みの分だけが増えます。
- ijson の C バックエンドが使えない環境では、純 Python に落ちて遅くなります（その場合は警告が出ます。倍率は未測定）。
