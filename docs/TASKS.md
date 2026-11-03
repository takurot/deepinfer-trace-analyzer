# 実装タスク計画: deepinfer-trace-analyzer v0.1

`docs/SPEC.md` v0.2（コミット `e49b264`）を、1 タスク 1 PR の単位に分けた計画。各タスクは、それ単独で検証できる状態で終わる。

## 依存関係

```
T0 SPEC 修正 ─────────────────────────────┐
T1 雛形 ─→ T2 読み込み ─→ T3 分類規則 ─┐   │
                     └─→ T4 step 分割 ─┴─→ T5 時間分解 ─→ T6 summary ─┬─→ T7 diff
                                                                    ├─→ T8 launch 遅延
                                                                    ├─→ T9 性能
                                                                    └─→ T10 実機検証
```

T4 と T5 は T0 の結論に従う。T3 と T4 は並行して進められる。

## SPEC との対応

| SPEC | タスク |
|---|---|
| FR-1、受け入れ基準 5 | T2 |
| FR-2、受け入れ基準 4 | T4 |
| FR-3 | T3 |
| FR-4、受け入れ基準 1 | T5 |
| FR-5 | T8 |
| FR-6、FR-8、FR-9 | T6（FR-9 の終了コードの仕組みは T1、各エラー条件は T2〜T4） |
| FR-7、受け入れ基準 3 | T7 |
| §5 非機能要件 | T1（依存）、T9（性能） |
| 受け入れ基準 2、§8 の実機で確かめる項目 | T10 |

## タスク

### T0. SPEC の FR-4 と FR-2 の矛盾を解消し、v0.2 を main に入れる

- 問題: FR-2 は GPU イベントを launch の時刻で step に帰属させる。FR-4 は `busy_union` を「帰属したイベントのうち step 窓と重なる部分」とする。zero-overhead scheduling で step k のカーネルが窓 k+1 で動くと、そのカーネルは窓 k では窓の外なので切り捨てられ、窓 k+1 では帰属が違うので数えられない。どちらの step にも入らず、両方の `idle` が過大になる。恒等式 `Σexclusive + idle = step_time` は形の上では成り立つが、値は実態と合わない。
- 推奨: 時間の分解（`busy_union`、`idle`、exclusive）は時刻で窓に切り分ける（壁時計の時間として正しい）。launch による帰属は、`unmatched_ratio` と、帰属と実行位置が窓をまたいだカーネルの割合（`spill_ratio`）の診断にだけ使う。benchmark の trace では、`_trim_events` が窓の外で launch された GPU イベントをすでに落としているので、時刻による切り分けで境界の混入は起きない。
- これは v0.2 で入れた判断の一部を戻すことになる。§7 の設計判断の表も直す。
- 完了条件: SPEC の FR-2、FR-4、§7、受け入れ基準 1 が 1 つの定義で矛盾なく読める。`docs/spec-v0.2` が main に入っている。

### T1. パッケージの雛形、CLI の骨格、CI

- `pyproject.toml`（Python 3.12 以上、実行時依存は `ijson` のみ、console script `deepinfer-trace-analyzer`）。
- `src/deepinfer_trace_analyzer/` と、`summary` / `diff` のサブコマンドを持つ argparse の骨格。終了コードを表す例外（入力エラー = 1、解析エラー = 2）と、それを標準エラーに出して終了する入口を作る（FR-9）。
- pytest、ruff、GitHub Actions（3.12 と 3.13）。
- 完了条件: `pip install -e .` のあと `deepinfer-trace-analyzer --help` が両方のサブコマンドを表示する。CI が緑。

### T2. trace の読み込みと正規化（FR-1）、合成 trace ビルダー

- コアの型: `Event`（name、cat、pid、tid、ts と dur は Decimal、correlation は int か None）、`Trace`（CPU イベント、GPU イベント、schedule の ts の列、device の一覧）。
- ijson によるストリーミング読み込み。`.json` と `.json.gz`、トップレベルが dict と配列の両方、文字列の ts / dur / correlation、プロファイラ自身の pid の除外、`ph == "X"` だけを残す、`gpu_user_annotation` を GPU イベントから外す。
- テスト用の合成 trace ビルダー（`tests/` 配下）。以降のすべてのタスクがこれで入力を作る。
- エラー: ファイルがない・JSON が壊れている・トップレベルが不明なら 1、`deepinfer:schedule` がなければ 2。
- 完了条件: 受け入れ基準 5（torch_npu 形式）を含むテストが通る。

### T3. カーネルの分類規則（FR-3）

- 既定の規則表、`memcpy` を `cat` で先に決める処理、大文字小文字を区別しない `re.search`。
- `--rules FILE`（TOML、`tomllib`）、`--no-default-rules`。不正な規則ファイル（必須キーがない、正規表現が壊れている）は終了コード 2。
- カテゴリの優先度の階層（計算 > memcpy > 通信）もここで規則と一緒に持つ。T5 が使う。
- 完了条件: 既定の規則で代表的なカーネル名が期待するカテゴリに入ることと、ユーザーの規則が既定より先に当たることをテストで示す。

### T4. step の分割と phase の判定（FR-2）

- step 窓、segment の最後の step の終端、`--prefill-steps` による境界。
- T0 の結論に従った GPU イベントの扱いと、`unmatched_ratio`、どの step にも入らないイベントの件数。
- phase の判定（`--prefill-steps`、`launch_forward_N{reqs}_B{tokens}` と `--prefill-threshold`、`unknown`）。`--skip-steps`。
- `--prefill-steps` が step 数より大きければ終了コード 2。
- 完了条件: 合成 trace で step 窓と phase がテストで一致する。受け入れ基準 4 は、DeepInfer の `examples/_trace.py` の `_step_starts` と比べる開発用スクリプト（DeepInfer のパスを引数で受け取る、CI では動かさない）で確かめる。

### T5. 時間の分解（FR-4）

- device ごと・step ごとの `busy_union`、`idle`、優先度による exclusive 時間（区間の端点を並べて掃引する）、inclusive 時間、`overlap`、CPU 側の `deepinfer:bubble` の合計。
- 完了条件: 受け入れ基準 1 の合成 trace（重なる計算と通信、correlation のあるものとないもの、窓をまたぐカーネル）で、すべての値が手計算と一致する。すべての step で `Σexclusive + idle = step_time` が成り立つことをテストで示す。

### T6. `summary` コマンドと出力形式（FR-6、FR-8、FR-9）

- phase ごとの集計、`--device`、`--show-unclassified [K]`。
- `table`、`markdown`、`json` の出力。JSON は `schema_version: 1`、入力ファイル名、適用したオプションと規則、step ごとの値を含み、時間は Decimal の文字列。
- 結果は標準出力、警告とエラーは標準エラー。GPU イベントが 1 つもないとき（idle 100%）の警告。
- 完了条件: 合成 trace に対する 3 形式の出力を golden ファイルで比べるテストが通る。JSON が `json.loads` で読め、step ごとの値を足すと summary の値になる。

### T7. `diff` コマンド（FR-7）

- 2 つの trace に同じ規則とオプションを適用し、phase × カテゴリの exclusive 時間の差分（μs/step、pt）を絶対値の降順に出す。最後に step 時間の差分の分解の行。inclusive 時間と `overlap` の差分。
- 警告: device 数や phase ごとの step 数の違い、`unmatched_ratio` > 5%、`other` > 10%。
- README の「使い方（予定）」を実際の使い方に更新する。
- 完了条件: 受け入れ基準 3（同じ trace 同士で差分がすべて 0）と、差分の分解 `ΣΔexclusive + Δidle = Δstep_time` をテストで示す。

### T8. launch の遅延（FR-5、参考値）

- correlation ごとに、GPU イベントの最も早い ts と launch の ts の差を取り、launch の名前ごとに p50 と p99 を出す。対応が取れた割合を併記する。summary の出力に足す。
- 完了条件: CUDA Graph の 1 対多の対応を含む合成 trace で、p50 と p99 と割合がテストで一致する。

### T9. 性能の確認（§5 非機能要件）

- 約 250 MB の合成 trace を生成するスクリプトを作り、`summary` の実行時間とピークメモリを測る。
- 完了条件: 60 秒以内、2 GB 以内。測定した環境、ijson のバックエンド、値を Issue に記録する。満たさないときは、どこで時間を使っているかのプロファイルを付けて、改善のタスクを別に起こす。

### T10. 実機の trace による検証と、既定の規則の改訂（受け入れ基準 2、§8）

- 必要なもの: GPU 実機で取った benchmark の trace（decode、EP8、zero-overhead scheduling の有効と無効の両方）と、profile server の trace。**これはオペレーターから提供してもらう必要がある。**
- `--show-unclassified` で `other` を見ながら既定の規則を改訂し、`other` を 10% 以下にする。
- §8 の未確認事項を確かめて SPEC に結果を書く: CUDA Graph のカーネルの correlation、`unmatched_ratio` の実際の値、profile server の最後の step、連結後の correlation の対応。
- 完了条件: 改訂した規則で `other` ≤ 10%。§8 の各項目に、確認した結果が書かれている。

## レビュー

各タスクを次の観点で見た。1 PR で完結し単独で検証できるか。完了条件が観測できるか。依存が正しいか。SPEC の要件の漏れと重複がないか。

| タスク | 判定 | レビューで直したこと |
|---|---|---|
| T0 | 追加 | タスクに分ける途中で、FR-2 の launch 帰属と FR-4 の窓による切り取りが矛盾することに気付いた。T4 と T5 の前に定義を決めないと実装できないので、先頭に SPEC 修正のタスクを足した。SPEC v0.2 はまだ `docs/spec-v0.2` ブランチにしかないので、main に入れる作業もここに含めた |
| T1 | OK | 終了コードの仕組みを T6 から前倒しした。T2〜T4 がそれぞれのエラー条件を実装するときに必要になるため |
| T2 | OK | 合成 trace ビルダーを独立したタスクにせず、最初に使う T2 に入れた。単独では検証できる振る舞いがないため |
| T3 | OK | 優先度の階層を T5 ではなく規則と一緒に持たせた。カテゴリの定義が 1 か所にまとまる |
| T4 | 修正 | 受け入れ基準 4 は DeepInfer のコードが要るので、CI の外の開発用スクリプトにした。DeepInfer を依存に入れないという §5 を守るため |
| T5 | OK | 中身が T0 の結論に依存することを明記した |
| T6 | OK | idle 100% の警告（§8 の Ascend の項目）の置き場所がなかったので、ここに入れた |
| T7 | OK | README の更新は、使い方が出そろうここで行う |
| T8 | OK | 参考値なので、T6 のあとに回した。他のタスクを止めない |
| T9 | OK | 目標を満たさないときの扱い（別タスクにする）を完了条件に入れた |
| T10 | 注意 | 実機の trace がないと着手できない。オペレーター待ちであることを Issue に明記する |
