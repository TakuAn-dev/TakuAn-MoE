[English README](./README.md)

# TakuAn-MoE

**Apple Silicon 上で巨大な MoE LLM を動かす**ための、依存関係ゼロの C++20 / Objective-C++ + Metal 4 (MSL) ローカル推論エンジンと、OpenAI 互換のバックエンドサーバーです。統合メモリを大きく超える総パラメータ量を持つモデルを想定して、M4 Max (CPU 16 コア / GPU 40 コア、64 GB) 向けに構築されています。ビルドは素の POSIX `Makefile` と Apple 純正 `clang++`、ツール類は Python 3 の標準ライブラリ（HTTP / BPE）だけで構成され、Homebrew、CMake、外部サーバー依存は一切使いません。MSL カーネルはロード時に `newLibraryWithSource:` でコンパイルされ、`MTLBinaryArchive` によりキャッシュされます。

リポジトリ: https://github.com/TakuAn-dev/TakuAn-MoE

## 主な特徴

- **Apple Silicon (M4 Max, 64 GiB)** — M4 Max 64 GiB 実機で検証済み。
- **外部依存ゼロ** — C++20 / Objective-C++ / Metal と Python 3 標準ライブラリのみ。
- **Inkling-2-Mini-Q8_0**（25.93 GiB / 680 テンソル / 48 論理スロット + MoE 24x128 top-8）— フル常駐（`--arena-gb 0`）と、メモリ上限付きの NVMe ストリーミング（`--arena-gb` + `pread` / `mtlio`）の両方に対応。
- **Inkling-2-Mini-Q4_K_S**（`15.71 GiB` の `.m64pack`）— `Q4_K` / `Q5_0` / `Q5_1` のパック変換・逆量子化・Metal MoE カーネル（`k_gemv_gateup_batch8_q4k_f32in`、`k_gemv_down_reduce8_q5_{0,1}_f32in`、`k_moe_grouped_f32in_sg_{q4k,q5_0,q5_1}`）に対応。`F32` の密行列サポート（`attn_sink_proj`、`attn_{k,v}_conv_proj`）と層ごとの `ExpertArena` ストライドにも対応。
- **最大 128K コンテキスト**（`--ctx 131072`）。
- **Milestone 3C / 3D 最適化**（実測値）:
  - Tensor-Core Grouped MoE（`B=128`）+ FP32 SiTU — バッチプリフィル **222〜246 tok/s**。
  - Split-K FlashDecoding — 長いコンテキストのデコードで **24.1 倍**高速化（カーネルのみ、`pos=20000`）。
  - SSD 永続プレフィックスキャッシュ（`.m64kv`）— **17.5k トークン**を約 **0.1 秒**で復元。
  - OpenAI 互換の HTTP/SSE サーバー — ツールコール、連続 `: keepalive` ハートビートに対応。Hermes Agent による検証済み（**45k+ トークン**、マルチターンでの連続ツール実行）。
- **Milestone 4A / 4B / 4C 最適化**（実測値）:
  - タイル化 FlashAttention プリフィル（`k_attn_f16_flash_prefill`、`engine/kernels/m64_prefill.metal`）を `B >= 8` でフルアテンション層（`is_swa=0`）と SWA 層（`is_swa=1`）の両方に有効化 — `p=20000`（`B=128`）でアテンションのプリフィルが最大 **7.63 倍**高速化（`5.93 ms` vs `45.25 ms/iter`）、`B=32` で **3.44 倍**、SWA 層の `B=128` で **4.93 倍**。
  - フルパック時の `--prefix-cache` 自動デフォルトを **2** から **4** チェックポイントに引き上げ（`--prefix-cache` で上書き可能）。
  - サーバーのストリーミング／切断安定性 — thinking 中の `delta.reasoning_content` を SSE でリアルタイム配信し、長時間の推論でもクライアントのアイドル／ストールタイムアウトが発生しない。クライアント切断・停止時には `SIGUSR1` で進行中の `GEN` をクリーンにキャンセル（`DONE ... cancelled`）し、`stdout` パイプをドレインしてワーカーのパイプデッドロックを防止。
- **Milestone 5 — KV キャッシュ dtype**: `--kv-type f16|q8_0` で Metal の KV キャッシュ形式を選択（`q8_0` は長いコンテキストで KV メモリを約半分に。Metal 限定、デフォルト `f16`）。
- **Milestone 6 — マルチモデル アーキテクチャディスパッチ**（Inkling-2-Mini に加え `gemma-4-26B-A4B`、`arch = gemma4` に対応）: `tools/m64pack.py` のパック変換、`m64run` の CPU オラクル + Metal FP16/Q8_0 KV カーネル、`takuan_server.py` の Gemma 4 サポート — SentencePiece-BPE トークナイザー（`tokenizer.ggml.model == "gemma4"`、`▁` 単語マーカー + `<0xXX>` バイトフォールバック）と、`<|turn>role ... <turn|>` チャットテンプレート、`<|channel>thought\n…<channel|>` の思考チャネルを `delta.reasoning_content` としてリアルタイム配信。
- **Milestone 7 — デコード・サービング最適化**（実測値）:
  - Inkling の `Q4_K` / `Q5_0` / `Q5_1` デコード用 MoE カーネル: アクティベーションレジスタ共有を 2 行（`k_gemv_gateup_batch8_q4k_f32in_m2`）/ 4 行（`k_gemv_down_reduce8_q5_{0,1}_f32in_m4`）へ拡張 — 実 Inkling シェイプで MoE gemv が **1.12〜1.22 倍**高速化し、m1 カーネルとビット単位で完全一致。
  - Prompt-lookup 投機的デコーディング（`--speculative N` / `--draft-tokens N`、0..8、デフォルト `0` = 無効）: n-gram 下書き検索 + バッチ検証 + スナップショット/リストアによるロールバック — 下書きモデルも追加メモリも不要で、検証時の argmax は通常グリーディとビット単位一致（Metal・Inkling・`temp==0`）。分岐トークンの再検証排除、アンカーの早期ストリーミング、適応的クールダウンも収録。
  - なめらかな SSE ストリーミング: サンプラーが top-p 候補を事前フィルタ（`prob >= 1e-6 * prob[max]`、平坦な分布では全語彙フォールバックで分布形状を厳密維持）し、語彙 201k 件の毎トークン全ソートを廃止 — サンプルコスト **2.09 → 0.825 ms/tok**（`nv=201,024`）。さらに SSE 接続へ `TCP_NODELAY` を設定し、トークン毎チャンクの Nagle 由来のバースト性を解消。
  - RAM プレフィックススタッシュ: 割り込む短いプロンプトで稼働中の会話 KV が破棄される場面では、最長の生きているチェックポイントを RAM に一時退避し、会話再開時に復元して長いコンテキストの再プリフィルを回避（自動動作、フラグ不要）。

## 検証済みベンチマーク（Apple M4 Max, 64 GB）

モデル: `Inkling-2-Mini-Q8_0`（`arch=inkling2_loop`、物理 24 層 / ループ 48 スロット、128 エキスパート、top-8）、パックサイズ **25.93 GiB**、680 テンソル。リファレンスオラクルは `llama.cpp`（Metal）。全データは [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) を参照。

### 常駐 Metal モード（Phase 2A-opt R4）

| 指標 | 結果 |
| :--- | :--- |
| デコードスループット（32 ステップ） | **45.33 tok/s** — `llama.cpp` Metal（**31.90 tok/s**）比 **+42.1%** |
| オラクルとのグリーディ argmax 一致 | **9/9** |
| スワップ増加量 | **0 MB** |

### Q4_K_S 量子化（`Inkling-2-Mini-Q4_K_S`、パックサイズ 15.71 GiB）

| 指標 | 結果 |
| :--- | :--- |
| デコードスループット（フル常駐 Metal） | **50.17 tok/s** — `Q8_0`（**43.80 tok/s**）比 **+14.5%** |
| `--arena-gb 4`（`ctx=256`） | 常駐 **約 4.29 GB** で **23.29 tok/s** |

### メモリ上限付きストリーミング（`--arena-gb`）

| 構成 | 常駐サイズ | デコード | スワップ増加量 | 備考 |
| :--- | ---: | ---: | ---: | :--- |
| `--arena-gb 4 --io pread` | **4.29 GB** | 1.25 tok/s | **0 MB** | 稼働中の約 45 GiB サーバーと同居、argmax 9/9 |
| `--arena-gb 8 --io mtlio` | 8.58 GB | **5.81 tok/s** | **0 MB** | エキスパートスロット 795、ヒット率 76.4% |
| `--arena-gb 12 --io mtlio` | 12.88 GB | **26.48 tok/s** | **0 MB** | スロット 1,302、ヒット率 85.9%、SSD 34.59 GB/s、argmax 9/9 |

## クイックスタート

1. **エンジンのビルド**（Apple 純正 `clang++` + Metal のみ、外部依存ゼロ）:
   ```sh
   make
   ```
2. **対応 GGUF モデルの取得**（単一 `.gguf` ファイル）:
   - `Inkling-2-Mini`（`Q8_0` または `Q4_K_S`、例：`Inkling-2-Mini-Q8_0.gguf`）
   - `gemma-4-26B-A4B`（`Q8_0` / `Q4_K_M` / `Q4_K_S`、例：`gemma-4-26b-a4b.gguf`）
3. **GGUF から `.m64pack` への変換**（手順 4 で `--pack` を省略して自動変換させることもできます）:
   ```sh
   python3 tools/m64pack.py pack <MODEL_DIR>/Inkling-2-Mini-Q8_0.gguf \
     <MODEL_DIR>/inkling-2-mini-q8_0.m64pack
   ```
   `m64pack.py` は追加パッケージなし（Python 3 標準ライブラリのみ）で動作します（大容量モデルを高速変換したい場合のみ任意で `pip install xxhash` が使えます）。
4. **サーバー起動**:
   ```sh
   # --pack を省略すると、初回起動時に <MODEL_DIR>/Inkling-2-Mini-Q8_0.m64pack を自動生成します:
   python3 tools/takuan_server.py --gguf <MODEL_DIR>/Inkling-2-Mini-Q8_0.gguf \
     --port 8080 --ctx 131072
   ```

## ビルド

```sh
make
```

## セルフテスト

```sh
./scripts/check.sh
```

`make test`（`scripts/check.sh` から呼び出されます）は、`tools/make_test_fixture.py`（Python 3 標準ライブラリのみ）を使って `build/fixture/` に極小の合成パックとゴールデンファイルを生成し、`test_dequant`、`test_metal_parity`、`test_p2a_kernels`、`test_prefill_kernels`、`test_arena`、`test_prefix_disk` およびサーバーテストスイートを実行します。

## サーバー

```sh
python3 tools/takuan_server.py --pack <MODEL_DIR>/inkling-2-mini-q8_0.m64pack \
  --gguf <MODEL_DIR>/Inkling-2-Mini-Q8_0.gguf \
  --port 8080 --ctx 131072 --cache-dir <CACHE_DIR>
```

メモリ上限付きストリーミングサーバー:

```sh
python3 tools/takuan_server.py --pack <MODEL_DIR>/inkling-2-mini-q8_0.m64pack \
  --gguf <MODEL_DIR>/Inkling-2-Mini-Q8_0.gguf \
  --port 8081 --ctx 4096 --arena-gb 12 --io mtlio
```

任意の OpenAI SDK クライアントは次の設定で接続できます。

- **Base URL:** `http://127.0.0.1:8080/v1`
- **Model:** `takuan-inkling-2-mini`
- **API Key:** 空でない任意の文字列

Gemma 4 パック（`--pack <MODEL_DIR>/gemma-4-26b-a4b*.m64pack` に対応する Gemma 4 GGUF を `--gguf` で指定）も同じ方法で提供できます — チャットテンプレートとトークナイザーは GGUF のメタデータから自動選択されます。

## ディレクトリ構成

```
engine/src      C++20 コア + Objective-C++ Metal ホスト
engine/kernels  MSL カーネル（ロード時にコンパイル、MTLBinaryArchive でキャッシュ）
engine/tests    パリティ / ユニットテスト
tools/          m64pack.py, bpe_gguf.py, takuan_server.py, test_server.py,
                make_test_fixture.py
scripts/        check.sh, swap_check.sh, gpu_validation_suite.sh
docs/           DESIGN.md, BENCHMARKS.md
```

## ライセンス

MIT — [LICENSE](LICENSE) を参照。

## 参考文献・謝辞（References & Acknowledgments）

- **[llama.cpp](https://github.com/ggml-org/llama.cpp)** — 数値検証のオラクルとして多大な助けを借りました。
- **strata** / **[mlx-od-moe](https://github.com/kqb/mlx-od-moe)** / **[baseRT](https://github.com/basecompute/baseRT)** / **swiftlm** — オンデマンド MoE 推論と Apple Silicon 向け推論の先駆的な知見を提供してくれました。
- **[FlashDecoding](https://arxiv.org/abs/2307.09179)** — Split-K Attention の手法の基礎。
- **Hermes Agent** — 開発・テスト・公開作業への協力に感謝します。
