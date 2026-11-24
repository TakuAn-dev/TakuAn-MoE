[English README](./README.md)

# TakuAn-MoE

**Apple Silicon 上で巨大な MoE LLM を動かす**ための、依存関係ゼロの C++20 / Objective-C++ + Metal 4 (MSL) ローカル推論エンジンと、OpenAI 互換のバックエンドサーバーです。統合メモリを大きく超える総パラメータ量を持つモデルを想定して、M4 Max (CPU 16 コア / GPU 40 コア、64 GB) 向けに構築されています。ビルドは素の POSIX `Makefile` と Apple 純正 `clang++`、ツール類は Python 3 の標準ライブラリ（HTTP / BPE）だけで構成され、Homebrew、CMake、外部サーバー依存は一切使いません。MSL カーネルはロード時に `newLibraryWithSource:` でコンパイルされ、`MTLBinaryArchive` によりキャッシュされます。

リポジトリ: https://github.com/TakuAn-dev/TakuAn-MoE

## 主な特徴

- **Apple Silicon (M4 Max, 64 GiB)** — M4 Max 64 GiB 実機で検証済み。
- **外部依存ゼロ** — C++20 / Objective-C++ / Metal と Python 3 標準ライブラリのみ。
- **Inkling-2-Mini-Q8_0**（25.93 GiB / 680 テンソル / 48 論理スロット + MoE 24x128 top-8）— フル常駐（`--arena-gb 0`）と、メモリ上限付きの NVMe ストリーミング（`--arena-gb` + `pread` / `mtlio`）の両方に対応。
- **最大 128K コンテキスト**（`--ctx 131072`）。
- **Milestone 3C / 3D 最適化**（実測値）:
  - Tensor-Core Grouped MoE（`B=128`）+ FP32 SiTU — バッチプリフィル **222〜246 tok/s**。
  - Split-K FlashDecoding — 長いコンテキストのデコードで **24.1 倍**高速化（カーネルのみ、`pos=20000`）。
  - SSD 永続プレフィックスキャッシュ（`.m64kv`）— **17.5k トークン**を約 **0.1 秒**で復元。
  - OpenAI 互換の HTTP/SSE サーバー — ツールコール、連続 `: keepalive` ハートビートに対応。Hermes Agent による検証済み（**45k+ トークン**、マルチターンでの連続ツール実行）。

## 検証済みベンチマーク（Apple M4 Max, 64 GB）

モデル: `Inkling-2-Mini-Q8_0`（`arch=inkling2_loop`、物理 24 層 / ループ 48 スロット、128 エキスパート、top-8）、パックサイズ **25.93 GiB**、680 テンソル。リファレンスオラクルは `llama.cpp`（Metal）。全データは [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) を参照。

### 常駐 Metal モード（Phase 2A-opt R4）

| 指標 | 結果 |
| :--- | :--- |
| デコードスループット（32 ステップ） | **45.33 tok/s** — `llama.cpp` Metal（**31.90 tok/s**）比 **+42.1%** |
| オラクルとのグリーディ argmax 一致 | **9/9** |
| スワップ増加量 | **0 MB** |

### メモリ上限付きストリーミング（`--arena-gb`）

| 構成 | 常駐サイズ | デコード | スワップ増加量 | 備考 |
| :--- | ---: | ---: | ---: | :--- |
| `--arena-gb 4 --io pread` | **4.29 GB** | 1.25 tok/s | **0 MB** | 稼働中の約 45 GiB サーバーと同居、argmax 9/9 |
| `--arena-gb 8 --io mtlio` | 8.58 GB | **5.81 tok/s** | **0 MB** | エキスパートスロット 795、ヒット率 76.4% |
| `--arena-gb 12 --io mtlio` | 12.88 GB | **26.48 tok/s** | **0 MB** | スロット 1,302、ヒット率 85.9%、SSD 34.59 GB/s、argmax 9/9 |

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
