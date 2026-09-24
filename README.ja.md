# h3c-app

[English](README.md) | [日本語](README.ja.md)

[MiniMax-H3](https://huggingface.co/lightx2v/Minimax-h3-Turbo)
（テキスト／画像／動画から、音声付き動画を生成する拡散トランスフォーマー）を
Apple Siliconだけで動かすネイティブmacOSアプリです（CLIと、アプリ自身が
実行中に提供するローカルHTTP APIも同梱）。すべてMetal/MPSGraph上で
プロセス内実行され、Python・PyTorch・クラウド通信は一切不要です。さらに
このforkでは、FFmpegへの依存も取り除いています — メディアの入出力は
ネイティブのAVFoundation/ImageIO経由です。

これはSalvatore Sanfilippo氏（antirez）の
[h3.c](https://github.com/antirez/h3.c)のforkです。C/Objective-Cのエンジン
本体（`libh3.a`）は2つのフロントエンドで共通・無変更のまま使われており、
このforkではネイティブSwiftUIアプリ（独自のモデルマネージャー、外部依存
なしのHugging Faceダウンローダー、組み込みの自動化API付き）、ネイティブ
（ffmpeg不要）な参照メディアのデコード、搭載メモリに応じた参照動画の
解像度上限自動調整、ライブラリ関数として呼び出せるint8アテンション
キャッシュビルダー（CLIツールとしてだけでなく）、そしてアプリ配布用の
Developer ID署名・公証対応を追加しています。以前あったPythonベースの
ローカルWeb GUIは、ネイティブアプリ自身の組み込みAPI（下記
[API](#api)参照）に置き換える形で廃止しました — このリポジトリには
もうPythonへの依存はどこにもありません。

## 2つの使い方

| フロントエンド | 場所 | 向いている用途 |
|---|---|---|
| **ネイティブアプリ**（`h3c-app.app`） | `native/H3Spike/` | エンドユーザー向け。フォーム入力（内容・画面の形・長さ・参照）、進捗＋残り時間表示、プレビュー付きで、フラグを覚える必要がない。実行中は`127.0.0.1`上の[HTTP API](#api)からも同じエンジン・ジョブ状態にアクセスできる。 |
| **CLI**（`./h3`） | リポジトリ直下 | スクリプト実行・ベンチマーク・全フラグ／環境変数を使った制御 — 以下の全機能はまずここで使えるようになる。 |

## 動作要件

- Apple Siliconを搭載したMac、macOS 13以降。M5クラスのGPU（Metal 4
  TensorOps）で最速のint8経路が有効になる。それより古いApple Siliconでも
  動作するが、BF16/MPSGraphに自動的にフォールバックする。
- Xcode Command Line Tools（`clang`、`swift`、`ar`）。
- Hugging Faceの
  [`MiniMaxAI/MiniMax-H3`](https://huggingface.co/MiniMaxAI/MiniMax-H3)
  チェックポイント — `FL2VA/`だけで約134GiB（約37GiBのBF16トランスフォーマー
  本体に加え、Qwen3-VLテキストエンコーダーと動画/音声VAEを含む）。参照
  条件付き生成用の`Ref2VA/`も加えると合計で約268GiBになります。ネイティブ
  アプリの「モデル管理」から、どちらも直接ダウンロードできます。
- 実行時に他の依存は不要。FFmpeg/FFprobeは**不要**です — 使われるのは
  オプションのクロスチェック用テスト1つだけです（[テスト](#テスト)を参照）。

## クイックスタート

### ネイティブアプリ

```sh
make -j8 libh3.a
cd native/H3Spike
./package_app.sh
open .build/h3c-app.app
```

アプリはモデルを固定パス
`~/Library/Application Support/h3c-analysis/MiniMax-H3` から探します。
初回起動前に、このパスにチェックポイント（内部に`FL2VA/`、任意で
`Ref2VA/`）を配置するかシンボリックリンクしてください。

対応GPUでの初回実行時は、アプリの計算方式選択からint8アテンション
キャッシュを作成（高速経路）するか、「省メモリ（SSDストリーミング）」を
選んでこの作成手順自体を省略し、元のBF16重みをストリーミングすることも
できます。

### CLI

```sh
make -j8
mkdir -p outputs
./h3 --info -d ./MiniMax-H3          # モデル/デバイスを検査するだけ（生成なし）
./h3 -d ./MiniMax-H3 -p "A red fox walks through fresh snow in a pine forest." \
  --width 512 --height 512 --frames 22 --steps 20 -o outputs/fox.mp4
```

`-p`を省略すると、`./h3 -d ./MiniMax-H3` は対話セッションとして起動します
（`!help`でコマンド一覧、`!ref-image`/`!first`/`!last`で条件付け、`!save`で
現在の結果を保存）。完全なフラグ一覧は`./h3 --help`で確認できます —
まず押さえておくべきものは下の[CLIフラグ早見表](#cliフラグ早見表)に
まとめています。

### API

ネイティブアプリを起動したまま、ウィンドウの代わりに（あるいは併用して）
スクリプトから操作できます。

```sh
curl -X POST http://127.0.0.1:8420/api/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A red fox walks through fresh snow.", "size_profile": "square", "seconds": 5}'

curl http://127.0.0.1:8420/api/status
curl http://127.0.0.1:8420/api/result/video -o fox.mp4
```

エンドポイントの全一覧は下記の[API](#api)を参照してください。

## 機能

- **テキストから動画+音声**（T2VA）: プロンプトだけで同期したH.264+AAC
  クリップを生成。
- **最初・最後フレーム条件付け**（FL2VA）: `--first-frame`/`--last-frame`
  （またはアプリの「最初・最後の画像」モード）で、生成の開始・終了フレーム
  を固定。
- **参照条件付け**（Ref2VA）、順序付きで組み合わせ可能: 画像
  （`--ref-image`）、音声あり／なしの動画（`--ref-video` /
  `--ref-silent-video`）、音声を差し替えた動画（`--ref-video-audio`）、
  単独の音声（`--ref-audio`）。参照は引数の順に`<Picture N>`/`<Video N>`
  としてモデルに提示されます。参照動画・画像の読み込みと、すべての動画・
  音声の書き出しはネイティブのAVFoundation/ImageIO経由で行われ、FFmpeg
  サブプロセスは一切使われません。
- **メモリ連動の参照動画サイズ調整**: 大きな参照動画は、マシンの物理
  メモリ搭載量に応じて（[h3_host.c](h3_host.c)内の
  `h3_reference_max_pixels()`）モデルに渡す前に自動的に縮小されます。
  マシンの余力に関わらず常に同じ固定上限を狙うのではありません。なぜ
  これが重要かは下の
  [参照条件付けのコスト](#参照条件付けのコストなぜ参照動画を大きくすると遅くなるのか)
  を参照してください。
- **LoRA**: `H3_LORA_PATH`は、どの常駐モードでもロード時にdiffusers/peft
  形式のアダプタを融合します（[h3_lora.c](h3_lora.c)参照）。
  `build_lora_cache`はint8キャッシュファイルへオフラインで事前融合します。
  4ステップのTurbo蒸留LoRA（[lightx2v/Minimax-h3-Turbo](https://huggingface.co/lightx2v/Minimax-h3-Turbo)
  由来）はエンドツーエンドで対応済みで、ネイティブアプリの「Turbo」
  オプションからも使えます。
- **2つの計算方式**（次節参照）: 高速なint8アテンションキャッシュと、
  キャッシュファイル不要で低メモリだが低速なSSDストリーミング。
- **モデルマネージャー**（ネイティブアプリのみ）: 固定の1パスではなく、
  複数のH3チェックポイントディレクトリとLoRAファイルを登録・切り替え可能。
  FL2VA/Ref2VAは外部依存なし（Python/huggingface_hub不要）でHugging Face
  から直接ダウンロードでき、初回起動時に自動的に案内されます — 素朴だが
  再開可能なURLSessionベースのダウンローダーです。詳細は
  [ModelDownloader.swift](native/H3Spike/Sources/H3cApp/ModelDownloader.swift)。
- **組み込みローカルAPI**（ネイティブアプリのみ）: `127.0.0.1`上で動く
  プレーンなHTTP/JSON API。生のPOSIXソケット上でアプリ自身が提供して
  おり（Network.framework・サードパーティサーバー不要）、ウィンドウと
  全く同じジョブ/エンジン状態を操作します。詳細は下記[API](#api)。
- **対話端末プレビュー**（`--show`、Kitty/Ghostty/iTerm2/WezTerm/Konsole
  対応）と、フェーズ別Metalタイミング/メモリ診断のための`--profile`。

## 計算方式

DiTの約37GiBのBF16重みは、生成のたびに何らかの形でGPUに供給する必要が
あります。互いに排他的な2つの方式があります。

| | int8アテンションキャッシュ | SSDストリーミング |
|---|---|---|
| フラグ／設定 | `H3_ATTENTION_CACHE=path`（アプリ: 「高速（int8キャッシュ）」） | `--ssd-streaming`（アプリ: 「省メモリ（SSDストリーミング）」） |
| 準備 | `build_attention_cache`を一度実行（CLIで約28秒、チェックポイントごとに約18GiBのint8キャッシュファイルを作成） | 不要 — チェックポイントをそのまま読む |
| 速度 | 最速の実測経路 | より遅い。22フレーム/512正方形のクリップで141秒 vs. キャッシュ方式の約78秒（いずれも20ステップ） |
| メモリ | int8量子化された重みをダブルバッファリングで常時ストリーミング | 常時DiTブロック2個分のみ常駐（追跡ストレージ約2GiB） |
| 必要条件 | M5クラスGPU（Metal 4 TensorOps／int8経路） | 任意のApple Silicon GPU |
| LoRA | 対応（`H3_LORA_PATH`、または事前融合済みキャッシュ） | 非対応 — エンジンは常駐/キャッシュブロックのロード時にしかLoRAを融合しない |

`build_attention_cache <FL2VA/transformer dir> <output file>`でキャッシュを
作成するか、モデルのルートディレクトリを指定してFL2VA/Ref2VA両方の
キャッシュを一度に作成できます。キャッシュのバージョン管理、
`model_kind`/`model_id`の不一致検出、`H3_INT8_STREAM_MLP`、キャッシュへの
LoRA融合といった詳細は[h3_attention_cache.c](h3_attention_cache.c)と
[h3_lora.c](h3_lora.c)に、CLIラッパーは
[h3_build_attention_cache.c](h3_build_attention_cache.c)にあります。

生成時間が無制限であることと、参照動画が大きすぎることは、どちらも同じ
制約に突き当たります — 次節を参照してください。

## 参照条件付けのコスト：なぜ参照動画を大きくすると遅くなるのか

大きな参照動画をRef2VAに渡す前に理解しておくべき最も重要な点です。DiT
自身のjoint self-attention（`h3_gpu_sdpa_bf16`）は**非因果的で、すべての
DiTレイヤー・評価されるすべてのdenoisingステップで全トークンに対して
計算されます** — メインの動画/音声潜在表現と参照条件付けトークンの
両方に対してです。参照は一度エンコードされてキャッシュされるのではなく、
全50 DiTブロックによって毎ステップ再度アテンション計算されます。
アテンションのコストはトークン総数に対して少なくとも2乗のオーダーで
増加するため、参照動画を大きく・長くすることは、一度きりのエンコード
コストではなく、生成**全体**のコストを何倍にもします。

実測: 672×384/10秒の生成を大きな参照動画に対して行うと、約4時間
かかりました（ただし正常に完了）。同じ参照動画をさらに15秒に拡大した
ケース、および別途1344×768/15秒のケースも、いずれも4時間以上かかって
完了しました。これらはいずれもバグではありません — `h3_gpu_gqa_causal_bf16`
と`h3_gpu_sdpa_bf16`を個別にベンチマークした結果（手法は`dff0763`前後の
git履歴を参照）、アテンションカーネル自体が異常に遅いという可能性は
除外されており、コストはアーキテクチャに起因するものです。上記の
メモリ連動の上限（`h3_reference_max_pixels`）は極端なケースの
**リスクを減らす**ものであり、このスケーリング自体を変えるものでは
ありません — 大きい・長い参照動画は遅いものと想定し、試行錯誤する際は
より小さい・短い参照動画にするか、`--reuse`/`--layers`をより軽い
プリセットにすることをおすすめします。

このスケーリングが引き起こしたより深刻な障害も一度ありました:
カスタムMetalカーネルのスレッドグループメモリ上限を超えたシーケンスで
動作する因果アテンションのフォールバック（`h3_gpu_gqa_mps`、明示的な
`O(シーケンス長²)`のマスクを構築するMPSGraphベースの経路）は、以前は
そのマスクをシーケンス長ごとにGPUの生存期間中ずっとキャッシュしており、
サイズ上限がありませんでした。十分に長い参照動画由来のシーケンスにより
このキャッシュが数十GBを確保しようとし、マシン全体がクラッシュしました
（アプリのクラッシュではなく、ウォッチドッグタイムアウトによる
カーネルパニックです — 診断する際は
`/Library/Logs/DiagnosticReports/*.panic`を参照してください）。
現在[h3_gpu.m](h3_gpu.m)は、このフォールバックのマスクが512MiBを
超える場合、確保する代わりに明確なエラーで生成を拒否します。ここから
得られる一般的な教訓は、呼び出し側が制御可能なサイズをキーとする
キャッシュすべてに当てはまります: **サイズが無制限な入力をキーとする
キャッシュには、確保前にチェックされる明示的な上限が必要であり、
「入力は小さいはず」というコメントだけでは不十分です。**

## リポジトリ構成

```
h3.c, h3_dit.c, h3_gpu.m, ...   コア推論エンジン（C + Objective-C/Metal）。libh3.aとCLIをビルド
h3.h                            公開C API（h3_load_dir、h3_generate、h3_build_attention_cache、...）
h3_shaders.metal                すべてのMetalコンピュートカーネル
main.c, h3_cli.c, linenoise.c   CLI引数解析＋対話セッションのフロントエンド -> ./h3
h3_build_attention_cache.c      h3_build_attention_cache()のCLIラッパー -> build_attention_cache
h3_build_lora_cache.c           LoRAをint8キャッシュへオフラインで融合するCLIツール -> build_lora_cache
tests/                          Cテストスイート（make test / make parity）
native/H3Spike/                 ネイティブmacOSアプリ（SwiftPM）
  Sources/CH3                   libh3.aのC APIをSwiftへ橋渡しするCシム
  Sources/H3Engine              C APIのSwift非同期ラッパー（AsyncThrowingStreamベースの進捗/キャンセル）
  Sources/H3cApp                SwiftUIアプリ本体（h3c-app.app）。ModelLibrary/ModelManagerView（登録済み
                                 モデル/LoRA）、ModelDownloader（Hugging Faceダウンロード）、
                                 HTTPServer/GenerationViewModel+API（組み込み自動化API）を含む
  Sources/H3Spike                H3Engine用の最小限のプロセス内スパイク/参照クライアント。配布アプリ本体ではない
  package_app.sh                h3c-app.appをビルド・パッケージング。署名・公証も行う（下記参照）
```

## ソースからのビルド

### ライブラリとCLI

```sh
make -j8            # ./h3 と libh3.a をビルド
make test            # 決定論的なホストテスト一式（+ フィクスチャがあればMetal/MLXパリティ）
make parity           # Metal/MLXの数値比較チェックのみ
```

### ネイティブアプリ

```sh
cd native/H3Spike
./package_app.sh
```

`libh3.a`はこのスクリプトでは再ビルドされません — リポジトリ直下で
`make libh3.a`を先に（エンジンのコードを変更した後も同様に）実行して
ください。`package_app.sh`はその後`swift build -c release`を実行し、
`h3c-app.app`を組み立てます。SwiftPM標準のリポジトリ内`.build`ではなく
リポジトリ外のスクラッチディレクトリ（`${TMPDIR}h3c-app-build-scratch`）
にビルドします。これは、リポジトリが同期フォルダ（Google Drive、
iCloud Drive、Dropboxなど）配下にあると、その同期デーモンがSwiftPMの
`build.db`のロックを保持してしまい、見せかけの「disk I/O error」で
ビルドが断続的に失敗することがあるためです — このエラーを見たら、
単なる不安定さと決めつける前に、チェックアウト先が同期対象ディレクトリ
内でないか確認してください。

#### コード署名と公証

署名と公証は2つの環境変数によるオプトイン方式なので、どちらも設定
しない`./package_app.sh`は従来通り未署名の開発ビルドを生成します。

```sh
H3C_SIGN_IDENTITY="Developer ID Application: NAME (TEAMID)" \
H3C_NOTARY_PROFILE="some-keychain-profile" \
./package_app.sh
```

- `H3C_SIGN_IDENTITY` — `security find-identity -v -p codesigning`で
  確認できる`Developer ID Application`証明書。Apple Developer Programへの
  登録と、この証明書の作成が必要です（Xcode → Settings → Accounts →
  Manage Certificates → `+` → Developer ID Application — Xcodeプロジェクト
  は不要で、アカウント管理機能を使うだけです）。署名だけ（
  `H3C_NOTARY_PROFILE`なし）でも、Hardened Runtime有効な状態でローカル
  実行するには十分です。
- `H3C_NOTARY_PROFILE` — `xcrun notarytool store-credentials <profile>
  --apple-id ... --team-id ...`で一度だけ保存するプロファイル名
  （通常のApple IDパスワードではなく[App用パスワード](https://appleid.apple.com)
  が必要）。両方の変数を設定すると、スクリプトはさらにzip化・公証提出・
  完了待ち・チケットのステープルまで行い、生成された`.app`はどのMacでも
  Gatekeeper（`spctl -a -vvv -t exec`）を通過するようになります。

ここでは気にする必要のあるネストされたフレームワークや埋め込みdylibは
ありません — `h3c-app`がリンクしているのはAppleのシステムフレームワークと
静的リンクされた`libh3.a`だけなので、バンドルに対する単純な
`codesign --deep`一回で十分です。

## テスト

```sh
make test
make parity
```

`make test`は決定論的なホストテスト一式を実行し、（gitで無視されている）
MLXフィクスチャが`misc/fixtures/`に配置されていれば、Metalソースを
実行時にコンパイルして、トイのH3ブロックを名前付きMLX出力と比較検証
します — Irisにならい意図的に実行時コンパイルとしており、Xcodeの
オプションであるオフラインMetalツールチェーンを必要としません。
`make parity`はそのMetal/MLXチェックのみを実行します。

`h3_av_mux_test`というテストは、ネイティブAVFoundationのmuxerを実際の
FFmpeg出力と突き合わせて検証するもので、`ffmpeg`が`PATH`上になければ
自動的にスキップされます — このプロジェクトでFFmpegが使われているのは
現在ここだけです。

## CLIフラグ早見表

完全な一覧は`./h3 --help`で確認できます。以下はまず押さえておくべき
ものです。デフォルト値: `--width 864 --height 480 --frames 56 --steps 20
--layers 50 --reuse 1`。

| フラグ | 効果 |
|---|---|
| `-d, --model-dir PATH` | MiniMax-H3チェックポイントのルート |
| `-p, --prompt TEXT` | 一度だけ実行して終了。省略すると対話セッションになる |
| `--width/--height N` | 出力キャンバス（32の倍数、積は768×1344以下） |
| `--frames N` / `--seconds N` | 長さの指定 — 互いに排他的。有効なH3の時間形状に切り上げられる |
| `--steps N` | denoisingパス数。デフォルト20。素早い試行には4〜7、厳密な基準には50 |
| `--reuse N` | denoiser全体の再利用（省略したステップを外挿）: 1が厳密、2が高速、3が積極的 |
| `--layers N` | 有効なDiTブロック数: 50が厳密、45が高速、40が積極的（最小35） |
| `--core-reuse N` | `--reuse`の代替: トランスフォーマーの残差は保持し、パッチ/ヘッドだけ毎ステップ更新 |
| `--token-reduction` | 中間ブロックで水平方向の動画トークンをペア化。高速だが構図が変わりうる |
| `--render-width/--render-height N` | モデル内部をより小さく実行し、vImageでアップスケール |
| `--ssd-streaming` | [計算方式](#計算方式)を参照 |
| `--use-int8-row-fc2` | M5専用、より高速（だが保守的でない）int8 FC2経路 |
| `--first-frame` / `--last-frame PATH` | FL2VAのアンカー条件付け |
| `--ref-image` / `--ref-video` / `--ref-silent-video` / `--ref-video-audio` / `--ref-audio` | Ref2VAの順序付き参照 — [機能](#機能)を参照 |
| `--show` | 対話端末プレビュー（Kitty/Ghostty/iTerm2/WezTerm/Konsole） |
| `--profile` | フェーズ別Metalタイミング・メモリ・ディスパッチ回数のレポート |
| `--info` | 生成せずモデル/デバイスを検査 |

環境変数（`H3_ATTENTION_CACHE`、`H3_ATTENTION_CACHE_DIR`、
`H3_LORA_PATH`、`H3_LORA_SCALE`、`H3_TOKEN_REFINER_LORA`、
`H3_INT8_STREAM_MLP`、`H3_QWEN_PREFETCH*`、`H3_ZERO_COPY_WEIGHTS`、
`H3_VAE_TILE_PIXELS`、`H3_DIT_COMMAND_BLOCKS`、`H3_PROFILE`、および
多数の`H3_DISABLE_*`/`H3_USE_SLOWER_*`系のA/B診断用スイッチ）は、
エンドユーザー向けのチューニングというよりベンチマークや数値比較の
ための代替コードパス選択用です。それぞれソース中の使用箇所
（[h3_dit.c](h3_dit.c)、[h3_gpu.m](h3_gpu.m)、
[h3_attention_cache.c](h3_attention_cache.c)から辿るのがおすすめ）と
コミット履歴にドキュメント化されています — いずれも、ある最適化を
同一プロセス内でA/B比較するオラクルが必要だったために存在するもので、
サポートされたエンドユーザー向けの機能ではありません。

## API

`h3c-app.app`が起動している間、`http://127.0.0.1:8420`でプレーンな
JSON APIを提供します — 生のPOSIXソケットで実装されており
（[HTTPServer.swift](native/H3Spike/Sources/H3cApp/HTTPServer.swift)参照）、
Network.frameworkやサードパーティ製サーバーは使っていません。別プロセスの
起動・停止も不要です。これは旧`gui/server.py`の置き換えです — ブラウザ
クライアント向けに`./h3`をサブプロセスとして包む別実装を用意するのでは
なく、このAPIはウィンドウと全く同じ`GenerationViewModel`/エンジンの
インスタンスを操作します
（[GenerationViewModel+API.swift](native/H3Spike/Sources/H3cApp/GenerationViewModel+API.swift)参照）。
ジョブは常に1つだけで、UIと共有されます — 「動画をつくる」を押すのと
`POST /api/generate`は同じ枠を奪い合い、負けた方には明確な`409`が
返ります。

クライアントとサーバーは常に同じMac上にあるため、メディア入力はブラウザ
向けのアップロードではなく、単なるファイルシステムパスです — 旧GUIと
異なり、アップロード用のエンドポイントはありません。

| メソッド | パス | 内容 |
|---|---|---|
| `POST` | `/api/generate` | ジョブを開始。JSONボディで現在のドラフトを完全に置き換えます（下記参照）。開始できれば`202`、リクエスト不正なら`400`、ジョブ実行中なら`409`。 |
| `GET` | `/api/status` | エンジン/ジョブの状態、進捗割合、フェーズ/ステージ文言、エラーメッセージ、結果が用意できているか。 |
| `POST` | `/api/cancel` | 実行中のジョブがあれば中止。 |
| `GET` | `/api/result/video` | 現在の結果を`video/mp4`としてストリーミング。結果が無いか、次のジョブの`generate()`呼び出しで削除された後は`404`。 |
| `GET` | `/api/models` | 登録済みH3モデルディレクトリ一覧（id・名前・パス・アクティブかどうか）。 |
| `GET` | `/api/loras` | 登録済みLoRAファイル一覧（id・名前・パス・強さ・アクティブかどうか）。 |

`POST /api/generate`のボディフィールド（`prompt`以外はすべて省略可）:

| フィールド | 既定値 | 備考 |
|---|---|---|
| `prompt` | — | 必須。 |
| `size_profile` | `"square"` | `smallSquare`、`square`、`landscapeUpscaled`、`landscapeNative`、`portraitUpscaled`、`portraitNative`のいずれか（[GenerationModels.swift](native/H3Spike/Sources/H3cApp/GenerationModels.swift)の`SizeProfile`参照）。 |
| `seconds` | `5` | 1〜15。 |
| `steps` | `20` | |
| `reuse` | `1` | 1〜3。 |
| `compute_mode` | このGPUでのアプリの既定値 | `attentionCache`または`ssdStreaming`。 |
| `seed` | ランダム | |
| `first_frame_path` / `last_frame_path` | なし | FL2VAのアンカー。`reference_paths`とは併用不可。 |
| `reference_paths` | `[]` | 順序付きのRef2VA参照。画像か動画かはパスごとに自動判定。 |
| `lora_name` | なし | `GET /api/loras`の名前と一致させる必要あり。省略（または`""`）すると、ウィンドウ側で選択済みでもこのジョブではLoRAなし扱いになる。 |
| `lora_scale` | そのLoRAの保存済みの強さ | `lora_name`指定時のみ意味を持つ。 |

```sh
curl -X POST http://127.0.0.1:8420/api/generate -H "Content-Type: application/json" -d '{
  "prompt": "A red fox walks through fresh snow in a pine forest.",
  "size_profile": "square", "seconds": 5, "steps": 20
}'

curl http://127.0.0.1:8420/api/status
curl http://127.0.0.1:8420/api/result/video -o fox.mp4
```

## ライセンス

MIT — [LICENSE](LICENSE)を参照。サードパーティ表示（`h3_shaders.metal`内の
一部設計はccvのFlashAttention実装を基にしています）は
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)にあります。
アップストリームプロジェクト: [antirez/h3.c](https://github.com/antirez/h3.c)。
