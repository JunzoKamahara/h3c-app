# h3c-app

[English](README.md) | [日本語](README.ja.md)

[MiniMax-H3](https://huggingface.co/lightx2v/Minimax-h3-Turbo)
（テキスト／画像／動画から、音声付き動画を生成する拡散トランスフォーマー）を
Apple Siliconだけで動かすネイティブmacOSアプリです。すべてMetal/MPSGraph上で
プロセス内実行され、Python・PyTorch・クラウド通信は一切不要です — メディアの
入出力はネイティブのAVFoundation/ImageIO経由です。基盤となるC/Objective-Cの
推論エンジン（`libh3.a`）はSalvatore
Sanfilippo氏（antirez）の[h3.c](https://github.com/antirez/h3.c)を
ベースにしています。詳しいライセンス表記は[ライセンス](#ライセンス)を
参照してください。

アプリを起動している間は、ウィンドウ操作に加えてローカルのJSON APIも
`127.0.0.1`上で提供されるので、スクリプトから操作することもできます。
詳しくは[API](#api)を参照してください。

## ダウンロード

もっとも簡単なインストール方法は、[リリースページ](https://github.com/JunzoKamahara/h3c-app/releases/latest)
にある署名・公証済みのビルド済み`.dmg`を使うことです — Xcode Command Line
Toolsもソースからのビルドも不要です。`.dmg`をダウンロードして開き、
`h3c-app`を`Applications`にドラッグしてください。初回起動時もGatekeeperに
よる「開発元が未確認」といった警告は出ません。

ソースからビルドする場合は、下記の
[ソースからのビルド](#ソースからのビルド)を参照してください。

## 動作要件

- Apple Siliconを搭載したMac、macOS 13以降。M5クラスのGPU（Metal 4
  TensorOps）で最速のint8経路が有効になる。それより古いApple Siliconでも
  動作するが、BF16/MPSGraphに自動的にフォールバックする。
- Xcode Command Line Tools（`clang`、`swift`、`ar`） —
  ソースからビルドする場合のみ必要。ビルド済み`.dmg`では不要。
- Hugging Faceの
  [`MiniMaxAI/MiniMax-H3`](https://huggingface.co/MiniMaxAI/MiniMax-H3)
  チェックポイント — `FL2VA/`だけで約134GiB（約37GiBのBF16トランスフォーマー
  本体に加え、Qwen3-VLテキストエンコーダーと動画/音声VAEを含む）。参照
  条件付き生成用の`Ref2VA/`も加えると合計で約268GiBになります。アプリの
  モデルマネージャーから、どちらも直接ダウンロードできます。
- 実行時に他の依存は不要。

## クイックスタート

```sh
make -j8 libh3.a
cd native/H3Spike
./package_app.sh
open .build/h3c-app.app
```

初回起動時、まだモデルが登録されていなければ、アプリがHugging Faceから
MiniMax-H3を直接ダウンロードするか尋ねます（[機能](#機能)を参照）。すでに
チェックポイントを持っている場合は「モデル管理」からその場所を指定して
ください。

対応GPUでは、計算方式選択からint8アテンションキャッシュを作成（高速経路）
するか、「省メモリ（SSDストリーミング）」を選んでこの作成手順自体を省略し、
元のBF16重みをストリーミングすることもできます。

## 機能

- **テキストから動画+音声**（T2VA）: プロンプトだけで同期したH.264+AAC
  クリップを生成。
- **最初・最後フレーム条件付け**（FL2VA）: 生成の開始・終了フレームを固定
  （アプリの「最初・最後の画像」モード、またはAPIの
  `first_frame_path`/`last_frame_path`）。
- **参照条件付け**（Ref2VA）、順序付きで組み合わせ可能: 画像・動画
  （音声あり／なし）に加えて音声単体のファイルもスタイル・内容の参照
  として使用できます（アプリの「参照画像・動画・音声」モード、または
  APIの`reference_paths`）。ただし音声だけを参照にすることはできず、
  画像か動画の参照が少なくとも1つ必要です。参照は順番に`<Picture N>`/
  `<Video N>`としてモデルに提示されます。参照の読み込みと生成動画の
  書き出しはすべてネイティブのAVFoundation/ImageIO経由です。
- **メモリ連動の参照動画サイズ調整**: 大きな参照動画は、マシンの物理
  メモリ搭載量に応じて（[h3_host.c](h3_host.c)内の
  `h3_reference_max_pixels()`）モデルに渡す前に自動的に縮小されます。
  マシンの余力に関わらず常に同じ固定上限を狙うのではありません。なぜ
  これが重要かは下の
  [参照条件付けのコスト](#参照条件付けのコストなぜ参照動画を大きくすると遅くなるのか)
  を参照してください。
- **LoRA**: アプリのモデルマネージャーが登録済みLoRAファイル（それぞれに
  保存された強さ付き）のライブラリを保持し、どの常駐モードでもロード時に
  融合します（[h3_lora.c](h3_lora.c)参照）。4ステップのTurbo蒸留LoRA
  （[lightx2v/Minimax-h3-Turbo](https://huggingface.co/lightx2v/Minimax-h3-Turbo)
  由来）はエンドツーエンドで動作します。`build_lora_cache`
  （`make build_lora_cache`でビルド）は、アプリを介さずエンジンを直接
  使う場合向けに、LoRAをint8キャッシュファイルへオフラインで事前融合する
  ツールです。
- **3つの計算方式**（[計算方式](#計算方式)参照）: 高速なint8アテンション
  キャッシュ、キャッシュ不要で大容量メモリのMac向けの常駐モード、
  低メモリだが低速なSSDストリーミング。
- **モデルマネージャー**: 固定の1パスではなく、複数のH3チェックポイント
  ディレクトリとLoRAファイルを登録・切り替え可能。FL2VA/Ref2VAは外部依存
  なし（Python/huggingface_hub不要）でHugging Faceから直接ダウンロード
  でき、初回起動時に自動的に案内されます — 素朴だが再開可能なURLSession
  ベースのダウンローダーです。詳細は
  [ModelDownloader.swift](native/H3Spike/Sources/H3cApp/ModelDownloader.swift)。
- **組み込みローカルAPI**: `127.0.0.1`上で動くプレーンなHTTP/JSON API。
  生のPOSIXソケット上でアプリ自身が提供しており（Network.framework・
  サードパーティサーバー不要）、ウィンドウと全く同じジョブ/エンジン状態を
  操作します。詳細は[API](#api)。

## 計算方式

DiTの約37GiBのBF16重みは、生成のたびに何らかの形でGPUに供給する必要が
あります。互いに排他的な3つの方式があり、いずれもアプリの計算方式選択
（またはAPIの`compute_mode`フィールド）から選べます。

| | int8アテンションキャッシュ | 常駐モード | SSDストリーミング |
|---|---|---|---|
| アプリ／APIでの値 | 「高速（int8キャッシュ）」／`attentionCache` | 「常駐（大容量メモリ向け、キャッシュ不要）」／`resident` | 「省メモリ（SSDストリーミング）」／`ssdStreaming` |
| 準備 | 一度だけのキャッシュ作成（必要な場合はアプリのウィンドウ内で提案されます。チェックポイントごとに約28秒、約18GiBのint8キャッシュファイルを作成） | ディスクへの準備は不要 — 起動のたびにその場で量子化（キャッシュ作成と同じ約28秒のコストを、起動・モデル切り替えのたびに払う） | 不要 — チェックポイントをそのまま読む |
| 速度 | 最速の実測経路 | キャッシュ方式と同程度（こちらもステップごとのディスクI/Oなし） | より遅い。22フレーム/512正方形のクリップで141秒 vs. キャッシュ方式の約78秒（いずれも20ステップ） |
| メモリ | int8量子化された重みをダブルバッファリングで常時ストリーミング | 全DiTブロックを常時メモリに展開 — Tensor演算ユニット搭載GPUではint8量子化で約18GiB、それ以外では完全なBF16のまま約37GiB | 常時DiTブロック2個分のみ常駐（追跡ストレージ約2GiB） |
| 必要条件 | M5クラスGPU（Metal 4 TensorOps／int8経路） | 任意のApple Silicon GPU（Tensor演算ユニットがない場合はBF16のまま常駐）— 大容量メモリのMac向け | 任意のApple Silicon GPU |
| LoRA | 対応 | 対応 | 非対応 — エンジンは常駐/キャッシュブロックのロード時にしかLoRAを融合しない |

キャッシュ形式の詳細 — バージョン管理、`model_kind`/`model_id`の不一致
検出、キャッシュへのLoRA融合 — は[h3_attention_cache.c](h3_attention_cache.c)
と[h3_lora.c](h3_lora.c)にあります。キャッシュは独立したステップとして
`build_attention_cache <FL2VA/transformer dir> <output file>`
（`make build_attention_cache`）で作成することもでき、モデルの
ルートディレクトリを指定すればFL2VA/Ref2VA両方のキャッシュを一度に
作成できます。

生成時間が無制限であることと、参照動画が大きすぎることは、どちらも同じ
制約に突き当たります — 次節を参照してください。

### ライブラリレベルの設定

エンジン自体（`h3_dit.c`）は、いくつかの環境変数を直接読み取ります —
`H3_ATTENTION_CACHE`、`H3_ATTENTION_CACHE_DIR`、`H3_LORA_PATH`、
`H3_LORA_SCALE`、`H3_TOKEN_REFINER_LORA`、`H3_INT8_STREAM_MLP`、
`H3_QWEN_PREFETCH*`、`H3_ZERO_COPY_WEIGHTS`、`H3_VAE_TILE_PIXELS`、
`H3_DIT_COMMAND_BLOCKS`、`H3_PROFILE`、および多数の
`H3_DISABLE_*`/`H3_USE_SLOWER_*`系のA/B診断用スイッチです。これらは
`libh3.a`を直接リンクする人向けのもので、アプリ自身は通常の利用では
これらに頼らず、同じ内部オプションを自前のUIと`H3GenerationParams`
経由で操作しています。それぞれソース中の使用箇所
（[h3_dit.c](h3_dit.c)、[h3_gpu.m](h3_gpu.m)、
[h3_attention_cache.c](h3_attention_cache.c)から辿るのがおすすめ）に
ドキュメント化されています。

## 参照条件付けのコスト：なぜ参照動画を大きくすると遅くなるのか

Ref2VAの参照は一度エンコードされてキャッシュされるのではなく、DiTが
全レイヤー・全denoisingステップで、メインの動画/音声潜在表現と共に
毎回再度アテンション計算します。アテンションコストはトークン総数に対して
少なくとも2乗で増加するため、参照動画を大きく・長くすると生成**全体**の
コストが跳ね上がります（一度きりのエンコードコストではありません）。
大きい・長い参照動画では実測で4時間以上かかったケースがあります（正常に
完了はします）。上記のメモリ連動の上限は極端なケースのリスクを減らします
が、このスケーリング自体は変わりません。

**参照動画使用時に生成が想定より遅い場合**: より小さい・短い参照動画に
するか、denoisingステップ数を減らすか、`reuse`の値を上げてください。
極端なケースは、メモリ不足を招く前に明確なエラーで拒否されます
（MPSGraphフォールバック経路の512MiBアテンションマスク上限、
[h3_gpu.m](h3_gpu.m)参照）。

## リポジトリ構成

```
h3.c, h3_dit.c, h3_gpu.m, ...   コア推論エンジン（C + Objective-C/Metal）。libh3.aをビルド
h3.h                            公開C API（h3_load_dir、h3_generate、h3_build_attention_cache、...）
h3_shaders.metal                すべてのMetalコンピュートカーネル
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
  make_dmg.sh                   ビルド済みh3c-app.appを配布用.dmgにまとめる
```

## ソースからのビルド

### ライブラリ

```sh
make -j8 libh3.a     # C/Objective-Cエンジンをビルド
make test             # 決定論的なホストテスト一式（+ フィクスチャがあればMetal/MLXパリティ）
make parity            # Metal/MLXの数値比較チェックのみ
```

`make build_attention_cache`と`make build_lora_cache`は、上記の独立した
キャッシュ準備ツール2つをビルドします。どちらもデフォルトの`make`
ターゲットには含まれません。

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
しない`./package_app.sh`は未署名の開発ビルドを生成します。

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

#### .dmgとして配布する

```sh
./make_dmg.sh
```

すでにビルド済みの`h3c-app.app`を、Applicationsへのドラッグ用ショート
カット付きの圧縮`.dmg`にまとめます。`hdiutil`のみを使用（サードパーティ製
のdmg作成ツールは不使用）。先に`package_app.sh`を実行しておいてください。
`H3C_SIGN_IDENTITY`と`H3C_NOTARY_PROFILE`が設定されていれば、
`make_dmg.sh`は`.dmg`ファイル自体にも署名・公証・ステープルを行います
— Gatekeeperの実際のチェックは起動時に`.app`自身の署名・ステープルに
対して行われるため必須ではありませんが、ダウンロード時の体験を完全に
クリーンにするため`.dmg`自体もGatekeeperを通過するようにできます。

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

## API

`h3c-app.app`が起動している間、`http://127.0.0.1:8420`でプレーンな
JSON APIを提供します — 生のPOSIXソケットで実装されており
（[HTTPServer.swift](native/H3Spike/Sources/H3cApp/HTTPServer.swift)参照）、
Network.frameworkやサードパーティ製サーバーは使っていません。別プロセスの
起動・停止も不要です。このAPIはウィンドウと全く同じ`GenerationViewModel`/
エンジンのインスタンスを操作します
（[GenerationViewModel+API.swift](native/H3Spike/Sources/H3cApp/GenerationViewModel+API.swift)参照）。
ジョブは常に1つだけで、UIと共有されます — 「動画をつくる」を押すのと
`POST /api/generate`は同じ枠を奪い合い、負けた方には明確な`409`が
返ります。

クライアントとサーバーは常に同じMac上にあるため、メディア入力はアップ
ロードではなく、単なるファイルシステムパスです。

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
| `compute_mode` | このGPUでのアプリの既定値 | `attentionCache`、`resident`、`ssdStreaming`のいずれか。 |
| `seed` | ランダム | |
| `first_frame_path` / `last_frame_path` | なし | FL2VAのアンカー。`reference_paths`とは併用不可。 |
| `reference_paths` | `[]` | 順序付きのRef2VA参照。画像/動画/音声はパスごとに自動判定。音声パスを含める場合、画像か動画を最低1つ含める必要あり。 |
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
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)にあります。推論エンジンは
Salvatore Sanfilippo氏による
[antirez/h3.c](https://github.com/antirez/h3.c)を起源としています。
