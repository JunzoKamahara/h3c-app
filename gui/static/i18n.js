/* Minimal i18n, matching the convention used by the h3movie web UI: English
   is the default; Japanese is used when the browser asks for it
   (navigator.language). ?lang=en|ja overrides, mainly for testing. */
"use strict";

const I18N = {
  en: {
    "title": "H3 Local GUI",
    "sub": "A local generation UI wrapping h3.c",

    "prompt.legend": "Prompt",

    "profile.legend": "Quality profile",
    "profile.square": "Square 512×512 (standard)",
    "profile.landscape": "Landscape 1344×768 (upscaled from 672×384)",
    "profile.portrait": "Portrait 768×1344 (upscaled from 384×672)",

    "firstframe.legend": "First frame image (optional – Image-to-Video)",
    "firstframe.drop": "Drop an image here, or click to choose",
    "firstframe.choose": "Choose Image",
    "firstframe.clear": "Remove Image",
    "firstframe.hint": "The image the model actually receives (after center-crop)",
    "firstframe.unsupported": "Only JPEG, PNG, or WebP are supported.",
    "firstframe.uploadFailed": "Could not upload the image.",

    "lastframe.legend": "Last frame image (optional)",
    "lastframe.drop": "Drop an image here, or click to choose",

    "refframe.legend": "Reference image (optional – Ref2VA)",
    "refframe.note": "Uses a different model path than the first/last frame image above - cannot be combined with either.",
    "refframe.drop": "Drop an image here, or click to choose",
    "refframe.unavailable": "Ref2VA checkpoint not found - reference images are unavailable.",

    "seconds.legend": "Length",
    "seconds.label": "Seconds (longer takes proportionally more time)",

    "turbo.legend": "Turbo (experimental)",
    "turbo.enable": "4-step Turbo LoRA (faster, lower fidelity - forces Steps to 4)",
    "turbo.note": "First/last-frame generation only - not available together with a reference image (Ref2VA).",
    "turbo.unavailable": "Turbo cache not found - build it with build_lora_cache first.",

    "advanced.summary": "Advanced",
    "advanced.layers": "Layers (35 = faster, slightly lower quality; 50 = full quality)",
    "advanced.reuse": "Reuse (1 = exact, 2 = fast)",
    "advanced.reuse1": "1 (exact, slower)",
    "advanced.reuse2": "2 (fast, verified)",
    "advanced.steps": "Steps",
    "advanced.seed": "Seed (leave blank for random)",
    "seed.placeholder": "Random",

    "generate": "Generate",
    "alert.promptRequired": "Please enter a prompt.",
    "alert.secondsRange": "Seconds must be a whole number between {min} and {max}.",
    "alert.rejected": "The request was rejected.",

    "progress.queued": "Queued...",
    "progress.elapsed": "Elapsed: {time}",
    "progress.back": "Back",

    "result.again": "Generate Again",
    "result.download": "Download",
    "result.seed": "Seed: {seed} (filled into Advanced → Seed for reuse)",

    "error.back": "Back",
    "error.default": "Generation failed.",

    "banner.modelDir": "Model directory not found: {path}",
    "banner.cache": "int8 cache not found (falling back to slower resident weight loading)",
    "banner.cacheBuildable": "int8 cache not found - it will be built automatically on the first job (adds time to that run only)",
  },

  ja: {
    "title": "H3 Local GUI",
    "sub": "h3.c をラップしたローカル生成UI",

    "prompt.legend": "プロンプト",

    "profile.legend": "画質プロファイル",
    "profile.square": "Square 512×512(標準)",
    "profile.landscape": "Landscape 1344×768(672×384からアップスケール)",
    "profile.portrait": "Portrait 768×1344(384×672からアップスケール)",

    "firstframe.legend": "最初のフレーム画像(任意・Image-to-Video)",
    "firstframe.drop": "ここに画像をドロップ、またはクリックして選択",
    "firstframe.choose": "画像を選択",
    "firstframe.clear": "画像を削除",
    "firstframe.hint": "実際にモデルへ渡される画像(短辺基準でクロップ後)",
    "firstframe.unsupported": "JPEG / PNG / WebP のみ対応しています。",
    "firstframe.uploadFailed": "画像のアップロードに失敗しました。",

    "lastframe.legend": "最後のフレーム画像(任意)",
    "lastframe.drop": "ここに画像をドロップ、またはクリックして選択",

    "refframe.legend": "参照画像(任意・Ref2VA)",
    "refframe.note": "上のFirst/Last frame画像とは別のモデル経路を使うため、どちらとも併用できません。",
    "refframe.drop": "ここに画像をドロップ、またはクリックして選択",
    "refframe.unavailable": "Ref2VAチェックポイントが見つからないため、参照画像は使用できません。",

    "seconds.legend": "長さ",
    "seconds.label": "秒数(長いほど時間がかかります)",

    "turbo.legend": "Turbo(実験的)",
    "turbo.enable": "4-step Turbo LoRA(高速・品質はやや低下 - ステップ数を4に固定)",
    "turbo.note": "First/last-frame生成専用 - 参照画像(Ref2VA)とは併用できません。",
    "turbo.unavailable": "Turboキャッシュが見つかりません - 先に build_lora_cache で作成してください。",

    "advanced.summary": "詳細設定",
    "advanced.layers": "層数(35=速い/品質やや低下, 50=フル品質)",
    "advanced.reuse": "reuse(1=厳密, 2=高速)",
    "advanced.reuse1": "1(厳密・遅い)",
    "advanced.reuse2": "2(高速・検証済み)",
    "advanced.steps": "ステップ数",
    "advanced.seed": "シード(空欄でランダム)",
    "seed.placeholder": "ランダム",

    "generate": "生成",
    "alert.promptRequired": "プロンプトを入力してください。",
    "alert.secondsRange": "秒数は{min}〜{max}の整数で指定してください。",
    "alert.rejected": "リクエストが拒否されました。",

    "progress.queued": "待機中...",
    "progress.elapsed": "経過: {time}",
    "progress.back": "戻る",

    "result.again": "もう一度生成",
    "result.download": "ダウンロード",
    "result.seed": "シード: {seed}(詳細設定→シード欄に反映済み・再利用可能)",

    "error.back": "戻る",
    "error.default": "生成に失敗しました。",

    "banner.modelDir": "モデルディレクトリが見つかりません: {path}",
    "banner.cache": "int8キャッシュが見つかりません(低速なレジデントロードにフォールバックします)",
    "banner.cacheBuildable": "int8キャッシュが見つかりません - 最初のジョブ実行時に自動でビルドします(その回だけ時間がかかります)",
  },
};

const LANG = (() => {
  const forced = new URLSearchParams(location.search).get("lang");
  const want = (forced || navigator.language || "en").toLowerCase();
  return want.startsWith("ja") ? "ja" : "en";
})();

function t(key, vars) {
  let s = I18N[LANG][key] ?? I18N.en[key] ?? key;
  if (vars) {
    for (const [k, v] of Object.entries(vars)) s = s.split(`{${k}}`).join(v);
  }
  return s;
}

function applyStaticTranslations() {
  document.documentElement.lang = LANG;
  document.title = t("title");
  document.querySelectorAll("[data-i18n]").forEach((el) => {
    el.textContent = t(el.dataset.i18n);
  });
  document.querySelectorAll("[data-i18n-placeholder]").forEach((el) => {
    el.placeholder = t(el.dataset.i18nPlaceholder);
  });
}
