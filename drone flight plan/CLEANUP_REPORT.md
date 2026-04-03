# Project Cleanup Report (Updated) / プロジェクト整理レポート（更新版）

---

## ファイル関係の全体図 / Complete File Relationship Map

```
main.py  ──────────────────────────────────→  main_v2.py
(手動操縦 + 距離推定のみ)        (+ eval_logger による毎フレームログ追加)
                                              ↑ これが最新・本命

main_tracking_copy.py  ──→  main_tracking.py  ═══  main_tracking_v2.py
(旧版: eval_logger旧スキーマ)   (現行版: 完全統合)    (完全に同一ファイル)
```

---

## ❌ 削除推奨ファイル / Files to DELETE

| ファイル | 理由 | 代替 |
|----------|------|------|
| `main_tracking_v2.py` | `main_tracking.py` と**完全に同一**（diff 0行） | `main_tracking.py` |
| `main_tracking_copy.py` | `main_tracking.py` の旧版。eval_logger 旧スキーマ・pinhole 未統合 | `main_tracking.py` |
| `hello.py` | 初期テスト用の簡易スクリプト。`main.py` が上位互換 | `main.py` |
| `testcam.py` | **中身が空**（0バイト） | 不要 |

---

## ⚠️ 統合・改名が必要なファイル / Files to MERGE or RENAME

### `main.py` → `main_v2.py` の関係

`main_v2.py` は `main.py` に **eval_logger による毎フレームログ** を追加した発展版です。
主な差分：

| 項目 | main.py | main_v2.py |
|------|---------|------------|
| ヘッダコメント | `# main.py` | `# main.py (LOG enabled full version)` |
| eval_logger import | なし | `create_logger, EvalRow, bbox_center` 等 |
| ログ出力 | なし | `logs/ics_main_YYYYMMDDHHMMSS.xlsx` |
| APPLE_SIZE_M | 0.08 | 0.05（実験で調整済み） |
| TARGET_DISTANCE_M | 0.10 m | 0.15 m |
| DIST_TOL_M | 0.08 m | 0.05 m |

**→ 対応**: `main_v2.py` を `main.py` にリネームして置き換え、旧 `main.py` は削除。

---

## ✅ 残すファイル（整理後） / Final File List

```
tello-fruit-tracker/
├── README.md
├── .gitignore
├── requirements.txt
│
├── main.py                       ← (旧 main_v2.py をリネーム) 手動操縦 + ログ
├── main_tracking.py              ← 自律追跡メイン
├── train_base_cfg_filterv1.py    ← 学習スクリプト (.py版)
├── train_base_cfg_filterv1.ipynb ← 学習ノートブック (.ipynb版)
├── test_Tello_state.py           ← Telloセンサー疎通確認
│
├── utils/
│   ├── __init__.py
│   ├── ctrl.py
│   ├── model.py
│   ├── tracking_continuity.py
│   ├── pinhole.py
│   ├── ssd_model.py
│   ├── ssd_predict_show.py
│   ├── match.py
│   ├── data_augumentation.py
│   ├── eval_logger.py
│   ├── eval_metrics.py
│   └── eval_batch.py
│
├── weights/
│   └── .gitkeep
└── logs/
    └── .gitkeep
```

---

## 学習ノートブックについて / About Training Notebook

`train_base_cfg_filterv1.ipynb` と `train_base_cfg_filterv1.py` は同一内容（.ipynb を変換したもの）。
両方 Git に残すのは冗長なので、以下のどちらかを選択：

- **研究用途（再現性重視）** → `.ipynb` のみ残す（セル出力・実験ログが記録される）
- **CI/スクリプト実行重視** → `.py` のみ残す
- **両方残す場合** → `.py` を `scripts/` 等に移動して整理

---

## .gitignore 補足 / nbstripout

ノートブックのセル出力（loss ログ等）を Git に含めたくない場合は `nbstripout` を推奨：

```bash
pip install nbstripout
nbstripout --install  # リポジトリに自動適用
```

---

## まとめ：実施コマンド例 / Summary: Commands to Execute

```bash
# 削除
rm main_tracking_v2.py
rm "main_tracking copy.py"
rm hello.py
rm testcam.py

# リネーム（旧 main.py を削除してから）
rm main.py
mv main_v2.py main.py

# utils パッケージ化
touch utils/__init__.py

# Git 管理用の空フォルダ保持
touch weights/.gitkeep
touch logs/.gitkeep

# (オプション) ノートブック出力の除外
pip install nbstripout && nbstripout --install
```
