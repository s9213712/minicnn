# 2026-04-24 Follow-up Issues

本文件整理自 2026-04-24 狀態檢查，可直接拆成 GitHub issues。

## Issue 1

Title: `Finalize the public ForwardExecutor contract for cuda_native`

Problem:

- 目前 `ForwardExecutor` 實作採用 `run(graph, feeds, params=None, mode='eval')`
- 部分測試仍假設舊式呼叫方式，例如 `ForwardExecutor(graph)` 後 `run(ctx)`
- `pytest -q` 因此在 `DropPath` 與 `LayerNorm` 測試上失敗

Why it matters:

- executor 是 `cuda_native` 的核心入口之一
- API 契約不明確，會讓文件、測試、呼叫端持續漂移

Proposed fix:

- 正式定稿 stateless executor 契約，並更新測試與文件
- 或提供一層明確標註 deprecated 的相容入口，但不要維持半新半舊狀態

Acceptance criteria:

- `DropPath` / `LayerNorm` 測試與文件使用同一種 executor 呼叫方式
- `pytest -q` 不再因 `ForwardExecutor.run()` 參數簽名失敗
- `docs/cuda_native.md` 或相關文件清楚寫明 public API

Priority: `P1`

## Issue 2

Title: `Rework cuda_native registry tests to distinguish stable minimum contract from feature growth`

Problem:

- 現有測試將 activation/kernel registry 完整列表視為 frozen contract
- 實際 registry 已擴張，例如 `GELU`、`AdaptiveAvgPool2d` 等
- 因此 `test_default_registry_keeps_activation_kernel_surface_stable` 與 `test_default_registry_exposes_kernel_metadata` 失敗

Why it matters:

- repo 正在擴張 `cuda_native`
- 舊測試如果要求完整列表相等，會把合理成長誤判成 regression

Proposed fix:

- 將測試改為驗證最小必要集合與 metadata 正確性
- 若需要 frozen contract，改成版本化 capability snapshot，而不是 phase1 固定清單

Acceptance criteria:

- registry 測試能容忍新增支援 op
- 必要核心 op 與 category metadata 仍被嚴格檢查
- 測試名稱與目的清楚區分「穩定契約」與「功能擴張」

Priority: `P1`

## Issue 3

Title: `Normalize structured validation errors for cuda_native shape and channel mismatches`

Problem:

- `Conv2d` mismatch 的實際錯誤訊息已升級為 group-aware 表述
- 但測試仍綁定舊字串 `weight expects 3 input channels, got 1`

Why it matters:

- 人類可讀訊息仍可能持續演化
- 若測試靠完整字串比對，會造成大量低價值失敗

Proposed fix:

- 導入穩定錯誤碼或結構化錯誤分類
- 測試改比對關鍵語意或 error code，而不是完整文字

Acceptance criteria:

- channel/group mismatch 有穩定 machine-readable contract
- 測試不再依賴完整人類訊息字串
- `Conv2d` 驗證錯誤仍保留足夠 debug 資訊

Priority: `P2`

## Issue 4

Title: `Clarify smoke output when cuda_legacy native artifacts are optional and missing`

Problem:

- `minicnn smoke --format json` 目前會回報 `native_available=false`
- 實際缺的是 `cuda_legacy` native artifact，不代表整個 repo 或 `cuda_native` 驗證不可用

Why it matters:

- 使用者容易把 optional component 缺失誤判成整體不可用

Proposed fix:

- 將 warning 名稱與說明改得更精準
- 明確標示缺的是 `cuda_legacy` optional artifact
- 補充不受影響的功能面，例如 `torch`、`autograd`、`cuda_native` validation

Acceptance criteria:

- smoke warning 能區分 optional/native-maintenance artifact 與核心功能故障
- JSON 欄位與文字輸出都更具體

Priority: `P2`

## Issue 5

Title: `Clean up untracked path-policy artifacts and standardize repo output paths`

Problem:

- 工作樹存在未追蹤目錄 `path-policy-artifacts/`
- 此類輸出未被現有 `artifacts/` / `outputs/` 規則吸收

Why it matters:

- 工作樹容易持續變髒
- repo output policy 不一致

Proposed fix:

- 若屬暫存輸出，加入 `.gitignore`
- 若屬正式分析產物，搬進既有 `artifacts/` 結構
- 補一份簡短輸出路徑政策文件

Acceptance criteria:

- 相同工作流不再於 repo 根目錄旁生成額外未管理資料夾
- `git status --short` 對常見工作流維持乾淨

Priority: `P3`
