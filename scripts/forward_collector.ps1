# Phase 2D forward-OOS data collector. Runs daily via a Windows Scheduled
# Task ("PumpExhaustionReversal_ForwardCollector") - delegates to
# scripts/run_daily_forward_collector.py, which does the SAME two-stage
# discipline as the historical backfill (run_stage_a_pump_discovery.py /
# run_stage_b_event_scoped_backfill.py): cheap OHLCV pull for the trailing
# window, frozen-pump-detector scan on just that window, and event-scoped
# OI/funding fetch ONLY for symbols/windows with a newly qualifying pump -
# never a full-universe OI/funding pull. Self-healing (7-day lookback inside
# the Python script) if a day's run is missed.
#
# This stream starts accumulating from the freeze date recorded in
# config/oi_volume_reversal_candidate_v1.yaml (frozen_at: 2026-08-27) and
# must NEVER be used to retune the OI_ROLLOVER + VOLUME_DECAY candidate -
# only to eventually evaluate it, once enough forward signals have
# accumulated (see that file's oos_acceptance_criteria).
#
# Per the user's explicit instruction: monitor only integrity/coverage/N
# while this accumulates - do not look at trading performance (win rate,
# net R) on partial forward data and do not adjust the frozen parameters.

$ErrorActionPreference = "Continue"
$repoRoot = "C:\Users\adama\Documents\zeph ai projects\pump_exhaustion_reversal"
Set-Location $repoRoot
$python = "C:\Users\adama\AppData\Local\Programs\Python\Python312\python.exe"
$logDir = Join-Path $repoRoot "data\forward_collector_logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$logFile = Join-Path $logDir "forward_collector.log"
$stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

Add-Content -Path $logFile -Encoding utf8 -Value "===== $stamp - forward collector run ====="

# Avoid PowerShell 5.1's *>> / 2>&1 on a native exe (known encoding/error-record
# quirks - see repo tooling notes) - capture stdout as plain text lines instead.
& $python scripts/run_daily_forward_collector.py --exchange bybit --config config/backtest.yaml | Out-File -FilePath $logFile -Encoding utf8 -Append
Add-Content -Path $logFile -Encoding utf8 -Value "--- bybit exit code: $LASTEXITCODE ---"

Add-Content -Path $logFile -Encoding utf8 -Value "===== $stamp - run complete ====="
