@echo off
set PYTHONPATH=%~dp0
if "%1"=="train" (
    python scripts/train.py %2 %3 %4 %5
) else if "%1"=="eval" (
    python scripts/evaluate.py %2 %3 %4 %5
) else if "%1"=="synthetic" (
    python scripts/generate_synthetic_plans.py %2 %3 %4 %5
) else if "%1"=="tsne" (
    python scripts/visualize.py tsne %2 %3 %4 %5
) else (
    echo Usage: run.bat [train^|eval^|synthetic^|tsne] [args...]
)
