@echo off
REM Navigate to the project root (parent of scritps)
cd /d "%~dp0.."

echo Running CGFM Training from Project Root...
python scritps/Contrastive_gate_fusionModel.py ^
    --epochs 10 ^
    --batch-size 32 ^
    --lr 0.0001 ^
    --lambda-contrast 0.1 ^
    --topk 5 10 20 ^
    --log-name cgfm_run_01

pause
