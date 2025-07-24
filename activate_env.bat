@echo off
REM 激活 summer_project_2025 conda 环境的批处理脚本
call conda activate summer_project_2025
echo 已激活 summer_project_2025 环境
python --version
echo Python 路径: 
python -c "import sys; print(sys.executable)"
