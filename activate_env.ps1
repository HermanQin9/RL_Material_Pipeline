# 激活 summer_project_2025 conda 环境的 PowerShell 脚本
conda activate summer_project_2025
Write-Host "已激活 summer_project_2025 环境" -ForegroundColor Green
python --version
Write-Host "Python 路径:" -ForegroundColor Yellow
python -c "import sys; print(sys.executable)"
