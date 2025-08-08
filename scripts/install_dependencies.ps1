# PowerShell脚本用于安装VALOR项目所需的Python依赖

# 检查Python是否已安装
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "错误: 未找到Python。请安装Python 3.6+并确保其在PATH中可用。" -ForegroundColor Red
    exit 1
}

Write-Host "检测到Python版本: $pythonVersion" -ForegroundColor Green

# 安装基本依赖
Write-Host "正在安装基本依赖..." -ForegroundColor Yellow
python -m pip install --upgrade pip
python -m pip install prettytable pandas numpy torch torchvision torchaudio matplotlib scikit-learn

# 安装可选依赖
Write-Host "正在安装可选依赖..." -ForegroundColor Yellow
python -m pip install wandb moviepy ffmpeg-python

Write-Host "依赖安装完成!" -ForegroundColor Green
Write-Host "现在您可以运行VALOR的PowerShell脚本了。例如:" -ForegroundColor Cyan
Write-Host ".\scripts\test_valor.ps1" -ForegroundColor Cyan