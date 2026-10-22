# AutoLesson Windows 一键部署：检查/安装 Python，创建 .venv，安装依赖
param([switch]$CheckOnly)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$PipIndexCN = "https://pypi.tuna.tsinghua.edu.cn/simple"
$PipIndexOfficial = "https://pypi.org/simple"
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}

function Step($text) { Write-Host "`n== $text ==" -ForegroundColor Cyan }
function Ok($text)   { Write-Host "  [OK] $text" -ForegroundColor Green }
function Warn($text) { Write-Host "  [!] $text" -ForegroundColor Yellow }

function Test-PythonExe([string]$exe) {
    if (-not $exe -or -not (Test-Path $exe)) { return $false }
    try {
        $ver = & $exe -c "import sys, tkinter; print(str(sys.version_info.major)+'.'+str(sys.version_info.minor))" 2>$null
        if ($LASTEXITCODE -ne 0 -or -not $ver) { return $false }
        $v = [version]($ver.Trim())
        # 项目依赖（onnxruntime/numpy 的 Windows wheel）在 3.13/3.14 上经常缺包，
        # 固定使用 3.10~3.12，优先 3.12。
        return ($v -ge [version]"3.10" -and $v -lt [version]"3.13")
    } catch { return $false }
}

function Find-Python {
    $candidates = @()
    if (Get-Command py -ErrorAction SilentlyContinue) {
        foreach ($v in @("3.12", "3.11", "3.10")) {
            try {
                $p = & py "-$v" -c "import sys; print(sys.executable)" 2>$null | Select-Object -First 1
                if ($LASTEXITCODE -eq 0 -and $p) { $candidates += $p.Trim() }
            } catch {}
        }
    }
    foreach ($name in @("python", "python3")) {
        $cmd = Get-Command $name -ErrorAction SilentlyContinue
        if ($cmd -and $cmd.Source -notlike "*WindowsApps*") { $candidates += $cmd.Source }
    }
    $candidates += @(
        "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
        "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe",
        "$env:LOCALAPPDATA\Programs\Python\Python310\python.exe",
        "C:\Python312\python.exe",
        "C:\Python311\python.exe"
    )
    foreach ($p in ($candidates | Select-Object -Unique)) {
        if (Test-PythonExe $p) { return $p }
    }
    return $null
}

function Install-Python {
    Step "未找到兼容的 Python 3.10~3.12，尝试自动安装 Python 3.12"
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        try {
            Write-Host "  使用 winget 安装，可能需要几分钟..."
            & winget install --id Python.Python.3.12 -e --silent --accept-package-agreements --accept-source-agreements | Out-Host
            if ($LASTEXITCODE -eq 0) {
                $p = Find-Python
                if ($p) { return $p }
            }
        } catch { Warn "winget 安装失败，改用官方安装包" }
    }

    $installer = Join-Path $env:TEMP "autolesson-python-3.12.10.exe"
    Remove-Item $installer -Force -ErrorAction SilentlyContinue
    $urls = @(
        "https://mirrors.tuna.tsinghua.edu.cn/python/3.12.10/python-3.12.10-amd64.exe",
        "https://mirrors.huaweicloud.com/python/3.12.10/python-3.12.10-amd64.exe",
        "https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe"
    )
    $downloaded = $false
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    foreach ($url in $urls) {
        try {
            Write-Host "  下载 $url"
            Invoke-WebRequest -Uri $url -OutFile $installer -UseBasicParsing -TimeoutSec 120
            if ((Test-Path $installer) -and (Get-Item $installer).Length -gt 10MB) {
                $downloaded = $true
                break
            }
        } catch { Warn "该镜像下载失败，换下一个源" }
    }
    if (-not $downloaded) { throw "Python 安装包下载失败，请检查网络后重试。" }
    Write-Host "  静默安装到当前用户，不需要管理员权限..."
    $args = @("/quiet", "InstallAllUsers=0", "PrependPath=1", "Include_launcher=1", "Include_test=0", "Include_pip=1")
    $proc = Start-Process -FilePath $installer -ArgumentList $args -Wait -PassThru
    Remove-Item $installer -Force -ErrorAction SilentlyContinue
    if ($proc.ExitCode -ne 0 -and $proc.ExitCode -ne 3010) {
        throw "Python 安装程序返回错误码 $($proc.ExitCode)。请手动安装 Python 3.12 后重新运行。"
    }
    $p = Find-Python
    if (-not $p) {
        throw "Python 安装完成但未找到 python.exe。请注销/重启 Windows 后重新运行；或手动安装 Python 3.12（勾选 Add python.exe to PATH）。"
    }
    return $p
}

Step "AutoLesson 一键部署"
Write-Host "项目目录: $Root"
$python = Find-Python
if (-not $python) { $python = Install-Python }
Ok "Python: $python"
if ($CheckOnly) {
    $checkVenv = Join-Path $Root ".venv\Scripts\python.exe"
    if (Test-Path $checkVenv) {
        & $checkVenv -c "import cv2, numpy, mss, pyautogui, rapidocr, onnxruntime, yaml, requests, tkinter; print('CHECK_ONLY_OK')"
    } else {
        Write-Host "CHECK_ONLY: .venv 尚未创建"
    }
    exit 0
}
Step "创建虚拟环境 .venv"
$venv = Join-Path $Root ".venv"
$venvPy = Join-Path $venv "Scripts\python.exe"
if (Test-Path $venvPy) {
    if (-not (Test-PythonExe $venvPy)) {
        $broken = "$venv.broken_" + (Get-Date -Format "yyyyMMddHHmmss")
        Warn "现有 .venv 不可用，重命名为 $broken"
        Rename-Item $venv $broken
    }
}
if (-not (Test-Path $venvPy)) {
    & $python -m venv $venv
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $venvPy)) {
        throw "创建虚拟环境失败。请确认磁盘可写，且 Python 安装包含 venv 模块。"
    }
}
Ok "虚拟环境: $venv"

Step "升级 pip / setuptools / wheel（默认清华源）"
& $venvPy -m pip install --upgrade pip setuptools wheel -i $PipIndexCN --timeout 60 --retries 3 --disable-pip-version-check
if ($LASTEXITCODE -ne 0) {
    Warn "清华源升级 pip 失败，改用官方源重试"
    & $venvPy -m pip install --upgrade pip setuptools wheel -i $PipIndexOfficial --timeout 60 --retries 3 --disable-pip-version-check
}
if ($LASTEXITCODE -ne 0) { Warn "pip 升级失败，继续尝试安装依赖" }

Step "安装项目依赖（默认清华源，首次约 5~15 分钟）"
& $venvPy -m pip install -r (Join-Path $Root "requirements.txt") -i $PipIndexCN --timeout 60 --retries 3 --disable-pip-version-check
if ($LASTEXITCODE -ne 0) {
    Warn "清华源安装失败，改用官方 PyPI 重试"
    & $venvPy -m pip install -r (Join-Path $Root "requirements.txt") -i $PipIndexOfficial --timeout 60 --retries 3 --disable-pip-version-check
}
if ($LASTEXITCODE -ne 0) { throw "依赖安装失败。请检查网络/代理后重新运行本脚本。" }

Step "验证安装"
$check = @"
import sys
import cv2, numpy, mss, pyautogui, rapidocr, onnxruntime, yaml, requests, tkinter
print('Python', sys.version.split()[0])
print('OpenCV', cv2.__version__)
print('RapidOCR import OK')
"@
& $venvPy -c $check
if ($LASTEXITCODE -ne 0) { throw "依赖验证失败，请把上面的错误信息发给维护者。" }

$configPath = Join-Path $Root "config.yaml"
$examplePath = Join-Path $Root "config.example.yaml"
if (-not (Test-Path $configPath) -and (Test-Path $examplePath)) {
    Copy-Item $examplePath $configPath
    Ok "已从 config.example.yaml 创建本地 config.yaml"
}
if (-not (Test-Path $configPath)) {
    Warn "未找到 config.yaml/config.example.yaml，请从项目压缩包恢复后再运行。"
}

Ok "部署完成"
Write-Host ""
Write-Host "下一步：" -ForegroundColor Cyan
Write-Host "  1. 双击 启动GUI.bat"
Write-Host "  2. 在“大模型”页填写 api_key（DeepSeek 平台申请）"
Write-Host "  3. 先打开“启动复刻测试页.bat”做一次试运行"
Write-Host "  4. 正式使用时打开智慧树课程页，选择模式后点“开始”"
Write-Host ""
Write-Host "停止：F9 或 ESC；GUI 也可点“停止”。" -ForegroundColor Yellow
