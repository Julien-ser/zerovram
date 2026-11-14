# Turn an Intel Iris Xe / Arc iGPU laptop into a local coding-agent box.
# Usage:  powershell -ExecutionPolicy Bypass -File setup.ps1 [-Model qwen3-coder:30b]
param([string]$Model = "")
$ErrorActionPreference = "Stop"

function Need($cmd, $hint) {
    if (-not (Get-Command $cmd -ErrorAction SilentlyContinue)) { throw "$cmd not found. $hint" }
}
Need ollama "Install from https://ollama.com/download"
Need uv     "Install with: powershell -c `"irm https://astral.sh/uv/install.ps1 | iex`""

# 1. Pick a model that fits. The iGPU can use about half of system RAM.
$ramGB = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB)
if (-not $Model) {
    $Model = if ($ramGB -ge 36) { "qwen3-coder:30b" } elseif ($ramGB -ge 16) { "qwen2.5-coder:7b" } else { "qwen2.5-coder:3b" }
}
Write-Host "RAM: $ramGB GB -> model: $Model"

# 2. The actual fix: Ollama ignores integrated GPUs unless both of these are set.
foreach ($kv in @{ OLLAMA_VULKAN = "1"; OLLAMA_IGPU_ENABLE = "1";
                   OLLAMA_API_BASE = "http://127.0.0.1:11434"; AIDER_MODEL = "ollama_chat/$Model" }.GetEnumerator()) {
    [Environment]::SetEnvironmentVariable($kv.Key, $kv.Value, "User")
    Set-Item "env:$($kv.Key)" $kv.Value
}

# 3. Restart Ollama so it picks up the variables.
Stop-Process -Name "ollama app", "ollama" -Force -ErrorAction SilentlyContinue
Start-Sleep 2
$app = Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama app.exe"
if (Test-Path $app) { Start-Process $app } else { Start-Process ollama -ArgumentList serve -WindowStyle Hidden }
Start-Sleep 10

# 4. Verify the iGPU was detected (not just CPU).
$log = Join-Path $env:LOCALAPPDATA "Ollama\server.log"
$gpu = if (Test-Path $log) { Select-String $log -Pattern 'msg="inference compute".*type=iGPU' | Select-Object -Last 1 }
if ($gpu) { Write-Host "OK: iGPU detected -> $(($gpu.Line -split 'description=')[1])" -ForegroundColor Green }
else      { Write-Warning "iGPU not detected. Update your Intel graphics driver, then re-run. Check $log" }

# 5. Model + agent.
ollama pull $Model
uv tool install --python 3.12 aider-chat
Write-Host "`nDone. Open a NEW terminal in your project folder and run:  aider" -ForegroundColor Green
Write-Host "Benchmark:  python bench.py $Model"
