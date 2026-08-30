[CmdletBinding()]
param(
    [string]$ProjectRoot,
    [string]$PagesOutput,
    [switch]$DryRun,
    [switch]$SkipFullTests
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$script:IsWindows = [System.Environment]::OSVersion.Platform -eq [System.PlatformID]::Win32NT

function Write-Step([string]$Label) {
    Write-Host "[GradLoop] $Label"
}

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)][string]$Label,
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $false)][string[]]$Arguments = @(),
        [switch]$AllowFailure
    )
    Write-Step $Label
    & $FilePath @Arguments | Out-Host
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0 -and -not $AllowFailure) {
        throw "$Label failed with exit code $exitCode"
    }
    return $exitCode
}

function Invoke-InteractiveChecked {
    param(
        [Parameter(Mandatory = $true)][string]$Label,
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $false)][string[]]$Arguments = @()
    )
    Write-Step $Label
    # Do not pipe or redirect this process: Wrangler requires a real terminal
    # for its hidden Secret prompt, and the secret must never enter our script.
    & $FilePath @Arguments
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw "$Label failed with exit code $exitCode"
    }
    return $exitCode
}

function ConvertTo-ProcessArgument([string]$Value) {
    if ($Value -notmatch '[\s"]') { return $Value }
    return '"' + ($Value -replace '(\\*)"', '$1$1\"' -replace '(\\+)$', '$1$1') + '"'
}

function Invoke-Captured {
    param(
        [Parameter(Mandatory = $true)][string]$Label,
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $false)][string[]]$Arguments = @()
    )
    Write-Step $Label
    $info = [System.Diagnostics.ProcessStartInfo]::new()
    $serializedArguments = (($Arguments | ForEach-Object { ConvertTo-ProcessArgument $_ }) -join ' ')
    if ($script:IsWindows -and [System.IO.Path]::GetExtension($FilePath) -ieq '.cmd') {
        $info.FileName = $env:ComSpec
        $info.Arguments = '/d /c call ' + (ConvertTo-ProcessArgument $FilePath)
        if ($serializedArguments) { $info.Arguments += ' ' + $serializedArguments }
    } else {
        $info.FileName = $FilePath
        $info.Arguments = $serializedArguments
    }
    $info.WorkingDirectory = $script:Root
    $info.UseShellExecute = $false
    $info.CreateNoWindow = $true
    $info.RedirectStandardOutput = $true
    $info.RedirectStandardError = $true
    $process = [System.Diagnostics.Process]::new()
    $process.StartInfo = $info
    if (-not $process.Start()) { throw "$Label could not start" }
    $stdoutTask = $process.StandardOutput.ReadToEndAsync()
    $stderrTask = $process.StandardError.ReadToEndAsync()
    $process.WaitForExit()
    $stdout = $stdoutTask.Result
    $stderr = $stderrTask.Result
    if ($process.ExitCode -ne 0) {
        throw "$Label failed with exit code $($process.ExitCode)"
    }
    # Return output to the caller only; never print or persist raw Wrangler output.
    return $stdout
}

function Get-SafeOutput([string]$BasePath, [string]$Purpose) {
    $base = [System.IO.Path]::GetFullPath($BasePath)
    $stamp = Get-Date -Format 'yyyyMMdd-HHmmssfff'
    $candidate = Join-Path $base "$Purpose-$stamp"
    if (Test-Path -LiteralPath $candidate) {
        throw "refusing to reuse output directory: $candidate"
    }
    New-Item -ItemType Directory -Path $candidate -Force | Out-Null
    return $candidate
}

function Get-PythonPath([string]$Root) {
    $local = Join-Path $Root '.venv\Scripts\python.exe'
    if (Test-Path -LiteralPath $local) { return $local }
    $repo = Split-Path (Split-Path $Root -Parent) -Parent
    $sibling = Join-Path $repo '.venv\Scripts\python.exe'
    if (Test-Path -LiteralPath $sibling) { return $sibling }
    throw 'Python virtual environment not found in worktree or sibling repository'
}

function Get-WorkerUrl([string]$Output) {
    $matches = [regex]::Matches($Output, 'https://[A-Za-z0-9.-]+\.workers\.dev(?:/[^\s]*)?')
    if ($matches.Count -eq 0) { throw 'Wrangler output did not contain a public Worker HTTPS URL' }
    $uri = [Uri]$matches[$matches.Count - 1].Value
    if ($uri.Query -or $uri.Fragment -or $uri.Scheme -ne 'https') { throw 'Worker URL is not a query-free HTTPS URL' }
    return $uri.AbsoluteUri.TrimEnd('/')
}

function Get-PagesUrl([string]$Output) {
    $matches = [regex]::Matches($Output, 'https://[A-Za-z0-9.-]+\.pages\.dev(?:/[^\s]*)?')
    if ($matches.Count -eq 0) { throw 'Wrangler output did not contain a public Pages HTTPS URL' }
    $uri = [Uri]$matches[$matches.Count - 1].Value
    if ($uri.Query -or $uri.Fragment -or $uri.Scheme -ne 'https') { throw 'Pages URL is not a query-free HTTPS URL' }
    return $uri.GetLeftPart([System.UriPartial]::Authority).TrimEnd('/')
}

function Get-PublicPagesProjectExists([string]$Npx, [string]$ProjectName) {
    $raw = Invoke-Captured -Label 'inspect Pages project list' -FilePath $Npx -Arguments @('wrangler', 'pages', 'project', 'list', '--json')
    try { $items = $raw | ConvertFrom-Json } catch { throw 'Pages project list was not valid JSON' }
    foreach ($item in @($items)) {
        if ([string]$item.name -eq $ProjectName) { return $true }
    }
    return $false
}

if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    $ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
}
if ([string]::IsNullOrWhiteSpace($PagesOutput)) {
    $PagesOutput = Join-Path $ProjectRoot 'dist\competition\one-key-pages'
}

$script:Root = [System.IO.Path]::GetFullPath($ProjectRoot)
$rootLeaf = Split-Path $script:Root -Leaf
$worktreesLeaf = Split-Path (Split-Path $script:Root -Parent) -Leaf
$repoLeaf = Split-Path (Split-Path (Split-Path $script:Root -Parent) -Parent) -Leaf
if ($rootLeaf -ne 'minicpmo-demo-integration' -or $worktreesLeaf -ne '.worktrees' -or $repoLeaf -ne 'Baoyan-RAGSDK-Agent') {
    throw 'refusing to run outside the expected isolated worktree layout'
}
if (-not (Test-Path -LiteralPath $script:Root -PathType Container)) { throw 'ProjectRoot does not exist' }

$branch = (git -C $script:Root branch --show-current).Trim()
if ($branch -ne 'feature/minicpmo-demo-integration') { throw 'unexpected Git branch' }
$remoteText = (git -C $script:Root remote -v | Out-String).Trim()
if ($remoteText) { throw 'Git remote is configured; refusing to continue' }

$python = Get-PythonPath $script:Root
$npx = if ($script:IsWindows) {
    (Get-Command 'npx.cmd' -ErrorAction Stop).Source
} else {
    (Get-Command 'npx' -ErrorAction Stop).Source
}
$env:npm_config_yes = 'true'
$config = Join-Path $script:Root 'deploy\map-realtime-worker\wrangler.one-key.jsonc'
$placeholderProxy = 'wss://placeholder.invalid/ws'
$projectName = 'gradloop-ragsdk-omni'
$workerName = 'gradloop-map-realtime-proxy'
$plannedPagesOrigin = 'https://gradloop-ragsdk-omni.pages.dev'
$startedAt = (Get-Date).ToUniversalTime().ToString('o')

Write-Step 'local preflight: runtime and repository safety'
if (-not (Get-Command $python -ErrorAction SilentlyContinue)) { throw 'configured Python executable is unavailable' }
if (-not (Get-Command $npx -ErrorAction SilentlyContinue)) { throw 'npx is unavailable' }

if (-not $SkipFullTests) {
    Invoke-Checked -Label 'focused Python tests' -FilePath $python -Arguments @('-m', 'pytest', 'tests/competition/test_one_key_map_setup.py', 'tests/competition/test_public_site_builder.py', 'tests/competition/test_map_realtime_smoke.py', '-q') | Out-Null
}
Invoke-Checked -Label 'Python lint' -FilePath $python -Arguments @('-m', 'ruff', 'check', 'scripts/competition/map_setup_core.py', 'scripts/competition/build_public_site.py', 'scripts/competition/smoke_map_realtime.py', 'tests/competition/test_one_key_map_setup.py') | Out-Null
Invoke-Checked -Label 'browser module tests' -FilePath 'node' -Arguments @('--test', 'web/tests/*.test.mjs') | Out-Null
Invoke-Checked -Label 'Worker module tests' -FilePath 'node' -Arguments @('--test', 'deploy/map-realtime-worker/test/*.test.mjs') | Out-Null
Invoke-Checked -Label 'repository public release scan' -FilePath $python -Arguments @('scripts/public_release_scan.py', '--root', '.') | Out-Null
Invoke-Checked -Label 'Git whitespace check' -FilePath 'git' -Arguments @('diff', '--check') | Out-Null

$validationOutput = Get-SafeOutput -BasePath $PagesOutput -Purpose 'validation'
Invoke-Checked -Label 'build validation-only Pages site' -FilePath $python -Arguments @('scripts/competition/build_public_site.py', '--output', $validationOutput, '--proxy-url', $placeholderProxy, '--project-url', $plannedPagesOrigin) | Out-Null
Invoke-Checked -Label 're-scan repository release allowlist after validation build' -FilePath $python -Arguments @('scripts/public_release_scan.py', '--root', '.') | Out-Null

Write-Host "Worker: $workerName"
Write-Host "Pages: $projectName"
Write-Host 'MAP host: minicpmo45.modelbest.cn'
Write-Host "Planned Pages origin: $plannedPagesOrigin"
if ($DryRun) {
    Write-Host 'DryRun complete; no Cloudflare login, deploy, Pages upload or Secret write was attempted.'
    exit 0
}

$confirmation = Read-Host '本地检查已通过。若确认外部部署，请输入 DEPLOY（其他输入将安全退出）'
if ($confirmation -cne 'DEPLOY') {
    Write-Host '未输入 DEPLOY，未执行任何外部操作。'
    exit 0
}

$whoamiExit = Invoke-Checked -Label 'Cloudflare identity check' -FilePath $npx -Arguments @('wrangler', 'whoami') -AllowFailure
if ($whoamiExit -ne 0) {
    Invoke-Checked -Label 'Cloudflare browser authorization' -FilePath $npx -Arguments @('wrangler', 'login') | Out-Null
    Invoke-Checked -Label 'Cloudflare identity re-check' -FilePath $npx -Arguments @('wrangler', 'whoami') | Out-Null
}

$workerOutput = Invoke-Captured -Label 'deploy public Worker configuration' -FilePath $npx -Arguments @('wrangler', 'deploy', '--config', $config)
$workerUrl = Get-WorkerUrl $workerOutput
$workerWs = ([Uri]$workerUrl).GetLeftPart([System.UriPartial]::Authority).Replace('https://', 'wss://') + '/ws'

Write-Host '请在 Wrangler 隐藏输入框中粘贴 API Key；不要粘贴到聊天。'
Invoke-InteractiveChecked -Label 'set encrypted MAP Worker Secret interactively' -FilePath $npx -Arguments @('wrangler', 'secret', 'put', 'MAP_API_KEY', '--config', $config, '--name', $workerName) | Out-Null

$finalOutput = Get-SafeOutput -BasePath $PagesOutput -Purpose 'final'
Invoke-Checked -Label 'build final Pages site with public Worker URL' -FilePath $python -Arguments @('scripts/competition/build_public_site.py', '--output', $finalOutput, '--proxy-url', $workerWs, '--project-url', $plannedPagesOrigin) | Out-Null
Invoke-Checked -Label 're-scan repository release allowlist after final build' -FilePath $python -Arguments @('scripts/public_release_scan.py', '--root', '.') | Out-Null

if (-not (Get-PublicPagesProjectExists -Npx $npx -ProjectName $projectName)) {
    Invoke-Checked -Label 'create Pages direct-upload project' -FilePath $npx -Arguments @('wrangler', 'pages', 'project', 'create', $projectName, '--production-branch', 'main') | Out-Null
}
$pagesOutputText = Invoke-Captured -Label 'deploy final Pages site' -FilePath $npx -Arguments @('wrangler', 'pages', 'deploy', $finalOutput, '--project-name', $projectName)
$pagesUrl = Get-PagesUrl $pagesOutputText
if ($pagesUrl -ne $plannedPagesOrigin) {
    throw "Pages origin mismatch: expected $plannedPagesOrigin"
}

$health = Invoke-WebRequest -Uri "$workerUrl/health" -UseBasicParsing
if ($health.StatusCode -ne 200) { throw 'Worker health check did not return HTTP 200' }
$healthBody = $health.Content | ConvertFrom-Json
if ($healthBody.status -ne 'ready' -or $healthBody.upstream -ne 'configured') { throw 'Worker health payload is not ready/configured' }

$smokeOutput = Join-Path $script:Root 'reports\local\competition\one-key-map-chat.json'
Invoke-Checked -Label 'run redacted chat smoke' -FilePath $python -Arguments @('scripts/competition/smoke_map_realtime.py', '--proxy-url', $workerWs, '--origin', $plannedPagesOrigin, '--mode', 'chat', '--asset-root', 'dist/competition/assets', '--output', $smokeOutput) | Out-Null
$smoke = Get-Content -LiteralPath $smokeOutput -Raw | ConvertFrom-Json
$finishedAt = (Get-Date).ToUniversalTime().ToString('o')
$summary = [ordered]@{
    schema_version = 'one-key-map-setup.v1'
    worker_url = $workerUrl
    pages_url = $pagesUrl
    health_status = [int]$health.StatusCode
    chat_smoke = if ($smoke.passed) { 'passed' } else { 'failed' }
    started_at = $startedAt
    finished_at = $finishedAt
    close_reason = [string]$smoke.close_reason
}
$summaryPath = Join-Path $script:Root 'reports\local\competition\one-key-map-latest.json'
New-Item -ItemType Directory -Path (Split-Path $summaryPath -Parent) -Force | Out-Null
$summary | ConvertTo-Json | Set-Content -LiteralPath $summaryPath -Encoding UTF8
Write-Host ('完成：Worker ' + $workerUrl + '；Pages ' + $pagesUrl + '；health ready；chat smoke ' + $summary.chat_smoke + '。')
