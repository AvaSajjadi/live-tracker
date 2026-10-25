param([Parameter(Mandatory = $true)][string]$Source, [Parameter(Mandatory = $true)][string]$Destination)
# Make an H.264 copy of a video with the converter that is built into Windows (Media Foundation). Nothing is installed.
$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Runtime.WindowsRuntime
$null = [Windows.Storage.StorageFile, Windows.Storage, ContentType = WindowsRuntime]
$null = [Windows.Storage.StorageFolder, Windows.Storage, ContentType = WindowsRuntime]
$null = [Windows.Media.Transcoding.MediaTranscoder, Windows.Media.Transcoding, ContentType = WindowsRuntime]
$null = [Windows.Media.MediaProperties.MediaEncodingProfile, Windows.Media.MediaProperties, ContentType = WindowsRuntime]

$methods = [System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object { $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 }
$asTaskOp = $methods | Where-Object { $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1' } | Select-Object -First 1
$asTaskProgress = $methods | Where-Object { $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncActionWithProgress`1' } | Select-Object -First 1

function Await-Op($op, [Type]$type) {
    $task = $asTaskOp.MakeGenericMethod($type).Invoke($null, @($op))
    $null = $task.Wait(-1)
    return $task.Result
}

$srcPath = (Resolve-Path -LiteralPath $Source).Path
$dstFull = [System.IO.Path]::GetFullPath($Destination)
$dstFolder = Split-Path -Parent $dstFull
$dstName = Split-Path -Leaf $dstFull

$srcFile = Await-Op ([Windows.Storage.StorageFile]::GetFileFromPathAsync($srcPath)) ([Windows.Storage.StorageFile])
$folder = Await-Op ([Windows.Storage.StorageFolder]::GetFolderFromPathAsync($dstFolder)) ([Windows.Storage.StorageFolder])
$dstFile = Await-Op ($folder.CreateFileAsync($dstName, [Windows.Storage.CreationCollisionOption]::ReplaceExisting)) ([Windows.Storage.StorageFile])

# the source's own size and frame rate, written as H.264 in an mp4 without sound
$profile = Await-Op ([Windows.Media.MediaProperties.MediaEncodingProfile]::CreateFromFileAsync($srcFile)) ([Windows.Media.MediaProperties.MediaEncodingProfile])
$profile.Video.Subtype = "H264"
$profile.Video.Bitrate = 3000000
$profile.Audio = $null
$profile.Container.Subtype = "MPEG4"

$transcoder = New-Object Windows.Media.Transcoding.MediaTranscoder
$prep = Await-Op ($transcoder.PrepareFileTranscodeAsync($srcFile, $dstFile, $profile)) ([Windows.Media.Transcoding.PrepareTranscodeResult])
if (-not $prep.CanTranscode) { throw "Windows cannot convert this video: $($prep.FailureReason)" }
$task = $asTaskProgress.MakeGenericMethod([double]).Invoke($null, @($prep.TranscodeAsync()))
$null = $task.Wait(-1)
if ($task.IsFaulted) { throw $task.Exception.InnerException }
"converted: $dstFull ($((Get-Item -LiteralPath $dstFull).Length) bytes)"
