Set-ExecutionPolicy Bypass -Force
Install-Module VcRedist -Force -AllowClobber
mkdir vcredist
$VcRedist = Get-VcList -Export All | Where-Object { $_.Release -eq '2010' -and $_.Architecture -eq 'x64' }
Save-VcRedist -Path 'vcredist' $VcRedist
Install-VcRedist -Path 'vcredist' -Silent $VcRedist
Invoke-WebRequest https://aka.ms/vs/16/release/vs_BuildTools.exe -UseBasicParsing -OutFile 'vs_BuildTools.exe'
./vs_BuildTools.exe --nocache --wait --quiet --norestart --includeRecommended --includeOptional --add Microsoft.VisualStudio.Workload.VCTools
git clone https://github.com/microsoft/vcpkg 'vcpkg'
vcpkg/bootstrap-vcpkg.bat -disableMetrics
vcpkg/vcpkg.exe install dlfcn-win32:x64-windows-static-md
