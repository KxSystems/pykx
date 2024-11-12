Invoke-WebRequest https://aka.ms/vs/16/release/vs_BuildTools.exe -UseBasicParsing -OutFile 'vs_BuildTools.exe'
./vs_BuildTools.exe --nocache --wait --quiet --norestart --includeRecommended --includeOptional --add Microsoft.VisualStudio.Workload.VCTools
if(Test-Path -Path .\vcpkg){
    Remove-Item -Recurse -Force .\vcpkg
}
git clone https://github.com/microsoft/vcpkg 'vcpkg'
vcpkg/bootstrap-vcpkg.bat -disableMetrics
vcpkg/vcpkg.exe install dlfcn-win32:x64-windows-static-md
