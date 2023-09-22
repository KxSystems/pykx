rem CAN NOT RUN A POWERSHELL SCRIPT ON A FD LAPTOP
rem set ROOT=%~dp0
rem powershell.exe -noexit %ROOT%\windows_prep.ps1
"%PYTHON%" -m pip install --no-deps --ignore-installed .
if errorlevel 1 exit 1
