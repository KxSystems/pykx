python .\parse_tests.py

cd ..

python -m pytest -vvv -n 0 --no-cov --junitxml=licensed_report.xml .\tests\win_tests\lic\licensed_tests.py
SET /A licensed =  %ERRORLEVEL%
python -m pytest -vvv -n 0 --no-cov --junitxml=unlicensed_report.xml .\tests\win_tests\unlic\unlicensed_tests.py
SET /A unlicensed =  %ERRORLEVEL%
python -m pytest -vvv -n 0 --no-cov --junitxml=ipc_licensed_report.xml .\tests\win_tests\ipc_lic\ipc_licensed_tests.py
SET /A ipc_licensed =  %ERRORLEVEL%
python -m pytest -vvv -n 0 --no-cov --junitxml=ipc_unlicensed_report.xml .\tests\win_tests\ipc_unlic\ipc_unlicensed_tests.py
SET /A ipc_unlicensed =  %ERRORLEVEL%
python -m pytest -vvv -n 0 --no-cov --junitxml=embedded_report.xml .\tests\win_tests\embedded\embedded_tests.py
SET /A embedded =  %ERRORLEVEL%
python -m pytest -vvv -n 0 --no-cov --junitxml=nep_licensed_report.xml .\tests\win_tests\nep_lic\nep_licensed_tests.py
SET /A nep_licensed =  %ERRORLEVEL%
python -m pytest -vvv -n 0 --no-cov --junitxml=nep_unlicensed_report.xml .\tests\win_tests\nep_unlic\nep_unlicensed_tests.py
SET /A nep_unlicensed =  %ERRORLEVEL%
python -m pytest -vvv -n 0 --no-cov --junitxml=pandas_licensed_report.xml .\tests\win_tests\pandas_lic\pandas_licensed_tests.py
SET /A pandas_licensed =  %ERRORLEVEL%
IF %licensed% NEQ 0 (
    exit %licensed%
)
IF %unlicensed% NEQ 0 (
    exit %unlicensed%
)
IF %ipc_licensed% NEQ 0 (
    exit %ipc_licensed%
)
IF %ipc_unlicensed% NEQ 0 (
    exit %ipc_unlicensed%
)
IF %embedded% NEQ 0 (
    exit %embedded%
)
IF %nep_licensed% NEQ 0 (
    exit %nep_licensed%
)
IF %nep_unlicensed% NEQ 0 (
    exit %nep_unlicensed%
)
IF %pandas_licensed% NEQ 0 (
    exit %pandas_licensed%
)

