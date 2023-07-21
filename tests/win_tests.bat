python .\parse_tests.py

cd ..

pytest .\tests\win_tests\lic\licensed_tests.py && pytest .\tests\win_tests\unlic\unlicensed_tests.py && pytest .\tests\win_tests\ipc_lic\ipc_licensed_tests.py && pytest .\tests\win_tests\ipc_unlic\ipc_unlicensed_tests.py && pytest .\tests\win_tests\embedded\embedded_tests.py && pytest .\tests\win_tests\nep_lic\nep_licensed_tests.py && pytest .\tests\win_tests\nep_unlic\nep_unlicensed_tests.py && pytest .\tests\win_tests\pandas_lic\pandas_licensed_tests.py
