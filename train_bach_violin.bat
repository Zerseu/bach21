@echo off
rem ---------------------------------------------------------------------------------------------
rem Trains the note model on Bach's violin parts.
rem
rem On first use it creates a virtual environment in %USERPROFILE%\.venvs\bach21 (your user
rem folder, not this one) and installs the dependencies into it.
rem The trained model is saved to bach21data\bach\violin\note_model.torch, the losses per epoch
rem to bach21data\bach\violin\note_log.csv. If a model already exists it is kept; pass
rem --retrain to train again:   train_bach_violin.bat --retrain
rem Settings are in the [note] section of config.ini.
rem ---------------------------------------------------------------------------------------------
setlocal
cd /d "%~dp0"

call "%~dp0scripts\setup_venv.bat"
if errorlevel 1 goto :failed

"%VENV_PYTHON%" train.py bach violin %*
if errorlevel 1 goto :failed

echo.
echo Training finished. The model is bach21data\bach\violin\note_model.torch
echo Run generate_bach_violin.bat to compose a piece with it.
set "EXIT_CODE=0"
goto :end

:failed
echo.
echo Training failed, see the messages above.
set "EXIT_CODE=1"

:end
rem Keep the window open when the script was started by double-clicking it.
echo %CMDCMDLINE% | findstr /i /c:" /c " >nul && pause
endlocal & exit /b %EXIT_CODE%
