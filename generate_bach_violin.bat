@echo off
rem ---------------------------------------------------------------------------------------------
rem Composes a new 1000-note piece with the model that train_bach_violin.bat trained.
rem
rem On first use it creates .venv in this folder and installs the dependencies into it.
rem The piece is written to bach21data\bach\violin\ as note_output.txt (the tokens),
rem midi_output.mid and xml_output.musicxml (open it in MuseScore or any score editor).
rem Extra options are passed on, for example a different piece:   generate_bach_violin.bat --seed 7
rem or a different length:   generate_bach_violin.bat --length 500
rem ---------------------------------------------------------------------------------------------
setlocal
cd /d "%~dp0"

if not exist "bach21data\bach\violin\note_model.torch" (
    echo There is no trained model yet. Run train_bach_violin.bat first.
    goto :failed
)

call "%~dp0scripts\setup_venv.bat"
if errorlevel 1 goto :failed

"%VENV_PYTHON%" generate.py bach violin --length 1000 %*
if errorlevel 1 goto :failed

echo.
echo Done. Open bach21data\bach\violin\midi_output.mid to listen to the piece.
set "EXIT_CODE=0"
goto :end

:failed
echo.
echo Generation failed, see the messages above.
set "EXIT_CODE=1"

:end
rem Keep the window open when the script was started by double-clicking it.
echo %CMDCMDLINE% | findstr /i /c:" /c " >nul && pause
endlocal & exit /b %EXIT_CODE%
