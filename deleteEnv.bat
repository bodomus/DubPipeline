@echo off

REM Удаляем виртуальное окружение, если существует
if exist .venv (
    rmdir /s /q .venv
    echo .venv removed.
) else (
    echo .venv does not exist.
)

pause
