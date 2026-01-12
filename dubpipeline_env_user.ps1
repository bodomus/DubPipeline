# dubpipeline_env_user.ps1
# Запускать в PowerShell (обычно можно без админа) — установит env для ТЕКУЩЕГО ПОЛЬЗОВАТЕЛЯ.

[Environment]::SetEnvironmentVariable("DUBPIPELINE_TTS_MAX_RU_CHARS", "260", "User")
[Environment]::SetEnvironmentVariable("DUBPIPELINE_TTS_TRY_SINGLE_CALL", "1", "User")
[Environment]::SetEnvironmentVariable("DUBPIPELINE_TTS_TRY_SINGLE_CALL_MAX_CHARS", "1200", "User")

Write-Host "OK. User environment variables set. Restart PyCharm to apply." -ForegroundColor Green
