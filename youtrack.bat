@echo off
echo === running curl ===
where curl
echo === request ===
curl -v -X POST https://bodomus.youtrack.cloud/mcp ^
  -H "Authorization: Bearer perm-Ym9kb211cw==.NDgtNA==.FKnJCoWMKp8vmWo2nX07okCLeSngMx" ^
  -H "Content-Type: application/json" ^
  --data "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/list\",\"params\":{}}" > youtrack.txt
echo === exit code: %errorlevel% ===
pause
