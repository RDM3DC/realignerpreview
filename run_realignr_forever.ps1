# PowerShell script to launch RealignR components

# Start the API server
Start-Process -NoNewWindow -FilePath "powershell" -ArgumentList "-Command uvicorn api_server:app --port 8080"

# Start the watcher
Start-Process -NoNewWindow -FilePath "python" -ArgumentList "watcher_gpt.py"

# Start the training loop
Start-Process -NoNewWindow -FilePath "python" -ArgumentList "realignr_train.py"

Write-Host "All components launched. Monitor logs for progress."
