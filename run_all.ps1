# Start the main training
Start-Process -NoNewWindow -FilePath "python" -ArgumentList "realignr_resume_crossdomain.py"

# Start TensorBoard
Start-Process -NoNewWindow -FilePath "tensorboard" -ArgumentList "--logdir=runs/cpr_reset_sanity"

# Start the watcher script
Start-Process -NoNewWindow -FilePath "python" -ArgumentList "examples/watch_tensorboard_feedback_live.py"

Write-Host "All components launched. Monitor TensorBoard at http://localhost:6006"