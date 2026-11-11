import subprocess; print(subprocess.run(["nvidia-smi"],capture_output=True,text=True).stdout)
