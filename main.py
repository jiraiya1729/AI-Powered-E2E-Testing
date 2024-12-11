import subprocess
import os

process = subprocess.Popen(
    ["flask", "run",],
    env={"FLASK_APP": "w.py", "FLASK_ENV": "development", **os.environ}
)