import os
import subprocess

file_name = input("Enter the file name: ")

if not file_name.endswith(".py"):
    file_name += ".py"

file_path = os.path.join("source", file_name)

if os.path.exists(file_path):
    subprocess.run(["python", file_path])
else:
    print(f"{file_name} not found!")