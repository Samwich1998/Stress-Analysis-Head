from natsort import natsorted
import subprocess
import os


if __name__ == "__main__":
    # Get the path of the folder where this file is located
    folderPath = os.path.dirname(os.path.abspath(__file__))

    for file in natsorted(os.listdir("./")):
        if not file.endswith('.py'): continue
        if 'global' in file: continue
        if '_' in file: continue

        # Get the full path of the file
        file_path = os.path.join(folderPath, file)

        result = subprocess.run(['python', file_path], shell=False)

        # If an error occurred.
        if result.returncode != 0:
            raise RuntimeError(f"Error running {file_path}")