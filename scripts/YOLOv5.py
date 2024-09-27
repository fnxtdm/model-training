
import os
import sys
import subprocess
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.config import tmp_dir, image_dir

# set WANDB_DISABLED=true
#python ../YOLOv5/train.py --img 640 --batch 16 --epochs 50 --data ../data/thinos9.yaml --weights yolov5s.pt
#python.exe ../YOLOv5/detect.py --weights ../YOLOv5/runs/train/exp2/weights/best.pt --img 640 --conf 0.25 --source ../datasets/raw

def run_python_file(python_file, *args):
    """
    Runs a Python file with specified arguments.

    Args:
        python_file (str): Path to the Python file.
        *args: Variable length argument list.

    Returns:
        None

    Raises:
        subprocess.CalledProcessError: If the script raises an error.
    """
    if os.path.exists(python_file):
        try:
            subprocess.run([
                sys.executable, python_file,
                *args
            ], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running the script: {e}")
    else:
        print(f"The file {python_file} does not exist in the current directory.")


if __name__ == "__main__":
    run_python_file('../YOLOv5/train.py',
                    '--img', '640',
                    '--batch', '1',
                    '--epochs', '100',
                    '--data', '../data/thinos9.yaml',
                    '--weights', 'yolov5x.pt',
                    '--exist-ok'
                    )

    run_python_file('../YOLOv5/detect.py',
                    '--img', '640',
                    '--conf', '0.25',
                    '--source', image_dir,
                    '--weights', '../YOLOv5/runs/train/exp/weights/best.pt'
                    )