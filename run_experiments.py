import subprocess
import argparse

if __name__ == "__main__":
    config = f"./configs/train_eval.yaml"
    subprocess_input = [
        "python",
        f"anomaly_detection/MADDR_dpa/main.py",
        "train_eval",
        config
    ]
    subprocess_input += [config]
    # print(subprocess_input)
    subprocess.run(subprocess_input)