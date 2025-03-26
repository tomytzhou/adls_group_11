import sys
import os

# Add the ADLS_project directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from kd import KD

def main():
        kd = KD(model_type='bert', checkpoint="bert-base-uncased", dataset_name="xu-song/cc100-samples")
        kd.run_rl_kd()

if __name__ == '__main__':
        main()