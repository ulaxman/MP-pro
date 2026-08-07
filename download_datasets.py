import os
import subprocess

def download_kaggle_dataset(dataset_name, download_path):
    """Download a dataset from Kaggle using the Kaggle CLI."""
    print(f"Downloading {dataset_name} to {download_path}...")
    os.makedirs(download_path, exist_ok=True)
    
    command = f"kaggle datasets download -d {dataset_name} -p {download_path} --unzip"
    
    try:
        subprocess.run(command, shell=True, check=True)
        print(f"Successfully downloaded and extracted {dataset_name}.")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading dataset: {e}")
        print("Please ensure kaggle CLI is installed and configured ( ~/.kaggle/kaggle.json ).")

if __name__ == "__main__":
    datasets = {
        "ASL Alphabet": "grassknoted/asl-alphabet",
        "Sign Language MNIST": "datamunge/sign-language-mnist",
        "FER2013": "msambare/fer2013"
    }
    
    base_dir = "datasets"
    
    for name, kaggle_id in datasets.items():
        dataset_dir = os.path.join(base_dir, name.lower().replace(" ", "_"))
        download_kaggle_dataset(kaggle_id, dataset_dir)
