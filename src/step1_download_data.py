import requests
from tqdm import tqdm
import os


def download_file(url, save_path):
    """
    Downloads a file from the given URL and saves it to the specified path.
    """
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(save_path, "wb") as file:
        for chunk in tqdm(response.iter_content(chunk_size=1024),
                          desc=f"Downloading {url}",
                          unit="B",
                          unit_scale=True,
                          unit_divisor=1024,
                          total=int(response.headers.get("content-length", 0))):
            file.write(chunk)


def download_files(file_paths):
    """
    Downloads multiple files from the given URLs and saves them to the specified paths.
    """
    print("Downloading files:\n")
    os.makedirs("data", exist_ok=True)

    for file_url, _ in file_paths:
        try:
            save_path = os.path.join("data", os.path.basename(file_url))
            download_file(file_url, save_path)
            print(f"\nDownloaded {file_url} to {save_path}")
        except Exception as e:
            print(f"\nError downloading {file_url}: {e}")


if __name__ == "__main__":
    files = [
        ('https://assets-datascientest.s3.eu-west-1.amazonaws.com/'
         'drift_monitoring/green_tripdata_2022-01.parquet', 'data'),
        ('https://assets-datascientest.s3.eu-west-1.amazonaws.com/'
         'drift_monitoring/green_tripdata_2022-02.parquet', 'data')
    ]
    download_files(files)
