import requests
from datetime import datetime
import numpy as np
        
def model_stealing(path_to_png_file: str):
    SERVER_URL = "http://34.71.138.79:9090"
    endpoint = "/modelstealing"
    TEAM_TOKEN = "X55E27lOG6LS3QRm"
    url = SERVER_URL + endpoint
    with open(path_to_png_file, "rb") as f:
        response = requests.get(url, files={"file": f}, headers={"token": TEAM_TOKEN})
        if response.status_code == 200:
            representation = response.json()["representation"]
            print("Request ok")
            return representation
        else:
            raise Exception(
                f"Model stealing failed. Code: {response.status_code}, content: {response.json()}"
            )
if __name__ == "__main__":

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    files_to_fetch = ["6661858_163933"]
    for file in files_to_fetch:
        representation = model_stealing("modelstealing/data/labels_ids_png/{}.png".format(file))
        np.savez("modelstealing/representations/{}_{}.npz".format(file, timestamp), representation=representation)