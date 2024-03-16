import requests
  
def model_stealing_submit(path_to_onnx_file: str):
    SERVER_URL = "http://34.71.138.79:9090"
    TEAM_TOKEN = "X55E27lOG6LS3QRm"
    endpoint = "/modelstealing/submit"
    url = SERVER_URL + endpoint
    with open(path_to_onnx_file, "rb") as f:
        response = requests.post(url, files={"file": f}, headers={"token": TEAM_TOKEN})
        if response.status_code == 200:
            print("Request ok")
            print(response.json())
        else:
            raise Exception(
                f"Model stealing submit failed. Code: {response.status_code}, content: {response.json()}"
            )
