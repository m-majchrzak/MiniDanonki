import requests

def model_stealing_reset():
    SERVER_URL = "http://34.71.138.79:9090"
    TEAM_TOKEN = "X55E27lOG6LS3QRm"
    endpoint = f"/modelstealing/reset"
    url = SERVER_URL + endpoint
    response = requests.post(url, headers={"token": TEAM_TOKEN})
    if response.status_code == 200:
        print("Request ok")
        print(response.json())
    else:
        raise Exception(
            f"Model stealing reset failed. Code: {response.status_code}, content: {response.json()}"
        )
