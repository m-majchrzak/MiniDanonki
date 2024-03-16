import requests
# Be careful. This can be done only once an hour.
# Computing this might take a few minutes. Be patient.
# Make sure your file has proper content.

def defense_submit(path_to_npz_file: str):
    endpoint = "/defense/submit"
    SERVER_URL = "http://34.71.138.79:9090"
    TEAM_TOKEN = "X55E27lOG6LS3QRm"
    url = SERVER_URL + endpoint
    with open(path_to_npz_file, "rb") as f:
        response = requests.post(url, files={"file": f}, headers={"token": TEAM_TOKEN})
        if response.status_code == 200:
            print("Request ok")
            print(response.json())
        else:
            raise Exception(
                f"Defense submit failed. Code: {response.status_code}, content: {response.json()}"
            )
