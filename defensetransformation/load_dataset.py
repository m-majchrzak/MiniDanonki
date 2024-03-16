import numpy as np
if __name__ == "__main__":
    data = np.load(
        "defensetransformation/data/DefenseTransformationEvaluate.npz"
    )
    #print(data["labels"], data["representations"].shape)
    #data = np.load("defensetransformation/data/DefenseTransformationSubmit.npz")
    #print(np.unique(data['labels']))
    print(data["representations"][0])
    print(data["labels"][0])