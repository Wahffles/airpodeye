import json

import numpy as np

CONSTANTS_FILE = "notacopy.json"


def load_constants():
    with open(CONSTANTS_FILE, "r") as f:
        data = json.load(f)

        HSV_BOUNDS.MAIN_BOUND_L = np.array(data["HSV_BOUNDS"]["MAIN_BOUND_L"])
        HSV_BOUNDS.MAIN_BOUND_U = np.array(data["HSV_BOUNDS"]["MAIN_BOUND_U"])

def dump_constants():
    with open(CONSTANTS_FILE, "w") as f:
        json.dump(
            { 
                "HSV_BOUNDS": {
                    "MAIN_BOUND_L": HSV_BOUNDS.MAIN_BOUND_L.tolist(),
                    "MAIN_BOUND_U": HSV_BOUNDS.MAIN_BOUND_U.tolist()
                },
            },
            f,
        )

class HSV_BOUNDS:

    MAIN_BOUND_L: np.array # constants.json
    MAIN_BOUND_U: np.array # constants.json
