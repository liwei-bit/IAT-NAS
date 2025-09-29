# lib/procedures/model_builder.py
import sys
sys.path.append("./lib/nas_201_api")
from api import get_cell_based_tiny_net

def get_model_from_arch_str(arch_str, num_classes=100):
    config = {
        "name": "cell_201",
        "num_classes": num_classes,
        "C": 16,
        "N": 5,
        "depth": 8,
    }
    model = get_cell_based_tiny_net(config)
    model.set_cal_mode('dynamic', arch_str)
    return model
