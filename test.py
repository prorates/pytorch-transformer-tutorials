#!/usr/bin/env python3

from config import get_config
from config import get_device

from tutorial1 import debug_code_model1
from tutorial2 import debug_code_model2
from tutorial3 import debug_code_model3
from tutorial4 import debug_code_model4
from tutorial5 import debug_code_model5
from tutorial6 import debug_code_model6
from tutorial7 import debug_code_model7

if __name__ == "__main__":
    # warnings.filterwarnings('ignore')
    config = get_config()
    device = get_device()

    debug_code_model1(config, device)
    debug_code_model2(config, device)
    debug_code_model3(config, device)
    debug_code_model4(config, device)
    debug_code_model5(config, device)
    debug_code_model6(config, device)
    debug_code_model7(config, device)
