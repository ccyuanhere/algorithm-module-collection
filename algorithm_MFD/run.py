from typing import Dict, Any, Optional
import numpy as np
from model import process
import json

def run(input_data: Dict[str, Any], config_path: Optional[str] = None) -> Dict[str, Any]:
    if config_path:
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        with open('config.json', 'r') as f:
            config = json.load(f)
    
    signal = input_data['signal']
    labels = input_data.get('labels')
    
    result = process(config, signal, labels)
    
    return {
        "result": result,
        "metrics": {},
        "log": "Matched filter detection completed successfully.",
        "success": True
    }
