import numpy as np

def process(config: dict, signal: np.ndarray, labels: Optional[np.ndarray] = None) -> np.ndarray:
    template_signal = np.array(config['template_signal'])
    matched_filter_output = np.correlate(signal, template_signal, mode='same')
    
    threshold = config['detection_threshold']
    detection_result = matched_filter_output > threshold
    
    return detection_result.astype(int)
