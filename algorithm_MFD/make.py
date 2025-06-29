import numpy as np

signal = np.random.randn(1000)
np.save('data/example_input.npy', signal)
np.save('data/example_output.npy', np.zeros_like(signal))
np.save('data/example_labels.npy', np.zeros_like(signal))
