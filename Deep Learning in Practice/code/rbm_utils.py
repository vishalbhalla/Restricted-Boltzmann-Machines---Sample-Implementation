import numpy as np

# Activation Function
def sigmoid(x):
    return 1. / (1. + np.exp(-1. * x))

# Error on Cross validation dataset.
def validation_error(valid_data, weights, hid_biases, vis_biases):
	# propagate up
    hid_input = np.dot(valid_data, weights) + hid_biases
    hid_states = sigmoid(hid_input) > np.random.uniform(size=hid_input.shape)
    # propagate down
    vis_probs = sigmoid(np.dot(hid_states, weights.T) + vis_biases)
    valid_err = np.sum((valid_data - vis_probs) ** 2) / valid_data.shape[0]
    return valid_err, vis_probs