import numpy as np

from numpy.random import choice, uniform, randn

import input_data
import plotting as plt
from plotting import ErrorPlot, ReconstructionPlot, ReceptiveFieldsPlot, ActivationPlot

from rbm_utils import sigmoid, validation_error

#
# Model Parameters
#

# Set the number of visible units
n_vis = 784
# Set the number of hidden units
n_hid = 250

#
# Training Parameters
#

# Set the number of epochs
# Unsupervised Training

n_epochs = 500
# Set the size a single batch
batch_size = 50
# Set the learning rate
lr = 0.1
# Set momentum for training
momentum = 0.5
# Set weight decay
decay = 0.0002

#
# Network Initialization
#

# Initialize the weights to random values
weights = 0.001 * randn(n_vis, n_hid)
# Initialize the hidden biases
hid_biases = np.zeros((1, n_hid))
# Initialize the visible biases
vis_biases = np.zeros((1, n_vis))
# Variable to save the weight deltas for momentum
weights_inc = np.zeros((n_vis, n_hid))
# Variable to save the hidden bias deltas for momentum
hid_biases_inc = np.zeros((1, n_hid))
# Variable to save the visible bias deltas for momentum
vis_biases_inc = np.zeros((1, n_vis))

#
# Obtain data set
#

# Load MNIST data set. The original dataset is converted into black and white (greyscale=False),
# so that we can use it with a binary Restricted Boltzmann Machine
data_sets = input_data.read_data_sets('MNIST_data', one_hot=True, greyscale=False)
# Get the number of minibatches
n_minibatches = data_sets.train.num_minibatches(batch_size)

#
# Plotting Setup
#

# Allow interactvie plotting
plt.allow_plot()
# Since we can't plot all the images in a batch, we pick some indices 
# that we use throughout the training
train_recon_indices = choice(range(batch_size), 5, replace=False)
valid_recon_indices = choice(range(data_sets.validation.num_examples), 5, replace=False)
recept_indices = choice(range(n_hid), 20, replace=False)
# Initialize the plots
error_plot = ErrorPlot(1)
recon_valid_plot = ReconstructionPlot(2)
recept_fields_plot = ReceptiveFieldsPlot(3, width=5, height=4)
activation_plot = ActivationPlot(4)

#
# Training
#


# K Step Contrasting Divergence.

for i_epoch in range(n_epochs):
    train_err = 0.0

    # Step 4
    for i_minibatch in range(n_minibatches):
        batch = data_sets.train.next_batch(batch_size, no_labels=True)

        #
        # Start Gibbs Step
        #
        pos_hid_inp = np.dot(batch, weights) + hid_biases
        pos_hid_probs = sigmoid(pos_hid_inp)

        # Hidden states will have binary values, that is 0 or 1.
        pos_hid_states = pos_hid_probs > uniform(size=pos_hid_probs.shape)

        # Calculate terms for Step 7 to 10.
        pos_prods = np.dot(batch.T, pos_hid_probs)

        pos_hid_act = np.sum(pos_hid_probs, axis=0)
        pos_vis_act = np.sum(batch, axis=0)


        # Sample Push the data back down. Lets calculate the value that comes back to the visible units.
        neg_vis_inp = np.dot(pos_hid_states, weights.T) + vis_biases
        neg_data = sigmoid(neg_vis_inp)

        # No need to sample here.
        neg_hid_inp = np.dot(neg_data, weights) + hid_biases
        neg_hid_probs = sigmoid(neg_hid_inp)

        neg_prods = np.dot(neg_data.T, neg_hid_probs)

        neg_hid_act = np.sum(neg_hid_probs, axis=0)
        neg_vis_act = np.sum(neg_data, axis=0)


        #
        # Update Parameters
        #

        weights_delta = (pos_prods - neg_prods)/batch_size
        vis_biases_delta = (pos_vis_act - neg_vis_act)/batch_size
        hid_biases_delta = (pos_hid_act - neg_hid_act)/batch_size

        # Increase the training process by adding a momentum by using weights in the previous iteration.
        weights += lr * weights_delta - decay *weights + momentum * weights_inc

        # Decay not required for the biases.
        vis_biases += lr * vis_biases_delta + momentum * vis_biases_inc
        hid_biases += lr * hid_biases_delta  + momentum * hid_biases_inc


        #
        #  Store Deltas
        #

        # Save deltas for momentum
        weights_inc = weights_delta
        vis_biases_inc = vis_biases_delta
        hid_biases_inc = hid_biases_delta

        #
        # Compute training error for batch
        #

        train_err += np.sum((batch - neg_data) ** 2)


    #
    # Error Reporting
    #

    # Compute total training error
    train_err /= batch_size * n_minibatches
    # Compute validation error and activation probabilities for the visible units (we will plot these later)
    valid_err, valid_vis_probs = validation_error(data_sets.validation.images, weights, hid_biases, vis_biases)
    print "Training Epoch %i - Train Error: %f - Valid Error: %f" % (i_epoch, train_err, valid_err)

    #
    # Plotting
    #

    error_plot.plot(train_err, valid_err, i_epoch+1)
    recon_valid_plot.plot([(data_sets.validation.images[x, :], valid_vis_probs[x, :]) for x in valid_recon_indices])
    recept_fields_plot.plot([weights[:, x] for x in recept_indices])
    activation_plot.plot(pos_hid_probs)