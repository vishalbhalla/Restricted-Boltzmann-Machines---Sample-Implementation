import matplotlib as mpl
mpl.use('TkAgg')

import matplotlib.cm as cm
from matplotlib import pyplot as plt

import numpy as np

FIGURE_TITLE_FONT_SIZE = 16


def allow_plot():
    plt.ion()


# Plot for Train vs Validation errors per epoch on the dataset.
class ErrorPlot(object):
    
    def __init__(self, fignum):
        self._fig = plt.figure(fignum)
        self._fig.suptitle('Mean Squared Error', fontsize=FIGURE_TITLE_FONT_SIZE)
        plt.grid()
        self._train_error_plot, = plt.plot([], [], label='Training Error')
        self._valid_error_plot, = plt.plot([], [], label='Validation Error')
        plt.legend()
        self._ax = plt.gca()

    def plot(self, train_err, valid_err, i_epoch):
        self._train_error_plot.set_xdata(np.append(self._train_error_plot.get_xdata(), i_epoch))
        self._train_error_plot.set_ydata(np.append(self._train_error_plot.get_ydata(), train_err))
    
        self._valid_error_plot.set_xdata(np.append(self._valid_error_plot.get_xdata(), i_epoch))
        self._valid_error_plot.set_ydata(np.append(self._valid_error_plot.get_ydata(), valid_err))
        self._ax.relim()
        self._ax.autoscale_view()
        self._fig.canvas.draw()


# Plot showing original vs reconstructed image from the dataset.
class ReconstructionPlot(object):
    
    def __init__(self, fignum):
        self._fig = plt.figure(fignum)
        self._fig.suptitle('Reconstruction Plots', fontsize=FIGURE_TITLE_FONT_SIZE)
        
    def plot(self, data):
        assert type(data) is list

        num_rows = len(data)
        for idx, elem in enumerate(data):
            assert type(elem) == tuple
            orig = np.reshape(elem[0], newshape=(28, 28))
            recon = np.reshape(elem[1], newshape=(28, 28))
            ax_orig = self._fig.add_subplot(num_rows, 2, 1 + 2 * idx)
            ax_recon = self._fig.add_subplot(num_rows, 2, 2 + 2 * idx)
            ax_orig.imshow(orig, cmap=cm.Greys_r)
            ax_recon.imshow(recon, cmap=cm.Greys_r)

        self._fig.canvas.draw()


# Plot depicting the Receptive Fields.
class ReceptiveFieldsPlot(object):

    def __init__(self, fignum, height, width):
        self._fig = plt.figure(fignum)
        self._fig.suptitle('Receptive Fields', fontsize=FIGURE_TITLE_FONT_SIZE)
        self._height = height
        self._width = width

    def plot(self, weights):
        assert type(weights) == list
        assert len(weights) == self._height * self._width

        for i_plot, weight_vec in enumerate(weights):
            field = np.reshape(weight_vec, newshape=(28, 28))
            ax = self._fig.add_subplot(self._height, self._width, i_plot + 1)
            ax.imshow(field, cmap=cm.Greys_r)

        self._fig.canvas.draw()


# Plot depicting the activation function on each neuron in an epoch during training phase.
class ActivationPlot(object):

    def __init__(self, fignum):
        self._fig = plt.figure(fignum)
        self._fig.suptitle('Hidden Unit Activation Over Batch', fontsize=FIGURE_TITLE_FONT_SIZE)
        self._ax = self._fig.add_subplot(111)
        self._cb = None

    def plot(self, pos_probs_hid):
        im = self._ax.imshow(pos_probs_hid, cmap=cm.Greys_r, aspect='auto')
        if not self._cb:
            self._cb = plt.colorbar(im)
        self._cb.set_clim(pos_probs_hid.min(), pos_probs_hid.max())
        self._cb.draw_all()
        self._fig.canvas.draw()
