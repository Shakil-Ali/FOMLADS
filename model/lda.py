import numpy as np
from FOMLADS.prepare_data import prepare_data
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm,show
from FOMLADS.plot.exploratory import plot_class_histograms
from FOMLADS.plot.evaluations import plot_roc

def max_lik_mv_gaussian(data):
    """
    Finds the maximum likelihood mean and covariance matrix for gaussian data
    samples (data)

    parameters
    ----------
    data - data array, 2d array of samples, each row is assumed to be an
      independent sample from a multi-variate gaussian

    returns
    -------
    mu - mean vector
    Sigma - 2d array corresponding to the covariance matrix
    """
    # the mean sample is the mean of the rows of data
    N, dim = data.shape
    mu = np.mean(data, 0)
    Sigma = np.zeros((dim, dim))
    # the covariance matrix requires us to sum the dyadic product of
    # each sample minus the mean.
    for x in data:
        # subtract mean from data point, and reshape to column vector
        # note that numpy.matrix is being used so that the * operator
        # in the next line performs the outer-product v * v.T
        x_minus_mu = np.matrix(x - mu).reshape((dim, 1))
        # the outer-product v * v.T of a k-dimentional vector v gives
        # a (k x k)-matrix as output. This is added to the running total.
        Sigma += x_minus_mu * x_minus_mu.T
    # Sigma is unnormalised, so we divide by the number of datapoints
    Sigma /= N
    # we convert Sigma matrix back to an array to avoid confusion later
    return mu, np.asarray(Sigma)


def project_data(data, weights):
    """
    Projects data onto single dimension according to some weight vector

    parameters
    ----------
    data - a 2d data matrix (shape NxD array-like)
    weights -- a 1d weight vector (shape D array like)

    returns
    -------
    projected_data -- 1d vector (shape N np.array)
    """
    N, D = data.shape
    data = np.matrix(data)
    weights = np.matrix(weights).reshape((D, 1))
    projected_data = np.array(data * weights).flatten()
    return projected_data


def fisher_linear_discriminant_projection(inputs, targets):
    """
    Finds the direction of best projection based on Fisher's linear discriminant

    parameters
    ----------
    inputs - a 2d input matrix (array-like), each row is a data-point
    targets - 1d target vector (array-like) -- can be at most 2 classes ids
        0 and 1

    returns
    -------
    weights - a normalised projection vector corresponding to Fisher's linear
        discriminant
    """
    # get the shape of the data
    N, D = inputs.shape
    # separate the classes
    inputs0 = inputs[targets == 0]
    inputs1 = inputs[targets == 1]
    # find maximum likelihood approximations to the two data-sets
    m0, S_0 = max_lik_mv_gaussian(inputs0)
    m1, S_1 = max_lik_mv_gaussian(inputs1)
    # convert the mean vectors to column vectors (type matrix)
    m0 = np.matrix(m0).reshape((D, 1))
    m1 = np.matrix(m1).reshape((D, 1))
    # calculate the total within-class covariance matrix (type matrix)
    S_W = np.matrix(S_0 + S_1)
    # calculate weights vector
    weights = np.array(np.linalg.inv(S_W) * (m1 - m0))
    # normalise
    weights = weights / np.sum(weights)
    # we want to make sure that the projection is in the right direction
    # i.e. giving larger projected values to class1 so:
    projected_m0 = np.mean(project_data(inputs0, weights))
    projected_m1 = np.mean(project_data(inputs1, weights))
    if projected_m0 > projected_m1:
        weights = -weights
    return weights


def import_for_classification(input_cols=None, target_col=None, classes=None):
    """
    Imports the iris data-set and generates exploratory plots

    parameters
    ----------
    ifname -- filename/path of data file.
    input_cols -- list of column names for the input data
    target_col -- column name of the target data
    classes -- list of the classes to plot

    returns
    -------
    inputs -- the data as a numpy.array object
    targets -- the targets as a 1d numpy array of class ids
    input_cols -- ordered list of input column names
    classes -- ordered list of classes
    """
    # if no file name is provided then use synthetic data
    dataframe = prepare_data("../data/wine.data")
    print("dataframe.columns = %r" % (dataframe.columns,))
    N = dataframe.shape[0]
    # if no target name is supplied we assume it is the last colunmn in the
    # data file
    if target_col is None:
        target_col = dataframe.columns[0]
        potential_inputs = dataframe.columns[1:,]
    else:
        potential_inputs = list(dataframe.columns)
        # target data should not be part of the inputs
        potential_inputs.remove(target_col)
    # if no input names are supplied then use them all
    if input_cols is None:
        input_cols = potential_inputs
    print("input_cols = %r" % (input_cols,))
    # if no classes are specified use all in the dataset
    if classes is None:
        # get the class values as a pandas Series object
        class_values = dataframe[target_col]
        classes = class_values.unique()
    else:
        # construct a 1d array of the rows to keep
        to_keep = np.zeros(N, dtype=bool)
        for class_name in classes:
            to_keep |= (dataframe[target_col] == class_name)
        # now keep only these rows
        dataframe = dataframe[to_keep]
        # there are a different number of dat items now
        N = dataframe.shape[0]
        # get the class values as a pandas Series object
        class_values = dataframe[target_col]
    print("classes = %r" % (classes,))
    # We now want to translate classes to targets, but this depends on our
    # encoding. For now we will perform a simple encoding from class to integer.
    targets = np.empty(N)
    for class_id, class_name in enumerate(classes):
        is_class = (class_values == class_name)
        targets[is_class] = class_id
    # print("targets = %r" % (targets,))

    # We're going to assume that all our inputs are real numbers (or can be
    # represented as such), so we'll convert all these columns to a 2d numpy
    # array object
    inputs = dataframe[input_cols].values
    return inputs, targets, input_cols, classes


def plot_scatter_array_classes(
        data, class_assignments=None, field_names=None, classes=None):
    """
    Plots scatter plots of input data, split according to class

    parameters
    ----------
    inputs - 2d data matrix of input values (array-like)
    class_assignments - 1d vector of class values as integers (array-like)
    field_names - list of input field names
    classes - list of class names (for axes labels)
    """
    # the number of dimensions in the data
    N, dim = data.shape
    dim = 5
    if class_assignments is None:
        class_assignments = np.ones(N)
    class_ids = np.unique(class_assignments)
    num_classes = len(class_ids)
    # acolor for each class
    colors = cm.rainbow(np.linspace(0, 1, num_classes))
    # create an empty figure object
    fig = plt.figure()
    # create a grid of four axes
    plot_id = 1
    for i in range(dim):
        for j in range(dim):
            if j > i:
                plot_id += 1
                continue
            ax = fig.add_subplot(dim, dim, plot_id)
            lines = []
            for class_id, class_color in zip(class_ids, colors):
                class_rows = (class_assignments == class_id)
                class_data = data[class_rows, :]
                # if it is a plot on the diagonal we histogram the data
                if i == j:
                    ax.hist(class_data[:, i], color=class_color, alpha=0.6)
                # otherwise we scatter plot the data
                else:
                    line, = ax.plot(
                        class_data[:, i], class_data[:, j], 'o', color=class_color,
                        markersize=1)
                    lines.append(line)
                # we're only interested in the patterns in the data, so there is no
                # need for numeric values at this stage
            if not classes is None and i == (dim - 1) and j == 0:
                fig.legend(lines, classes)
            ax.set_xticks([])
            ax.set_yticks([])
            # if we have field names, then label the axes
            if not field_names is None:
                if i == (dim - 1):
                    ax.set_xlabel(field_names[j])
                if j == 0:
                    ax.set_ylabel(field_names[i])
            # increment the plot_id
            plot_id += 1
    show()
    plt.tight_layout()


def project_and_histogram_data(
        inputs, targets, method, title=None, classes=None):
    """
    Takes input and target data for classification and projects
    this down onto 1 dimension according to the given method,
    then histograms the projected data.

    parameters
    ----------
    inputs - a 2d input matrix (array-like), each row is a data-point
    targets - 1d target vector (array-like) -- can be at most 2 classes ids
        0 and 1
    """
    weights = get_projection_weights(inputs, targets, method)
    projected_inputs = project_data(inputs, weights)
    ax = plot_class_histograms(projected_inputs, targets)
    # label x axis
    ax.set_xlabel(r"$\mathbf{w}^T\mathbf{x}$")
    ax.set_title("Projected Data: %s" % method)
    if not classes is None:
        ax.legend(classes)


def maximum_separation_projection(inputs, targets):
    """
    Finds the projection vector that maximises the distance between the
    projected means

    parameters
    ----------
    inputs - a 2d input matrix (array-like), each row is a data-point
    targets - 1d target vector (array-like) -- can be at most 2 classes ids
        0 and 1

    returns
    -------
    weights - a normalised projection vector
    """
    # get the shape of the data
    N, D = inputs.shape
    # separate the classes
    inputs0 = inputs[targets == 0]
    inputs1 = inputs[targets == 1]
    # find maximum likelihood approximations to the two data-sets
    m0, _ = max_lik_mv_gaussian(inputs0)
    m1, _ = max_lik_mv_gaussian(inputs1)
    # calculate weights vector
    weights = m1 - m0
    return weights


def get_projection_weights(inputs, targets, method):
    """
    Takes input and target data for classification and projects
    this down onto 1 dimension according to the given method


    parameters
    ----------
    inputs - a 2d input matrix (array-like), each row is a data-point
    targets - 1d target vector (array-like) -- can be at most 2 classes ids
        0 and 1
    returns
    -------
    weights - the projection vector
    """
    if len(np.unique(targets)) > 2:
        raise ValueError("This method only supports data with two classes")
    if method == 'maximum_separation':
        weights = maximum_separation_projection(inputs, targets)
    elif method == 'fisher':
        weights = fisher_linear_discriminant_projection(inputs, targets)
    else:
        raise ValueError("Unrecognised projection method")
    return weights
def construct_and_plot_roc(
        inputs, targets, method='maximum_separation', **kwargs):
    """
    Takes input and target data for classification and projects
    this down onto 1 dimension according to the given method,
    then plots roc curve for the data.

    parameters
    ----------
    inputs - a 2d input matrix (array-like), each row is a data-point
    targets - 1d target vector (array-like) -- can be at most 2 classes ids
        0 and 1
    """
    weights = get_projection_weights(inputs, targets, method)
    projected_inputs = project_data(inputs, weights)
    new_ordering = np.argsort(projected_inputs)
    projected_inputs = projected_inputs[new_ordering]
    targets = np.copy(targets[new_ordering])
    N = targets.size
    num_neg = np.sum(1-targets)
    num_pos = np.sum(targets)
    false_positive_rates = np.empty(N)
    true_positive_rates = np.empty(N)
    for i, w0 in enumerate(projected_inputs):
        false_positive_rates[i] = np.sum(1-targets[i:])/num_neg
        true_positive_rates[i] = np.sum(targets[i:])/num_pos
    fig, ax = plot_roc(
        false_positive_rates, true_positive_rates, **kwargs)
    return fig, ax

def main(input_cols=None, target_col=None, classes=None):
    """
    Imports the iris wine dataset and generates exploratory plots

    parameters
    ----------
    input_cols -- list of column names for the input data
    target_col -- column name of the target data
    classes -- list of the classes to plot

    """
    inputs, targets, field_names, classes = import_for_classification(input_cols=input_cols, target_col=target_col,
                                                                      classes=classes)
    # print("inputs = %r" % (inputs,))
    plot_scatter_array_classes(
        inputs, targets, field_names=field_names, classes=classes)


    plt.show()

def code_not_from_lab():

if __name__ == '__main__':
    main()