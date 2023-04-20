def get_model_data(model):
    """
    Extracts the weights, biases, and layer information from a Keras model.

    Args:
        model: A Keras model object.

    Returns:
        A tuple containing the weights, biases, and layer information.
    """
    weights = []
    biases = []
    layers = []

    for layer in model.layers:
        layer_weights = layer.get_weights()
        if len(layer_weights) > 0:
            weights.append(layer_weights[0])
            biases.append(layer_weights[1])
            layers.append(layer)

    return weights, biases, layers

def get_neurons(layers):
    """
    Generates the neuron positions for each layer.

    Args:
        layers: A list of Keras layer objects.

    Returns:
        A list of lists containing the neuron positions for each layer.
    """
    neurons = []
    for layer in layers:
        layer_neurons = []
        for i in range(layer.output_shape[1]):
            layer_neurons.append((i,))
        neurons.append(layer_neurons)
    return neurons

def plot_neurons(neurons, biases=None, figsize=(6, 4)):
    """
    Plots the neurons as circles layer by layer using Matplotlib.

    Args:
        neurons: A list of lists containing the neuron positions for each layer.
        biases: A list of arrays containing the bias values for each layer.
        figsize: A tuple specifying the width and height of the figure in inches.

    Returns:
        A Matplotlib figure object.
    """
    fig = plt.figure(figsize=figsize)

    # Create custom colormap
    colors = ['blue', 'white', 'red']
    cmap = LinearSegmentedColormap.from_list('bwr', colors)

    # Set axis limits
    plt.xlim(-1, len(neurons))
    plt.ylim(-1, max([len(layer) for layer in neurons]))

    # Plot neurons
    for i, layer in enumerate(neurons):
        for j, neuron in enumerate(layer):
            if biases is not None:
                bias = biases[i][j]
                norm_bias = (bias - biases[i].min()) / (biases[i].max() - biases[i].min())
                color = cmap(norm_bias)
            else:
                color = 'black'
            circle = Circle((i, j), 0.3, facecolor=color, edgecolor='purple')
            plt.gca().add_patch(circle)

    # Set aspect ratio
    plt.gca().set_aspect('equal')

    return fig

def add_biases(biases, neurons):
    """
    Adds the biases as text annotations next to the neurons.

    Args:
        biases: A list of arrays containing the bias values for each layer.
        neurons: A list of lists containing the neuron positions for each layer.

    Returns:
        None
    """
    # Add biases
    for i in range(len(biases)):
        for j in range(biases[i].shape[0]):
            x, y = i, neurons[i][j][0]
            bias = biases[i][j]
            plt.text(x, y, f'{bias:.2f}', fontsize=8, ha='center', va='center')

def plot_weights(weights, neurons):
    """
    Plots the weights as lines between the neurons with varying color based on their values.

    Args:
        weights: A list of arrays containing the weight values for each layer.
        neurons: A list of lists containing the neuron positions for each layer.

    Returns:
        None
    """
    # Create custom colormap
    colors = ['blue', 'white', 'red']
    cmap = LinearSegmentedColormap.from_list('bwr', colors)

    # Normalize weights
    all_weights = np.concatenate([w.flatten() for w in weights])
    min_weight = all_weights.min()
    max_weight = all_weights.max()
    norm_weights = [(w - min_weight) / (max_weight - min_weight) for w in weights]

    # Plot weights
    for i in range(len(weights)):
        if i + 1 < len(neurons):
            for j in range(weights[i].shape[0]):
                for k in range(weights[i].shape[1]):
                    if k < len(neurons[i + 1]):
                        x1, y1 = i, neurons[i][j][0]
                        x2, y2 = i + 1, neurons[i + 1][k][0]
                        weight = norm_weights[i][j, k]
                        color = cmap(weight)
                        plt.plot([x1, x2], [y1, y2], '-', lw=3, color=color)

def get_weight_label_positions(weights, neurons, shift=0.25):
    """
    Finds the positions for the weight labels so that they overlap less.

    Args:
        weights: A list of arrays containing the weight values for each layer.
        neurons: A list of lists containing the neuron positions for each layer.
        shift: A float value between -1 and 1 that controls the position of the weight labels along the segments connecting the centers of the circles.

    Returns:
        A list of lists containing the weight label positions for each layer.
    """
    label_positions = []
    for i in range(len(weights)):
        if i + 1 < len(neurons):
            layer_label_positions = []
            for j in range(weights[i].shape[0]):
                neuron_label_positions = []
                for k in range(weights[i].shape[1]):
                    if k < len(neurons[i + 1]):
                        x1, y1 = i, neurons[i][j][0]
                        x2, y2 = i + 1, neurons[i + 1][k][0]
                        x = x1 + (0.5 + shift / 2) * (x2 - x1)
                        y = y1 + (0.5 + shift / 2) * (y2 - y1)
                        neuron_label_positions.append((x, y))
                layer_label_positions.append(neuron_label_positions)
            label_positions.append(layer_label_positions)
    return label_positions

def add_weights(weights, neurons):
    """
    Adds the weight values as text annotations at the corresponding locations.

    Args:
        weights: A list of arrays containing the weight values for each layer.
        neurons: A list of lists containing the neuron positions for each layer.

    Returns:
        None
    """
    # Add weights
    label_positions = get_weight_label_positions(weights, neurons)
    for i in range(len(weights)):
        if i + 1 < len(neurons):
            for j in range(weights[i].shape[0]):
                for k in range(weights[i].shape[1]):
                    if k < len(neurons[i + 1]):
                        x, y = label_positions[i][j][k]
                        weight = weights[i][j, k]
                        plt.text(x, y, f'{weight:.2f}', fontsize=8)

def vis_model_weights(model):
    weights, biases, layers = get_model_data(model)
    neurons = get_neurons(layers)

    # Plot neurons with larger size
    fig = plot_neurons(neurons, biases=biases, figsize=(12, 8))

    # Plot weights
    plot_weights(weights, neurons)

    # Add biases
    add_biases(biases, neurons)

    # Add weights
    add_weights(weights, neurons)

    # Show plot
    plt.show()
