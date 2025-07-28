
# MyML

MyML is my own framework for training neural networks. It is a toy project intended to help me understand how neural networks work, I have no ambition of competing with established deep learning frameworks. One of the reasons for this is that there is no GPU acceleration in my framework and I'm not intending to add it. Many features found in the established frameworks are currently missing in my framework (see below), but I'm intending to add them eventually. My framework is built on top of numpy and tested against PyTorch. I used knowledge I gained from the "Machine Learning 2" course on FIT CTU, "Deep Learning" course on MFF UK and consultations with ChatGPT (but all code I've written myself!).

## Implemented features

The features that are already implemented:

- building & training fully connected neural networks:
    - gradient computation for various array operations
    - MSE loss, cross entropy loss
    - various optimizers: SGD (optionally with momentum), RMSProp (optionally with momentum), Adam
- some regularization methods:
    - $L_2$ and $L_1$ weight penalization
    - label smoothing
    - decoupled weight decay for Adam

## TODO list

The features currently not implemented that I intend to add:

1. dropout
2. batch normalization
3. convolutions
4. recurrent neural networks, LSTM
5. attention
6. self-attention, transformers

## Requirements

All that is required is numpy. The unit tests as well as some classes intended exclusively for testing purposes also require PyTorch. Furthermore, I use PyTorch (`torch` and `torchvision` packages) for working with data (obtaining datasets and splitting them into batches) when testing my framework, but one could do all this manually.

## Building a computational graph

Central to my framework are the classes derived from `TensorNode`. Each of these represents an $N$-dimensional array. The class `ConstantNode` represents a constant $N$-dimensional array with no dependencies on other arrays and is constructed from a numpy array, e.g.

    import numpy as np
    import nodes

    A = nodes.ConstantNode(np.random.random((4, 4, 4)))
    B = nodes.ConstantNode(np.random.random((4, 4, 4)))

Other classes derived from `TensorNode` represent various combinations of other instances of `TensorNode`. For example, the `TensorDotNode` represents a generalization of matrix multiplication to $N$-dimensional arrays:

    d = 1
    AB = nodes.TensorDotNode(A, B, d)

The `d` means that we match the last `d` indices of `A` with the first `d` indices of `B` and sum over all the possible values of these matched indices for each combination of the other indices. Mathematically:

$$
(AB)_{i_1,i_2,i_3,i_4} = \sum_{j} A_{i_1,i_2,j} B_{j,i_3,i_4}.
$$

The resulting shape is thus $(4,4,4,4)$.

If we had `d = 2` instead, we would have

$$
(AB)_{i_1,i_2} = \sum_{j_1,j_2} A_{i_1,j_1,j_2} B_{j_1,j_2,i_2}
$$

and the resulting shape would be $(4,4)$.

The `SumNode` can be used to perform a sum over the first `d` dimensions of the arrays:

    d = 1
    AB_sum = nodes.SumNode(AB, d)

Since `AB` is of shape $(4,4,4,4)$ and `d = 1`, `AB_sum` will have shape $(4,4,4)$. 

The `ElementwiseNode` can be used to perform various elementwise operations with the arrays. The possible elementwise operations are found in the `elementwise` module. Each elementwise operation takes a certain number of arrays of the exact same shape as input and its output is of that shape as well. The unary elementwise operations take exactly one such array, e.g. trigonometric functions:

    import elementwise
    A_sin = nodes.ElementwiseNode(elementwise.ElementwiseSin(), [A])

Other elementwise operations can be applied to an arbitrary number of arrays, for example addition:

    crazy_1 = nodes.ElementwiseNode(elementwise.ElementwiseAdd(3), [A, B, AB_sum])

Or multiplication:

    crazy_2 = nodes.ElementwiseNode(elementwise.ElementwiseMul(2), [A_sin, crazy_1])

The constructor argument of `ElementwiseAdd` and `ElementwiseMul` is the expected number of arguments.

Finally, we may obtain the values of the nodes as numpy arrays:

    A_val = A.get_value()
    A_sin_val = A_sin.get_value()
    crazy_2_val = crazy_2.get_value()

Thus far, nothing particularly mindblowing has been demonstrated -- all the presented operations can be easily done with numpy without wrapping the arrays in my `TensorNode`s. However, the `TensorNode`s have a certain trick up their sleeve: they can compute the gradient, which is essential for the training of neural networks. This functionality is accessed via the `get_gradients_against` method. When this method is called on a `TensorNode` representing a scalar (or any array of "volume" 1), it will evaluate the gradient of the function represented by that `TensorNode`. 

    crazy_sum = nodes.SumNode(crazy_2, 3) # shape of crazy_2 is (4,4,4), so this sums over all axes and returns a scalar
    grad_A, grad_B = crazy_sum.get_gradients_against([A, B])

A list of `ConstantNode`s against which the gradient is to be computed shall be passed to the `get_gradients_against` method. It will then return the gradients against the individual `ConstantNodes` as numpy arrays in the corresponding order.

Here the design of my framework differs from PyTorch somewhat. In PyTorch, this functionality is performed via the `backward` method, which doesn't return anything and instead saves the gradient to the `grad` attribute of the individual node objects. Meanwhile my framework treats the node objects as immutable, so the gradients must be returned. (The immutability is not checked, but should be complied with.)

The `get_gradients_against` method can also be called on a node (say $X$) of arbitrary shape. In that case however, we still assume that there is some scalar function that we are computing the gradient against, and must pass the gradient of that function with respect to the output of node $X$ as an additional argument.

    crazy_2_grad = np.random.random((4, 4, 4))
    grad_A, grad_B = crazy_2.get_gradients_against([A, B], crazy_2_grad)

Precisely this additional argument allows the `get_gradients_against` method to be implemented recursively, utilizing the magical multivariable chain rule. Each type of node implements its particular way of computing the gradient using the `get_input_gradients` method, which is called from `get_gradients_against`.

## Training a neural network

A neural network is an instance of the `NeuralNetwork` class. In my framework, I define a neural network as something that

- specifies its parameters, where each parameter has a name and a shape (this is done by the `get_params` method),
- given an input of a particular type and concrete arrays as values for the parameters, constructs a computational graph which computes some output from the input and allows the gradient against the parameters to be computed (this is done by the `construct` method).

The `NeuralNetwork` class is generic -- it has one type parameter, which specifies the type of input. So it is possible to define a neural network that takes e.g. a string as input, and the necessary procedure of converting that string to a numeric array will be implemented inside the neural network. But the conversion of the output back into some adequate representation must be done outside of the neural network. I will think about if I'll stick to this design or change it in the future.

You may define your particular neural network either by implementing the `NeuralNetwork` class, or by composing some of its existing derived classes. I show the latter approach here:

    import neural_network

    composed_model = neural_network.InputNumpyModule(
        neural_network.SequentialModule([
            neural_network.FlattenModule(),
            neural_network.LinearModule(784, 100),
            neural_network.ElementwiseModule(elementwise.ElementwiseReLU()),
            neural_network.LinearModule(100, 100),
            neural_network.ElementwiseModule(elementwise.ElementwiseReLU()),
            neural_network.LinearModule(100, 10)
        ])
    )

The `SequentialModule`, `FlattenModule`, `LinearModule` and `ElementwiseModule(elementwise.ElementwiseReLU())` are analogous to PyTorch's `Sequential`, `Flatten`, `Linear` and `ReLU` classes from the `torch.nn` module. These classes take some `TensorNode` and produce a computational graph. However, we don't want to pass `TensorNode`s to the network, we want to pass raw numpy arrays to it -- that's what the outer `InputNumpyModule` does, it takes a numpy array as input, wraps it in a `ConstantNode` and passes it to the wrapped `NeuralNetwork`.

To train the neural network, we need an instance of `NeuralNetworkOptimizer`. For example, we may use the basic `SGDOptimizer`:

    import optim
    import loss

    optimizer = optim.SGDOptimizer(
        composed_model,
        loss.OneHotEncodeWrapLoss(
            loss.CrossEntropyLoss(), 10
        ), {
            "1.weight": (np.random.random((784, 100)) * 2 - 1) * ((6 / (100 + 784)) ** 0.5),
            "3.weight": (np.random.random((100, 100)) * 2 - 1) * ((6 / (100 + 100)) ** 0.5),
            "5.weight": (np.random.random((100, 10)) * 2 - 1) * ((6 / (100 + 10)) ** 0.5),
            "1.bias": np.zeros((100,)),
            "3.bias": np.zeros((100,)),
            "5.bias": np.zeros((10,)),
        },
    )

Notice that apart from the model to be trained, we need to pass two more things to the optimizer:
- The loss function: an instance of the `LossFunction` abstract class, which takes a computational graph (outputted from a neural network) and a numpy array with the target outputs and returns a scalar `TensorNode` quantifying how far the model's outputs are from the desired outputs. The gradient is then computed against this function, i.e. it is this `TensorNode` that the optimizer will call `get_gradients_against` on.
- The initial parameter values: unlike in PyTorch, in my framework the neural network doesn't remember its parameters. A dictionary of the parameter values must be passed with each evaluation of the neural network. It is the optimizer that remembers the parameter values in my framework. So we need to pass the initial parameter values to the optimizer. The distribution from which the individual parameter values are generated is very important for how well the training will go, here I'm using the renowned Xavier initialization.

To demystify the names of the parameters -- `1.weight`, `3.weight`, etc. -- these came into being in the following way:
1. Each instance of `LinearModule` defines two parameters -- `weight` and `bias`. Its `get_params` method returns these parameters. `FlattenModule` and `ElementwiseModule` have no parameters.
2. To determine its parameters, the `SequentialModule` calls `get_params` on each of its child modules. However, to ensure that each parameter name is unique, it prefixes these parameter names with the index of the module it comes from in the list passed to the constructor (and a period). It then also uses these prefixes to determine which parameter value is destined for which child when it's constructing the computational graph (and strips the prefixes before passing the parameters to the child).
3. Because the `InputNumpyModule` has only one child and no parameters of its own, it doesn't have to change the parameter names in any way -- so it just forwards the parameters returned from calling `get_params` on its child.

To train the network, we need some dataset. This network is actually designed to work with the Fashion MNIST dataset, which I we can get via the `torchvision` package:

    import torchvision
    
    training_data = torchvision.datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(lambda x: x[0,:,:]),
        ]),
    )

And then I use a PyTorch dataloader to split this dataset into batches:

    import torch

    batch_size = 64
    train_dl = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True)

(Perhaps I could add my own analogue of PyTorch's `DataLoader` so I don't have to rely on PyTorch for this, but I put higher priority on tasks that actually help me understand modern deep learning, see TODO list above).

We may then iterate through the batches and call the optimizer to tune the model's parameters. This is done many times, each passage through all the batches in the dataset is called an epoch.

    EPOCHS = 10
    for j in range(EPOCHS):
        for i, (X_torch, y_torch) in enumerate(train_dl):
            X, y = X_torch.detach().numpy(), y_torch.detach().numpy()

            relevant_info = optimizer.prepare_step(X, y)
            optimizer.perform_step(relevant_info)

            loss_num = relevant_info.loss_node.get_value().item()
            print(f"\r -> batch ({i + 1}/{len(train_dl)}), loss is {loss_num:.3f}", end="")
        print(f"\rEpoch {j} done" + " " * 30)

Notice:
- Because PyTorch's DataLoader returns PyTorch tensors and my framework works with numpy arrays, we need to convert the PyTorch tensors to numpy arrays.
- The optimizer step consists of two method calls -- first the `prepare_step` method, which returns an `OptimizationStepRelevantInfo` object, which contains information that is relevant to determining the parameter updates, for example the gradient values. Then the `perform_step` method, which actually updates the model's parameters according to the passed `OptimizationStepRelevantInfo`. The `prepare_step` method is the same for all optimizers, but the `perform_step` is different for each optimizer (each uses the gradient in a different way to update the model's parameters).

The optimizer also has a method called `step`, which first calls `prepare_step` and then `perform_step`, and then also returns the `OptimizationStepRelevantInfo`. So we could equivalently write

    EPOCHS = 10
    for j in range(EPOCHS):
        for i, (X_torch, y_torch) in enumerate(train_dl):
            X, y = X_torch.detach().numpy(), y_torch.detach().numpy()

            relevant_info = optimizer.step(X, y)

            loss_num = relevant_info.loss_node.get_value().item()
            print(f"\r -> batch ({i + 1}/{len(train_dl)}), loss is {loss_num:.3f}", end="")
        print(f"\rEpoch {j} done" + " " * 30)

Note that you would also probably want to measure the total loss against the entire training dataset as well as the accuracy after each epoch. Furthermore, you would want to have a testing dataset and also compute the loss and accuracy against the testing dataset. I want to keep the examples simple however.
