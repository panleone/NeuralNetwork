Implement a compile time expression system and a neural network engine on top of it.

Currently supports:

- Basic operations (+, -, /, *, exp, log, ReLU, etc...) and broadcasting
- Matrix multiplication
- Convolution 1d and 2d
- Residual connections (... or more generally any acyclic topology)

External libraries:

- OpenBlas https://github.com/OpenMathLib/OpenBLAS (for efficient matrix multiplication)

As an example application, a neural network is trained on the MNIST-1D dataset https://github.com/greydanus/mnist1d.
The design of the expression system is inspired from the paper https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=ca2f8a9b7407de039957a358f995265ec8b769a9