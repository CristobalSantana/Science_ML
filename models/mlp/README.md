# MLP

A standard fully-connected feedforward network (`mlp.py`), used as the
baseline architecture experiments compare against.

Uses `tanh` activations - not a stylistic choice: physics-informed losses
need a second derivative of the network output taken via autograd, and
ReLU-family activations have zero second derivative almost everywhere,
which starves a PDE residual of gradient information. `tanh` is the
standard choice in the physics-informed neural network literature for
exactly this reason (Raissi, Perdikaris & Karniadakis, 2019).
