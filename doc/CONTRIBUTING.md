Please complete the below steps to be able to contribute to the development of this project:

1. Install Marabou using these [instructions](https://neuralnetworkverification.github.io/Marabou/Setup/0_Installation.html).

2. Create a Python virtual environment for the project.

_Note: For macOS, use Python version X.Y where X and Y are defined in MarabouCore.cpython-XY-darwin.so, which is located in the separate directory `Marabou/maraboupy/` once step 1 is complete._

3. Install the required dependencies via `pip install -r requirements.txt`

The steps for development are complete. Please find the next steps for development at [todo.md](todo.md).

Other notes for development:
- MarabouUtils.Equation() may be deprecated and replaced with MarabouCore.Equation() in the near future. At the moment, `MarabouUtils.Equation()` is what works with `MarabouNetwork.addEquation()` and, thus, is used in the main module.