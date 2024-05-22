### How to Contribute

To contribute to the development of this project, please follow these steps:

1. **Install Marabou**
   - Follow the installation instructions provided [here](https://neuralnetworkverification.github.io/Marabou/Setup/0_Installation.html).

2. **Set up a Python Virtual Environment**
   - Create and activate a virtual environment for the project.
   - Note: Make sure to use Python version `A`.`B` where `A` and `B` are specified in the maraboupy binary (e.g. `MarabouCore.cpython-AB-darwin.so` for macOS or `MarabouCore.cpython-AB-x86_64-linux-gnu.so` for Linux)
        - The maraboupy binary can be found in the `Marabou/maraboupy/` directory after completing step 1.
        - See [Issue #538](https://github.com/NeuralNetworkVerification/Marabou/issues/538) for troubleshooting.

3. **Install Required Dependencies**
   - Run `pip install -r requirements.txt` to install all necessary dependencies.

You are now ready to start development. For the next steps, please refer to the [todo.md](todo.md) file.

### Additional Notes for Development
- Be aware that `MarabouUtils.Equation()` might be deprecated and replaced with `MarabouCore.Equation()` in the near future. Currently, `MarabouUtils.Equation()` is compatible with `MarabouNetwork.addEquation()`, which is why it is used in the main module. (Interpreted from [Issue #652](https://github.com/NeuralNetworkVerification/Marabou/issues/652) and [Issue #717](https://github.com/NeuralNetworkVerification/Marabou/issues/717))