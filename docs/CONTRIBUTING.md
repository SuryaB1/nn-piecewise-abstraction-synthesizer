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

4. **Test Execution**
   - After activating your virtual environment, run `python src/nn_piecewise_abstraction_synthesizer/main.py` from the top-level directory.
   - After several seconds, you should see a directory called `tess_form_gif/` in the top-level directory containing the files `cegis_iteration_0.png` and `tessellation_formation.gif`. The PNG and GIF should both show the same plot of three red points, three blue points, two yellowish-green points, two dashed lines, one solid green line, and two axes with labels from -10.0 to 10.0.

You are now ready to start development. For the next steps, please refer to the [todo.md](todo.md) file.

### Additional Notes for Development
- Be aware that `MarabouUtils.Equation()` might be deprecated and replaced with `MarabouCore.Equation()` in the near future. Currently, `MarabouUtils.Equation()` is compatible with `MarabouNetwork.addEquation()`, which is why it is used in the main module. (Interpreted from [Issue #652](https://github.com/NeuralNetworkVerification/Marabou/issues/652) and [Issue #717](https://github.com/NeuralNetworkVerification/Marabou/issues/717))