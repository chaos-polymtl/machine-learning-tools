# machine-learning-tools
Machine learning tools in Python

## Artificial Neural Network (ANN)

In the folder `ann`, the Python file `ann.py` contains the definition of the functions used to build an ANN model.

- `read_data` reads the data in a `.csv` file where the first columns are the features and the last column is the output. The function uses the [Pandas](https://pandas.pydata.org/) library to read the file and converts the DataFrame into two [Numpy](https://numpy.org/) tensors, X and Y. X is the tensor of the inputs and Y is the tensor of the output.
- `initial_setup` normalizes X and Y using a MinMax scaling from the [Scikit-learn](https://scikit-learn.org/stable/) library. It also splits the database into a training and testing set.
- `fit_model` builds and trains the ANN's model with the [TensorFlow](https://www.tensorflow.org/) library.

The Python file `mixer_main.py` builds and trains the ANN model with the dataset of CFD simulations of mixers. In the `database` folder, the file `mixer_database.csv` includes approximately 100k Lethe simulations of pitched blade impeller mixers. The database has 7 features (Reynolds number and geometric ratios) and 1 output (power number).

### Grid search

Within the `ann` folder, a Python file `grid_search.py` is included to show an example of a grid search algorithm to tune the hyperparameters of the ANN model. The grid search method uses a cross-validation technique. This example uses the database of mixers.

## Physics-informed Neural Network (PINN)

In the folder `pinn`, the Python file `pinn.py` contains the definition of the functions used to build a PINN model. The PINN is built for a set of ODEs describing the molar balance of chemical species. The ODEs depends on rate constants $k$. The set of reactions at study is:

$A \leftrightarrow C + B$ and

$C \leftrightarrow D$

The PINN tries to identify 4 kinetic constants ($k_1$, $k_2$, $k_3$ and $k_4$) of the above reactions. The PINN model is coded using the [PyTorch](https://pytorch.org/) library.

- The class `PINeuralNet` uses the base class `torch.nn.Module`. It allows to build the architecture, set up the 4 parameters and define the forward pass.
- The class `Curiosity` (named after the rover [Curiosity](https://mars.nasa.gov/msl/home/) from NASA that went on planet Mars to discover some wonders) trains the PINN model. The loss function is define in this class. Note that Curiosity can be easily change to satisfy any kind of ODEs.

Other files goes with the PINN model:
- `data.py` reads the data in the `kinetic_database.csv` file where the first column is the indenpendant variable and the next columns are the dependant variables, or the concentrations in this case.
- `ode.py` computes the numerical solution of the set of ODEs. It uses a Runge-Kutta method.
- `main_kinetic.py` builds and trains the PINN model with the dataset of species concentrations.