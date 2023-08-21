# machine-learning-tools
Machine learning tools in Python

## Artificial Neural Network (ANN)

In folder `ann`, the Python file `ann.py` contains the definition of the functions used to build the ANN model.

- `read_data` reads the data in a `.csv` file where the first columns are the features and the last column is the output. The function uses the [Pandas](https://pandas.pydata.org/) library to read the file and converts the DataFrame into two [Numpy](https://numpy.org/) tensors, X and Y. X is the tensor of the inputs and Y is the tensor of the output.
- `initial_setup` normalizes X and Y using a MinMax scaling from the [Scikit-learn](https://scikit-learn.org/stable/) library. It also splits the database into a training and testing set.
- `fit_model` builds and trains the ANN's model with the [TensorFlow](https://www.tensorflow.org/) library.

The Python file `mixer_main.py` builds and trains the ANN model with the dataset of CFD simulations of mixers. In the `database` folder, the file `mixer_database.csv` includes approximately 100k Lethe simulations of pitched blade impeller mixers. The database has 7 features (Reynolds number and geometric ratios) and 1 output (power number).

### Grid search

Within the `ann` folder, a Python file `grid_search.py` is included to show an example of a grid search algorithm to tune the hyperparameters of the ANN model. The grid search method uses a cross-validation technique. This example uses the database of mixers.

## Physics-informed Neural Network (PINN)

In construction...