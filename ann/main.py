# =================================================================================
# Main program to train the ANN model.
# Author: Valérie Bibeau, Polytechnique Montréal, H4ck4th0n 2023
# =================================================================================

# ---------------------------------------------------------------------------
# Imports
from ann import *
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
# ---------------------------------------------------------------------------

# Read the data
file_path = '../database/mixer_database.csv'
X, Y = read_data(file_path)

# Set the features tensor and the target tensor for the training and testing set
X_train, X_test, y_train, y_test, scaler_X, scaler_y = initial_setup(X, Y,
                                                                     test_size=0.3,
                                                                     random_state=42)

# Compile and fit the model
history, model, params = fit_model(X_train, y_train,
                                   units=40, layers=2, activation='tanh', lr=1e-3,
                                   epochs=5000, batch_size=200, val_frac=0.0, l2=1e-10,
                                   verbose=1)

# Calculate the MAPE for the training set
train_predictions = model.predict(X_train)
train_mse = mean_squared_error(y_true=scaler_y.inverse_transform(y_train),
                               y_pred=scaler_y.inverse_transform(train_predictions))
train_mae = mean_absolute_error(y_true=scaler_y.inverse_transform(y_train), 
                                y_pred=scaler_y.inverse_transform(train_predictions))
train_mape = mean_absolute_percentage_error(y_true=scaler_y.inverse_transform(y_train),
                                            y_pred=scaler_y.inverse_transform(train_predictions))
# Calculate the metrics for the testing set
test_predictions = model.predict(X_test)
test_mse = mean_squared_error(y_true=scaler_y.inverse_transform(y_test),
                              y_pred=scaler_y.inverse_transform(test_predictions))
test_mae = mean_absolute_error(y_true=scaler_y.inverse_transform(y_test), 
                               y_pred=scaler_y.inverse_transform(test_predictions))
test_mape = mean_absolute_percentage_error(y_true=scaler_y.inverse_transform(y_test),
                                           y_pred=scaler_y.inverse_transform(test_predictions))

print("Mean Squared Error:")
print("     Training set:   {:5.4e}".format(train_mse))
print("     Testing set:    {:5.4e}".format(test_mse))
print("Mean Absolute Error:")
print("     Training set:   {:5.6f}".format(train_mae))
print("     Testing set:    {:5.6f}".format(test_mae))
print("Mean Absolute Percentage Error:")
print("     Training set:   {:5.4f}".format(train_mape))
print("     Testing set:    {:5.4f}".format(test_mape))

# Check evolution of training
plt.plot(history.history['loss'],label='Training')
try: 
    val_loss = history.history['val_loss']
    plt.plot(val_loss,label='Validation')
except:
    pass
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()