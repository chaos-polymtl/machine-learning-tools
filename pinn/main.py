# ============================================================================
# Physics-informed Neural Network functions using PyTorch
# Author : Valérie Bibeau, Polytechnique Montréal, 2023
# PINN with 1 feature (time) and 4 outputs.
# ============================================================================

# ---------------------------------------------------------------------------
# Imports
from ode import *
from data import *
import matplotlib.pyplot as plt
import torch
from pinn import *
# ----------------------------------------------------------------------------

# Initial condition
y0 = np.array([1., 0., 0.2, 0.])

# Real kinetic parameters
class parameters():
    k1 = 1.5
    k2 = 0.5
    k3 = 1.
    k4 = 0.1

prm = parameters()

# Data
files = ['../database/kinetic_data.csv']

X, Y, idx_C, idx_y0 = gather_data(files, 10)

device = torch.device('cpu')
X, Y = put_in_device(X, Y, device)

# PINN architecture
f_hat = torch.zeros(X.shape[0], 1).to(device)
k = [1., 1., 1., 1.]
PINN = Curiosity(X, Y, idx_C, idx_y0, f_hat, 1e-2, k, 10, 1, device)

# Make all outputs positive at the beginning of training
for i, p in enumerate(PINN.PINN.parameters()):
    p.data.clamp_(min=0.)

# Training
epoch = 0
max_epochs = 20000
while epoch <= max_epochs:

    try:

        PINN.optimizer.step(PINN.closure)

        # Positive constraint on all kinetic constants
        PINN.PINN.k1.data.clamp_(min=0.)
        PINN.PINN.k2.data.clamp_(min=0.)
        PINN.PINN.k3.data.clamp_(min=0.)
        PINN.PINN.k4.data.clamp_(min=0.)

        epoch += 1

        if epoch % 100 == 0:
            print(f'Epoch {epoch} \t loss_c_data: {PINN.loss_c_data:.4e} \t loss_c_ode: {PINN.loss_c_ode:.4e}')

        # Adaptive learning rate
        if epoch == 5000:
            PINN.optimizer = torch.optim.Adam(PINN.params, lr=1e-3)

        if epoch == 10000:
            PINN.optimizer = torch.optim.Adam(PINN.params, lr=1e-4)

    except KeyboardInterrupt:
        break

# PINN predictions
c_pred = PINN.PINN(X)
t_pred = X.detach().numpy()
cA_pred = c_pred[:,0].detach().numpy()
cB_pred = c_pred[:,1].detach().numpy()
cC_pred = c_pred[:,2].detach().numpy()
cD_pred = c_pred[:,3].detach().numpy()

# Training dataset (data points)
t_train = X[idx_C].detach().numpy()
cA_train = Y[idx_C,0].detach().numpy()
cB_train = Y[idx_C,1].detach().numpy()
cC_train = Y[idx_C,2].detach().numpy()
cD_train = Y[idx_C,3].detach().numpy()

# Numerical evaluation
prm.k1 = float(PINN.PINN.k1.detach().numpy())
prm.k2 = float(PINN.PINN.k2.detach().numpy())
prm.k3 = float(PINN.PINN.k3.detach().numpy())
prm.k4 = float(PINN.PINN.k4.detach().numpy())
t, mat_y = runge_kutta(y0, prm, 0.01, 10)

# Graphs and Prints
plt.plot(t, mat_y[:,0], '--r')
plt.plot(t, mat_y[:,1], '--b')
plt.plot(t, mat_y[:,2], '--k')
plt.plot(t, mat_y[:,3], '--c')
plt.plot(t_pred, cA_pred, '-r')
plt.plot(t_pred, cB_pred, '-b')
plt.plot(t_pred, cC_pred, '-k')
plt.plot(t_pred, cD_pred, '-c')
plt.plot(t_train, cA_train, 'or', label='[A]')
plt.plot(t_train, cB_train, 'ob', label='[B]')
plt.plot(t_train, cC_train, 'ok', label='[C]')
plt.plot(t_train, cD_train, 'oc', label='[D]')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Concentration (mol/L)')
plt.show()

print(f'k1 = {PINN.PINN.k1.detach().numpy()}')
print(f'k2 = {PINN.PINN.k2.detach().numpy()}')
print(f'k3 = {PINN.PINN.k3.detach().numpy()}')
print(f'k4 = {PINN.PINN.k4.detach().numpy()}')
