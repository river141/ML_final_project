import numpy as np
import matplotlib.pyplot as plt

# loss_dir = '.\\CycleGAN-custom\\output_train_0\\losses.txt'
loss_dir = '.\\CycleGAN-custom\\output_train_0_no_cycle\\losses_no_cycle.txt'
avg_over = 50

data = np.loadtxt(loss_dir)
iterations = data[:, 0]
variables = data[:, 1:]
num_variables = variables.shape[1]
if num_variables == 6:
    fig, axs = plt.subplots(3, 2, sharex=True)
    names = ['G_A', 'G_B', 'Cycle_A', 'Cycle_B', 'D_A', 'D_B']
elif num_variables == 4:
    fig, axs = plt.subplots(2, 2, sharex=True)
    names = ['G_A', 'G_B', 'D_A', 'D_B']

for i in range(num_variables):
    axs.flatten()[i].plot(iterations, variables[:, i], linewidth=0.5)
    # calculate and plot running average
    running_avg = np.convolve(variables[:, i], np.ones(avg_over)/avg_over, mode='valid')
    axs.flatten()[i].plot(iterations[int(avg_over-1):], running_avg, label='Running Average')
    axs.flatten()[i].set_xlabel('Iterations')
    axs.flatten()[i].set_ylabel(names[i])
    axs.flatten()[i].legend()

plt.tight_layout()
plt.show()