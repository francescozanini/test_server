import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

model_name = 'Model_2021-06-10_23-24-35' #'Model_2021-06-10_22-47-29'

def create_graph(model_name):
    contents = open('model30.log', 'r').read().split('\n')
    times = []
    accs = []
    losses = []
    acc_vals = []
    loss_vals = []
    for c in contents:
        if model_name in c:
            model_name, timestamp, acc, loss, acc_val, loss_val = c.split(',')
            times.append(float(timestamp))
            accs.append(float(acc))
            losses.append(float(loss))
            acc_vals.append(float(acc_val))
            loss_vals.append(float(loss_val))

    fig = plt.figure()
    ax1 = plt.subplot2grid((2, 1), (0, 0))
    ax2 = plt.subplot2grid((2, 1), (1, 0), sharex=ax1)

    ax1.plot(times, accs, label='accs')
    ax1.plot(times, acc_vals, label='acc_vals')
    ax1.legend(loc='best')

    ax2.plot(times, losses, label='losses')
    ax2.plot(times, loss_vals, label='loss_vals')
    ax2.legend(loc='best')

    plt.show()

create_graph(model_name)
