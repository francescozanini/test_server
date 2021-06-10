import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

model_name = 'Model_2021-06-10_22-47-29'

def create_graph(model_name):
    contents = open('model.log', 'r').read().split('\n')
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
            
