import pickle
import matplotlib.pyplot as plt


with open('inv_exp.pickle', 'rb') as file:
    dict = pickle.load(file)

plt.plot(dict['dim'], dict['time_cpu'], label='CPU')
plt.plot(dict['dim'], dict['time_gpu'], label='GPU')
plt.legend(loc='best')
