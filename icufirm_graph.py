import pickle
import matplotlib.pyplot as plt


with open('inv_exp.pickle', 'rb') as file:
    dict = pickle.load(file)

plt.plot(dict['dim'], dict['cpu'], label='CPU')
plt.plot(dict['dim'], dict['gpu'], label='GPU')
plt.legend(loc='best')
plt.show()
