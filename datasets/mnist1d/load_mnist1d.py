from mnist1d.data import make_dataset, get_dataset_args
import numpy as np

default_args = get_dataset_args()
data = make_dataset(default_args)
x_train, y_train = data['x'], data['y']
x_test, y_test = data['x_test'], data['y_test']

np.savetxt('x_train.txt', x_train, delimiter=' ', fmt='%.8e')
np.savetxt('y_train.txt', y_train, delimiter=' ', fmt='%u')

np.savetxt('x_test.txt', x_test, delimiter=' ', fmt='%.8e')
np.savetxt('y_test.txt', y_test, delimiter=' ', fmt='%u')


