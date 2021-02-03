import numpy as np
from Model import Network

net = Network([2, 2, 3], 4)
x = np.array([1, 2, 3, 4])
wynik = net.forward(x)
print(wynik)
