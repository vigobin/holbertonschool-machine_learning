#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

x = ['Farrah', 'Fred', 'Felicia']
fruits = ['apples', 'bananas', 'oranges', 'peaches']
plt.bar(range(len(fruit[0])), fruit[0], width=0.5, color='red', label=fruits[0])
plt.bar(range(len(fruit[0])), fruit[1], width=0.5, color='yellow',
        label=fruits[1], bottom=fruit[0])
plt.bar(range(len(fruit[0])), fruit[2], width=0.5, color='orange',
        label=fruits[2], bottom=fruit[0] + fruit[1])
plt.bar(range(len(fruit[0])), fruit[3], width=0.5, color='#ffe5b4',
        label=fruits[3], bottom=fruit[0] + fruit[1] + fruit[2])

plt.ylabel('Quantity of Fruit')
plt.ylim(0, 80)
plt.yticks(np.arange(0, 81, 10))
plt.xticks(np.arange(len(x)), x)
plt.title('Number of Fruit per Person')
plt.legend()
plt.show()
