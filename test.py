from typing import Iterable, Iterator
from collections import deque

class A:
    def __init__(self, x):
        self.x = x
        print('A')
    
    def __iter__(self):
        return B(x=self.x)

    def __repr__(self):
        return 'class A with x = ' + str(self.x)
    

class B:
    def __init__(self, x):
        self.x = x

    def __next__(self):
        print(self.x)

    def __repr__(self):
        return 'class B with x = ' + str(self.x)

a = A(3)
print(isinstance(a, Iterable))
for i in a:
    continue