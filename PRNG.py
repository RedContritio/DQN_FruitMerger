import base64
import math
import datetime

## 伪随机数生成器
class PRNG():
    def __init__(self):
        self.s = 1234
        self.p = 999979
        self.q = 999983
        self.m = self.p * self.q
    
    def hash(self, x: any):
        y = base64.encodebytes(bytes(str(x).encode('utf8')))
        z = 0
        for i, v in enumerate(y):
            z += v * math.pow(128, i)
        return z
    
    def seed(self, seed: any = datetime.datetime.now()):
        y = 0
        z = 0
        while y % self.p == 0 or y % self.q == 0 or y == 0 or y == 1:
            y = (self.hash(seed) + z) % self.m
            z += 1
        
        self.s = y

        [self.next() for _ in range(10)]

    def next(self):
        self.s = (self.s * self.s) % self.m
        return self.s / self.m

    def random(self, l: float =  0, r: float = 1):
        return self.next() * (r - l) + l
    
    def randint(self, l: int = 0, r: int = 2):
        return int(math.ceil(self.random(l, r)))
    
    def randsign(self) -> int:
        return -1 if self.random() > .5 else 1