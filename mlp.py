#!/usr/bin/env python

import math
import pickle
import gzip
import numpy as np
import pandas
import matplotlib.pylab as plt
import pytest
#%matplotlib inline

class xnor_mlp:
    #From the expression, we can say that the XNOR gate consists of an AND gate (x1x2), a NOR gate (x1`x2`), and an OR gate
    def __init__(self) -> None:
        pass
    
    def calculate(self, x1, x2):
        # run x1, x2 through mlp network
        output_layer1 = self.layer1(x1, x2)
        final_output = self.layer2(output_layer1)
        return final_output
    
    def indicator(self, x):
        return 1 if x > 0 else 0

    def layer1(self, x1, x2):
        # layer is specific to breaking XNOR into 2 steps, this first step is to break it into AND and OR
        # node for NAND 
        z1 = (-2 * x1) + (-2 * x2) + 3
        output1 = self.indicator(z1)
        
        # node for OR 
        z2 = (2 * x1) + (2 * x2) - 1
        output2 = self.indicator(z2)
        
        return output1, output2

    def layer2(self, l1out):
        # inputs from layer 1, used to determine if this is XNOR or not
        out = (2 * l1out[0]) + (2 * l1out[1]) - 3
        l2out = self.indicator(out)
        return l2out

class mlp_xnor_v2:
    def __init__(self) -> None:
        self.weights = np.array([1,1])
        self.bias = -0.5
    
    def activation_function(self, x):
        return 1 if x >= 0 else 0

    def forward(self, inputs):
        weighted_sum = np.dot(self.weights, inputs) + self.bias
        output = self.activation_function(weighted_sum)
        return output
    
    
class xnor_mlp_v2:
    #From the expression, we can say that the XNOR gate consists of an AND gate (x1x2), a NOR gate (x1`x2`), and an OR gate
    #This means we will have to combine 3 perceptrons:
    # AND (x1+x2–1)
    # NOR (-x1-x2+1)
    # OR (2x1+2x2–1)
    
    def __init__(self) -> None:
        pass
    
    def calculate(self, x1, x2):
        # run x1, x2 through mlp network
        output_layer1 = self.layer1(x1, x2)
        final_output = self.layer2(output_layer1)
        return final_output
    
    def indicator(self, x):
        return 1 if x > 0 else 0

    def layer1(self, x1, x2):
        # layer is specific to breaking XNOR into 2 steps, this first step is to break it into AND and OR
        # node for AND 
        z1 = (1 * x1) + (1 * x2) -1
        output1 = self.indicator(z1)
        
        # node for NOR 
        z2 = (-1 * x1) + (-1 * x2) + 1
        output2 = self.indicator(z2)
        
        return output1, output2

    def layer2(self, l1out):
        # inputs from layer 1, used to determine if this is OR or not
        out = (2 * l1out[0]) + (2 * l1out[1]) - 1
        l2out = self.indicator(out)
        return l2out

if __name__ == "__main__":
    #do something
    # Testing the network with all combinations of x1 and x2
    combinations = [(0, 0), (0, 1), (1, 0), (1, 1)]
    results = {}

    mlp = xnor_mlp()

    for x1, x2 in combinations:
        results[(x1, x2)] = mlp.calculate(x1, x2)

    print(f"v1 results: {results}")
    
    v2_mlp = mlp_xnor_v2()
    for x1, x2 in combinations:
        results[(x1, x2)] = v2_mlp.forward(np.array([x1, x2]))
        
    print(f"v2 results: {results}")
    
    v3_mlp = xnor_mlp_v2()
    for x1, x2 in combinations:
        results[(x1, x2)] = v3_mlp.calculate(x1, x2)
    # NOTE: this last version gives the correct results, the others do not
        
    print(f"v3 results: {results}")