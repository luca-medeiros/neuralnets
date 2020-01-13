
from miniflow import *

# x, y, z = Input(), Input(), Input()
# f = Add(x, y, z)
# feed_dict = {x: 4, y: 5, z: 10}


inputs, weights, bias = Input(), Input(), Input()

f = Linear(inputs, weights, bias)

feed_dict = {
    inputs: [6, 14, 3],
    weights: [0.5, 0.25, 1.4],
    bias: 2
}
graph = topological_sort(feed_dict)
output = forward_pass(f, graph)

print(output)