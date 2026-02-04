import nevergrad as ng

def square(x):
    return sum((x - .5)**2)

optimizer = ng.optimizers.NGOpt(parametrization=2, budget=100)
recommendation = optimizer.minimize(square)
print(recommendation.value)  # recommended value