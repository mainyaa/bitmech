
from pymc3 import  *

import numpy as np
import matplotlib.pyplot as plt
size = 200
true_intercept = 1
true_slope = 2

x = np.linspace(0, 1, size)
# y = a + b*x
true_regression_line = true_intercept + true_slope * x
# add noise
y = true_regression_line + np.random.normal(scale=.5, size=size)

data = dict(x=x, y=y)

with Model() as model:
    # specify glm and pass in data. The resulting linear model, its likelihood and
    # and all its parameters are automatically added to our model.
    GLM.from_formula('y ~ x', data)
    trace = sample(progressbar=False, tune=1000, njobs=4) # draw posterior samples using NUTS sampling

    plt.figure(figsize=(7, 7))
    traceplot(trace)
    plt.tight_layout();

    plt.figure(figsize=(7, 7))
    plt.plot(x, y, 'x', label='data')
    plots.plot_posterior_predictive_glm(trace, samples=100,
                                        label='posterior predictive regression lines')
    plt.plot(x, true_regression_line, label='true regression line', lw=3., c='y')

    plt.title('Posterior predictive regression lines')
    plt.legend(loc=0)
    plt.xlabel('x')
    plt.ylabel('y');
    plt.show()

