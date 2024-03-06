'''
runGlass.py

Michael Langer  COMP 546 Winter 2024   Assignment 3  
'''

import numpy as np
from makeGlassLRorTB import makeGlassLRorTB
from Q2 import Q2
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

N = 128    #  image width
dx = 2     #  translation defined by the dot pattern e.g. (2, 0)
dy = 0

p_noise = .02

#  You will need to increase the number of trials to get a good fit in your psychometric curve.
numtrials = 10

#  The following variable p0_list is the list of probabilities of a pixel being 1 for the different density Glass patter conditions.
#  A pixel has value 0 with probability p0. 
#  A pixel has value 1 with probability 1-p0.
#  In order to get a good fit to a psychometric function, you will need to use more p0 levels than this,
#  and the levels you choose will need to depend on other parameters such as dx, dy.
p0_list = np.array([0.01, .02, 0.03])

numcorrect = np.zeros(len(p0_list) )
for i, p0 in enumerate(p0_list):
    for t in range(numtrials):
        if t % 2 == 1:
            condition = 'lr'
        else:
            condition = 'tb'

        I = makeGlassLRorTB(N, p_noise, p0, condition,dx,dy)
        # show an example Glass patterns for this p0 
        if t == 1:
            fig = plt.figure(i)
            ax = fig.add_subplot(111)
            plt.imshow(I,cmap='gray')
            ax.set_title('p0 is ' + str(p0));

        #  Here is where you call the function that decides whether the Glass pattern is top/bottom or left/right.
        response = Q2(I)
        
        #  Compare the actual condition to the response and determine
        #  if the response was correct.
        if  condition == response:
            numcorrect[i] += 1

frac_correct = numcorrect/numtrials

'''
Now we want to fit a psychometric function to the data.
First, do a least squares linear fit to the first three data points to get the y intercept
and slope.   We expect the psychometric function at p0 = 0 to have value about 0.5 (guessing)
and for the slope to ramp up linearly at first.  This is the linear part of the logistic function.
That's why we do a linear fit for the first few data points!
Then use the estimated y-intercept and slope as the initial estimate for the
logistic function (sigmoid shape) fit.
[ML: This seems to work, but I don't guarantee it.]
'''

def linear_fit(x, a, b):   
    return a*x + b

def sigmoid(x, x0, k):
    return 1/(1+ np.exp(-k*(x - x0)))

a_b, pcov = curve_fit(linear_fit, p0_list[:3], frac_correct[:3])
x0_k, goodness = curve_fit(sigmoid, p0_list, frac_correct, [a_b[1], a_b[0]])
x0 = x0_k[0]
k  = x0_k[1]
# solve for  0.75 = 1/(1+exp(-k*x - k*x0))
thres =  - np.log(1/.75 - 1) / k + x0

fig = plt.figure(p0_list.shape[0])
ax = fig.add_subplot(111)
plt.plot( p0_list, 1/(1+np.exp(-k*p0_list - k*x0)))
plt.plot(p0_list, frac_correct,'*k')
ax.set_title('threshold is ' + str(thres));
plt.show()
