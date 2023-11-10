
## Question 1.b

The probability that the first parent does not contain a given allele can be written as $P( x_1 \neq A)$. The second probability is the conditional probability $P(x_2 \neq A | x_1 \neq A )$, the probabilites of these, respectively are $(2N-1)/2N$ and $(2N-2)/2N$, and we run this 100 times, so the probability that NONE of the children contain allele $A$ can be written as follows:

$(P( x_1 \neq A) * P(x_2 \neq A | x_1 \neq A ))^N = ((2N-1)/(2N) * (2N-2)/(2N-1))^N = ((2N-2)/2N)^N$

This is the probability of survival, so the probability of extinction is:

$1 - ((2N-2)/2N)^N$  = 0.366  when N=100

This is very close to the number we get through experimentation, which ranges from 0.61 to 0.65.