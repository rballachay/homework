
## Question 1.b

The probability that the first parent does not contain a given allele can be written as $P( x_1 \neq A)$. The second probability is the conditional probability $P(x_2 \neq A | x_1 \neq A )$, the probabilites of these, respectively are $(2N-1)/2N$ and $(2N-2)/2N$, and we run this 100 times, so the probability that NONE of the children contain allele $A$ can be written as follows:

$(P( x_1 \neq A) * P(x_2 \neq A | x_1 \neq A ))^N = ((2N-1)/(2N) * (2N-2)/(2N-1))^N = ((2N-2)/2N)^N$

This is the probability of survival, so the probability of extinction is:

$1 - ((2N-2)/2N)^N$  = 0.366  when N=100

This is very close to the number we get through experimentation, which ranges from 0.61 to 0.65.

## Question 1.f

The probability that the SNP42 is extinct after one generation can be calculated as follows:

$P( x_1 \neq SNP42 ) * P(x_2 \neq SNP42 | x_1 \neq SNP42) = (2N-1.5)/(2N+0.5) * (2N-2.5)/(2N-0.5)$


## Question 3.a 

![Assembled Genome](./results/sequence.png)

## Question 3.b
Original: S1-R1-S2-R2-S3-R1-S4-R2-S5-R1-S6

1. Switch S3\S5: 
    S1-R1-S2-R2-S5-R1-S4-R2-S3-R1-S6

2. Switch S2\S4
    S1-R1-S4-R2-S3-R1-S2-R2-S5-R1-S6

3. Switch R2\R4 + R3\R5
    S1-R1-S4-R2-S5-R1-S2-R2-S3-R1-S6

## Question 3.c 

Assuming our algorithm is producing reads of 100 base pairs and our sequence contains a 1000 bp short tandem repeat, we will have a lot of fragments that will contain exactly the same bases. Because the goal of our algorithm is to make the shortest path possible, these reads will overlap perfectly and collapse down to a total repeat region of closer to 100, instead of 1000. 