
## Question 1.b

The probability that the first parent does not contain a given allele can be written as $P( x_1 \neq A)$. The second probability is the conditional probability $P(x_2 \neq A | x_1 \neq A )$, the probabilites of these, respectively are $(2N-1)/2N$ and $(2N-2)/2N$, and we run this 100 times, so the probability that NONE of the children contain allele $A$ can be written as follows:

$(P( x_1 \neq A) * P(x_2 \neq A | x_1 \neq A ))^N = ((2N-1)/(2N) * (2N-2)/(2N-1))^N = ((2N-2)/2N)^N$

This is the probability of survival, so the probability of extinction is:

$1 - ((2N-2)/2N)^N$  = 0.366  when N=100

This is very close to the number we get through experimentation, which ranges from 0.61 to 0.65.

## Question 1.f

The probability that the SNP42 is extinct after one generation can be calculated as follows:

$(P( x_1 \neq SNP42 ) * P(x_2 \neq SNP42 | x_1 \neq SNP42))^N = ((2N-1.5)/(2N+0.5) * (2N-2.5)/(2N-0.5))^N$

This is the probability of survival, so the probability of extinction is:

$1 - (())$

# Question 2.d

Mathematically, the chi-squared test checks each variable in our 2-D table of alle counts (0,1,2) and the class observed (disease, no disease). It then uses the calculated expected value (the expected number of counts assuming no correlation) and the observed values (actual allele counts) and the following formula to calculate chi-squared:

$$\chi^2 = \sum \frac {(O - E)^2}{E}$$

This is summarized over each of our three alleles. Meanwhile, we calculate the odds ratio as the number of effected individuals given that they have a certain allele. This test, however ignores the expected and observed likelihood of disease vs non-disease, and thus measures the effect size given a disease/non-disease exists, but doesn't help to know if a correlation between allele type and the disease exists.


## Question 3.a 

![Assembled Genome](./results/sequence.png)

## Question 3.b
Original: S1-R1-S2-R2-S3-R1-S4-R2-S5-R1-S6

1. Switch S3\S5: 

    S1-R1-S2-R2-S5-R1-S4-R2-S3-R1-S6

2. Switch S2\S4:

    S1-R1-S4-R2-S3-R1-S2-R2-S5-R1-S6

3. Switch R2\R4 + R3\R5:

    S1-R1-S4-R2-S5-R1-S2-R2-S3-R1-S6

## Question 3.c 

Assuming our algorithm is producing reads of 100 base pairs and our sequence contains a 1000 bp short tandem repeat, we will have a lot of fragments that will contain exactly the same bases. Because the goal of our algorithm is to make the shortest path possible, these reads will overlap perfectly and collapse down to a total repeat region of closer to 100, instead of 1000. 