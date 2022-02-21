# Recreation of instances proving Theorem 3.1

This code is run by calling `python InfiniteExample/neverending_example.py`.
Change the values of a, and d in the file and play around with the results.

This code creates and solves the instance:

min x_1 - (10 + d) x_2 - a x_3

-0.5 x_2 + 3 x_3 <= 0

-x_3 <= 0

-0.5 x_1 + 0.5 x_2 - 3.5 x_3 <= 0

0.5 x_1 + 1.5 x_3 <= 0.5

x_1 in Z, x_2 in R, x_3 in {0, 1}

At each round, the highest scoring cut of the following is applied and the LP re-solved:

-10 x_1 + 10 x_2 + x_3 <= 0

-1 x_1 + x_3 <= 1 - eps(n)

-1 x_1 + 10 x_2 <= 30.5 - eps(n)

The cut-scoring rule is: \lambda * integer_support + (1 - \lambda) * objective_parallelism

Using the results of Theorem 3.1, output is given which tells us what will happen. The solve process then occurs,
and the result should shadow what we predict will occur.