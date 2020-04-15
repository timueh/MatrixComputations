[![Build Status](https://travis-ci.org/timueh/MatrixComputations.svg?branch=master)](https://travis-ci.org/timueh/MatrixComputations)

# Matrix computations

A Julia package that explores basic algorithms for matrix computations (LU decomposition, QR decomposition, ...).
This code is *not meant to be production ready*; it's a side project I started.
Why bother with such elementary matrix computations?
The answer is simple: to get a better understanding for the elementary algorithms that are under the hood of pretty much all numerical tools.

The algorithms are based on two prominent sources:

- "Matrix Computations", G.H. Golub and C.F. Van Loan, Johns Hopkins University Press, 1983
- "Fundamentals of Matrix Computations", D.S. Watkins, John Wiley & Sons, 1991

The code has been tested for Julia versions >= 1.2.