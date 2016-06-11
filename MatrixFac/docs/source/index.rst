Matrix Factorization Intro
===========================================================================
	
The need of finding most relevant items on the internet with little efforts has became increasingly popular in the last years. Recommender systems are bringing a contribution to this issue by using algorithms that will analyze the user data and give recommendations based on the results.

Matrix factorization is one of the methods used in recommender systems, and is able to generate recommendations using the difference between the expected result and the actual one. 

In the next chapters, a more thorough analysis of the Matrix Factorization method will be presented, together with the algorithm implementation and a practical example. 

Implementation of Matrix Factorization     
=======================================

In the sections below, a step-by-step implementation of the Matrix Factorization algorithm is presented. 
The programming language used for the implementation of the algorithm is Python, version 3.5.
(download page here: https://www.python.org/downloads/)

Parameters definition
^^^^^^^^^^^^^^^^^^^^^^

For performing the algorithm logic, we first need to define the following parameters: 

	* A matrix *R*, containing ratings from all users is defined and will be factorized;
	* *P* and *Q*, two matrices of which product need to be approximating R. The rows in the two matrices show he associations strength between the users, respecively items and features;
	* *K*, representing the number of latent features;
	* *alpha*, a constant indicating the rate of approaching to the minimum. A small value is recommended for alpha, so that small steps are taken towards the minimum;
	* *beta*, a regularization parameter that helps in avoiding overfitting. It is part of the algorithm extension implementation.

	Here is the main algorithm definition, containing all parameters mentioned above::

		def mat_fact(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02)
	
	insert table here with real data set!

Finding the estimated error
^^^^^^^^^^^^^^^^^^^^^^^^^^^
Once a matrix containing all user information has been defined, we need to find the composition of *P* and *Q*, taking into account the latent features, *K*

    * to begin with, the main matrix *R*, is searched at every iteration we make, so that only populated cells are taken into account::
    	
    	for step in range(steps)
       		for i in range(len(R))
        		for j in range(len(R[i])) 
        			if R[i][j] > 0

    * the dot product of the arrays corresponding to *P* and *Q* is processed in the next step and the result is substracted from the original matrix::

    	eij = R[i][j] - numpy.dot(P[i,:],Q[:,j])

Updating rules definition
^^^^^^^^^^^^^^^^^^^^^^^^^

Extension of algorithm: Regularization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Practical Demonstration     
=======================
Contents:

.. toctree::
   :maxdepth: 2

   help

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

