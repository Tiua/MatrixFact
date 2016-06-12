# An implementation of matrix factorization
#
try:
    import psutil
    import numpy
    from memory_profiler import profile #only here for the purpose of finding the memory allocation.
    #remove line above if you wish to find the line-by-line runtime ($kernprof -l -v filename.py)
except:
    print("This implementation requires the numpy module.")
    exit(0)
###############################################################################

"""
@INPUT:
    R     : a matrix to be factorized, dimension N x M. Contains all ratings obtained from all users so far
    P     : an initial matrix of dimension N x K. Each row will represent the strength of the associations between users and features
    Q     : an initial matrix of dimension M x K. Each row will represent the strength of the associations between items and features
    K     : the number of latent features
    steps : the maximum number of steps taken to perform the optimisation
    alpha : a constant whose value determines the rate of approaching the minimum
    beta  : the regularization parameter to avoid overfitting. Addition to the original algorithm.
@OUTPUT:
    the updated matrices P and Q

"""

@profile
def matrix_factorization(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02):
    Q = Q.T
    for step in range(steps):
        for i in range(len(R)):  # looping through the initial R matrix
            for j in range(len(R[i])):
                if R[i][j] > 0:  # as long as the cells are populated
                    eij = R[i][j] - numpy.dot(P[i, :], Q[:,
                                                       j])  # finds the error between the estimated rating and the real rating

                    for k in range(K):  # for each feature in the number of latent features
                        P[i][k] = P[i][k] + alpha * (
                        2 * eij * Q[k][j] - beta * P[i][k])  # formulate the update rules for both p_ik and q_kj:
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        eR = numpy.dot(P, Q)  # addition to the initial algorithm definition: Regularization steps
        e = 0
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - numpy.dot(P[i, :], Q[:, j]), 2)
                    for k in range(K):
                        e = e + (beta / 2) * (pow(P[i][k], 2) + pow(Q[k][j], 2))
        if e < 0.001:
            break
    return P, Q.T


###############################################################################
"A small test with simple data to check the algorithm functionality"

if __name__ == "__main__":

#     R = [
#         [5, 0, 4, 0, 1],
#         [0, 2, 1, 1, 3],
#         [1, 0, 0, 0, 4],
#         [2, 0, 3, 5, 0],
#         [5, 1, 0, 2, 2],
#         [2, 4, 1, 5, 3]
#     ]
#
#     R = numpy.array(R)
#
#     N = len(R)
#     M = len(R[0])
#     K = 2
#
#     P = numpy.random.rand(N, K)
#     Q = numpy.random.rand(M, K)
#
#     nP, nQ = matrix_factorization(R, P, Q, K)

# @profile
# def twotest1():
#     R = [
#         [1, 3, 1, 0, 0, 0, 4, 5, 5, 1, 0, 2],
#         [0, 0, 3, 5, 0, 2, 1, 0, 0, 0, 4, 5],
#         [1, 0, 0, 0, 3, 0, 5, 0, 4, 0, 1, 3],
#         [5, 1, 0, 0, 0, 4, 5, 0, 2, 1, 3, 0],
#         [5, 0, 4, 0, 1, 3, 1, 0, 0, 0, 4, 5],
#     ]
#
#     R = numpy.array(R)
#     N = len(R)
#     M = len(R[0])
#     K = 2
#     P = numpy.random.rand(N, K)
#     Q = numpy.random.rand(M, K)
#
#     nP, nQ = matrix_factorization(R, P, Q, K)

# @profile
# def threetest1():
    R = [
        [5, 0, 4, 0, 1, 5, 5, 1, 0, 2, 2, 1, 2, 0, 3, 5, 0, 2, 5, 0, 4, 0, 1, 5, 5, 1, 0, 2, 2, 1, 2, 0, 3, 5, 0, 2],
        [0, 2, 1, 1, 3, 0, 5, 0, 4, 0, 0, 2, 1, 0, 0, 0, 4, 5, 1, 0, 0, 0, 4, 5, 0, 2, 1, 1, 3, 0, 5, 0, 4, 0, 1, 3],
        [2, 0, 3, 5, 0, 2, 1, 0, 0, 0, 4, 5, 0, 2, 1, 1, 3, 0, 5, 1, 0, 2, 2, 1, 2, 0, 3, 5, 1, 3, 1, 0, 0, 0, 4, 5],
        [5, 0, 4, 0, 1, 5, 5, 1, 0, 2, 2, 1, 2, 0, 3, 5, 0, 2, 0, 2, 1, 1, 3, 0, 5, 0, 4, 0, 0, 2, 1, 0, 0, 0, 4, 5],
        [1, 0, 0, 0, 4, 5, 0, 2, 1, 1, 3, 0, 5, 0, 4, 0, 1, 3, 2, 0, 3, 5, 0, 2, 1, 0, 0, 0, 4, 5, 0, 2, 1, 1, 3, 0],
        [5, 1, 0, 2, 2, 1, 2, 0, 3, 5, 1, 3, 1, 0, 0, 0, 4, 5, 5, 0, 4, 0, 1, 5, 5, 1, 0, 2, 2, 1, 2, 0, 3, 5, 0, 2],
        [0, 2, 1, 1, 3, 0, 5, 0, 4, 0, 0, 2, 1, 0, 0, 0, 4, 5, 1, 0, 0, 0, 4, 5, 0, 2, 1, 1, 3, 0, 5, 0, 4, 0, 1, 3],
        [2, 0, 3, 5, 0, 2, 1, 0, 0, 0, 4, 5, 0, 2, 1, 1, 3, 0, 5, 1, 0, 2, 2, 1, 2, 0, 3, 5, 1, 3, 1, 0, 0, 0, 4, 5],
        [0, 2, 1, 1, 3, 0, 5, 0, 4, 0, 0, 2, 1, 0, 0, 0, 4, 5, 5, 1, 0, 2, 2, 1, 2, 0, 3, 5, 1, 3, 1, 0, 0, 0, 4, 5],
        [1, 0, 0, 0, 4, 5, 0, 2, 1, 1, 3, 0, 5, 0, 4, 0, 1, 3, 2, 0, 3, 5, 0, 2, 1, 0, 0, 0, 4, 5, 0, 2, 1, 1, 3, 0],
        [5, 1, 0, 2, 2, 1, 2, 0, 3, 5, 1, 3, 1, 0, 0, 0, 4, 5, 5, 0, 4, 0, 1, 5, 5, 1, 0, 2, 2, 1, 2, 0, 3, 5, 0, 2],
        [0, 2, 1, 1, 3, 0, 5, 0, 4, 0, 0, 2, 1, 0, 0, 0, 4, 5, 1, 0, 0, 0, 4, 5, 0, 2, 1, 1, 3, 0, 5, 0, 4, 0, 1, 3],
        [2, 0, 3, 5, 0, 2, 1, 0, 0, 0, 4, 5, 0, 2, 1, 1, 3, 0, 5, 1, 0, 2, 2, 1, 2, 0, 3, 5, 1, 3, 1, 0, 0, 0, 4, 5],
        [5, 0, 4, 0, 1, 5, 5, 1, 0, 2, 2, 1, 2, 0, 3, 5, 0, 2, 0, 2, 1, 1, 3, 0, 5, 0, 4, 0, 0, 2, 1, 0, 0, 0, 4, 5],
        [1, 0, 0, 0, 4, 5, 0, 2, 1, 1, 3, 0, 5, 0, 4, 0, 1, 3, 2, 0, 3, 5, 0, 2, 1, 0, 0, 0, 4, 5, 0, 2, 1, 1, 3, 0],
        [5, 0, 4, 0, 1, 5, 5, 1, 0, 2, 2, 1, 2, 0, 3, 5, 0, 2, 5, 0, 4, 0, 1, 5, 5, 1, 0, 2, 2, 1, 2, 0, 3, 5, 0, 2],
        [0, 2, 1, 1, 3, 0, 5, 0, 4, 0, 0, 2, 1, 0, 0, 0, 4, 5, 1, 0, 0, 0, 4, 5, 0, 2, 1, 1, 3, 0, 5, 0, 4, 0, 1, 3],
        [2, 0, 3, 5, 0, 2, 1, 0, 0, 0, 4, 5, 0, 2, 1, 1, 3, 0, 5, 1, 0, 2, 2, 1, 2, 0, 3, 5, 1, 3, 1, 0, 0, 0, 4, 5],
        [5, 0, 4, 0, 1, 5, 5, 1, 0, 2, 2, 1, 2, 0, 3, 5, 0, 2, 0, 2, 1, 1, 3, 0, 5, 0, 4, 0, 0, 2, 1, 0, 0, 0, 4, 5],
        [1, 0, 0, 0, 4, 5, 0, 2, 1, 1, 3, 0, 5, 0, 4, 0, 1, 3, 2, 0, 3, 5, 0, 2, 1, 0, 0, 0, 4, 5, 0, 2, 1, 1, 3, 0],
        [5, 1, 0, 2, 2, 1, 2, 0, 3, 5, 1, 3, 1, 0, 0, 0, 4, 5, 5, 0, 4, 0, 1, 5, 5, 1, 0, 2, 2, 1, 2, 0, 3, 5, 0, 2],
        [0, 2, 1, 1, 3, 0, 5, 0, 4, 0, 0, 2, 1, 0, 0, 0, 4, 5, 1, 0, 0, 0, 4, 5, 0, 2, 1, 1, 3, 0, 5, 0, 4, 0, 1, 3],
        [2, 0, 3, 5, 0, 2, 1, 0, 0, 0, 4, 5, 0, 2, 1, 1, 3, 0, 5, 1, 0, 2, 2, 1, 2, 0, 3, 5, 1, 3, 1, 0, 0, 0, 4, 5],
        [0, 2, 1, 1, 3, 0, 5, 0, 4, 0, 0, 2, 1, 0, 0, 0, 4, 5, 5, 1, 0, 2, 2, 1, 2, 0, 3, 5, 1, 3, 1, 0, 0, 0, 4, 5],
        [1, 0, 0, 0, 4, 5, 0, 2, 1, 1, 3, 0, 5, 0, 4, 0, 1, 3, 2, 0, 3, 5, 0, 2, 1, 0, 0, 0, 4, 5, 0, 2, 1, 1, 3, 0],
        [5, 1, 0, 2, 2, 1, 2, 0, 3, 5, 1, 3, 1, 0, 0, 0, 4, 5, 5, 0, 4, 0, 1, 5, 5, 1, 0, 2, 2, 1, 2, 0, 3, 5, 0, 2],
        [0, 2, 1, 1, 3, 0, 5, 0, 4, 0, 0, 2, 1, 0, 0, 0, 4, 5, 1, 0, 0, 0, 4, 5, 0, 2, 1, 1, 3, 0, 5, 0, 4, 0, 1, 3],
        [2, 0, 3, 5, 0, 2, 1, 0, 0, 0, 4, 5, 0, 2, 1, 1, 3, 0, 5, 1, 0, 2, 2, 1, 2, 0, 3, 5, 1, 3, 1, 0, 0, 0, 4, 5],
        [5, 0, 4, 0, 1, 5, 5, 1, 0, 2, 2, 1, 2, 0, 3, 5, 0, 2, 0, 2, 1, 1, 3, 0, 5, 0, 4, 0, 0, 2, 1, 0, 0, 0, 4, 5],
        [1, 0, 0, 0, 4, 5, 0, 2, 1, 1, 3, 0, 5, 0, 4, 0, 1, 3, 2, 0, 3, 5, 0, 2, 1, 0, 0, 0, 4, 5, 0, 2, 1, 1, 3, 0]

    ]

    R = numpy.array(R)
    N = len(R)
    M = len(R[0])
    K = 2

    P = numpy.random.rand(N, K)
    Q = numpy.random.rand(M, K)

    nP, nQ = matrix_factorization(R, P, Q, K)

# @profile
# def fourtest1():
#     R = [
#         [5, 0, 4, 0, 1, 3, 1, 0, 0, 0, 4, 1],
#         [0, 2, 1, 1, 3, 0, 5, 0, 4, 0, 1, 3],
#         [1, 0, 0, 0, 4, 5, 0, 2, 1, 1, 3, 0],
#         [2, 0, 3, 5, 0, 2, 1, 0, 0, 0, 4, 5],
#         [5, 1, 0, 2, 2, 1, 2, 0, 3, 5, 0, 2],
#         [5, 5, 1, 0, 2, 2, 1, 2, 0, 3, 5, 0],
#         [2, 5, 0, 4, 0, 1, 3, 1, 0, 0, 0, 1],
#         [5, 1, 0, 2, 2, 1, 2, 0, 3, 5, 0, 2],
#         [2, 0, 3, 5, 0, 2, 1, 0, 0, 0, 4, 2],
#         [5, 0, 2, 1, 1, 3, 0, 5, 0, 4, 0, 1],
#         [3, 2, 0, 3, 5, 0, 2, 1, 0, 0, 0, 4],
#
#     ]
#
#     R = numpy.array(R)
#
#     N = len(R)
#     M = len(R[0])
#     K = 2
#     P = numpy.random.rand(N, K)
#     Q = numpy.random.rand(M, K)
#
#     nP, nQ = matrix_factorization(R, P, Q, K)

    #if __name__ == "__main__":
   #     test1
    #    twotest1
    #    threetest1
    #    fourtest1
