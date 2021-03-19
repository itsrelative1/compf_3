import numpy as np
import matplotlib.pyplot as plt

# functions to use here

def solve_system(B, A, V, f, timesteps):
    """
    Takes the linear systems specified by the FTCS/CN methods and solves them.
    """

    # save all the solved data for later so we can plot
    V_data = [V]

    # invert matrix B only once to save comp time
    B_inv = np.linalg.inv(B)

    # now we also need identity matrix for the solver
    I = np.identity(len(A))

    for _ in range(1, timesteps):
        V_new = np.linalg.solve(I, B_inv.dot(A.dot(V) + f))
        V = V_new

        # manually set the boundaries
        V_new[-1] = np.exp(7)-110; V_new[0] = 0

        # save the data
        V_data.append(V)

    # return the data as an array
    return np.array(V_data)


def initialize_FTCS_matrix(timesteps, spacesteps, X_max, X_min, T, r, sigma):
    """Sets up the matrices for the FTCS method"""

    # first determine dx and dt
    dt = T/(timesteps-1)
    dx = (X_max-X_min)/(spacesteps-1)

    # for FTCS, the B-matrix is just the identity
    B = np.identity(spacesteps)

    # set up the vectors for the A-matrix including boundary conditions

    # upper diag
    a_1 = np.array([(r-0.5*sigma**2)*dt/(2*dx) + 0.5*sigma**2*(dt/dx**2) for _ in range(spacesteps-1)])
    #a_1[0] = 0; a_1[-1] = 0

    # diag
    a_2 = np.array([1 - r*dt - sigma**2*(dt/dx**2) for _ in range(spacesteps)])
    #a_2[0] = 0; a_2[-1] = S_max

    # lower diag
    a_3 = np.array([-(r-0.5*sigma**2)*dt/(2*dx) + 0.5*sigma**2*(dt/dx**2) for _ in range(spacesteps-1)])
    #a_3[0] = 0; a_3[-1] = 0

    # set up the sparse matrix by adding the diagonals
    A = np.diagflat(a_1, k=1) + np.diagflat(a_2, k=0) + np.diagflat(a_3, k=-1)

    print(A)
    # set up the extra vector for the boundary condition
    f = np.array([0 for i in range(spacesteps)])
    f[-1] = np.exp(X_max)
    
    # return the necessary matrices/vectors for the FTCS
    return B, A, f

    
def initialize_CN_matrix(timesteps, spacesteps, X_max, X_min, T, r, sigma):
    """Sets up the matrices for the CN method"""

    # first determine dx and dt
    dt = T/(timesteps-1)
    dx = (X_max-X_min)/(spacesteps-1)

    # set up B matrix
    B = np.identity(spacesteps)

    #upper diag
    b_1 = np.array([-(r-0.5*sigma**2)*dt/(4*dx) - 0.25*sigma**2*(dt/dx**2) for _ in range(spacesteps-1)])

    # diag
    b_2 = np.array([1+0.5*sigma**2*(dt/dx**2) + r*dt/2 for _ in range(spacesteps)])

    # lower diag
    b_3 = np.array([(r-0.5*sigma**2)*dt/(4*dx) - 0.25*sigma**2*(dt/dx**2) for _ in range(spacesteps-1)])

    B = np.diagflat(b_1, k=1) + np.diagflat(b_2, k=0) + np.diagflat(b_3, k=-1) 

    # set up the vectors for the A-matrix including boundary conditions

    # upper diag
    a_1 = np.array([(r-0.5*sigma**2)*dt/(4*dx) + 0.25*sigma**2*(dt/dx**2) for _ in range(spacesteps-1)])
    #a_1[0] = 0; a_1[-1] = 0

    # diag
    a_2 = np.array([1 - 0.5*sigma**2*dt/(dx**2) - 0.5*r*dt  for _ in range(spacesteps)])
    #a_2[0] = 0; a_2[-1] = S_max

    # lower diag
    a_3 = np.array([-(r-0.5*sigma**2)*dt/(4*dx) + 0.25*sigma**2*(dt/dx**2) for _ in range(spacesteps-1)])
    #a_3[0] = 0; a_3[-1] = 0

    # set up the sparse matrix by adding the diagonals
    A = np.diagflat(a_1, k=1) + np.diagflat(a_2, k=0) + np.diagflat(a_3, k=-1)

    print(A)
    # set up the extra vector for the boundary condition
    f = np.array([0 for i in range(spacesteps)])
    f[-1] =  ((r-0.5*sigma**2)*dt/(4*dx) + 0.25*sigma**2*(dt/dx**2) -(r-0.5*sigma**2)*dt/(4*dx) - 0.25*sigma**2*(dt/dx**2)) * np.exp(X_max)
    
    # return the necessary matrices/vectors for the FTCS
    return B, A, f


if __name__ == "__main__":

    # details for the algorithm
    spacesteps = 500
    timesteps = 1000
    r = 0.04
    T = 1
    sigma = 0.3
    X_min = -2
    X_max = 7
    K = 110
    S_min = 0.0001
    S_max = 10000
    S_0 = 100

    dx = (X_max - X_min)/(spacesteps-1)

    S_list = np.linspace(S_min, S_max, spacesteps)
    #print(S_list)
    X_list = [np.log(S) for S in S_list]

    # initialize the FTCS or CN details
    #B, A, f = initialize_FTCS_matrix(timesteps, spacesteps, X_max, X_min, T, r, sigma)
    B, A, f = initialize_CN_matrix(timesteps, spacesteps, X_max, X_min, T, r, sigma)

    # initialize the first time vector, which is just the terminal payoff for all S
    X_list = np.linspace(X_min, X_max, spacesteps)

    # make an X_list such that we have exactly our wanted S_0 price for comparison reasons
    #X_list = [np.log(S_0) - dx*i for i in range(int(spacesteps*0.75)+1)][::-1] + [np.log(S_0) + dx*i for i in range(1, int(spacesteps*0.25))]
    print(X_list)
    print(len(X_list))
    V = np.array([max(0, np.exp(X)-K) for X in X_list])
    V[-1] = np.exp(X_max) - K
    #print(V)

    # solve the system
    data = solve_system(B,A,V,f,timesteps)
    #print(data[-1])

    plt.imshow(np.transpose(data), cmap='Reds')
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.show()

    S_list = [np.exp(X) for X in X_list]
    print(S_list)
    plt.plot(S_list, data[-1])
    plt.show()