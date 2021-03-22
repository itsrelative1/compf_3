import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
        V_new[-1] = np.exp(7); V_new[0] = 0

        # save the data
        V_data.append(V)

    # return the data as an array
    return np.array(V_data)

def get_FTCS_delta(V, dx, S_list):
    """get the delta's from a V vector, FTCS method"""

    D_vector = []

    for i in range(1,len(V)-1):
        D = (1/(2*dx*S_list[i])) * (V[i+1] - V[i-1])
        D_vector.append(D)
    return D_vector

def get_CN_delta(V1, V2, dx, S_list):
    """get the delta's from V vector, CN method"""

    D_vector = []

    for i in range(1,len(V)-1):
        D = (1/(4*dx*S_list[i])) * (V1[i+1] - V1[i-1] + V2[i+1] - V2[i-1])
        D_vector.append(D)
    return D_vector

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

    # diag
    a_2 = np.array([1 - r*dt - sigma**2*(dt/dx**2) for _ in range(spacesteps)])

    # lower diag
    a_3 = np.array([-(r-0.5*sigma**2)*dt/(2*dx) + 0.5*sigma**2*(dt/dx**2) for _ in range(spacesteps-1)])

    # set up the sparse matrix by adding the diagonals
    A = np.diagflat(a_1, k=1) + np.diagflat(a_2, k=0) + np.diagflat(a_3, k=-1)

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

    # upper diag
    b_1 = np.array([-(r-0.5*sigma**2)*dt/(4*dx) - 0.25*sigma**2*(dt/dx**2) for _ in range(spacesteps-1)])

    # diag
    b_2 = np.array([1+0.5*sigma**2*(dt/dx**2) + r*dt/2 for _ in range(spacesteps)])

    # lower diag
    b_3 = np.array([(r-0.5*sigma**2)*dt/(4*dx) - 0.25*sigma**2*(dt/dx**2) for _ in range(spacesteps-1)])

    B = np.diagflat(b_1, k=1) + np.diagflat(b_2, k=0) + np.diagflat(b_3, k=-1) 

    # set up the vectors for the A-matrix including boundary conditions

    # upper diag
    a_1 = np.array([(r-0.5*sigma**2)*dt/(4*dx) + 0.25*sigma**2*(dt/dx**2) for _ in range(spacesteps-1)])

    # diag
    a_2 = np.array([1 - 0.5*sigma**2*dt/(dx**2) - 0.5*r*dt  for _ in range(spacesteps)])

    # lower diag
    a_3 = np.array([-(r-0.5*sigma**2)*dt/(4*dx) + 0.25*sigma**2*(dt/dx**2) for _ in range(spacesteps-1)])

    # set up the sparse matrix by adding the diagonals
    A = np.diagflat(a_1, k=1) + np.diagflat(a_2, k=0) + np.diagflat(a_3, k=-1)

    # set up the extra vector for the boundary condition
    f = np.array([0 for i in range(spacesteps)])
    f[-1] =  ((r-0.5*sigma**2)*dt/(4*dx) + 0.25*sigma**2*(dt/dx**2) - -(r-0.5*sigma**2)*dt/(4*dx) - 0.25*sigma**2*(dt/dx**2)) * np.exp(X_max)
    
    # return the necessary matrices/vectors for the FTCS
    return B, A, f


if __name__ == "__main__":

    # details for the algorithm
    method = 'FTCS' #'CN'
    spacesteps = 300
    timesteps = 3333
    r = 0.04
    T = 1
    sigma = 0.3
    X_min = -2
    X_max = 7
    K = 110
    S_0 = 120
    dx = (X_max - X_min)/(spacesteps-1)

    # initialize the FTCS or CN details
    if method == 'FTCS':
        B, A, f = initialize_FTCS_matrix(timesteps, spacesteps, X_max, X_min, T, r, sigma)
    else:
        B, A, f = initialize_CN_matrix(timesteps, spacesteps, X_max, X_min, T, r, sigma)

    # make an X_list such that we have exactly our wanted S_0 price for comparison reasons
    X_list = [np.log(S_0) - dx*i for i in range(int(spacesteps*0.75)+1)][::-1] + [np.log(S_0) + dx*i for i in range(1, int(spacesteps*0.25))]

    # if the array isn't exactly spacesteps long because of rounding, add an additional term
    if len(X_list) != spacesteps:
        X_list.append(np.log(S_0)+dx*(int(spacesteps*0.25)))

    # initialize the first time vector, which is just the terminal payoff for all S
    V = np.array([max(0, np.exp(X)-K) for X in X_list])
    V[-1] = np.exp(X_max)

    # solve the system
    data = solve_system(B,A,V,f,timesteps)

    # print the option value at t=0 for the wanted S_0
    S_list = [np.exp(X) for X in X_list]
    print("Found option value: ", data[-1][int(spacesteps*0.75)], "at S_0 = ", S_list[int(spacesteps*0.75)])
    
    #print(S_list)

    delta_list = []
    for i in range(1, len(data)-1):

        # for FTCS
        if method == 'FTCS':
            deltas=get_FTCS_delta(data[i-1], dx, S_list)

        else:
            deltas=get_CN_delta(data[i], data[i-1], dx, S_list)

        delta_list.append(deltas)

    # convert delta list to array
    delta_list = np.array(delta_list)

    ############ Pretty plots #############
    # we cut off the lower/higher S values since they're of no use to us and only used for calculation

    plt.plot(S_list[:int(spacesteps/1.2)], data[-1][:int(spacesteps/1.2)])
    plt.xlabel('Stock price ($)')
    plt.ylabel('Option value ($)')
    plt.title(f'Option values at t=0 ({method} method)')
    plt.show()

    # for option values
    data = data[:, int(spacesteps/1.45):int(spacesteps/1.2)]
    ny, nx = data.shape
    x = np.array(S_list[int(spacesteps/1.45):int(spacesteps/1.2)])
    y = np.linspace(0,1,ny)
    xv, yv = np.meshgrid(x,y)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    data3d = ax.plot_surface(xv,yv,data, cmap='afmhot')
    ax.set_xlabel('Stock price ($)')
    ax.set_ylabel('Time until expiration (yrs)')
    ax.set_zlabel('Option value ($)')
    ax.set_title(f'Option values ({method} method)')
    fig.colorbar(data3d)
    plt.show()

    # for deltas
    delta_list = delta_list[:, int(spacesteps/1.45):int(spacesteps/1.2)]
    ny, nx = np.array(delta_list).shape
    x = np.array(S_list[int(spacesteps/1.45):int(spacesteps/1.2)])
    y = np.linspace(0,1,ny)
    xv, yv = np.meshgrid(x,y)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    data3d = ax.plot_surface(xv,yv,np.array(delta_list), cmap='afmhot')
    ax.set_xlabel('Stock price ($)')
    ax.set_ylabel('Time until expiration (yrs)')
    ax.set_zlabel('Delta value')
    ax.set_title(f'Delta values ({method} method)')
    fig.colorbar(data3d)
    plt.show()

    # some simple functions for errors
    def rel(val):
        BS_val = 9.62534492
        return abs(BS_val-val)

    def rel_110(val):
        BS_val = 15.12858760
        return abs(BS_val-val)

    def rel_120(val):
        BS_val = 21.78881954
        return abs(BS_val-val)

    # plot the errors, results obtained manually from individual runs

    # CN method results
    plt.semilogy([100,200,500,1000], [rel(val) for val in [9.524381475654614, 9.607473374668396, 9.624860270215237,\
         9.625452779297067]], label='S_0 = 100')
    plt.semilogy([100,200,500,1000], [rel_110(val) for val in [14.985994607256139, 15.09379860635135, 15.123078730633424,\
         15.127216809767091]], label='S_0 = 110')
    plt.semilogy([100,200,500,1000], [rel_120(val) for val in [21.676899240522815, 21.765456752955526, 21.786739047180166,\
         21.78873015422363]], label='S_0 = 120')
    plt.xlabel('Grid points (x 1000)')
    plt.ylabel('Absolute error')
    plt.title('Absolute error of CN method compared \nto Black-Scholes values')
    plt.legend()
    plt.show()

    # FTCS method results
    plt.semilogy([100,200,500,1000], [rel(val) for val in [9.525788013950654, 9.608854910621005, 9.626237002098552, 9.625548093936741]], label='S_0 = 100')
    plt.semilogy([100,200,500,1000], [rel_110(val) for val in [14.987816102941991,15.095571233889768,15.12483901207327,15.12395819227632]], label='S_0 = 110')
    plt.semilogy([100,200,500,1000], [rel_120(val) for val in [21.678628278552363,21.76715830587018,21.788434351375287,21.787586034091788]], label='S_0 = 120')
    plt.xlabel('Grid points (x 1000)')
    plt.ylabel('Absolute error')
    plt.title('Absolute error of FTCS method compared \nto Black-Scholes values')
    plt.legend()
    plt.show()