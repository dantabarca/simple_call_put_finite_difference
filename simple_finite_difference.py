import numpy as np
from matplotlib import pyplot as plt

###########################################
# A simple example of a finite difference
# code to price European options.
# Uses explicit or implicit time evolution.
###########################################

class putcall_fd():
    def __init__(self,
                 rate = 0.1,
                 volatility=0.4,
                 stike = 50,
                 iscall = True,
                 num_times = 1000,
                 num_prices =100,
                 tfinal = 1,
                 maxprice = 100):

        self.r = rate
        self.s = volatility
        self.strike = strike
        self.iscall = iscall
        self.n = num_times
        self.m = num_prices
        self.t = tfinal
        self.smax = maxprice




    #construct the evolution matrix
    def construct_imex_operator(self, explicit = True):
        r = self.r
        s = self.s
        dt = t/n

        alpha = (lambda i: 0.5*i*dt*(i*s*s-r)) if explicit else (lambda i: 0.5*i*dt*(r-i*s*s))
        beta = (lambda i: -dt*(i*i*s*s+r)) if explicit else (lambda i: dt*(i*i*s*s+r))
        gamma = (lambda i: 0.5*i*dt*(i*s*s+r)) if explicit else (lambda i: -0.5*i*dt*(i*s*s+r))


        A = np.zeros((m-1,m-1))

        for i in range(0,m-1):
            A[i,i] = 1+beta(i+1)

        for i in range(0,m-2):
            A[i+1,i] = alpha(i+2)

        for i in range(0,m-2):
            A[i,i+1] = gamma(i+1)

        A[0,0] += 2*alpha(1)
        A[0,1] += -alpha(1)

        A[m-2,m-3] += -gamma(m-1)
        A[m-2,m-2] += 2*gamma(m-1)
        return A


    #final condition for a put/call
    def set_final_condition(self):
        strike = self.strike
        sgrid = np.linspace(0, self.smax, m+1)
        s = sgrid[1:-1]
        ffinal = np.where(s>strike, s-strike,0) if iscall else np.where(s<strike, strike-s,0)
        return s,ffinal

    #does the explicit evolution
    def evolve_explicit(self, ffinal):
        fex = ffinal.copy()
        A = self.construct_imex_operator(explicit=True)
        fjm1_ex = np.dot(A,fex)
        for i in range(self.n):
            fjm1_ex = np.dot(A,fex)
            fex = fjm1_ex.copy()
        return fex

    #does the implicit evolution
    def evolve_implicit(self, ffinal):
        fim = ffinal.copy()
        A = self.construct_imex_operator(explicit=False)
        fjm1_im = np.dot(A,fim)
        for i in range(self.n):
            fjm1_im = np.linalg.solve(A,fim)
            fim = fjm1_im.copy()
        return fim


#simple example
if __name__ == '__main__':

    # risk-free rate
    r = 0.1
    # volatility
    s = 0.4
    # strike prices
    strike = 50
    #call or put
    iscall = True

    #number of grid points in t (n) and s (m)
    # if n is too small, then the explicit
    # evolution will become unstable
    n = 1000
    m = 100

    #time grid
    t = 1/2.
    tgrid = np.linspace(0,t,n+1)

    #grid of underlying prices
    smax = 100
    sgrid = np.linspace(0, smax, m+1)



    #initialize the finite differencing object
    call_fd = putcall_fd(r, s, strike, iscall, n, m, t, smax)

    #set the final condition (K-S) for put or (S-K) for call
    s, ffinal = call_fd.set_final_condition()

    #do the explicit finite differncing
    finit_ex = call_fd.evolve_explicit(ffinal)

    #do the implicit finite differncing
    finit_im = call_fd.evolve_implicit(ffinal)



    #plot the results
    fig, ax = plt.subplots(1)
    ax.plot(s, ffinal)
    ax.plot(s, finit_ex)
    ax.plot(s, finit_im)

    ax.axis([0, smax, -10, smax/2+10])
    ax.grid()
    ax.set_xlabel("Underlying Price")
    if iscall:
        ax.set_ylabel("Call value")
    else:
        ax.set_ylabel("Put value")


    plt.show()


