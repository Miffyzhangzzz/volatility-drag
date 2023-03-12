import numpy as np
import matplotlib.pyplot as plt

# number of steps
n = 100
# time in years
T = 1
# number of sims
M = 1000000
# initial stock price
S0 = 1
# volatility sigma

# calc each time step
dt = T/n


def gbm_sim(mu, sigma):
    
    np.random.seed(123)
    St = np.exp(
        (mu - sigma ** 2 / 2) * dt
        + sigma * np.random.normal(0, np.sqrt(dt), size=(M,n)).T)
    

    St = np.vstack([np.ones(M), St])

    St = S0 * St.cumprod(axis=0)

    
    return  St

sim_1 = gbm_sim(0.1,0.1)[-1]
sim_2 = gbm_sim(0.1,0.25)[-1]

def plot_sim_hist():
    
    hist_1 = sim_1
    hist_2 = sim_2
    # plt.hist(hist_1,bins=50 ,alpha = 0.8,color ='blue', label = 'simulation 1')
    # plt.hist(hist_2,bins=50,  alpha = 0.3,color = 'green', label = 'simulation 2')   
    
    plt.hist([hist_1,hist_2], bins=100, label = ['sim1', 'sim2 - higher vol'])
    plt.title("Terminal Wealth - vol drag effect")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend(loc='upper right')
    plt.show()

plot_sim_hist()        
    