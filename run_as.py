"""
Script runs numerical simulation described by Avellaneda and Stoikov
https://www.math.nyu.edu/faculty/avellane/HighFrequencyTrading.pdf
"""
from env.price import SimplePriceProcess
from helper import plot_hist
from Strategies.symetric import SymmetricTrader
from Strategies.inventory import InventoryTrader
import matplotlib.pyplot as plt
import numpy as np

# simulation settings
SIMS = 1000  # number of price simulations
DT = 0.005  # time between quote updates
T = 1
# price process settings
SIGMA = 2
K = 1.5
A = 140

# market making settings
GAMMA = 0.1  # risk aversion for symmetric and inventory strategies

# create strategies
labels = ("Symmetric", "Inventory")
colors = ("tab:orange", "tab:blue")
n_strategies = 2
trader_sym = SymmetricTrader(GAMMA, SIGMA, K, K)  # symmetric strategy
trader_inv = InventoryTrader(GAMMA, SIGMA, K, K)  # inventory strategy

# PL & Q log for symmetric strategy
final_pl_sym = np.empty(shape=SIMS)
final_q_sym = np.empty(shape=SIMS)

# PL & Q log for inventory strategy
final_pl_inv = np.empty(shape=SIMS)
final_q_inv = np.empty(shape=SIMS)


for i in range(SIMS):

    tmp_s = []
    tmp_pl = 0

    # create price process simulation
    pp = SimplePriceProcess(n_strategies, SIGMA, K, A, K, A, DT, total_time=T)

    # get initial state
    q, x, s, remain_t, done = pp.state()

    bid = np.empty(shape=n_strategies)  # bid values quoted by strategies
    ask = np.empty(shape=n_strategies)  # ask values quoted by strategies

    while not done:
        # fill bid / ask vectors with quotes
        bid[0], ask[0] = trader_sym.get_quotes(s, remain_t)
        bid[1], ask[1] = trader_inv.get_quotes(q[1], s, remain_t)

        # send quotes to market env
        q, x, s, remain_t, done = pp.quote(bid, ask)

    # calculate final pl
    final_pl_sym[i], final_pl_inv[i] = x + q * s
    final_q_sym[i], final_q_inv[i] = q


# print results and show plots
print("\nSymmetric")
print("PL mean: %.2f std: %.2f  " % (final_pl_sym.mean(), final_pl_sym.std()))
print("Q mean: %.2f std: %.2f" % (final_q_sym.mean(), final_q_sym.std()))
print("\nInventory")
print("PL mean: %.2f std: %.2f  " % (final_pl_inv.mean(), final_pl_inv.std()))
print("Q mean: %.2f std: %.2f" % (final_q_inv.mean(), final_q_inv.std()))


def plot_hist_(name, data, data_labels):
    """plots multiple histograms"""
    plt.figure(name)
    plt.clf()
    min_data = min([d.min() for d in data])
    max_data = max([d.max() for d in data])
    range_data = max_data - min_data
    bins = np.linspace(min_data - 0.1 * range_data, max_data + 0.1 * range_data, 50)

    for d, l in zip(data, data_labels):
        plt.hist(d, bins=bins, alpha=0.2, label=l)

    plt.grid()
    plt.legend()
    plt.title(name)
    plt.savefig("results/%s.png" % name, bbox_inches='tight')


# show histograms
plot_hist("terminal PL histogram", (final_pl_sym, final_pl_inv), labels)


plt.show()
