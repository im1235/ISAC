"""
Script performs:
- Training of ISAC model
- Benchmark against Inventory and Symmetric strategies
- Visualization of results
"""
from env.price import SimplePriceProcess, DetailedPriceProcess
from Strategies.symetric import SymmetricTrader
from Strategies.inventory import InventoryTrader
from Strategies.inventory_sac import InventoryAITrader
from control.sac_e import SoftActorCritic
from helper import plot_hist
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d  # don't remove
import numpy as np
import torch
import time

# simulation settings
T = 1           # duration
DT = 0.005      # time between quote updates
SIMS = 2000     # number of simulations in benchmark

# price process settings
SIGMA = 2
K = 1.5
A = 140

# market making settings
GAMMA = 0.1  # risk aversion for symmetric and inventory

# sac settings
TAU = 0.01  # soft update parameters
LEARN_RATE = 5e-5
HIDDEN_DIM = 128
TRAIN_EPOCHS = 2000
FORCE_CPU = False  # force execution on cpu

# configure execution device
if not FORCE_CPU and torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

# create soft actor critic advisor
sac = SoftActorCritic(2, 1, HIDDEN_DIM, device,
                      lr=LEARN_RATE,
                      tau=TAU,
                      mem_capacity=200_000,
                      mem_batch_size=10_000,
                      mem_batch_abs_mode=True,
                      mem_min_samples=20000,
                      mem_update_interval=1,
                      )

# create strategies
labels = ("Symmetric", "Inventory", "ISAC")
colors = ("r", "b", "g")
n_strategies = 3
trader_sym = SymmetricTrader(GAMMA, SIGMA, K, K)  # symmetric strategy
trader_inv = InventoryTrader(GAMMA, SIGMA, K, K)  # inventory strategy
trader_inv_sac = InventoryAITrader(SIGMA, K, A, K, A, DT, sac)  # inventory strategy with sac


"""
train inventory strategy with sac advisor
"""


train_rewards = np.empty(shape=TRAIN_EPOCHS)

train_start = time.time()
epochs_start = time.time()
for i in range(TRAIN_EPOCHS):

    bid = np.empty(shape=1)  # ask quoted by strategy
    ask = np.empty(shape=1)  # bid quoted by str.

    # create price process simulation
    pp = SimplePriceProcess(1, SIGMA, K, A, K, A, DT, total_time=T)

    # get initial state
    q, x, s, remain_t, done = pp.state()

    # simulation loop, training logic is inside get_quotes
    while not done:
        # obtain quotes from model
        bid[0], ask[0] = trader_inv_sac.get_quotes(x[0], q[0], s, remain_t, train_mode=True)
        # send quotes to market simulation
        q, x, s, remain_t, done = pp.quote(bid, ask)

    # save cumulative reward for epoch
    train_rewards[i] = trader_inv_sac.reward_total

    # finish training epoch
    trader_inv_sac.reset()

    # plot rewards after training epoch
    plt.figure("Rewards")
    plt.clf()
    plt.plot(train_rewards[:i + 1])
    plt.xlabel("epoch")
    plt.ylabel("reward")
    plt.grid()
    plt.savefig("results/rewards.png", bbox_inches='tight')
    plt.pause(0.001)

    # optionally display epoch info
    if i % 10 == 0 and i > 0:
        now = time.time()
        epoch_dur = now - epochs_start
        epochs_start = now
        train_dur = (now - train_start) / 60
        quotes_ps = 1 / DT * 10 / epoch_dur
        print("training %.2f%% done in %.2f min, running %.2f quote/s"
              % (100. * i / TRAIN_EPOCHS, train_dur, quotes_ps))
        print("MEM %d items, sample size %d\n" % (trader_inv_sac.advisor.replay_buffer.current_capacity,
                                                  trader_inv_sac.advisor.replay_buffer.current_batch_size))


""" 
benchmark ISAC against other strategies
"""


# PL & Q log for symmetric strategy
final_pl_sym = np.empty(shape=(SIMS, 1))
final_q_sym = np.empty(shape=(SIMS, pp.steps))

# PL & Q log for inventory strategy
final_pl_inv = np.empty(shape=(SIMS, 1))
final_q_inv = np.empty(shape=(SIMS, pp.steps))

# PL & Q log for inventory sac strategy
final_pl_invs = np.empty(shape=(SIMS, 1))
final_q_invs = np.empty(shape=(SIMS, pp.steps))

# variables for plotting details of first simulation
plot_gamma = []
plot_bid = []
plot_ask = []
plot_bid_f = []
plot_ask_f = []
plot_q = []
plot_s = []
plot_pl = []

sim_start = time.time()
epochs_start = time.time()

for i in range(SIMS):

    tmp_s = []
    tmp_pl = 0

    # create price process simulation
    pp = DetailedPriceProcess(n_strategies, SIGMA, K, A, K, A, DT,  total_time=T)

    # get initial state
    q, x, s, remain_t, done = pp.state()

    bid = np.empty(shape=n_strategies)  # bid values quoted by strategies
    ask = np.empty(shape=n_strategies)  # ask values quoted by strategies

    while not done:
        # fill bid / ask vectors with quotes
        bid[0], ask[0] = trader_sym.get_quotes(s, remain_t)
        bid[1], ask[1] = trader_inv.get_quotes(q[1], s, remain_t)
        bid[2], ask[2] = trader_inv_sac.get_quotes(x[2], q[2], s, remain_t, train_mode=False)

        # record data if first simulation
        if i == 0:
            plot_gamma.append(trader_inv_sac.gamma)
            plot_bid.append(bid[2])
            plot_ask.append(ask[2])

            if pp.bid_hit[2]:
                plot_bid_f.append(plot_bid.__len__()-1)
            if pp.ask_lift[2]:
                plot_ask_f.append(plot_ask.__len__()-1)

            plot_q.append(q[2])
            plot_s.append(s)
            plot_pl.append(pp.last_pl[2])

        # send quotes to market env
        q, x, s, remain_t, done = pp.quote(bid, ask)

    # log final pl and q
    final_pl_sym[i], final_pl_inv[i], final_pl_invs[i] = pp.log_pl[-1]
    final_q_sym[i, ], final_q_inv[i, ], final_q_invs[i, ] = np.abs(pp.log_q[:, 0]), np.abs(pp.log_q[:, 1]), \
        np.abs(pp.log_q[:, 2])

    # optionally display simulation progress
    if i % 10 == 0 and i > 0:
        now = time.time()
        epoch_dur = now - epochs_start
        epochs_start = now
        sim_dur = (now - sim_start) / 60
        quotes_ps = 1 / DT * 10 / epoch_dur
        print("simulation %.2f%% done in %.2f min, running %.2f quote/s" %
              (100. * i / SIMS, sim_dur, quotes_ps))


"""
visualization of obtained results 
"""


def print_info(pl, q, name):
    """ prints strategy info"""
    pl_m = pl.mean()
    pl_std = pl.std()
    pl_std_pr = 100 * pl_std / pl_m

    q_m = q.mean()
    q_std = q.std()
    q_std_pr = 100 * q_std / q_m

    print("\n" + name)
    print("PL mean: %.2f std: %.2f (%.2f%%) " % (pl_m, pl_std, pl_std_pr))
    print("abs(Q) mean: %.2f std: %.2f (%.2f%%) " % (q_m, q_std, q_std_pr))


# print results
print("\nResults:")
print_info(final_pl_sym[:, -1], final_q_sym[:, -1], labels[0])
print_info(final_pl_inv[:, -1], final_q_inv[:, -1], labels[1])
print_info(final_pl_invs[:, -1], final_q_invs[:, -1], labels[2])

# show histograms
plot_hist("terminal PL histogram", (final_pl_sym[:, -1], final_pl_inv[:, -1], final_pl_invs[:, -1]), labels)
plot_hist("terminal Q histogram", (final_q_sym[:, -1], final_q_inv[:, -1], final_q_invs[:, -1]), labels)

# show mean pl & q plots
plt.figure("Mean and 95th percentile of inventory q")
plt.plot(final_q_sym.mean(axis=0), colors[0], label=labels[0] + " mean")
plt.plot(np.percentile(final_q_sym, 95, axis=0), ":"+colors[0], label=labels[0] + " 95th prc.")
plt.plot(final_q_inv.mean(axis=0), colors[1], label=labels[1] + " mean")
plt.plot(np.percentile(final_q_inv, 95, axis=0), ":"+colors[1], label=labels[1] + " 95th prc.")
plt.plot(final_q_invs.mean(axis=0), colors[2], label=labels[2] + " mean")
plt.plot(np.percentile(final_q_invs, 95, axis=0), ":"+colors[2], label=labels[2] + " 95th prc.")
plt.grid()
plt.legend()
plt.ylabel("inventory q")
plt.xlabel("step")
plt.savefig("results/inventory_evolution.png", bbox_inches='tight')

# show example of price evolution, quotes, q and gamma
plt.figure("Example of simulated path", figsize=(6, 11))
plt.subplot(411)
plt.plot(plot_ask, ":rD", label="ask", markevery=plot_ask_f)
plt.plot(plot_bid, ":gD", label="bid", markevery=plot_bid_f)
plt.plot(plot_s, "b", label="midprice")

plt.xlabel("step")
plt.grid()
plt.legend(loc='lower left')
plt.subplot(412)
plt.plot(plot_gamma, "b", label="gamma")
plt.xlabel("step")
plt.grid()
plt.legend()
plt.subplot(413)
plt.plot(plot_q, "g", label="inventory")
plt.xlabel("step")
plt.legend()
plt.grid()
plt.subplot(414)
plt.plot(plot_pl, "g", label="PL")
plt.xlabel("step")
plt.legend()
plt.grid()
plt.savefig("results/single_price_sim.png", bbox_inches='tight')

# gamma surface plot
max_q = final_q_invs.max()
rts = np.linspace(0, pp.steps, pp.steps+1)
qs = np.linspace(max_q, 0, max_q+1)
x, y = np.meshgrid(qs, rts)
z = np.empty(shape=x.shape)
for i, j in np.ndindex(*x.shape):
    trader_inv_sac.get_quotes(0, x[i, j], 100, 1-y[i, j]*DT)
    z[i, j] = trader_inv_sac.gamma

fig = plt.figure("gamma surface")
ax = plt.axes(projection='3d')
surf = ax.plot_surface(x, y, z, cmap='viridis', edgecolor='none')
ax.set_ylabel('step')
ax.set_xlabel('inventory')
ax.set_zlabel('gamma')
fig.colorbar(surf)
plt.savefig("results/policy_surface.png", bbox_inches='tight')

print("\nTotal duration %.2f min" % ((time.time() - train_start) / 60))
plt.show()
