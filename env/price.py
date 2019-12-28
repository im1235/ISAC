import numpy as np
import random


class SimplePriceProcess:
    """
    simple price process environment, keeps track of multiple traders inventory (q) and cash (x)

    limit bid filled - > increase inventory q = q+1 ,decrease cash x = x - price
    limit ask filled - > decrease inventory q = q-1 , increase cash x = x + price

    """

    def __init__(self, dim, sigma, bid_k, bid_a, ask_k, ask_a, dt, s0=100, total_time=1):

        # price process parameters
        self.sigma = sigma
        self.bid_k = bid_k
        self.bid_a = bid_a
        self.ask_k = ask_k
        self.ask_a = ask_a
        self.dt = dt

        self.q = np.zeros(dim)      # vector of inventories
        self.x = np.zeros(dim)      # vector of cash
        self.bid_hit = np.zeros(dim, dtype=bool)
        self.ask_lift = np.zeros(dim, dtype=bool)

        # simulate random price change
        self.price_process = s0 + np.cumsum(sigma * np.sqrt(dt) * np.random.choice([1, -1],  int(total_time / dt)))
        self.price_process = np.insert(self.price_process, 0, s0)

        self.steps = len(self.price_process)    # total number of simulation steps
        self.step_idx = 0
        self.step_idx_max = self.steps-1
        self.done = False

    @property
    def s(self):
        return self.price_process[self.step_idx]

    @property
    def remaining_time(self):
        return 1 - self.step_idx * self.dt

    def quote(self, bid, ask):

        spread_bid = self.s - bid
        spread_ask = ask - self.s

        # fill probability for quoted spreads
        p_bid_hit = self.bid_a * np.exp(-self.bid_k * spread_bid) * self.dt
        p_ask_lift = self.ask_a * np.exp(-self.ask_k * spread_ask) * self.dt

        # simulation of bid fill
        self.bid_hit = random.random() < p_bid_hit
        self.q += self.bid_hit
        self.x -= self.bid_hit * bid

        # simulation of ask fill
        self.ask_lift = random.random() < p_ask_lift
        self.q -= self.ask_lift
        self.x += self.ask_lift * ask

        # price evolution
        if self.step_idx < self.step_idx_max:
            # move index to next simulated price
            self.step_idx += 1
        else:
            # end of simulation
            self.done = True

        return self.state()

    def state(self):
        """
        :return: inventory[], cash[], price, remaining time (0.-1), is time done
        """
        return self.q, self.x, self.s, self.remaining_time, self.done


class DetailedPriceProcess(SimplePriceProcess):
    """
    Wrapper around price process that adds logging capabilities
    """

    def __init__(self, dim, sigma, bid_k, bid_a, ask_k, ask_a, dt, s0=100, total_time=1):
        super().__init__(dim, sigma, bid_k, bid_a, ask_k, ask_a, dt, s0, total_time)

        log_shape = (self.steps, dim)               # length of logs

        # arrays for logging
        self.log_pl = np.empty(shape=log_shape)         # profit loss
        self.log_q = np.empty(shape=log_shape)          # inventory
        self.log_bid = np.empty(shape=log_shape)  # quoted bid prices
        self.log_ask = np.empty(shape=log_shape)  # quoted ask prices

        self.log_s = np.empty(shape=log_shape[0])       # midprice

    def quote(self, bid, ask):

        # set logged values, step_idx is incremented by superclass
        self.log_pl[self.step_idx] = self.last_pl
        self.log_q[self.step_idx] = self.q
        self.log_bid[self.step_idx] = bid
        self.log_ask[self.step_idx] = ask
        self.log_s[self.step_idx] = self.s

        return super().quote(bid, ask)

    @property
    def last_pl(self):
        return self.x + self.q * self.s

    @property
    def last_q(self):
        return self.log_q[self.step_idx]
