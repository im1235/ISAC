import math
import torch
import numpy as np
import random


class InventoryAITrader:
    """
    Inventory strategy with gamma regulated by Soft Actor Critic
    """
    def __init__(self, sigma, bid_k, bid_a, ask_k, ask_a, dt, advisor):

        self.sigma = sigma
        self.bid_k = bid_k
        self.bid_a = bid_a
        self.ask_k = ask_k
        self.ask_a = ask_a
        self.dt = dt
        self.advisor = advisor

        # variables used to build training samples
        self.prev_state = None
        self.prev_action = None
        self.prev_pl = None
        self.prev_q = None

        # epoch reward
        self.reward_total = 0

        # regulated risk aversion parameter
        self.gamma = None

        # pre computed values
        self.sigma_sqrt_dt = self.sigma * np.sqrt(self.dt)
        self.sigma_squared = self.sigma ** 2

    def reset(self):
        """
        resets traders internal training variables after training epoch
        :return:
        """
        self.prev_state = None
        self.prev_action = None
        self.prev_pl = None
        self.prev_q = None
        self.reward_total = 0

    def get_quotes(self, x, q, s, rt, train_mode=False):
        """

        :param x: cash
        :param q: inventory
        :param s: price
        :param rt: remaining time
        :param train_mode: True/False
        :return: bid, ask limit order prices
        """
        # state is defined with inventory q and remaining time rt
        cur_state = scale_state(q, rt)

        if train_mode:
            # current pl is cash + inventory * price
            cur_pl = x + q * s

            if self.prev_action is not None:
                # reward is delta in profit - risk
                risk_penalty = 0.5 * abs(q) * self.sigma_sqrt_dt

                delta_pl = cur_pl - self.prev_pl
                reward = delta_pl - risk_penalty

                # add reward to total epoch reward
                self.reward_total += reward

                # push training sample into memory
                if self.advisor.push_buffer(self.prev_state, self.prev_action, reward, cur_state, rt <= 0):
                    self.advisor.update()  # update weights if there is more than N new samples in memory

            if self.advisor.replay_buffer.initialized:
                # buffer is fully initialized, exploration is stopped
                explore = False
            else:
                # take random sample with probability 1 - buffer populated %
                explore = random.random() > self.advisor.replay_buffer.buffer_idx / self.advisor.replay_buffer.capacity

            # if in exploration take random sample, otherwise estimate action
            action = torch.FloatTensor(1).uniform_(-1, 1) if explore else self.advisor.get_action(scale_state(q, rt))

            # set states for next training sample
            self.prev_state = cur_state
            self.prev_pl = cur_pl
            self.prev_action = action

        else:
            # trader is not training, estimate action
            action = self.advisor.get_action(cur_state)

        # scale action to gamma value
        self.gamma = scale_gamma(action.numpy())

        # pre computed values
        two_div_gamma = 2 / self.gamma
        g_ss_rt = self.gamma * self.sigma_squared * rt

        # calculate reservation price and spreads
        reservation_price = s - q * g_ss_rt
        spread_bid = (g_ss_rt + two_div_gamma * math.log(1 + self.gamma / self.bid_k)) / 2
        spread_ask = (g_ss_rt + two_div_gamma * math.log(1 + self.gamma / self.ask_k)) / 2

        return reservation_price - spread_bid, reservation_price + spread_ask  # return bid quote, ask quote


def scale_state(q, rt):
    """
    scales state, assumes max inventory is 100
    """
    return [abs(q)/100, rt]


def scale_gamma(action):
    """
    scales action to gamma
    """
    return max(1e-5, ((action + 1)/2))
