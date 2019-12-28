import math


class SymmetricTrader:
    """
    Symmetric strategy
    https://www.math.nyu.edu/faculty/avellane/HighFrequencyTrading.pdf
    """
    def __init__(self, gamma, sigma, bid_k, ask_k):

        # pre computed values
        self.gamma_sigma_sq = gamma * (sigma**2)
        self.spread_constant_bid = (2 / gamma) * math.log(1 + gamma / bid_k)
        self.spread_constant_ask = (2 / gamma) * math.log(1 + gamma / ask_k)

    def get_quotes(self, s, rt):
        """
        :param s: price
        :param rt: remaining time
        :return:  bid, ask limit order prices
        """
        spread_bid, spread_ask = self.get_bid_ask(rt)
        return s - spread_bid, s + spread_ask

    def get_bid_ask(self, remain_time):
        """
        :param remain_time: remaining time
        :return: reservation price, bid (positive) spread , ask spread
        """
        g_ss_rt = self.gamma_sigma_sq * remain_time
        spread_bid = (g_ss_rt + self.spread_constant_bid) / 2
        spread_ask = (g_ss_rt + self.spread_constant_ask) / 2
        return spread_bid, spread_ask
