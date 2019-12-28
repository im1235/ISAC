import math


class InventoryTrader:
    """
    Inventory strategy
    https://www.math.nyu.edu/faculty/avellane/HighFrequencyTrading.pdf
    """
    def __init__(self, gamma, sigma, bid_k, ask_k):

        # pre computed values
        self.gamma_sigma_sq = gamma * (sigma**2)
        self.spread_constant_bid = (2 / gamma) * math.log(1 + gamma / bid_k)
        self.spread_constant_ask = (2 / gamma) * math.log(1 + gamma / ask_k)

    def get_quotes(self, q, s, rt):
        """
        :param q: inventory
        :param s: price
        :param rt: remaining time
        :return: bid, ask limit order prices
        """
        reservation_price, spread_bid, spread_ask = self.get_rp_bid_ask(q, s, rt)
        return reservation_price - spread_bid, reservation_price + spread_ask

    def get_rp_bid_ask(self, q, s, rt):
        """
        :param q: inventory
        :param s: price
        :param rt: remaining time
        :return: reservation price, bid (positive) spread , ask spread
        """
        g_ss_rt = self.gamma_sigma_sq * rt

        reservation_price = s - q * g_ss_rt
        spread_bid = (g_ss_rt + self.spread_constant_bid) / 2
        spread_ask = (g_ss_rt + self.spread_constant_ask) / 2

        return reservation_price, spread_bid, spread_ask
