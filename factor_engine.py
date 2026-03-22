"""
This file houses the FactorEngine class.
"""

import pandas as pd
# Python's standard library includes a module named `operator`
# Don't confuse with it, `operators` is my our customised operators
from operators import *


class FactorEngine:
    """
    FactorEngine takes in pandel data (stacked on tickers, columns are metrics)
    """
    def __init__(self, df: pd.DataFrame):
        """
        Parameters:
            df: The panel_data required is a stacked long table. (Stacked on tickers, columns are OHLCV)
        """
        # From data directly
        self.open   = df.pivot(index="Date", columns="Ticker", values="Open")
        self.high   = df.pivot(index="Date", columns="Ticker", values="High")
        self.low    = df.pivot(index="Date", columns="Ticker", values="Low")
        self.close  = df.pivot(index="Date", columns="Ticker", values="Close")
        self.volume = df.pivot(index="Date", columns="Ticker", values="Volume")
        # Some processing
        self.ret    = self.close.pct_change()
        self.adv20  = self.volume.rolling(20).mean()


    """WoldQuant Factors"""
    def wq001(self):
        """
        rank{Ts_ArgMax[SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5]} - 0.5
        """
        inner = self.close
        inner[self.ret < 0] = ts_stddev(self.ret, 20)
        return cs_rank(ts_argmax(signed_power(inner, 2), 5)) - 0.5
    
    def wq002(self):
        """
        -1 * correlation{rank(delta(log(volume), 2)), rank(((close - open) / open)), 6}
        """
        inner_l = cs_rank(delta(np.log(self.volume), 2))
        inner_r = cs_rank((self.close - self.open) / self.open)
        corr = correlation(inner_l, inner_r, 6)
        return -1 * corr
    
    def wq003(self):
        """
        -1 * correlation(rank(open), rank(volume), 10)
        """
        return -1 * correlation(cs_rank(self.open), cs_rank(self.volume), 10)
    
    def wq004(self):
        """
        -1 * Ts_Rank(rank(low), 9)
        """
        return -1 * ts_rank(cs_rank(self.low), 9)
    
    def wq005(self):
        """
        rank((open - (sum(vwap, 10) / 10))) * (-1 * abs(rank((close - vwap))))
        """
        # No access to VWAP data right now via yfinance
        return None
    
    def wq006(self):
        """
        -1 * correlation(open, volume, 10)
        """
        return -1 * correlation(self.open, self.volume, 10)
    
    def wq007(self):
        """
        (adv20 < volume) ? ((-1 * ts_rank(abs(delta(close, 7)), 60)) * sign(delta(close, 7))) : (-1 * 1)
        """
        # This factor looks weird: -1 * 1, likely something generated systematically
        result = self.close
        result = -1
        result[self.adv20 < self.volume] = -1 * ts_rank(np.abs(delta(self.close, 7)), 60) * np.sign(delta(self.close, 7))
        return result
    
    def wq008(self):
        """
        -1 * rank(sum(open, 5) * sum(returns, 5) - delay(sum(open, 5) * sum(returns, 5), 10))
        """
        return -1 * cs_rank(ts_sum(self.open, 5) * ts_sum(self.ret, 5) - delay(ts_sum(self.open, 5) * ts_sum(self.ret, 5), 10))

    def wq009(self):
        """
        0 < ts_min(delta(close, 1), 5) ? delta(close, 1) : ts_max(delta(close, 1), 5) < 0 ? delta(close, 1) : -1 * delta(close, 1)
        ------------------------------   ---------------   -----------------------------------------------------------------------
                                                           -------------------------------   ---------------   -------------------
        """
        inner1 = -1 * delta(self.close, 1)
        inner1[ts_max(delta(self.close, 1), 5) < 0] = delta(self.close, 1)

        inner2 = inner1
        inner2[ts_min(delta(self.close, 1), 5) > 0] = delta(self.close, 1)

        return inner2

    def wq010(self):
        """
        rank(((0 < ts_min(delta(close, 1), 4)) ? delta(close, 1) : ((ts_max(delta(close, 1), 4) < 0) ? delta(close, 1) : (-1 * delta(close, 1)))))

        Translated version, replace delta(close, 1) as diff
        rank{
            0 < ts_min(diff,4)
            ? diff
            : (
                ts_max(diff, 4) < 0)
                ? diff
                : -diff
              )
        }
        """
        diff = delta(self.close, 1)

        inner1 = -diff
        inner1[ts_max(diff, 4) < 0] = diff

        inner2 = inner1
        inner2[ts_min(diff, 4) > 0] = diff

        return cs_rank(inner2)

    def wq011(self):
        """
        (rank(ts_max((vwap - close), 3)) + rank(ts_min((vwap - close), 3))) * rank(delta(volume, 3))
        """
        # Missing VWAP data now, skip this one
        return None
    
    def wq012(self):
        """
        sign(delta(volume, 1)) * (-1 * delta(close, 1))
        """
        return np.sign(delta(self.close, 1) * -1 * delta(self.close, 1))
    
    def wq013(self):
        """
        -1 * rank(covariance(rank(close), rank(volume), 5))
        """
        return -1 * cs_rank(covariance(cs_rank(self.close), cs_rank(self.volume), 5))
    
    def wq014(self):
        """
        (-1 * rank(delta(returns, 3))) * correlation(open, volume, 10)
        """
        return -1 * cs_rank(delta(self.ret, 3)) * correlation(self.open, self.volume, 10)
    
    def wq015(self):
        """
        -1 * sum(rank(correlation(rank(high), rank(volume), 3)), 3)
        """
        return -1 * ts_sum(cs_rank(correlation(cs_rank(self.high), cs_rank(self.volume), 3)), 3)
    
    def wq016(self):
        """
        -1 * rank(covariance(rank(high), rank(volume), 5))
        """
        return -1 * cs_rank(covariance(cs_rank(self.high), cs_rank(self.volume), 5))
    
    def wq017(self):
        """
        
        """
        return