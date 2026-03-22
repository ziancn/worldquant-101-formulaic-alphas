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
        result[:] = -1
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
        (-1 * rank(ts_rank(close, 10))) * rank(delta(delta(close, 1), 1)) * rank(ts_rank(volume / adv20, 5))
        """
        return (-1 
                * cs_rank(ts_rank(self.close, 10))
                * cs_rank(delta(self.close, 1) - delay(delta(self.close, 1), 1))
                * cs_rank(ts_rank(self.volume / self.adv20, 5)))
    
    def wq018(self):
        """
        (-1 * rank(((stddev(abs((close - open)), 5) + (close - open)) + correlation(close, open, 10))))

        Translated version:
        diff = close - open

        -1 * rank{ stddev(abs(diff), 5) + diff + correlation(close, open, 10) }
        """
        diff = self.close - self.open
        return -1 * cs_rank(ts_stddev(np.abs(diff), 5) + diff + correlation(self.close, self.open, 10))
    
    def wq019(self):
        """
        ((-1 * sign(((close - delay(close, 7)) + delta(close, 7)))) * (1 + rank((1 + sum(returns, 250)))))
        """
        direction = -1 * np.sign(delta(self.close, 7))
        weight = 1 + cs_rank(ts_sum(self.ret, 250))
        return direction * weight
    
    def wq020(self):
        """
        (((-1 * rank((open - delay(high, 1)))) * rank((open - delay(close, 1)))) * rank((open - delay(low, 1))))

        Translated:
        -1 * rank(open - delay(high, 1)) 
           * rank(open - delay(close, 1)) 
           * rank(open - delay(low, 1))
        """
        return (-1 
                * cs_rank(self.open - delay(self.high, 1))
                * cs_rank(self.open - delay(self.close, 1))
                * cs_rank(self.open - delay(self.low, -1)))
    
    def wq021(self):
        """
        ((((sum(close, 8) / 8) + stddev(close, 8)) < (sum(close, 2) / 2)) ? (-1 * 1) : 
        (((sum(close, 2) / 2) < ((sum(close, 8) / 8) - stddev(close, 8))) ? 1 : (((1 < (volume / adv20)) || 
        ((volume / adv20) == 1)) ? 1 : (-1 * 1))))

        Translated:
        if (MA(8) + STD(8)) < MA(2): 
            return -1
        elif MA(2) < (MA(8) - STD(8)): 
            return 1
        elif (volume / adv20) >= 1:
            return 1
        else:
            return -1
        """
        result = self.close
        result[:] = -1

        ma_2 = ts_sum(self.close, 2) / 2
        ma_8 = ts_sum(self.close, 8) / 8
        std_8 = ts_stddev(self.close, 8)

        result[self.volume / self.adv20 >= 1] = 1
        result[ma_2 < (ma_8 - std_8)] = 1
        result[(ma_8 + std_8) < ma_2] = -1

        return result
    
    def wq022(self):
        """
        (-1 * (delta(correlation(high, volume, 5), 5) * rank(stddev(close, 20)))) 

        Translated:
        -1 * (
            delta(correlation(high, volume, 5), 5) * 
            rank(stddev(close, 20))
        )
        """
        return (-1 
                * delta(correlation(self.high, self.volume, 5), 5)
                * cs_rank(ts_stddev(self.close, 20)))
    
    def wq023(self):
        """
        (((sum(high, 20) / 20) < high) ? (-1 * delta(high, 2)) : 0) 
        
        Translated:
        if (MA(high, 20) < high):
            return -1 * delta(high, 2)
        else:
            return 0
        """

        result = self.close
        result[:] = 0

        ma = ts_sum(self.high, 20) / 20

        result[ma < self.high] = -1 * delta(self.high, 2)

        return result
    
    def wq024(self):
        """
        ((((delta((sum(close, 100) / 100), 100) / delay(close, 100)) < 0.05) || 
        ((delta((sum(close, 100) / 100), 100) / delay(close, 100)) == 0.05)) ? (-1 * (close - ts_min(close, 
        100))) : (-1 * delta(close, 3))) 

        Translated:
        MA_100 = sum(close, 100) / 100
        MA_Return = delta(MA_100, 100) / delay(close, 100)

        if MA_Return <= 0.05:
            return -1 * (close - ts_min(close, 100))
        else:
            return -1 * delta(close, 3)
        """
        ma_100 = ts_sum(self.close, 100) / 100
        ma_ret = delta(ma_100, 100) / delay(self.close, 100)

        result = -1 * delta(self.close, 3)
        result[ma_ret <= 0.05] = -1 * (self.close - ts_min(self.close, 100))

        return result
    
    def wq025(self):
        """
        rank(((((-1 * returns) * adv20) * vwap) * (high - close)))
        """
        # Missing VWAP data, skip this one
        return None
    
    def wq026(self):
        """
        (-1 * ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3)) 
        """
        vol_strength  = ts_rank(self.volume, 5)
        high_strength = ts_rank(self.high, 5)

        return -1 * ts_max(correlation(vol_strength, high_strength, 5), 3)
    
    def wq027(self):
        """
        ((0.5 < rank((sum(correlation(rank(volume), rank(vwap), 6), 2) / 2.0))) ? (-1 * 1) : 1) 

        Translated:
        # 1. Measure the "Sync" between Volume and VWAP (using ranks to remove outliers)
        Sync = correlation(rank(volume), rank(vwap), 6)

        # 2. Smooth the signal over 2 days
        Avg_Sync = sum(Sync, 2) / 2.0

        # 3. Final Decision
        if rank(Avg_Sync) > 0.5:
            return -1  # Sell/Short: If the price-volume sync is in the top 50% of the market
        else:
            return 1   # Buy/Long: If the sync is weak
        """
        # Missing VWAP data, skip this one
        return None
    
    def wq028(self):
        """
        scale(((correlation(adv20, low, 5) + ((high + low) / 2)) - close))
        """
        return
