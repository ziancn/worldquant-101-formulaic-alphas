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

        self.vwap = df.pivot(index="Date", columns="Ticker", values="VWAP") if "VWAP" in df.columns else None
        self.cap = df.pivot(index="Date", columns="Ticker", values="Cap") if "Cap" in df.columns else None

        # Some processing
        self.ret    = self.close.pct_change()
        self.adv20  = self.volume.rolling(20).mean()
        self.adv15  = self.volume.rolling(15).mean()
        self.adv30  = self.volume.rolling(30).mean()
        self.adv40  = self.volume.rolling(40).mean()
        self.adv50  = self.volume.rolling(50).mean()
        self.adv60  = self.volume.rolling(60).mean()
        self.adv120 = self.volume.rolling(120).mean()
        self.adv180 = self.volume.rolling(180).mean()

        # Temp hijack
        self.vwap = (self.high + self.low + self.close) / 3


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

        return cs_rank(inner2) - 0.5

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
        mid_px = (self.high + self.low) / 2
        corr = correlation(self.adv20, self.low)
        inner = corr + mid_px - self.close
        return scale(inner)
    
    def wq029(self):
        """
        (min(product(rank(rank(scale(log(sum(ts_min(rank(rank((-1 * rank(delta((close - 1), 
        5))))), 2), 1))))), 1), 5) + ts_rank(delay((-1 * returns), 6), 5))
        """
        signal_A = ts_min(cs_rank(-delta(self.close, 5)), 2)
        signal_A_clean = np.log(scale(cs_rank(signal_A)))
        signal_B = ts_rank(delay(-self.ret, 6), 5)

        return min(signal_A_clean, 5) + signal_B
    
    def wq030(self):
        """
        (((1.0 - rank(((sign((close - delay(close, 1))) + sign((delay(close, 1) - delay(close, 2)))) + 
        sign((delay(close, 2) - delay(close, 3)))))) * sum(volume, 5)) / sum(volume, 20))
        """
        price_trend = (np.sign(delta(self.close, 1))
                       + np.sign(delay(delta(self.close, 1), 1))
                       + np.sign(delay(delta(self.close, 1), 2)))

        vol_ratio = ts_sum(self.volume, 5) / ts_sum(self.volume, 20)

        return (1.0 - cs_rank(price_trend)) * vol_ratio
    
    def wq031(self):
        """
        ((rank(rank(rank(decay_linear((-1 * rank(rank(delta(close, 10)))), 10)))) + rank((-1 * 
        delta(close, 3)))) + sign(scale(correlation(adv20, low, 12))))
        """
        long_term_reversal = cs_rank(decay_linear(-cs_rank(delta(self.close, 10)), 10))
        short_term_reversal = cs_rank(-delta(self.close, 3))
        liquidity_signal = np.sign(scale(correlation(self.adv20, self.low, 12)))

        return long_term_reversal + short_term_reversal + liquidity_signal
    
    def wq032(self):
        """
        (scale(((sum(close, 7) / 7) - close)) + (20 * scale(correlation(vwap, delay(close, 5), 230))))
        """
        # Missing VWAP data, skip now
        # short_gap = (ts_sum(self.close, 7) / 7.0) - self.close
        # short_signal = scale(short_gap)

        # price_sync = correlation(self.vwap, delay(self.close, 5), 230)
        # long_signal = 20 * scale(price_sync)

        # return short_signal + long_signal
        ...

    def wq033(self):
        """
        rank((-1 * ((1 - (open / close))^1)))
        """
        return cs_rank(-1 * (1 - (self.open / self.close)))
    
    def wq034(self):
        """
        rank(((1 - rank((stddev(returns, 2) / stddev(returns, 5)))) + (1 - rank(delta(close, 1)))))
        """
        vol_instability = ts_stddev(self.ret, 2) / ts_stddev(self.ret, 5)
        price_shock = delta(self.close, 1)

        return cs_rank(
            (1 - cs_rank(vol_instability)) +
            (1 - cs_rank(price_shock))
        )
    
    def wq035(self):
        """
        ((Ts_Rank(volume, 32) * (1 - Ts_Rank(((close + high) - low), 16))) * (1 -
        Ts_Rank(returns, 32)))
        """
        vol_peak = ts_rank(self.volume, 32)
        price_stretch = ts_rank((self.close + self.high - self.low), 16)
        return_peak = ts_rank(self.ret, 32)

        return vol_peak * (1 - price_stretch) * (1 - return_peak)
    
    def wq036(self):
        """
        (((((2.21 * rank(correlation((close - open), delay(volume, 1), 15))) + (0.7 * rank((open
        - close)))) + (0.73 * rank(Ts_Rank(delay((-1 * returns), 6), 5)))) + rank(abs(correlation(vwap,
        adv20, 6)))) + (0.6 * rank((((sum(close, 200) / 200) - open) * (close - open)))))
        """
        vol_price_sync = cs_rank(correlation((self.close - self.open), delay(self.volume, 1), 15))
        intraday_reversal = cs_rank(self.open - self.close)
        lagged_shock_reversal = cs_rank(ts_rank(delay(-self.ret, 6), 5))
        liquidity_intensity = cs_rank(abs(correlation(self.vwap, self.adv20, 6)))
        trend_pullback = cs_rank(((ts_sum(self.close, 200) / 200) - self.open) * (self.close - self.open))

        # Final Weighted Ensemble
        return ((2.21 * vol_price_sync)
                + (0.70 * intraday_reversal)
                + (0.73 * lagged_shock_reversal)
                + (1.00 * liquidity_intensity)
                + (0.60 * trend_pullback))
    
    def wq037(self):
        """
        (rank(correlation(delay((open - close), 1), close, 200)) + rank((open - close)))
        """
        historical_bias = cs_rank(correlation(delay(self.open - self.close, 1), self.close, 200))
        intraday_reversal = cs_rank(self.open - self.close)

        return historical_bias + intraday_reversal
    
    def wq038(self):
        """
        -1 * rank(Ts_Rank(close, 10)) * rank((close / open))
        """
        return -1 * cs_rank(ts_rank(self.close, 10)) * cs_rank(self.close / self.open)
    
    def wq039(self):
        """
        ((-1 * rank((delta(close, 7) * (1 - rank(decay_linear((volume / adv20), 9)))))) * (1 +
        rank(sum(returns, 250))))
        """
        short_term_move = delta(self.close, 7)
        volume_exhaustion = 1 - cs_rank(decay_linear(self.volume / self.adv20, 9))
        reversal_signal = -1 * cs_rank(short_term_move * volume_exhaustion)
        long_term_strength = 1 + cs_rank(ts_sum(self.ret, 250))

        return reversal_signal * long_term_strength
    
    def wq040(self):
        """
        (-1 * rank(stddev(high, 10))) * correlation(high, volume, 10)
        """
        return -1 * cs_rank(ts_stddev(self.high, 10)) * correlation(self.high, self.volume, 10)
    
    def wq041(self):
        """
        (((high * low)^0.5) - vwap)
        """
        return (self.high - self.low) ** 0.5 - self.vwap
    
    def wq042(self):
        """
        (rank((vwap - close)) / rank((vwap + close)))
        """
        return cs_rank(self.vwap - self.close) / cs_rank(self.vwap - self.close)
    
    def wq043(self):
        """
        (ts_rank((volume / adv20), 20) * ts_rank((-1 * delta(close, 7)), 8))
        """
        relative_vol_rank = ts_rank(self.volume / self.adv20, 20)
        price_drop_rank = ts_rank(-delta(self.close, 7), 8)

        return relative_vol_rank * price_drop_rank
    
    def wq044(self):
        """
        (-1 * correlation(high, rank(volume), 5))
        """
        return -1 * correlation(self.high, cs_rank(self.volume, 5))
    
    def wq045(self):
        """
        (-1 * ((rank((sum(delay(close, 5), 20) / 20)) * correlation(close, volume, 2)) *
        rank(correlation(sum(close, 5), sum(close, 20), 2))))
        """
        historical_price_level = cs_rank(ts_sum(delay(self.close, 5), 20) / 20.0)
        price_vol_sync = correlation(self.close, self.volume, 2)
        trend_consistency = cs_rank(correlation(ts_sum(self.close, 5), ts_sum(self.close, 20), 2))

        return -1 * (historical_price_level * price_vol_sync * trend_consistency)
    
    def wq046(self):
        """
        ((0.25 < (((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10))) ?
        (-1 * 1) : (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < 0) ? 1 :
            ((-1 * 1) * (close - delay(close, 1)))))
        """

        x = ((delay(self.close, 20) - delay(self.close, 10)) / 10) - ((delay(self.close, 10) - self.close) / 10)
        result = pd.DataFrame(index=self.close.index, columns=self.close.columns, dtype=float)
        result.loc[x > 0.25] = -1
        result.loc[x < 0] = 1
        mask_else = (~(x > 0.25)) & (~(x < 0))
        result.loc[mask_else] = -1 * (self.close - delay(self.close, 1))
        return result

    def wq047(self):
        """
        ((((rank((1 / close)) * volume) / adv20) * ((high * rank((high - close))) / (sum(high, 5) / 5))) - rank((vwap - delay(vwap, 5))))
        """

        if self.vwap is None:
            return None
        term1 = cs_rank(1 / self.close) * self.volume / self.adv20
        term2 = self.high * cs_rank(self.high - self.close) / (ts_sum(self.high, 5) / 5)
        return term1 * term2 - cs_rank(self.vwap - delay(self.vwap, 5))

    def wq048(self):
        """
        (indneutralize(((correlation(delta(close, 1), delta(delay(close, 1), 1), 250) * delta(close, 1)) / close), IndClass.subindustry) / sum(((delta(close, 1) / delay(close, 1))^2), 250))
        """
        # need IndNeutralize/industry classification, unavailable
        return None

    def wq049(self):
        """
        (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-1 * 0.1)) ? 1 : ((-1 * 1) * (close - delay(close, 1))))
        """

        x = ((delay(self.close, 20) - delay(self.close, 10)) / 10) - ((delay(self.close, 10) - self.close) / 10)
        result = pd.DataFrame(index=self.close.index, columns=self.close.columns, dtype=float)
        result.loc[x < -0.1] = 1
        mask_else = ~(x < -0.1)
        result.loc[mask_else] = -1 * (self.close - delay(self.close, 1))
        return result

    def wq050(self):
        """
        (-1 * ts_max(rank(correlation(rank(volume), rank(vwap), 5)), 5))
        """
        if self.vwap is None:
            return None
        return -1 * ts_max(cs_rank(correlation(cs_rank(self.volume), cs_rank(self.vwap), 5)), 5)

    def wq051(self):
        """
        (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-1 * 0.05)) ? 1 : ((-1 * 1) * (close - delay(close, 1))))
        """

        x = ((delay(self.close, 20) - delay(self.close, 10)) / 10) - ((delay(self.close, 10) - self.close) / 10)
        result = pd.DataFrame(index=self.close.index, columns=self.close.columns, dtype=float)
        result.loc[x < -0.05] = 1
        mask_else = ~(x < -0.05)
        result.loc[mask_else] = -1 * (self.close - delay(self.close, 1))
        return result

    def wq052(self):
        """
        ((((-1 * ts_min(low, 5)) + delay(ts_min(low, 5), 5)) * rank(((sum(returns, 240) - sum(returns, 20)) / 220))) * ts_rank(volume, 5))
        """

        return (((-1 * ts_min(self.low, 5)) + delay(ts_min(self.low, 5), 5)) *
                cs_rank((ts_sum(self.ret, 240) - ts_sum(self.ret, 20)) / 220) *
                ts_rank(self.volume, 5))

    def wq053(self):
        """
        (-1 * delta((((close - low) - (high - close)) / (close - low)), 9))
        """

        ratio = ((self.close - self.low) - (self.high - self.close)) / (self.close - self.low)
        return -1 * delta(ratio, 9)

    def wq054(self):
        """
        ((-1 * ((low - close) * (open^5))) / ((low - high) * (close^5)))
        """

        return (-1 * ((self.low - self.close) * (self.open ** 5))) / ((self.low - self.high) * (self.close ** 5))

    def wq055(self):
        """
        (-1 * correlation(rank(((close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low, 12)))), rank(volume), 6))
        """

        z = (self.close - ts_min(self.low, 12)) / (ts_max(self.high, 12) - ts_min(self.low, 12))
        return -1 * correlation(cs_rank(z), cs_rank(self.volume), 6)

    def wq056(self):
        """
        (0 - (1 * (rank((sum(returns, 10) / sum(sum(returns, 2), 3))) * rank((returns * cap)))))
        """

        if self.cap is None:
            return None
        num = ts_sum(self.ret, 10)
        denom = ts_sum(ts_sum(self.ret, 2), 3)
        return -1 * (cs_rank(num / denom) * cs_rank(self.ret * self.cap))

    def wq057(self):
        """
        (0 - (1 * ((close - vwap) / decay_linear(rank(ts_argmax(close, 30)), 2))))
        """

        if self.vwap is None:
            return None
        return -1 * ((self.close - self.vwap) / decay_linear(cs_rank(ts_argmax(self.close, 30)), 2))

    def wq058(self):
        """
        (-1 * Ts_Rank(decay_linear(correlation(IndNeutralize(vwap, IndClass.sector), volume, 3.92795), 7.89291), 5.50322))
        """
        return None

    def wq059(self):
        """
        (-1 * Ts_Rank(decay_linear(correlation(IndNeutralize(((vwap * 0.728317) + (vwap * (1 - 0.728317))), IndClass.industry), volume, 4.25197), 16.2289), 8.19648))
        """
        return None

    def wq060(self):
        """
        (0 - (1 * ((2 * scale(rank(((((close - low) - (high - close)) / (high - low)) * volume)))) - scale(rank(ts_argmax(close, 10))))))
        """

        a = scale(cs_rank(((((self.close - self.low) - (self.high - self.close)) / (self.high - self.low)) * self.volume)))
        b = scale(cs_rank(ts_argmax(self.close, 10)))
        return -1 * ((2 * a) - b)

    def wq061(self):
        """
        (rank((vwap - ts_min(vwap, 16.1219))) < rank(correlation(vwap, adv180, 17.9282)))
        """

        if self.vwap is None:
            return None
        return (cs_rank(self.vwap - ts_min(self.vwap, 16)) < cs_rank(correlation(self.vwap, self.adv180, 18))).astype(float)

    def wq062(self):
        """
        ((rank(correlation(vwap, sum(adv20, 22.4101), 9.91009)) < rank(((rank(open) + rank(open)) < (rank(((high + low) / 2)) + rank(high))))) * -1)
        """

        if self.vwap is None:
            return None
        left = cs_rank(correlation(self.vwap, ts_sum(self.adv20, 22), 10))
        right_bool = (cs_rank(self.open) + cs_rank(self.open)) < (cs_rank((self.high + self.low) / 2) + cs_rank(self.high))
        right = cs_rank(right_bool.astype(float))
        return (left < right) * -1

    def wq063(self):
        """
        ((rank(decay_linear(delta(IndNeutralize(close, IndClass.industry), 2.25164), 8.22237)) - rank(decay_linear(correlation(((vwap * 0.318108) + (open * (1 - 0.318108))), sum(adv180, 37.2467), 13.557), 12.2883))) * -1)
        """
        return None

    def wq064(self):
        """
        ((rank(correlation(sum(((open * 0.178404) + (low * (1 - 0.178404))), 12.7054), sum(adv120, 12.7054), 16.6208)) < rank(delta(((((high + low) / 2) * 0.178404) + (vwap * (1 - 0.178404))), 3.69741))) * -1)
        """

        if self.vwap is None:
            return None
        left = cs_rank(correlation(ts_sum((self.open * 0.178404) + (self.low * (1 - 0.178404)), 13), ts_sum(self.adv120, 13), 17))
        right = cs_rank(delta((((self.high + self.low) / 2) * 0.178404) + (self.vwap * (1 - 0.178404)), 3))
        return (left < right) * -1

    def wq065(self):
        """
        ((rank(correlation(((open * 0.00817205) + (vwap * (1 - 0.00817205))), sum(adv60, 8.6911), 6.40374)) < rank((open - ts_min(open, 13.635)))) * -1)
        """

        if self.vwap is None:
            return None
        left = cs_rank(correlation((self.open * 0.00817205) + (self.vwap * (1 - 0.00817205)), ts_sum(self.adv60, 9), 6))
        right = cs_rank(self.open - ts_min(self.open, 14))
        return (left < right) * -1

    def wq066(self):
        """
        ((rank(decay_linear(delta(vwap, 3.51013), 7.23052)) + Ts_Rank(decay_linear(((((low * 0.96633) + (low * (1 - 0.96633))) - vwap) / (open - ((high + low) / 2))), 11.4157), 6.72611)) * -1)
        """

        if self.vwap is None:
            return None
        left = cs_rank(decay_linear(delta(self.vwap, 3), 7))
        right = ts_rank(decay_linear((((self.low * 0.96633) + (self.low * (1 - 0.96633))) - self.vwap) / (self.open - ((self.high + self.low) / 2)), 11), 7)
        return (left + right) * -1

    def wq067(self):
        """
        ((rank((high - ts_min(high, 2.14593))) ^ rank(correlation(IndNeutralize(vwap, IndClass.sector), IndNeutralize(adv20, IndClass.subindustry), 6.02936))) * -1)
        """
        return None

    def wq068(self):
        """
        ((Ts_Rank(correlation(rank(high), rank(adv15), 8.91644), 13.9333) < rank(delta(((close * 0.518371) + (low * (1 - 0.518371))), 1.06157))) * -1)
        """

        left = ts_rank(correlation(cs_rank(self.high), cs_rank(self.adv15), 9), 14)
        right = cs_rank(delta((self.close * 0.518371) + (self.low * (1 - 0.518371)), 1))
        return (left < right) * -1

    def wq069(self):
        """
        ((rank(ts_max(delta(IndNeutralize(vwap, IndClass.industry), 2.72412), 4.79344)) ^ Ts_Rank(correlation(((close * 0.490655) + (vwap * (1 - 0.490655))), adv20, 4.92416), 9.0615)) * -1)
        """
        return None

    def wq070(self):
        """
        ((rank(delta(vwap, 1.29456)) ^ Ts_Rank(correlation(IndNeutralize(close, IndClass.industry), adv50, 17.8256), 17.9171)) * -1)
        """
        return None

    def wq071(self):
        """
        max(Ts_Rank(decay_linear(correlation(Ts_Rank(close, 3.43976), Ts_Rank(adv180, 12.0647), 18.0175), 4.20501), 15.6948), Ts_Rank(decay_linear((rank(((low + open) - (vwap + vwap)))^2), 16.4662), 4.4388))
        """
        return None

    def wq072(self):
        """
        (rank(decay_linear(correlation(((high + low) / 2), adv40, 8.93345), 10.1519)) / rank(decay_linear(correlation(Ts_Rank(vwap, 3.72469), Ts_Rank(volume, 18.5188), 6.86671), 2.95011)))
        """
        if self.vwap is None:
            return None
        return (cs_rank(decay_linear(correlation((self.high + self.low) / 2, self.adv40, 9), 10)) /
                cs_rank(decay_linear(correlation(ts_rank(self.vwap, 4), ts_rank(self.volume, 19), 7), 3)))

    def wq073(self):
        """
        (max(rank(decay_linear(delta(vwap, 4.72775), 2.91864)), Ts_Rank(decay_linear(((delta(((open * 0.147155) + (low * (1 - 0.147155))), 2.03608) / ((open * 0.147155) + (low * (1 - 0.147155)))) * -1), 3.33829), 16.7411)) * -1)
        """
        return None

    def wq074(self):
        """
        ((rank(correlation(close, sum(adv30, 37.4843), 15.1365)) < rank(correlation(rank(((high * 0.0261661) + (vwap * (1 - 0.0261661)))), rank(volume), 11.4791))) * -1)
        """
        if self.vwap is None:
            return None
        left = cs_rank(correlation(self.close, ts_sum(self.adv30, 37), 15))
        right = cs_rank(correlation(cs_rank((self.high * 0.0261661) + (self.vwap * (1 - 0.0261661))), cs_rank(self.volume), 11))
        return (left < right) * -1

    def wq075(self):
        """
        (rank(correlation(vwap, volume, 4.24304)) < rank(correlation(rank(low), rank(adv50), 12.4413)))
        """
        if self.vwap is None:
            return None
        return (cs_rank(correlation(self.vwap, self.volume, 4)) < cs_rank(correlation(cs_rank(self.low), cs_rank(self.adv50), 12))).astype(float)

    def wq076(self):
        """
        (max(rank(decay_linear(delta(vwap, 1.24383), 11.8259)), Ts_Rank(decay_linear(Ts_Rank(correlation(IndNeutralize(low, IndClass.sector), adv81, 8.14941), 19.569), 17.1543), 19.383)) * -1)
        """
        return None

    def wq077(self):
        """
        min(rank(decay_linear(((((high + low) / 2) + high) - (vwap + high)), 20.0451)), rank(decay_linear(correlation(((high + low) / 2), adv40, 3.1614), 5.64125)))
        """
        if self.vwap is None:
            return None
        return np.minimum(
            cs_rank(decay_linear((((self.high + self.low) / 2) + self.high) - (self.vwap + self.high), 20)),
            cs_rank(decay_linear(correlation((self.high + self.low) / 2, self.adv40, 3), 6)))

    def wq078(self):
        """
        (rank(correlation(sum(((low * 0.352233) + (vwap * (1 - 0.352233))), 19.7428), sum(adv40, 19.7428), 6.83313)) ^ rank(correlation(rank(vwap), rank(volume), 5.77492)))
        """
        if self.vwap is None:
            return None
        left = cs_rank(correlation(ts_sum((self.low * 0.352233) + (self.vwap * (1 - 0.352233)), 20), ts_sum(self.adv40, 20), 7))
        right = cs_rank(correlation(cs_rank(self.vwap), cs_rank(self.volume), 6))
        return left ** right

    def wq079(self):
        """
        (rank(delta(IndNeutralize(((close * 0.60733) + (open * (1 - 0.60733))), IndClass.sector), 1.23438)) < rank(correlation(Ts_Rank(vwap, 3.60973), Ts_Rank(adv150, 9.18637), 14.6644)))
        """
        return None

    def wq080(self):
        """
        ((rank(Sign(delta(IndNeutralize(((open * 0.868128) + (high * (1 - 0.868128))), IndClass.industry), 4.04545))) ^ Ts_Rank(correlation(high, adv10, 5.11456), 5.53756)) * -1)
        """
        return None

    def wq081(self):
        """
        ((rank(Log(product(rank((rank(correlation(vwap, sum(adv10, 49.6054), 8.47743))^4)), 14.9655))) < rank(correlation(rank(vwap), rank(volume), 5.07914))) * -1)
        """
        return None

    def wq082(self):
        """
        (min(rank(decay_linear(delta(open, 1.46063), 14.8717)), Ts_Rank(decay_linear(correlation(IndNeutralize(volume, IndClass.sector), ((open * 0.634196) + (open * (1 - 0.634196))), 17.4842), 6.92131), 13.4283)) * -1)
        """
        return None

    def wq083(self):
        """
        ((rank(delay(((high - low) / (sum(close, 5) / 5)), 2)) * rank(rank(volume))) / (((high - low) / (sum(close, 5) / 5)) / (vwap - close)))
        """
        if self.vwap is None:
            return None
        numerator = cs_rank(delay(((self.high - self.low) / (ts_sum(self.close, 5) / 5)), 2)) * cs_rank(cs_rank(self.volume))
        denominator = ((self.high - self.low) / (ts_sum(self.close, 5) / 5)) / (self.vwap - self.close)
        return numerator / denominator

    def wq084(self):
        """
        SignedPower(Ts_Rank((vwap - ts_max(vwap, 15.3217)), 20.7127), delta(close, 4.96796))
        """
        if self.vwap is None:
            return None
        return signed_power(ts_rank(self.vwap - ts_max(self.vwap, 15), 21), delta(self.close, 5))

    def wq085(self):
        """
        (rank(correlation(((high * 0.876703) + (close * (1 - 0.876703))), adv30, 9.61331)) ^ rank(correlation(Ts_Rank(((high + low) / 2), 3.70596), Ts_Rank(volume, 10.1595), 7.11408)))
        """
        left = cs_rank(correlation((self.high * 0.876703) + (self.close * (1 - 0.876703)), self.adv30, 10))
        right = cs_rank(correlation(ts_rank((self.high + self.low) / 2, 4), ts_rank(self.volume, 10), 7))
        return left ** right

    def wq086(self):
        """
        ((Ts_Rank(correlation(close, sum(adv20, 14.7444), 6.00049), 20.4195) < rank(((open + close) - (vwap + open)))) * -1)
        """
        if self.vwap is None:
            return None
        left = ts_rank(correlation(self.close, ts_sum(self.adv20, 15), 6), 20)
        right = cs_rank((self.open + self.close) - (self.vwap + self.open))
        return (left < right) * -1

    def wq087(self):
        """
        (max(rank(decay_linear(delta(((close * 0.369701) + (vwap * (1 - 0.369701))), 1.91233), 2.65461)), Ts_Rank(decay_linear(abs(correlation(IndNeutralize(adv81, IndClass.industry), close, 13.4132)), 4.89768), 14.4535)) * -1)
        """
        return None

    def wq088(self):
        """
        min(rank(decay_linear(((rank(open) + rank(low)) - (rank(high) + rank(close))), 8.06882)), Ts_Rank(decay_linear(correlation(Ts_Rank(close, 8.44728), Ts_Rank(adv60, 20.6966), 8.01266), 6.65053), 2.61957))
        """
        left = cs_rank(decay_linear((cs_rank(self.open) + cs_rank(self.low)) - (cs_rank(self.high) + cs_rank(self.close)), 8))
        right = ts_rank(decay_linear(correlation(ts_rank(self.close, 8), ts_rank(self.adv60, 21), 8), 7), 3)
        return np.minimum(left, right)

    def wq089(self):
        """
        (Ts_Rank(decay_linear(correlation(IndNeutralize(vwap, IndClass.industry), adv81, 17.4193), 19.848), 7.54455) / rank(decay_linear(delta(((close * 0.524434) + (vwap * (1 - 0.524434))), 2.77377), 16.2664)))
        """
        return None

    def wq090(self):
        """
        ((rank((close - ts_max(close, 4.66719))) ^ Ts_Rank(correlation(IndNeutralize(adv40, IndClass.subindustry), low, 5.38375), 3.21856)) * -1)
        """
        return None

    def wq091(self):
        """
        ((Ts_Rank(decay_linear(decay_linear(correlation(IndNeutralize(close, IndClass.industry), volume, 9.74928), 16.398), 3.83219), 4.8667) - rank(decay_linear(correlation(vwap, adv30, 4.01303), 2.6809))) * -1)
        """
        return None

    def wq092(self):
        """
        min(Ts_Rank(decay_linear(((((high + low) / 2) + close) < (low + open)), 14.7221), 18.8683), Ts_Rank(decay_linear(correlation(rank(low), rank(adv30), 7.58555), 6.94024), 6.80584))
        """
        # logical comparison inside decay_linear not directly supported
        return None

    def wq093(self):
        """
        (Ts_Rank(decay_linear(correlation(IndNeutralize(vwap, IndClass.industry), adv81, 17.4193), 19.848), 7.54455) / rank(decay_linear(delta(((close * 0.524434) + (vwap * (1 - 0.524434))), 2.77377), 16.2664)))
        """
        return None

    def wq094(self):
        """
        ((rank((vwap - ts_min(vwap, 11.5783))) ^ Ts_Rank(correlation(Ts_Rank(vwap, 19.6462), Ts_Rank(adv60, 4.02992), 18.0926), 2.70756)) * -1)
        """
        return None

    def wq095(self):
        """
        (rank((open - ts_min(open, 12.4105))) < Ts_Rank((rank(correlation(sum(((high + low) / 2), 19.1351), sum(adv40, 19.1351), 12.8742)) ^ 5), 11.7584))
        """

        left = cs_rank(self.open - ts_min(self.open, 12))
        right = ts_rank(cs_rank(correlation(ts_sum((self.high + self.low) / 2, 19), ts_sum(self.adv40, 19), 13)) ** 5, 12)
        return (left < right).astype(float)

    def wq096(self):
        """
        (max(Ts_Rank(decay_linear(correlation(rank(vwap), rank(volume), 3.83878), 4.16783), 8.38151), Ts_Rank(decay_linear(Ts_Rank(Ts_ArgMax(correlation(rank(open), rank(adv15), 20.8187), 8.62571), 6.95668), 8.07206), 14.0365), 13.4143)) * -1)
        """
        return None

    def wq097(self):
        """
        ((rank(decay_linear(delta(IndNeutralize(((low * 0.721001) + (vwap * (1 - 0.721001))), IndClass.industry), 3.3705), 20.4523)) - Ts_Rank(decay_linear(Ts_Rank(correlation(Ts_Rank(low, 7.87871), Ts_Rank(adv60, 17.255), 4.97547), 18.5925), 15.7152), 6.71659)) * -1)
        """
        return None

    def wq098(self):
        """
        (rank(decay_linear(correlation(vwap, sum(adv5, 26.4719), 4.58418), 7.18088)) - rank(decay_linear(Ts_Rank(Ts_ArgMin(correlation(rank(open), rank(adv15), 20.8187), 8.62571), 6.95668), 8.07206)))
        """
        return None

    def wq099(self):
        """
        ((rank(correlation(sum(((high + low) / 2), 19.8975), sum(adv60, 19.8975), 8.8136)) < rank(correlation(low, volume, 6.28259))) * -1)
        """
        left = cs_rank(correlation(ts_sum((self.high + self.low) / 2, 20), ts_sum(self.adv60, 20), 9))
        right = cs_rank(correlation(self.low, self.volume, 6))
        return (left < right) * -1

    def wq100(self):
        """
        (0 - (1 * (((1.5 * scale(indneutralize(indneutralize(rank(((((close - low) - (high - close)) / (high - low)) * volume)), IndClass.subindustry), IndClass.subindustry))) - scale(indneutralize((correlation(close, rank(adv20), 5) - rank(ts_argmin(close, 30))), IndClass.subindustry))) * (volume / adv20))))
        """
        return None

    def wq101(self):
        """
        ((close - open) / ((high - low) + .001))
        """
        return (self.close - self.open) / ((self.high - self.low) + 0.001)
