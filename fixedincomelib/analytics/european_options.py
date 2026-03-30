import math
from enum import Enum
from typing import Optional, Dict
from scipy.stats import norm


class CallOrPut(Enum):

    CALL = "call"
    PUT = "put"
    INVALID = "invalid"

    @classmethod
    def from_string(cls, value: str) -> "CallOrPut":
        if not isinstance(value, str):
            raise TypeError("value must be a string")
        try:
            return cls(value.lower())
        except ValueError:
            raise ValueError(f"Invalid token: {value}")

    def to_string(self) -> str:
        return self.value


class SimpleMetrics(Enum):

    ## valuations
    PV = "pv"
    ## vol
    IMPLIED_NORMAL_VOL = "implied_normal_vol"
    IMPLIED_LOG_NORMAL_VOL = "implied_log_normal_vol"
    ## pv sensitivities
    DELTA = "delta"
    GAMMA = "gamma"
    VEGA = "vega"
    TTE_RISK = "tte_risk"
    STRIKE_RISK = "strike_risk"
    STRIKE_RISK_2 = "strike_risk_2"
    THETA = "theta"

    ## vol sensitivities
    # nv = f(ln_vol, f, k, tte)
    D_N_VOL_D_LN_VOL = "d_n_vol_d_ln_vol"
    D_N_VOL_D_FORWARD = "d_n_vol_d_forward"
    D_N_VOL_D_TTE = "d_n_vol_d_tte"
    D_N_VOL_D_STRIKE = "d_n_vol_d_strike"
    # ln_vol = f^-1(nv, f, k, tte)
    D_LN_VOL_D_N_VOL = "d_ln_vol_d_n_vol"
    D_LN_VOL_D_FORWARD = "d_ln_vol_d_forward"
    D_LN_VOL_D_TTE = "d_ln_vol_d_tte"
    D_LN_VOL_D_STRIKE = "d_ln_vol_d_strike"

    @classmethod
    def from_string(cls, value: str) -> "SimpleMetrics":
        if not isinstance(value, str):
            raise TypeError("value must be a string")
        try:
            return cls(value.lower())
        except ValueError:
            raise ValueError(f"Invalid token: {value}")

    def to_string(self) -> str:
        return self.value


class EuropeanOptionAnalytics:

    @staticmethod
    def european_option_log_normal(
        forward: float,
        strike: float,
        time_to_expiry: float,
        log_normal_sigma: float,
        option_type: Optional[CallOrPut] = CallOrPut.CALL,
        calc_risk: Optional[bool] = False,
    ) -> Dict[SimpleMetrics, float]:
        """
        Computes the Black-76 price and analytic Greeks of a European call or put option
        in the forward measure, using lognormal implied volatility.

        res should include
        - SimpleMetrics.PV: present value
        - SimpleMetrics.DELTA: delta
        - SimpleMetrics.GAMMA: gamma
        - SimpleMetrics.VEGA: vega
        - SimpleMetrics.THETA: theta
        - SimpleMetrics.TTE_RISK: time to expiry risk
        - SimpleMetrics.STRIKE_RISK: strike risk

        use calc_risk to control whether to compute the risk metrics or not
        """

        if time_to_expiry <= 0 or log_normal_sigma <= 0:
            raise ValueError("Time to expiry and implied log-normal sigma must be positive")

        sqrt_t = math.sqrt(time_to_expiry)
        std_dev = log_normal_sigma * sqrt_t
        d1 = (math.log(forward / strike) + 0.5 * std_dev**2) / std_dev
        d2 = d1 - std_dev
        
        cp = 1 if option_type == CallOrPut.CALL else -1
        pv = cp * (forward * norm.cdf(cp * d1) - strike * norm.cdf(cp * d2))
        
        res = {SimpleMetrics.PV: pv}
        if calc_risk:
            pdf_d1 = norm.pdf(d1)
            res[SimpleMetrics.DELTA] = norm.cdf(d1) if option_type == CallOrPut.CALL else norm.cdf(d1) - 1
            res[SimpleMetrics.GAMMA] = pdf_d1 / (forward * std_dev)
            res[SimpleMetrics.VEGA] = forward * sqrt_t * pdf_d1
            res[SimpleMetrics.STRIKE_RISK] = -cp * norm.cdf(cp * d2)
            # Theta = -dV/dT, TTE_RISK = dV/dT
            res[SimpleMetrics.THETA] = -(forward * pdf_d1 * log_normal_sigma) / (2 * sqrt_t)
            res[SimpleMetrics.TTE_RISK] = -res[SimpleMetrics.THETA]

        # pricing

        # risk

        return res

    @staticmethod
    def european_option_normal(
        forward: float,
        strike: float,
        time_to_expiry: float,
        normal_sigma: float,
        option_type: Optional[CallOrPut] = CallOrPut.CALL,
        calc_risk: Optional[bool] = False,
    ) -> Dict[SimpleMetrics, float]:
        """
        Computes the Bachelier (normal) price and analytic Greeks of a European call or put option
        in the forward measure, using normal implied volatility.

        res should include
        - SimpleMetrics.PV: present value
        - SimpleMetrics.DELTA: delta
        - SimpleMetrics.GAMMA: gamma
        - SimpleMetrics.VEGA: vega
        - SimpleMetrics.THETA: theta
        - SimpleMetrics.TTE_RISK: time to expiry risk
        - SimpleMetrics.STRIKE_RISK: strike risk

        use calc_risk to control whether to compute the risk metrics or not
        """

        if time_to_expiry <= 0 or normal_sigma <= 0:
            raise ValueError("Time to expiry and implied normal sigma must be positive")

        sqrt_t = math.sqrt(time_to_expiry)
        std_dev = normal_sigma * sqrt_t
        d = (forward - strike) / std_dev
        
        cp = 1 if option_type == CallOrPut.CALL else -1
        pv = cp * (forward - strike) * norm.cdf(cp * d) + std_dev * norm.pdf(d)
        
        res = {SimpleMetrics.PV: pv}
        if calc_risk:
            pdf_d = norm.pdf(d)
            res[SimpleMetrics.DELTA] = norm.cdf(d) if option_type == CallOrPut.CALL else norm.cdf(d) - 1
            res[SimpleMetrics.GAMMA] = pdf_d / std_dev
            res[SimpleMetrics.VEGA] = sqrt_t * pdf_d
            res[SimpleMetrics.STRIKE_RISK] = -cp * norm.cdf(cp * d)
            res[SimpleMetrics.THETA] = -(normal_sigma * pdf_d) / (2 * sqrt_t)
            res[SimpleMetrics.TTE_RISK] = -res[SimpleMetrics.THETA]

        # pricing

        # risk

        return res

    @staticmethod
    def implied_lognormal_vol_sensitivities(
        pv: float,
        forward: float,
        strike: float,
        time_to_expiry: float,
        option_type: Optional[CallOrPut] = CallOrPut.CALL,
        calc_risk: Optional[bool] = False,
        tol: Optional[float] = 1e-8,
    ) -> Dict[SimpleMetrics, float]:
        """
        Computes the implied lognormal volatility from option PV under the Black-76 model and its sensitivities.

        res should include
        - SimpleMetrics.IMPLIED_LOG_NORMAL_VOL: implied lognormal volatility
        - SimpleMetrics.D_LN_VOL_D_FORWARD: sensitivity of implied lognormal volatility to forward
        - SimpleMetrics.D_LN_VOL_D_TTE: sensitivity of implied lognormal volatility to time to expiry
        - SimpleMetrics.D_LN_VOL_D_STRIKE: sensitivity of implied lognormal volatility to strike

        use calc_risk to control whether to compute the risk metrics or not

        """
        vol = EuropeanOptionAnalytics._implied_lognormal_vol_black(pv, forward, strike, time_to_expiry, option_type, tol)
        res = {SimpleMetrics.IMPLIED_LOG_NORMAL_VOL: vol}
        if calc_risk:
            g = EuropeanOptionAnalytics.european_option_log_normal(forward, strike, time_to_expiry, vol, option_type, True)
            vega = g[SimpleMetrics.VEGA]
            res[SimpleMetrics.D_LN_VOL_D_FORWARD] = -g[SimpleMetrics.DELTA] / vega
            res[SimpleMetrics.D_LN_VOL_D_STRIKE] = -g[SimpleMetrics.STRIKE_RISK] / vega
            res[SimpleMetrics.D_LN_VOL_D_TTE] = -g[SimpleMetrics.TTE_RISK] / vega

        # 1) compute implied vol

        # 2) compute greeks at implied vol

        # 3) compute sensitivities of implied vol using implicit function theorem
        # G(\sigma_imp(f, k, tte, pv), f, k, tte) = pv, where G is the pricing function
        # For instance, for f risk, we have
        # dG/dsigma * dsigma / df = - dG/df => - dG/df / dG/dsigma

        return res

    @staticmethod
    def implied_normal_vol_sensitivities(
        pv: float,
        forward: float,
        strike: float,
        time_to_expiry: float,
        option_type: Optional[CallOrPut] = CallOrPut.CALL,
        calc_risk: Optional[bool] = False,
        tol: Optional[float] = 1e-8,
    ) -> Dict[SimpleMetrics, float]:
        """
        Computes the implied normal volatility from option PV under the Bachelier model and,
        optionally, its sensitivities using the implicit function theorem.

        res should include
        - SimpleMetrics.IMPLIED_NORMAL_VOL: implied normal volatility
        - SimpleMetrics.D_N_VOL_D_FORWARD: sensitivity of implied normal volatility to forward
        - SimpleMetrics.D_N_VOL_D_TTE: sensitivity of implied normal volatility to time to expiry
        - SimpleMetrics.D_N_VOL_D_STRIKE: sensitivity of implied normal volatility to strike

        use calc_risk to control whether to compute the risk metrics or not
        """


        # 1) Compute implied normal vol

        

        # 2) Compute Greeks at implied vol

        # 3) Compute sensitivities of implied vol
        # G(\sigma_imp(f, k, tte), f, k, tte) = pv, where G is the pricing function
        # For instance, for f risk, we have
        # dG/dsigma * dsigma / df = - dG/df => - dG/df / dG/dsigma


        vol = EuropeanOptionAnalytics._implied_normal_vol_bachelier(pv, forward, strike, time_to_expiry, option_type, tol)
        res = {SimpleMetrics.IMPLIED_NORMAL_VOL: vol}
        if calc_risk:
            g = EuropeanOptionAnalytics.european_option_normal(forward, strike, time_to_expiry, vol, option_type, True)
            vega = g[SimpleMetrics.VEGA]
            res[SimpleMetrics.D_N_VOL_D_FORWARD] = -g[SimpleMetrics.DELTA] / vega
            res[SimpleMetrics.D_N_VOL_D_STRIKE] = -g[SimpleMetrics.STRIKE_RISK] / vega
            res[SimpleMetrics.D_N_VOL_D_TTE] = -g[SimpleMetrics.TTE_RISK] / vega
        return res

    @staticmethod
    def lognormal_vol_to_normal_vol(
        forward: float,
        strike: float,
        time_to_expiry: float,
        log_normal_sigma: float,
        calc_risk: Optional[bool] = False,
        shift: Optional[float] = 0.0,
        tol: Optional[float] = 1e-8,
    ) -> Dict[SimpleMetrics, float]:
        """
        Converts lognormal implied volatility into normal (Bachelier) implied volatility
        via price equivalence, and compute sensitivities.

        res should include
        - SimpleMetrics.IMPLIED_NORMAL_VOL: equivalent normal implied volatility
        - SimpleMetrics.D_N_VOL_D_LN_VOL: sensitivity of normal vol to lognormal vol
        - SimpleMetrics.D_N_VOL_D_FORWARD: sensitivity of normal vol to forward
        - SimpleMetrics.D_N_VOL_D_STRIKE: sensitivity of normal vol to strike
        - SimpleMetrics.D_N_VOL_D_TTE: sensitivity of normal vol to time to expiry
        """

        res: Dict[SimpleMetrics, float] = {}

        option_type = CallOrPut.PUT if forward > strike else CallOrPut.CALL

        # 1) black price (BS'76)
        # V = BS(f, k, tte, log_normal_sigma)

        # 2) implied normal vol (Bachelier)
        # nv = Imp(f, k, tte, V)
        # notice dnv/dV = 1 / vega


        # 1) Get PV from Black model
        res_ln = EuropeanOptionAnalytics.european_option_log_normal(forward + shift, strike + shift, time_to_expiry, log_normal_sigma, CallOrPut.CALL, True)
        # 2) Get Normal Vol and sensitivities from that PV
        res_n = EuropeanOptionAnalytics.implied_normal_vol_sensitivities(res_ln[SimpleMetrics.PV], forward + shift, strike + shift, time_to_expiry, CallOrPut.CALL, calc_risk, tol)
        
        res = {SimpleMetrics.IMPLIED_NORMAL_VOL: res_n[SimpleMetrics.IMPLIED_NORMAL_VOL]}
        if calc_risk:
            # d_N_vol / d_LN_vol = (dV/d_LN_vol) / (dV/d_N_vol) = Vega_LN / Vega_N
            vega_n = EuropeanOptionAnalytics.european_option_normal(forward + shift, strike + shift, time_to_expiry, res_n[SimpleMetrics.IMPLIED_NORMAL_VOL], CallOrPut.CALL, True)[SimpleMetrics.VEGA]
            res[SimpleMetrics.D_N_VOL_D_LN_VOL] = res_ln[SimpleMetrics.VEGA] / vega_n
            res[SimpleMetrics.D_N_VOL_D_FORWARD] = res_n[SimpleMetrics.D_N_VOL_D_FORWARD] + res_ln[SimpleMetrics.DELTA] / vega_n
            res[SimpleMetrics.D_N_VOL_D_STRIKE] = res_n[SimpleMetrics.D_N_VOL_D_STRIKE] + res_ln[SimpleMetrics.STRIKE_RISK] / vega_n
            res[SimpleMetrics.D_N_VOL_D_TTE] = res_n[SimpleMetrics.D_N_VOL_D_TTE] + res_ln[SimpleMetrics.TTE_RISK] / vega_n
        return res


    @staticmethod
    def normal_vol_to_lognormal_vol(
        forward: float,
        strike: float,
        time_to_expiry: float,
        normal_sigma: float,
        calc_risk: Optional[bool] = False,
        shift: Optional[float] = 0.0,
        tol: Optional[float] = 1e-8,
    ) -> Dict[SimpleMetrics, float]:
        """
        Converts normal implied volatility into lognormal implied volatility
        via price equivalence, and computes sensitivities.

        res should include
        - SimpleMetrics.IMPLIED_LOG_NORMAL_VOL: equivalent lognormal implied volatility
        - SimpleMetrics.D_LN_VOL_D_N_VOL: sensitivity of lognormal vol to normal vol
        - SimpleMetrics.D_LN_VOL_D_FORWARD: sensitivity of lognormal vol to forward
        - SimpleMetrics.D_LN_VOL_D_STRIKE: sensitivity of lognormal vol to strike
        - SimpleMetrics.D_LN_VOL_D_TTE: sensitivity of lognormal vol to time to expiry
        """

        res: Dict[SimpleMetrics, float] = {}

        option_type = CallOrPut.PUT if forward > strike else CallOrPut.CALL

        # 1) bachelier
        # V = Bachelier(f, k, tte, normal_sigma)

        # 2) implied log normal vol (BS'76)
        # ln_nv = Imp(f, k, tte, V)
        # notice dln_nv/dV = 1 / vega

        # risk

        res_n = EuropeanOptionAnalytics.european_option_normal(forward + shift, strike + shift, time_to_expiry, normal_sigma, CallOrPut.CALL, True)
        res_ln = EuropeanOptionAnalytics.implied_lognormal_vol_sensitivities(res_n[SimpleMetrics.PV], forward + shift, strike + shift, time_to_expiry, CallOrPut.CALL, calc_risk, tol)
        
        res = {SimpleMetrics.IMPLIED_LOG_NORMAL_VOL: res_ln[SimpleMetrics.IMPLIED_LOG_NORMAL_VOL]}
        if calc_risk:
            vega_ln = EuropeanOptionAnalytics.european_option_log_normal(forward + shift, strike + shift, time_to_expiry, res_ln[SimpleMetrics.IMPLIED_LOG_NORMAL_VOL], CallOrPut.CALL, True)[SimpleMetrics.VEGA]
            res[SimpleMetrics.D_LN_VOL_D_N_VOL] = res_n[SimpleMetrics.VEGA] / vega_ln
            res[SimpleMetrics.D_LN_VOL_D_FORWARD] = res_ln[SimpleMetrics.D_LN_VOL_D_FORWARD] + res_n[SimpleMetrics.DELTA] / vega_ln
            res[SimpleMetrics.D_LN_VOL_D_STRIKE] = res_ln[SimpleMetrics.D_LN_VOL_D_STRIKE] + res_n[SimpleMetrics.STRIKE_RISK] / vega_ln
            res[SimpleMetrics.D_LN_VOL_D_TTE] = res_ln[SimpleMetrics.D_LN_VOL_D_TTE] + res_n[SimpleMetrics.TTE_RISK] / vega_ln
        return res

    ### utilities below

    @staticmethod
    def _implied_lognormal_vol_black(
        pv: float,
        forward: float,
        strike: float,
        time_to_expiry: float,
        option_type: Optional[CallOrPut] = CallOrPut.CALL,
        tol: Optional[float] = 1e-8,
        vol_min: Optional[float] = 0.0,
        vol_max: Optional[float] = 10.0,
        max_iter: Optional[int] = 1000,
    ) -> float:
        """
        Solves for the Black-76 implied lognormal volatility from a European option price using a
        hybrid Newton-Raphson and bisection method, subject to arbitrage bounds and convergence
        controls.

        Return "sigma" implied lognormal volatility

        """
        # Use the provided guess method
        vol = EuropeanOptionAnalytics._initial_log_normal_implied_vol_guess(forward, time_to_expiry, pv)
        
        # Ensure the guess is within sensible bounds before starting
        vol = max(vol_min, min(vol_max, vol))

        for _ in range(max_iter):
            res = EuropeanOptionAnalytics.european_option_log_normal(
                forward, strike, time_to_expiry, vol, option_type, calc_risk=True
            )
            diff = res[SimpleMetrics.PV] - pv
            
            if abs(diff) < tol:
                return vol
            
            vega = res[SimpleMetrics.VEGA]
            
            # Prevent division by zero
            if abs(vega) < 1e-12:
                break
                
            vol -= diff / vega
            
            # Keep vol within physical bounds during iteration
            vol = max(vol_min, min(vol_max, vol))
            
        return vol

    @staticmethod
    def _implied_normal_vol_bachelier(
        pv: float,
        forward: float,
        strike: float,
        time_to_expiry: float,
        option_type: Optional[CallOrPut] = CallOrPut.CALL,
        tol: Optional[float] = 1e-8,
        vol_min: Optional[float] = 1e-8,
        vol_max: Optional[float] = 0.1,
        max_iter: Optional[int] = 100,
    ) -> float:
        """
        Solves for the Bachelier implied normal volatility from a European option price using a
        hybrid Newton-Raphson and bisection method, subject to arbitrage bounds and convergence
        controls.

        Return "sigma" implied lognormal volatility
        """
        # Use the provided guess method
        vol = EuropeanOptionAnalytics._initial_normal_implied_vol_guess(time_to_expiry, pv)
        
        vol = max(vol_min, min(vol_max, vol))

        for _ in range(max_iter):
            res = EuropeanOptionAnalytics.european_option_normal(
                forward, strike, time_to_expiry, vol, option_type, calc_risk=True
            )
            diff = res[SimpleMetrics.PV] - pv
            
            if abs(diff) < tol:
                return vol
            
            vega = res[SimpleMetrics.VEGA]
            
            if abs(vega) < 1e-12:
                break
                
            vol -= diff / vega
            vol = max(vol_min, min(vol_max, vol))
            
        return vol

    @staticmethod
    def _initial_log_normal_implied_vol_guess(forward: float, time_to_expiry: float, pv: float):
        return math.sqrt(2 * math.pi / time_to_expiry) * pv / forward

    @staticmethod
    def _initial_normal_implied_vol_guess(time_to_expiry: float, pv: float):
        return pv * math.sqrt(2 * math.pi / time_to_expiry)
