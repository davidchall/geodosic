# third-party imports
import numpy as np
from scipy.stats import norm


class LKBModel(object):
    """Lyman-Kutcher-Burman NTCP model.

    Lyman (1985) Radiat Res Suppl, 8, S13
    Kutcher, Burman et al (1991) Int J Radiat Oncol Biol Phys, 21, 137
    """
    def __init__(self, TD50, m, n):
        """Instantiate LKB NTCP model.

        Args:
            TD50: EUD which results in 50% complication probability
            m: normalized slope of dose-response curve
            n: describes volume dependence of organ

        Note: reduces to the probit model when n=1
        """
        self.TD50 = TD50
        self.m = m
        self.n = n

    def ntcp(self, dvh):
        eud = dvh.eud(a=1./self.n)
        t = (eud - self.TD50) / (self.m * self.TD50)
        return norm.cdf(t)


class RelativeSerialityModel(object):
    """Relative seriality NTCP model.

    Kallman, Agren, Brahme (1992) Int J Radiat Biol, 62, 249
    """
    def __init__(self, D50, gamma, s):
        """Instantiate relative seriality NTCP model.

        Args:
            D50: dose which results in 50% complication probability
            gamma: normalized slope of dose-reponse curve
            s: relative seriality factor

        Note: an approximation to the Poisson model
        """
        self.D50 = D50
        self.gamma = gamma
        self.s = s

    def response_curve(self, dose):
        tmp = np.e * self.gamma * (1. - dose / self.D50)
        return np.power(2, -np.exp(tmp))

    def ntcp(self, dvh):
        probs = self.response_curve(dvh.dose_centers)
        prod = np.prod(np.power(1. - np.power(probs, self.s), dvh.dDVH))
        return np.power(1. - prod, 1./self.s)


class LogisticDVHMetricModel(object):
    """Logistic DVH metric NTCP model.

    Bentzen, Tucker (1997) Int J Radiat Biol, 71, 531
    """
    def __init__(self, D50, gamma50, metric_func):
        """Instantiate logistic NTCP model using single DVH metric.

        Args:
            D50: dose which results in 50% complication probability
            gamma50: normalized slope of dose-reponse curve at this point
            metric_func: function returning DVH metric
        """
        self.D50 = D50
        self.gamma50 = gamma50
        self.metric_func = metric_func

    def ntcp(self, dvh):
        dvh_metric = self.metric_func(dvh)
        return 1 / (1 + np.exp(4 * self.gamma50 * (1 - dvh_metric / self.D50)))
