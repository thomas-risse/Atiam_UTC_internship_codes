import numpy as np
import scipy.signal as sig

class inputReconstruction():
    def __init__(self, sr):
        """Initialize the class with the sampling frequency used

        Args:
            sr (int): sampling frequency
        """
        # Sampling frequency and timestep
        self.sr = sr
        self.timestep = 1 / self.sr

        # Optimal shift value
        self.tOpt = 0.5-np.sqrt(3)/6
        # pre-filter coefficients
        self.num = [1 / (1-self.tOpt)]
        self.den = [1, self.tOpt / (1-self.tOpt)]

    def _preFilter(self, x):
        """Filters signal x with the iir pre-filter 
        defined by self?num and self.den

        Args:
            x (ND array): signals to filter. Size : [Nsignals, len(signals)]

        Returns:
            ND array: filtered signals
        """
        xFiltered = sig.lfilter(self.num, self.den, x)
        return xFiltered

    def DAC(self,x):
        """Convert a digital signal to an analog representation
        using a pre filter and linear interpolation

        Args:
            x (2D array): Signals to convert

        Returns:
            function: function returning signals estimates at time t
        """
        # pre-filter digital signal
        xFiltered = self._preFilter(x)
        # Creates an interpolating function and returns it
        def xFunc(t):
            steps = (t // self.timestep).astype(int)
            relativePosition = t % self.timestep / self.timestep
            relativePosition = relativePosition[steps<int(len(xFiltered)-2)]
            steps = steps[steps<int(len(xFiltered)-2)]
            return xFiltered[steps] * (1 - relativePosition) +\
                     xFiltered[steps+1] * relativePosition
        return xFunc
