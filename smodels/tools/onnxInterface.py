#!/usr/bin/env python3

"""
.. module:: onnxInterface
   :synopsis: Code that delegates the computation of limits and likelihoods to
              machine-learned onnx models.

.. moduleauthor:: Wolfgang Waltenberger <wolfgang.waltenberger@gmail.com>

"""

try:
    import onnxruntime
except ModuleNotFoundError:
    print("[SModelS:onnxInterface] onnxruntime import failed. Is the module installed?")
    import sys
    sys.exit(-1)

ver = onnxruntime.__version__.split(".")
if ver[1] == "0":
    print("[SModelS:onnxInterface] WARNING you are using onnx v%s." % onnx.__version__)
    print("[SModelS:onnxInterface] We recommend onnx >= 1.0.0. Please try to update onnxruntime ASAP!")

import numpy as np
from smodels.tools.smodelsLogging import logger

def getOnnxComputer(dataset, nsig ):
    """create the onnx ul computer object
    :returns: onnx upper limit computer, and combinations of signal regions
    """
    from smodels.tools.onnxInterface import OnnxData, OnnxUpperLimitComputer
    keys = list ( dataset.globalInfo.onnxFiles.keys() )
    if len(keys)>1:
        logger.error ( "I have not yet implemented more than a single onnx file" )
        import sys
        sys.exit(-1)
    # oxf = dataset.globalInfo.onnxFiles [ keys[0] ]
    fname = "/tmp/test.onnx"
    with open ( fname, "wb" ) as f:
        f.write ( dataset.globalInfo.onnx[0] )
        f.close()
    import onnxruntime
    oxsession = onnxruntime.InferenceSession( fname )
    data = OnnxData(nsig, oxsession, fname )
    ulcomputer = OnnxUpperLimitComputer(data, lumi=dataset.getLumi() )
    return ulcomputer

class OnnxData:
    """
    Holds data for use in onnx
    :ivar nsignals: signal predictions list divided into sublists, 
                    one for each onnx file
    :ivar inputOnnx: list of onnx instances
    :ivar onnxFiles: optional list of json files, mostly for debugging
    """

    def __init__(self, nsignals, inputOnnx, onnxFiles=None ):
        self.nsignals = nsignals  # fb
        self.inputOnnx = inputOnnx
        self.cached_likelihoods = {}  ## cache of likelihoods (actually twice_nlls)
        self.cached_lmaxes = {}  # cache of lmaxes (actually twice_nlls)
        self.cachedULs = {False: {}, True: {}, "posteriori": {}}
        self.onnxFiles = onnxFiles
        self.combinations = None

    def totalYield ( self ):
        """ the total yield in all signal regions """
        S = sum ( self.nsignals )
        return S

class OnnxUpperLimitComputer:
    """
    Class that computes the upper limit using the onnx models and 
    signal informations in the `data` instance of `OnnxData`
    """

    def __init__(self, data, cl=0.95, lumi=None ):
        """
        :param data: instance of `OnnxData` holding the signals information
        :param cl: confdence level at which the upper limit is desired to be computed
        """
        self.data = data
        self.lumi = lumi
        self.sigma_mu = 1.

    def likelihood(self, mu=1.0, nll=False, expected=False):
        """
        Returns the value of the likelihood.
        Inspired by the `onnx.infer.mle` module but for non-log likelihood
        :param nll: if true, return nll, not llhd
        :param expected: if False, compute expected values, if True,
            compute a priori expected, if "posteriori" compute posteriori
            expected
        """
        # FIXME implemented expectation values
        inp = [ self.data.nsignals ]
        ort_outs = self.data.inputOnnx.run(None, { "dense_input": inp } )
        ret = float ( ort_outs[0][0][0] ) # nll
        if nll: # return nll
            return ret
        return self.exponentiateNLL ( ret, doIt=True )

    def chi2(self):
        """
        Returns the chi square
        """
        return 2 * (
            self.lmax(nll=True) - self.likelihood(nll=True)
        )

    def exponentiateNLL(self, twice_nll, doIt):
        """if doIt, then compute likelihood from nll,
        else return nll"""
        if twice_nll == None:
            return None
            #if doIt:
            #    return 0.0
            #return 9000.0
        if doIt:
            return np.exp(-twice_nll / 2.0)
        return twice_nll / 2.0

    def getSigmaMu(self ):
        """given a workspace, compute a rough estimate of sigma_mu,
        the uncertainty of mu_hat"""
        return 1.

    def lmax(self, nll=False, expected=False, allowNegativeSignals=False):
        """
        Returns the negative log max likelihood
        :param nll: if true, return nll, not llhd
        :param expected: if False, compute expected values, if True,
            compute a priori expected, if "posteriori" compute posteriori
            expected
        :param allowNegativeSignals: if False, then negative nsigs are replaced with 0.
        """
        return 1.

    def getUpperLimitOnSigmaTimesEff(self, expected=False ):
        """
        Compute the upper limit on the fiducial cross section sigma times efficiency:
            - by default, the combination of the workspaces contained into self.workspaces
            - if workspace_index is specified, self.workspace[workspace_index]
              (useful for computation of the best upper limit)

        :param expected:  - if set to `True`: uses expected SM backgrounds as signals
                          - else: uses `self.nsignals`
        :return: the upper limit on sigma times eff at `self.cl` level (0.95 by default)
        """

        ul = self.getUpperLimitOnMu( expected=expected )
        if ul == None:
            return ul
        if self.lumi is None:
            logger.error(f"asked for upper limit on fiducial xsec, but no lumi given with the data")
            return ul
        xsec = self.data.totalYield() / self.lumi
        return ul * xsec

    # Trying a new method for upper limit computation :
    # re-scaling the signal predictions so that mu falls in [0, 10] instead of
    # looking for mu bounds
    # Usage of the index allows for rescaling
    def getUpperLimitOnMu(self, expected=False ):
        """
        Compute the upper limit on the signal strength modifier with:
        :param expected:  - if set to `True`: uses expected SM backgrounds as signals
                          - else: uses `self.nsignals`
        :return: the upper limit at `self.cl` level (0.95 by default)
        """
        return -1.

if __name__ == "__main__":
    oxfile = "../../test/database_onnx/13TeV/ATLAS/ATLAS-SUSY-2018-04-eff/model.onnx"
    yields = [[ 15., 15. ]]
    oxsession = onnxruntime.InferenceSession( oxfile )
    oxdata = OnnxData ( yields, oxsession, oxfile )
    computer = OnnxUpperLimitComputer ( oxdata )
    print ( "likelihood", computer.likelihood ( mu = 1. ) )
