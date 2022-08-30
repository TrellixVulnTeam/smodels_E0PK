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
    print ( "keys", dataset.globalInfo.onnxFiles )
    import onnxruntime
    data = OnnxData(nsig, dataset.globalInfo.onnx[0] )
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
        inp = self.data.nsignals
        ort_outs = self.data.inputOnnx.run(None, { "dense_input": inp } )
        ret = float ( ort_outs[0][0][0] ) # nll
        if nll: # return nll
            return ret
        return self.exponentiateNLL ( ret, doIt=True )

    def chi2(self, workspace_index=None):
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

    def getSigmaMu(self, workspace):
        """given a workspace, compute a rough estimate of sigma_mu,
        the uncertainty of mu_hat"""
        obss, bgs, bgVars, nsig = {}, {}, {}, {}
        channels = workspace.channels
        for chdata in workspace["channels"]:
            if not chdata["name"] in channels:
                continue
            bg = 0.0
            var = 0.0
            for sample in chdata["samples"]:
                if sample["name"] == "Bkg":
                    tbg = sample["data"][0]
                    bg += tbg
                    hi = sample["modifiers"][0]["data"]["hi_data"][0]
                    lo = sample["modifiers"][0]["data"]["lo_data"][0]
                    delta = max((hi - bg, bg - lo))
                    var += delta**2
                if sample["name"] == "bsm":
                    ns = sample["data"][0]
                    nsig[chdata["name"]] = ns
            bgs[chdata["name"]] = bg
            bgVars[chdata["name"]] = var
        for chdata in workspace["observations"]:
            if not chdata["name"] in channels:
                continue
            obss[chdata["name"]] = chdata["data"][0]
        vars = []
        for c in channels:
            # poissonian error
            if nsig[c]==0.:
                nsig[c]=1e-5
            poiss = abs(obss[c]-bgs[c]) / nsig[c]
            gauss = bgVars[c] / nsig[c]**2
            vars.append ( poiss + gauss )
        var_mu = np.sum ( vars )
        n = len ( obss )
        # print ( f" sigma_mu from onnx uncorr {var_mu} {n} "  )
        sigma_mu = float ( np.sqrt ( var_mu / (n**2) ) )
        self.sigma_mu = sigma_mu
        #import IPython
        #IPython.embed()
        #sys.exit()

    def lmax(self, nll=False, expected=False, allowNegativeSignals=False):
        """
        Returns the negative log max likelihood
        :param nll: if true, return nll, not llhd
        :param workspace_index: supply index of workspace to use. If None,
            choose index of best combo
        :param expected: if False, compute expected values, if True,
            compute a priori expected, if "posteriori" compute posteriori
            expected
        :param allowNegativeSignals: if False, then negative nsigs are replaced with 0.
        """
        # logger.error("expected flag needs to be heeded!!!")
        logger.debug("Calling lmax")
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                "Values in x were outside bounds during a minimize step, clipping to bounds",
            )

            self.__init__(self.data)
            if workspace_index == None:
                workspace_index = self.getBestCombinationIndex()
            if workspace_index != None:
                if self.zeroSignalsFlag[workspace_index] == True:
                    logger.warning("Workspace number %d has zero signals" % workspace_index)
                    return None
                else:
                    workspace = self.updateWorkspace(workspace_index, expected=expected)
            else:
                return None
            # Same modifiers_settings as those used when running the 'onnx cls' command line
            msettings = {"normsys": {"interpcode": "code4"}, "histosys": {"interpcode": "code4p"}}
            model = workspace.model(modifier_settings=msettings)
            # obs = workspace.data(model)
            self.getSigmaMu(workspace)
            try:
                bounds = model.config.suggested_bounds()
                if allowNegativeSignals:
                    bounds[model.config.poi_index] = (-5., 10. )
                muhat, maxNllh = onnx.infer.mle.fit(workspace.data(model), model,
                        return_fitted_val=True, par_bounds = bounds )
                if False: # get sigma_mu from hessian
                    onnx.set_backend(onnx.tensorlib, 'minuit')
                    muhat, maxNllh,o = onnx.infer.mle.fit(workspace.data(model), model,
                            return_fitted_val=True, par_bounds = bounds,
                            return_result_obj = True )
                    sigma_mu = float ( np.sqrt ( o.hess_inv[0][0] ) ) * self.scale
                    # print ( f"\n>>> sigma_mu from hessian {sigma_mu:.2f}" )
                    onnx.set_backend(onnx.tensorlib, 'scipy')

                muhat = muhat[model.config.poi_index]*self.scale

            except (onnx.exceptions.FailedMinimization, ValueError) as e:
                logger.error(f"onnx mle.fit failed {e}")
                muhat, maxNllh = float("nan"), float("nan")
            self.muhat = muhat
            try:
                ret = maxNllh.tolist()
            except:
                ret = maxNllh
            try:
                ret = float(ret)
            except:
                ret = float(ret[0])
            self.data.cached_lmaxes[workspace_index] = ret
            ret = self.exponentiateNLL(ret, not nll)
            return ret

    def updateWorkspace(self, workspace_index=None, expected=False):
        """
        Small method used to return the appropriate workspace

        :param workspace_index: the index of the workspace to retrieve from the corresponding list
        :param expected: if False, retuns the unmodified (but patched) workspace. Used for computing observed or aposteriori expected limits.
                        if True, retuns the modified (and patched) workspace, where obs = sum(bkg). Used for computing apriori expected limit.
        """
        if self.nWS == 1:
            if expected == True:
                return self.workspaces_expected[0]
            else:
                return self.workspaces[0]
        else:
            if workspace_index == None:
                logger.error("No workspace index was provided.")
            if expected == True:
                return self.workspaces_expected[workspace_index]
            else:
                return self.workspaces[workspace_index]

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
