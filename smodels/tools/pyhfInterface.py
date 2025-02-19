#!/usr/bin/env python3

"""
.. module:: pyhfInterface
   :synopsis: Code that delegates the computation of limits and likelihoods to
              pyhf.

.. moduleauthor:: Gael Alguero <gaelalguero@gmail.com>
.. moduleauthor:: Wolfgang Waltenberger <wolfgang.waltenberger@gmail.com>

"""
import jsonpatch
import warnings
import jsonschema
import copy

jsonver=""
try:
    import importlib.metadata
    jsonver = importlib.metadata.version("jsonschema")
except Exception as e:
    try:
        from jsonschema import __version__ as jsonver
    except Exception as e:
        pass
if jsonver[0] == "2":
#if jsonschema.__version__[0] == "2": ## deprecated
    print ( "[SModelS:pyhfInterface] jsonschema is version %s, we need > 3.x.x" % \
            ( jsonschema.__version__ ) )
    sys.exit()

import time, sys, os
try:
    import pyhf
except ModuleNotFoundError:
    print ( "[SModelS:pyhfInterface] pyhf import failed. Is the module installed?" )
    sys.exit(-1)

ver = pyhf.__version__.split(".")
if ver[1]=="4" or (ver[1]=="5" and ver[2] in [ "0", "1" ]):
    print ( "[SModelS:pyhfInterface] WARNING you are using pyhf v%s." % pyhf.__version__ )
    print ( "[SModelS:pyhfInterface] We recommend pyhf >= 0.5.2. Please try to update pyhf ASAP!" )

pyhfinfo = { "backend": "numpy", "hasgreeted": False, "backendver": "?", "ver": ver,
             "required": "0.6.1".split(".") }

try:
    pyhf.set_backend(b"pytorch")
    import torch
    pyhfinfo["backend"] = "pytorch"
    pyhfinfo["backendver"] = torch.__version__
except pyhf.exceptions.ImportBackendError as e:
    print ( "[SModelS:pyhfInterface] WARNING could not set pytorch as the pyhf backend, falling back to the default." )
    print ( "[SModelS:pyhfInterface] We however recommend that pytorch be installed." )
    import numpy
    pyhfinfo["backendver"]=numpy.version.full_version

    warnings.filterwarnings('ignore', r'invalid value encountered in log')

from scipy import optimize
import numpy as np
from smodels.tools.smodelsLogging import logger
import logging
logging.getLogger("pyhf").setLevel(logging.CRITICAL)

def getLogger():
    """
    Configure the logging facility. Maybe adapted to fit into
    your framework.
    """

    import logging

    logger = logging.getLogger("pyhfInterface")
    # formatter = logging.Formatter('%(module)s - %(levelname)s: %(message)s')
    # ch = logging.StreamHandler()
    # ch.setFormatter(formatter)
    # ch.setLevel(logging.DEBUG)
    # logger.addHandler(ch)
    logger.setLevel(logging.DEBUG)
    return logger

countWarning = { "llhdszero": 0 }
#logger=getLogger()

class PyhfData:
    """
    Holds data for use in pyhf
    :ivar nsignals: signal predictions list divided into sublists, one for each json file
    :ivar inputJsons: list of json instances
    :ivar jsonFiles: optional list of json files
    :ivar nWS: number of workspaces = number of json files
    """
    def __init__ (self, nsignals, inputJsons, jsonFiles = None ):
        self.nsignals = nsignals # fb
        self.inputJsons = inputJsons
        self.cached_likelihoods = {} ## cache of likelihoods (actually twice_nlls)
        self.cached_lmaxes = {} # cache of lmaxes (actually twice_nlls)
        self.cachedULs = { False: {}, True: {}, "posteriori": {} }
        self.jsonFiles = jsonFiles
        self.combinations = None
        if jsonFiles != None:
            self.combinations = [os.path.splitext(os.path.basename(js))[0] for js in jsonFiles]

        self.nWS = len(inputJsons)
        self.errorFlag = False
        self.getWSInfo()
        self.checkConsistency()

    def getWSInfo(self):
        """
        Getting informations from the json files

        :ivar channelsInfo: list of dictionaries (one dictionary for each json file) containing useful information about the json files
            - :key signalRegions: list of dictonaries with 'json path' and 'size' (number of bins) of the 'signal regions' channels in the json files
            - :key otherRegions: list of strings indicating the path to the control and validation region channels
        """
        # Identifying the path to the SR and VR channels in the main workspace files
        self.channelsInfo = [] # workspace specifications
        if not isinstance(self.inputJsons, list):
            logger.error("The `inputJsons` parameter must be of type list")
            self.errorFlag = True
            return
        for ws in self.inputJsons:
            wsChannelsInfo = {}
            wsChannelsInfo['signalRegions'] = []
            wsChannelsInfo['otherRegions'] = []
            if not 'channels' in ws.keys():
                logger.error("Json file number {} is corrupted (channels are missing)".format(self.inputJsons.index(ws)))
                self.channelsInfo = None
                return
            for i_ch, ch in enumerate(ws['channels']):
                if ch['name'][:2] == 'SR': # if channel name starts with 'SR'
                    wsChannelsInfo['signalRegions'].append({'path':'/channels/'+str(i_ch)+'/samples/0', # Path of the new sample to add (signal prediction)
                                                            'size':len(ch['samples'][0]['data'])}) # Number of bins
                else:
                    wsChannelsInfo['otherRegions'].append('/channels/'+str(i_ch))
            wsChannelsInfo['otherRegions'].sort(key=lambda path: path.split('/')[-1], reverse=True) # Need to sort correctly the paths to the channels to be removed
            self.channelsInfo.append(wsChannelsInfo)

    def checkConsistency(self):
        """
        Check various inconsistencies of the PyhfData attributes

        :param zeroSignalsFlag: boolean identifying if all SRs of a single json are empty
        """
        if not isinstance(self.nsignals, list):
            logger.error("The `nsignals` parameter must be of type list")
            self.errorFlag = True
        if self.nWS != len(self.nsignals):
            logger.error('The number of subsignals provided is different from the number of json files')
            self.errorFlag = True
        self.zeroSignalsFlag = list()
        if self.channelsInfo == None:
            return
        for wsInfo, subSig in zip(self.channelsInfo, self.nsignals):
            if not isinstance(subSig, list):
                logger.error("The `nsignals` parameter must be a two dimensional list")
                self.errorFlag = True
            nBinsJson = 0
            for sr in wsInfo['signalRegions']:
                nBinsJson += sr['size']
            if nBinsJson != len(subSig):
                logger.error('The number of signals provided is different from the number of bins for json number {} and channel number {}'.format(self.channelsInfo.index(wsInfo), self.nsignals.index(subSig)))
                self.errorFlag = True
            allZero = all([s == 0 for s in subSig])
            # Checking if all signals matching this json are zero
            self.zeroSignalsFlag.append(allZero)

class PyhfUpperLimitComputer:
    """
    Class that computes the upper limit using the jsons files and signal informations in the `data` instance of `PyhfData`
    """
    def __init__ ( self, data, cl=0.95, includeCRs = False):
        """
        :param data: instance of `PyhfData` holding the signals information
        :param cl: confdence level at which the upper limit is desired to be computed

        :ivar data: created from :param data:
        :ivar nsignals: signal predictions list divided into sublists, one for each json file
        :ivar inputJsons: list of input json files as python json instances
        :ivar channelsInfo: list of channels information for the json files
        :ivar zeroSignalsFlag: list boolean flags in case all signals are zero for a specific json
        :ivar nWS: number of workspaces = number of json files
        :ivar patches: list of patches to be applied to the inputJsons as python dictionary instances
        :ivar workspaces: list of workspaces resulting from the patched inputJsons
        ;ivar workspaces_expected: list of patched workspaces with observation yields replaced by the expected ones
        :ivar cl: created from :param cl:
        :ivar scale: scale that is applied to the signal predictions, dynamically changes throughout the upper limit calculation
        :ivar alreadyBeenThere: boolean flag that identifies when the :ivar nsignals: accidentally passes twice at two identical values
        """
        self.data = data
        self.nsignals = self.data.nsignals
        logger.debug("Signals : {}".format(self.nsignals))
        self.inputJsons = self.data.inputJsons
        self.channelsInfo = self.data.channelsInfo
        self.zeroSignalsFlag = self.data.zeroSignalsFlag
        self.nWS = self.data.nWS
        self.includeCRs = includeCRs
        self.patches = self.patchMaker()
        self.workspaces = self.wsMaker()
        self.workspaces_expected = self.wsMaker(apriori = True)
        self.cl = cl
        self.scale = 1.
        self.alreadyBeenThere = False # boolean to detect wether self.signals has returned to an older value
        self.checkPyhfVersion()
        self.welcome()

    def welcome ( self ):
        """ greet the world """
        if pyhfinfo["hasgreeted"]:
            return
        logger.info ( f"Pyhf interface, we are using v{'.'.join(pyhfinfo['ver'])}, with {pyhfinfo['backend']} v{pyhfinfo['backendver']} as backend." )
        pyhfinfo["hasgreeted"] = True

    def checkPyhfVersion ( self ):
        """ check the pyhf version, currently we need 0.6.1+ """
        if pyhfinfo["ver"] < pyhfinfo["required"]:
            logger.warning ( f"pyhf version is {'.'.join(pyhfinfo['ver'])}. SModelS currently requires pyhf>={'.'.join(pyhfinfo['required'])}. You have been warned." )

    def rescale(self, factor):
        """
        Rescales the signal predictions (self.nsignals) and processes again the patches and workspaces

        :return: updated list of patches and workspaces (self.patches, self.workspaces and self.workspaces_expected)
        """
        self.nsignals = [[sig*factor for sig in ws] for ws in self.nsignals]
        try:
            self.alreadyBeenThere = self.nsignals == self.nsignals_2
        except AttributeError:
            pass
        self.scale *= factor
        logger.debug('new signal scale : {}'.format(self.scale))
        self.patches = self.patchMaker()
        self.workspaces = self.wsMaker()
        self.workspaces_expected = self.wsMaker(apriori = True)
        try:
            self.nsignals_2 = self.nsignals_1.copy() # nsignals at previous-to-previous loop
        except AttributeError:
            pass
        self.nsignals_1 = self.nsignals.copy() # nsignals at previous loop

    def patchMaker(self):
        """
        Method that creates the list of patches to be applied to the `self.inputJsons` workspaces, one for each region given the `self.nsignals` and the informations available in `self.channelsInfo` and the content of the `self.inputJsons`
        NB: It seems we need to include the change of the "modifiers" in the patches as well

        :return: the list of patches, one for each workspace
        """
        if self.channelsInfo == None:
            return None
        nsignals = self.nsignals
        # Constructing the patches to be applied on the main workspace files
        patches = []
        for ws, info, subSig in zip(self.inputJsons, self.channelsInfo, self.nsignals):
            patch = []
            for srInfo in info['signalRegions']:
                nBins = srInfo['size']
                operator = {}
                operator["op"] = "add"
                operator["path"] = srInfo['path']
                value = {}
                value["data"] = subSig[:nBins]
                subSig = subSig[nBins:]
                value["modifiers"] = []
                value["modifiers"].append({"data": None, "type": "normfactor", "name": "mu_SIG"})
                value["modifiers"].append({"data": None, "type": "lumi", "name": "lumi"})
                value["name"] = "bsm"
                operator["value"] = value
                patch.append(operator)
            if self.includeCRs:
                logger.debug("keeping the CRs")
            else:
                for path in info['otherRegions']:
                    patch.append({'op':'remove', 'path':path})
            patches.append(patch)
        return patches

    def wsMaker(self, apriori = False):
        """
        Apply each region patch (self.patches) to his associated json (self.inputJsons) to obtain the complete workspaces
        :param apriori: - If set to `True`: Replace the observation data entries of each workspace by the corresponding sum of the expected yields
                        - Else: The observed yields put in the workspace are the ones written in the corresponfing json dictionary

        :returns: the list of patched workspaces
        """
        if self.patches == None:
            return None
        if self.nWS == 1:
            try:
                wsDict = jsonpatch.apply_patch(self.inputJsons[0], self.patches[0])
                if apriori == True:
                    # Replace the observation data entries by the corresponding sum of the expected yields
                    for obs in wsDict['observations']:
                        for ch in wsDict['channels']:
                            # Finding matching observation and bkg channel
                            if obs['name'] == ch['name']:
                                bkg = [0.]*len(obs['data'])
                                for sp in ch['samples']:
                                    if sp['name'] == 'bsm': continue
                                    for iSR in range(len(obs['data'])):
                                        # Summing over all bkg samples for each bin/SR
                                        bkg[iSR] += sp['data'][iSR]
                                # logger.debug('bkgs for channel {} :\n{}'.format(obs['name'], bkg))
                                obs['data'] = bkg
                return [pyhf.Workspace(wsDict)]
            except (pyhf.exceptions.InvalidSpecification, KeyError) as e:
                logger.error("The json file is corrupted:\n{}".format(e))
                return None
        else:
            workspaces = []
            for js, patch in zip(self.inputJsons, self.patches):
                wsDict = jsonpatch.apply_patch(js, patch)
                if apriori == True:
                    # Replace the observation data entries by the corresponding sum of the expected yields
                    for obs in wsDict['observations']:
                        for ch in wsDict['channels']:
                            # Finding matching observation and bkg channel
                            if obs['name'] == ch['name']:
                                bkg = [0.]*len(obs['data'])
                                for sp in ch['samples']:
                                    if sp['name'] == 'bsm': continue
                                    for iSR in range(len(obs['data'])):
                                        # Summing over all bkg samples for each bin/SR
                                        bkg[iSR] += sp['data'][iSR]
                                # logger.debug('bkgs for channel {} :\n{}'.format(obs['name'], bkg))
                                obs['data'] = bkg
                try:
                    ws = pyhf.Workspace(wsDict)
                except (pyhf.exceptions.InvalidSpecification, KeyError) as e:
                    logger.error("Json file number {} is corrupted:\n{}".format(self.inputJsons.index(json), e))
                    return None
                workspaces.append(ws)
            return workspaces

    def backup ( self ):
        self.bu_signal = copy.deepcopy ( self.data.nsignals )

    def restore ( self ):
        if not hasattr ( self, "bu_signal" ):
            return
        self.data.nsignals = copy.deepcopy ( self.bu_signal )
        del self.bu_signal

    def likelihood( self, mu=1., workspace_index=None, nll=False,
                    expected=False ):
        """
        Returns the value of the likelihood.
        Inspired by the `pyhf.infer.mle` module but for non-log likelihood
        :param workspace_index: supply index of workspace to use. If None,
                                choose index of best combo
        :param nll: if true, return nll, not llhd
        :param expected: if False, compute expected values, if True,
            compute a priori expected, if "posteriori" compute posteriori
            expected
        """
        mumin,mumax = -10., 20.
        if mu>mumax:
            if countWarning["llhdszero"]<1:
                logger.warning ( f"likelihoods of signal strengths mu= {mu:.1f} > {mumax} are automatically set to 0 (will suppress similar msgs)" )
            countWarning["llhdszero"]+=1
            return self.exponentiateNLL ( None, not nll )
        if mu<mumin:
            if countWarning["llhdszero"]<1:
                logger.warning ( f"likelihoods of signal strengths mu= {mu:.1f} < {mumin} are automatically set to 0 (will suppress similar msgs)" )
            countWarning["llhdszero"]+=1
            return self.exponentiateNLL ( None, not nll )
        # print ( "pyhf likelihood for", mu )
        logger.debug("Calling likelihood")
        if type(workspace_index ) == float:
            logger.error ( "workspace index is float" )
        # logger.error("expected flag needs to be heeded!!!")
        with warnings.catch_warnings():
            warnings.filterwarnings ( "ignore", "Values in x were outside bounds during a minimize step, clipping to bounds" )
            # warnings.filterwarnings ( "ignore", "", module="pyhf.exceptions" )
            if workspace_index == None:
                workspace_index = self.getBestCombinationIndex()
            if workspace_index == None:
                return None
            self.backup()
            try:
                if abs ( mu-1.) > 1e-6:
                    for i,ns in enumerate ( self.data.nsignals ):
                        for j,v in enumerate ( ns ):
                            self.data.nsignals[i][j]=v*mu
                self.__init__(self.data)
                ### allow this, for computation of l_SM
                #if self.zeroSignalsFlag[workspace_index] == True:
                #    logger.warning("Workspace number %d has zero signals" % workspace_index)
                #    return None
                workspace = self.updateWorkspace(workspace_index, expected = expected)
                # Same modifiers_settings as those used when running the 'pyhf cls' command line
                msettings = { 'normsys': {'interpcode': 'code4'}, 
                              'histosys': {'interpcode': 'code4p'}}
                model = workspace.model(modifier_settings=msettings)
                d = workspace.data(model)
                indices = []
                slices = list ( workspace.channel_slices.values() )
                for slce in slices:
                    for i in range ( slce.start, slce.stop ):
                        indices.append ( i )
                total = np.array( [ d[i]+self.data.nsignals[0][i] for i in indices ])
                if np.any ( total[total<0] ):
                    # we have negative total yields. return a llhd of 0 for that
                    self.restore()
                    return self.exponentiateNLL ( None, not nll )
                _, nllh = pyhf.infer.mle.fixed_poi_fit( 1., d,
                        model, return_fitted_val=True, maxiter=200 )
            except (pyhf.exceptions.FailedMinimization, ValueError) as e:
                logger.debug ( f"pyhf fixed_poi_fit failed for mu={mu}: {e}" )
                # lets try with different initialisation
                init, n_ = pyhf.infer.mle.fixed_poi_fit( 0., workspace.data(model),
                        model, return_fitted_val=True, maxiter=200 )
                initpars=init.tolist()
                initpars[1]=1
                for i in [ 0, 2 ]:
                    initpars[i]=1.
                try:
                    _, nllh = pyhf.infer.mle.fixed_poi_fit( 1., workspace.data(model),
                        model, return_fitted_val=True, init_pars = initpars, maxiter=200 )
                except (pyhf.exceptions.FailedMinimization, ValueError) as e:
                    logger.info ( f"pyhf fixed_poi_fit failed twice for mu={mu}: {e}" )
                
                    self.restore()
                    return self.exponentiateNLL ( None, not nll )
            except:
                self.restore()
                return self.exponentiateNLL ( None, not nll )

            # print ( "likelihood best fit", _ )
            ret = nllh.tolist()
            try:
                ret = float(ret)
            except:
                ret = float(ret[0])
            self.data.cached_likelihoods[workspace_index]=ret #THIS CAN STAY BC IT MAY BE NEEDED ELSEWHERE IN THE CODE
            ret = self.exponentiateNLL ( ret, not nll )
            # print ( "now leaving the fit mu=", mu, "llhd", ret, "nsig was", self.data.nsignals )
            self.restore()
            return ret

    def getBestCombinationIndex( self ):
        """ find the index of the best expected combination """
        if self.nWS == 1:
            return 0
        logger.debug("Finding best expected combination among %d workspace(s)" % self.nWS)
        ulMin = float('+inf')
        i_best = None
        for i_ws in range(self.nWS):
            if self.zeroSignalsFlag[i_ws] == True:
                logger.debug("Workspace number %d has zero signals" % i_ws)
                continue
            else:
                ul = self.ulSigma(expected=True, workspace_index=i_ws)
            if ul == None:
                continue
            if ul < ulMin:
                ulMin = ul
                i_best = i_ws
        return i_best

    def chi2(self, workspace_index=None):
        """
        Returns the chi square
        """
        return 2 * ( self.lmax ( workspace_index, nll=True ) - self.likelihood ( workspace_index, nll=True ) )

    def exponentiateNLL ( self, twice_nll, doIt ):
        """ if doIt, then compute likelihood from nll,
            else return nll """
        if twice_nll == None:
            if doIt:
                return 0.
            return 9000.
        if doIt:
            return np.exp(-twice_nll/2.)
        return twice_nll / 2.

    def lmax( self, workspace_index=None, nll=False,
              expected=False, allowNegativeSignals = False ):
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
            warnings.filterwarnings ( "ignore", "Values in x were outside bounds during a minimize step, clipping to bounds" )

            self.__init__(self.data)
            if workspace_index == None:
                workspace_index = self.getBestCombinationIndex()
            if workspace_index != None:
                if self.zeroSignalsFlag[workspace_index] == True:
                    logger.warning("Workspace number %d has zero signals" % workspace_index)
                    return None
                else:
                    workspace = self.updateWorkspace(workspace_index, expected = expected)
            else:
                return None
            # Same modifiers_settings as those used when running the 'pyhf cls' command line
            msettings = {'normsys': {'interpcode': 'code4'}, 'histosys': {'interpcode': 'code4p'}}
            model = workspace.model(modifier_settings=msettings)
            try:
                bounds = model.config.suggested_bounds()
                if allowNegativeSignals:
                    bounds[model.config.poi_index] = (-5., 10. )
                muhat, maxNllh = pyhf.infer.mle.fit(workspace.data(model), model, return_fitted_val=True, par_bounds = bounds )
                muhat = muhat[model.config.poi_index]*self.scale

            except (pyhf.exceptions.FailedMinimization, ValueError) as e:
                logger.error ( f"pyhf mle.fit failed {e}" )
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
            self.data.cached_lmaxes [ workspace_index ] = ret
            ret = self.exponentiateNLL ( ret, not nll )
            return ret

    def updateWorkspace(self, workspace_index = None, expected = False):
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

    # Trying a new method for upper limit computation :
    # re-scaling the signal predictions so that mu falls in [0, 10] instead of
    # looking for mu bounds
    # Usage of the index allows for rescaling
    def ulSigma (self, expected=False, workspace_index=None):
        """
        Compute the upper limit on the signal strength modifier with:
            - by default, the combination of the workspaces contained into self.workspaces
            - if workspace_index is specified, self.workspace[workspace_index]
              (useful for computation of the best upper limit)

        :param expected:  - if set to `True`: uses expected SM backgrounds as signals
                          - else: uses `self.nsignals`
        :param workspace_index: - if different from `None`: index of the workspace to use
                                  for upper limit
                                - else: choose best combo
        :return: the upper limit at `self.cl` level (0.95 by default)
        """
        if workspace_index in self.data.cachedULs[expected]:
            return self.data.cachedULs[expected][workspace_index]
        with warnings.catch_warnings():
            warnings.filterwarnings ( "ignore", "Values in x were outside bounds during a minimize step, clipping to bounds" )
            startUL = time.time()
            logger.debug("Calling ulSigma")
            if self.data.errorFlag or self.workspaces == None:
                # For now, this flag can only be turned on by PyhfData.checkConsistency
                return None

            if all( [self.zeroSignalsFlag[workspace_index] for workspace_index in range(self.nWS)] ) == True:
                logger.debug("There is (are) %d workspace(s) and no signal(s) was (were) found" % self.nWS)
                return None
            if workspace_index == None:
                workspace_index = self.getBestCombinationIndex()
            if workspace_index == None:
                logger.debug("Best combination index not found")
                return None

            def root_func(mu ):
                # If expected == False, use unmodified (but patched) workspace
                # If expected == True, use modified workspace where observations = sum(bkg) (and patched)
                # If expected == posteriori, use unmodified (but patched) workspace
                workspace = self.updateWorkspace(workspace_index, expected = expected)
                # Same modifiers_settings as those use when running the 'pyhf cls' command line
                msettings = {'normsys': {'interpcode': 'code4'}, 'histosys': {'interpcode': 'code4p'}}
                model = workspace.model(modifier_settings=msettings)
                bounds = model.config.suggested_bounds()
                bounds[model.config.poi_index] = (0,10)
                start = time.time()
                args = {}
                args["return_expected"] = ( expected == "posteriori" )
                args["par_bounds"] = bounds
                # args["maxiter"]=100000
                pver = float ( pyhf.__version__[:3] )
                stat = "qtilde"
                if pver < 0.6:
                    args["qtilde"]=True
                else:
                    args["test_stat"]=stat
                with np.testing.suppress_warnings() as sup:
                    if pyhfinfo["backend"] == "numpy":
                        sup.filter ( RuntimeWarning, r'invalid value encountered in log')
                    # print ("expected", expected, "return_expected", args["return_expected"], "mu", mu, "\nworkspace.data(model) :", workspace.data(model, include_auxdata = False), "\nworkspace.observations :", workspace.observations, "\nobs[data] :", workspace['observations'])
                    try:
                        result = pyhf.infer.hypotest(mu, workspace.data(model), model, **args )
                    except Exception as e:
                        logger.info ( f"when testing hypothesis {mu}, caught exception: {e}" )
                        result = float("nan")
                        if expected == "posteriori":
                            result = [ float("nan") ]*2
                end = time.time()
                logger.debug("Hypotest elapsed time : %1.4f secs" % (end - start))
                logger.debug(f"result for {mu} {result}" )
                if expected == "posteriori":
                    logger.debug('computing a-posteriori expected limit')
                    logger.debug("expected = {}, mu = {}, result = {}".format(expected, mu, result))
                    try:
                        CLs = float(result[1].tolist())
                    except TypeError:
                        CLs = float(result[1][0])
                else:
                    logger.debug("expected = {}, mu = {}, result = {}".format(expected, mu, result))
                    CLs = float(result)
                # logger.debug("Call of root_func(%f) -> %f" % (mu, 1.0 - CLs))
                return 1.0 - self.cl - CLs
            # Rescaling singals so that mu is in [0, 10]
            factor = 3.
            wereBothLarge = False
            wereBothTiny = False
            nattempts = 0
            nNan = 0
            lo_mu, med_mu, hi_mu = .2, 1., 5.
            # print ( "starting with expected", expected )
            while "mu is not in [lo_mu,hi_mu]":
                nattempts += 1
                if nNan > 5:
                    #logger.warning("encountered NaN 5 times while trying to determine the bounds for brent bracketing. now trying with q instead of qtilde test statistic")
                    return None
                    # nattempts = 0
                if nattempts > 10:
                    logger.warning ( "tried 10 times to determine the bounds for brent bracketing. we abort now." )
                    return None
                # Computing CL(1) - 0.95 and CL(10) - 0.95 once and for all
                rt1 = root_func(lo_mu )
                # rt5 = root_func(med_mu)
                rt10 = root_func(hi_mu )
                # print ( "we are at",lo_mu,med_mu,hi_mu,"values at", rt1, rt5, rt10, "scale at", self.scale,"factor at", factor )
                if rt1 < 0. and 0. < rt10: # Here's the real while condition
                    break
                if self.alreadyBeenThere:
                    factor = 1 + (factor-1)/2
                    logger.debug("Diminishing rescaling factor")
                if np.isnan(rt1):
                    rt5 = root_func ( med_mu )
                    if rt5 < 0. and rt10 > 0.:
                        lo_mu = med_mu
                        med_mu = np.sqrt (lo_mu * hi_mu)
                        continue
                    if rt10 < 0.: ## also try to increase hi_mu
                        hi_mu = hi_mu + ( 10. - hi_mu ) * .5
                        med_mu = np.sqrt (lo_mu * hi_mu)
                    nNan += 1
                    self.rescale(factor)
                    continue
                if np.isnan(rt10):
                    rt5 = root_func ( med_mu )
                    if rt5 > 0. and rt1 < 0.:
                        hi_mu = med_mu
                        med_mu = np.sqrt (lo_mu * hi_mu)
                        continue
                    if rt1 > 0.: ## also try to decrease lo_mu
                        lo_mu = lo_mu * .5
                        med_mu = np.sqrt (lo_mu * hi_mu)
                    nNan += 1
                    self.rescale(1/factor)
                    continue
                # Analyzing previous values of wereBoth***
                if rt10 < 0 and rt1 < 0 and wereBothLarge:
                    factor = 1 + (factor-1)/2
                    logger.debug("Diminishing rescaling factor")
                if rt10 > 0 and rt1 > 0 and wereBothTiny:
                    factor = 1 + (factor-1)/2
                    logger.debug("Diminishing rescaling factor")
                # Preparing next values of wereBoth***
                wereBothTiny = rt10 < 0 and rt1 < 0
                wereBothLarge = rt10 > 0 and rt1 > 0
                # Main rescaling code
                if rt10 < 0.:
                    self.rescale(factor)
                    continue
                if rt1 > 0.:
                    self.rescale(1/factor)
                    continue
            # Finding the root (Brent bracketing part)
            logger.debug("Final scale : %f" % self.scale)
            logger.debug("Starting brent bracketing")
            ul = optimize.brentq(root_func, lo_mu, hi_mu, rtol=1e-3, xtol=1e-3)
            endUL = time.time()
            logger.debug("ulSigma elpased time : %1.4f secs" % (endUL - startUL))
            self.data.cachedULs[expected][workspace_index]=ul*self.scale
            return ul*self.scale # self.scale has been updated within self.rescale() method

if __name__ == "__main__":
    C = [ 18774.2, -2866.97, -5807.3, -4460.52, -2777.25, -1572.97, -846.653, -442.531,
       -2866.97, 496.273, 900.195, 667.591, 403.92, 222.614, 116.779, 59.5958,
       -5807.3, 900.195, 1799.56, 1376.77, 854.448, 482.435, 258.92, 134.975,
       -4460.52, 667.591, 1376.77, 1063.03, 664.527, 377.714, 203.967, 106.926,
       -2777.25, 403.92, 854.448, 664.527, 417.837, 238.76, 129.55, 68.2075,
       -1572.97, 222.614, 482.435, 377.714, 238.76, 137.151, 74.7665, 39.5247,
       -846.653, 116.779, 258.92, 203.967, 129.55, 74.7665, 40.9423, 21.7285,
       -442.531, 59.5958, 134.975, 106.926, 68.2075, 39.5247, 21.7285, 11.5732]
    nsignal = [ x/100. for x in [47,29.4,21.1,14.3,9.4,7.1,4.7,4.3] ]
    m=Data( observed=[1964,877,354,182,82,36,15,11],
              backgrounds=[2006.4,836.4,350.,147.1,62.0,26.2,11.1,4.7],
              covariance= C,
#              third_moment = [ 0.1, 0.02, 0.1, 0.1, 0.003, 0.0001, 0.0002, 0.0005 ],
              third_moment = [ 0. ] * 8,
              nsignal = nsignal,
              name="ATLAS-SUSY-2018-31 model" )
    ulComp = PyhfUpperLimitComputer(cl=.95)
    #uls = ulComp.ulSigma ( Data ( 15,17.5,3.2,0.00454755 ) )
    #print ( "uls=", uls )
    ul_old = 131.828*sum(nsignal) #With respect to the older refernece value one must normalize the xsec
    print ( "old ul=", ul_old )
    ul = ulComp.ulSigma ( m )
    print ( "ul (marginalized)", ul )
    ul = ulComp.ulSigma ( m, marginalize=False )
    print ( "ul (profiled)", ul )
