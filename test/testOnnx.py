#!/usr/bin/env python3

"""
.. module:: testOnny
   :synopsis: Test the onnx-based likelihoods

.. moduleauthor:: Wolfgang Waltenberger <wolfgang.waltenberger@gmail.com>

"""

import sys
sys.path.insert(0,"../")
import unittest

class OnnxTest(unittest.TestCase):
    def mestDirectly ( self ):
        """ compute the likelihood directly, not via txnames """
        oxfile = "./database_onnx/13TeV/ATLAS/ATLAS-SUSY-2018-04-eff/model.onnx"
        yields = [ 15., 15. ]
        import onnxruntime
        from smodels.tools.onnxInterface import OnnxData, OnnxUpperLimitComputer
        oxdata = OnnxData ( yields, oxfile )
        computer = OnnxUpperLimitComputer ( oxdata )
        llhd = computer.likelihood ( mu = 1. )
        self.assertAlmostEqual ( llhd / 1.306750516840411e-23, 1., 3 )
        nll = computer.likelihood ( mu = 1., nll = True )
        self.assertAlmostEqual ( nll, 105.38382720947266, 3 )

    def testDirectly1831 ( self ):
        """ compute the likelihood directly, not via txnames """
        oxfile = "./database_onnx/13TeV/ATLAS/ATLAS-SUSY-2018-31-eff/modelA.onnx"
        yields = [ 15., 15., 15. ]
        import onnxruntime
        from smodels.tools.onnxInterface import OnnxData, OnnxUpperLimitComputer
        oxdata = OnnxData ( yields, oxfile )
        computer = OnnxUpperLimitComputer ( oxdata )
        llhd = computer.likelihood ( mu = 1. )
        self.assertAlmostEqual ( llhd / 8.121422344963753e-30, 1.,  3 )
        nll = computer.likelihood ( mu = 1., nll = True )
        self.assertAlmostEqual ( nll, 66.98304748535156, 3 )

    def mest201804 ( self ):
        """ test the model of ATLAS-SUSY-2018-04 """
        from smodels.experiment.databaseObj import Database
        from smodels.theory.model import Model
        from smodels.share.models.SMparticles import SMList
        from smodels.share.models.mssm import BSMList
        from smodels.theory import decomposer
        from smodels.theory.theoryPrediction import theoryPredictionsFor

        db = Database ( "./database_onnx/", force_load = None )
        expRes = db.getExpResults ( analysisIDs = [ "ATLAS-SUSY-2018-04" ],
                                    dataTypes = "efficiencyMap" )
        filename = "./testFiles/slha_onnx/TStauStau.slha"
        model = Model(BSMList, SMList)
        model.updateParticles(filename)
        smstoplist = decomposer.decompose(model, sigmacut=0)
        predictions = theoryPredictionsFor(expRes[0], smstoplist, 
                         useBestDataset = False, combinedResults = True)
        # print ( "predictions", predictions )
        prediction = predictions[0]
        pred_signal_strength = prediction.xsection.value
        prediction.computeStatistics()
        nll = prediction.likelihood( nll = True )
        # for pyhf that would be 100.0526662621055
        self.assertAlmostEqual ( nll, 100.06201171875, 3 )
        llhd = prediction.likelihood()
        # for pyhf i get exp(-nll) = 3.5292233396179906e-44
        self.assertAlmostEqual ( llhd / 3.4963947738083063e-44, 1., 3 )

    def mest201831 ( self ):
        """ test the model of ATLAS-SUSY-2018-31 """
        from smodels.experiment.databaseObj import Database
        from smodels.theory.model import Model
        from smodels.share.models.SMparticles import SMList
        from smodels.share.models.mssm import BSMList
        from smodels.theory import decomposer
        from smodels.theory.theoryPrediction import theoryPredictionsFor

        db = Database ( "./database_onnx/", force_load = "txt" )
        expRes = db.getExpResults ( analysisIDs = [ "ATLAS-SUSY-2018-31" ],
                                    dataTypes = "efficiencyMap" )
        filename = "./testFiles/slha_onnx/T6bbHH.slha"
        model = Model(BSMList, SMList)
        model.updateParticles(filename)
        smstoplist = decomposer.decompose(model, sigmacut=0)
        predictions = theoryPredictionsFor(expRes[0], smstoplist, 
                         useBestDataset = False, combinedResults = True)
        # print ( "predictions", predictions )
        prediction = predictions[0]
        pred_signal_strength = prediction.xsection.value
        prediction.computeStatistics()
        nll = prediction.likelihood( nll = True )
        # for pyhf that would be 24.807966704239153
        self.assertAlmostEqual ( nll, 24.807966704239153, 3 )
        llhd = prediction.likelihood()
        # for pyhf i get exp(-nll) = 1.6828172419917045e-11
        self.assertAlmostEqual ( llhd / 1.6828172419917045e-11, 1., 3 )

if __name__ == "__main__":
    unittest.main()
