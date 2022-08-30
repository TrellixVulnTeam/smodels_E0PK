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
        oxfile = "./database_onnx/13TeV/ATLAS/ATLAS-SUSY-2018-04-eff/model.onnx"
        yields = [[ 15., 15. ]]
        import onnxruntime
        from smodels.tools.onnxInterface import OnnxData, OnnxUpperLimitComputer
        oxsession = onnxruntime.InferenceSession( oxfile )
        oxdata = OnnxData ( yields, oxsession, oxfile )
        computer = OnnxUpperLimitComputer ( oxdata )
        llhd = computer.likelihood ( mu = 1. )
        self.assertAlmostEqual ( llhd, 1.306750516840411e-23, 3 )
        nll = computer.likelihood ( mu = 1., nll = True )
        self.assertAlmostEqual ( nll, 105.38382720947266, 3 )

    def testOnnx ( self ):
        from smodels.experiment.databaseObj import Database
        from smodels.theory.model import Model
        from smodels.share.models.SMparticles import SMList
        from smodels.share.models.mssm import BSMList
        from smodels.theory import decomposer
        from smodels.theory.theoryPrediction import theoryPredictionsFor

        db = Database ( "./database_onnx/" )
        expRes = db.getExpResults ( analysisIDs = [ "ATLAS-SUSY-2018-04" ],
                                dataTypes = "efficiencyMap" )
        filename = "./testFiles/slha/TStauStau.slha"
        model = Model(BSMList, SMList)
        model.updateParticles(filename)
        smstoplist = decomposer.decompose(model, sigmacut=0)
        prediction = theoryPredictionsFor(expRes[0], smstoplist)[0]
        pred_signal_strength = prediction.xsection.value
        prediction.computeStatistics()
        llhd = prediction.likelihood()
        # for pyhf i get 3.5292233396179906e-44
        self.assertAlmostEqual ( llhd, 0.0074667034648726646, 3 )
        nll = prediction.likelihood( nll = True )
        self.assertAlmostEqual ( nll, 4.897301680470236, 3 )

if __name__ == "__main__":
    unittest.main()
