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
    def testOnnx ( self ):
        from smodels.experiment.databaseObj import Database
        db = Database ( "./database_onnx/" )
        print ( db )

if __name__ == "__main__":
    unittest.main()
