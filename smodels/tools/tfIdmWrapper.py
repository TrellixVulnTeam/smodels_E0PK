#!/usr/bin/env python3

"""
.. module:: tfIdmWrapper
   :synopsis: Wrapper for Humberto's tensorflow based xsec regressor for the IDM model.

.. moduleauthor:: Humberto Reyes <humberto.reyes-gonzalez@lpsc.in2p3.fr>

"""

from smodels.tools.wrapperBase import WrapperBase
from smodels.tools import wrapperBase
from smodels.tools.physicsUnits import pb, TeV
from smodels.tools.smodelsLogging import logger
from smodels.theory import crossSection
from smodels.theory.crossSection import LO
from smodels import installation
from smodels.theory.exceptions import SModelSTheoryError as SModelSError

class TfIdmWrapper(WrapperBase):
    """
    This is the wrapper around the tensorflow-based xsec regressor 
    for the IDM model

    """
    def __init__(self ):
        """
        """
        WrapperBase.__init__(self)
        self.name = "tfidm"

    def checkFileExists(self, inputFile):
        """
        Check if file exists, raise an IOError if it does not.

        :returns: absolute file name if file exists.

        """
        nFile = self.absPath(inputFile)
        if not os.path.exists(nFile):
            raise IOError("file %s does not exist" % nFile)
        return nFile

    def __str__(self):
        """
        Describe the current status

        """
        ret = "tool: %s\n" % (self.name)
        ret += "executable: %s\n" % (self.executablePath)
        ret += "temp dir: %s\n" % self.tempdir
        ret += "nevents: %d\n" % self.nevents
        return ret

    def run( self, slhafile ):
        """
        Execute pythia_lhe with n events, at sqrt(s)=sqrts.

        :param slhafile: input SLHA file
        :returns: List of cross sections

        """
        xsecs = crossSection.XSectionList()
        return xsecs


if __name__ == "__main__":
    import os
    tool = TfIdmWrapper()
    print("[tfIdmWrapper] installed: " + str(tool.installDirectory()))
    print("[tfIdmWrapper] check: " + wrapperBase.ok(tool.checkInstallation()))
    print("[tfIdmWrapper] seconds per event: %d" % tool.secondsPerEvent)
    slhafile = "inputFiles/slha/simplyGluino.slha"
    slhapath = os.path.join ( installation.installDirectory(), slhafile )
    xsec = tool.run(slhapath, unlink=True )
    isok = abs ( xsec[0].value.asNumber ( pb ) - 2.80E-01  ) < 1e-5
    print("[tfIdmWrapper] run: " + wrapperBase.ok (isok) )
