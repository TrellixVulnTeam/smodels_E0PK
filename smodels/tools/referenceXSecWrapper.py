#!/usr/bin/env python3

"""
.. module:: referenceXSecWrapper
   :synopsis: code to retrieve cross sections from LHC Xsec group's
              references, published at
              https://twiki.cern.ch/twiki/bin/view/LHCPhysics/SUSYCrossSections

.. moduleauthor:: Wolfgang Waltenberger <wolfgang.waltenberger@gmail.com>

"""

from __future__ import print_function
from smodels.tools.wrapperBase import WrapperBase
from smodels.tools import wrapperBase
from smodels.tools.smodelsLogging import logger, setLogLevel
from smodels.tools.physicsUnits import fb, pb, TeV, mb
from smodels import installation
from smodels.theory.exceptions import SModelSTheoryError as SModelSError
import os, sys, io, shutil, pyslha

class ReferenceXSecWrapper:
    """
    An instance of this class represents the installation of pythia8.
    """

    def __init__(self ):
        """ we dont derive from WrapperBase because it is not really a wrapper.
            It's something simpler. """
        self.name = "refxsec"
        self.sqrtses = [ 13 ]
        self.shareDir = os.path.join ( installation.installDirectory(), \
                                       "smodels", "share", "refxsecs" )

    def installDirectory ( self ):
        """ not really needed. """
        return None

    def pathOfExecutable( self ):
        """ dummy implementation """
        return "N/A"

    def checkInstallation ( self ):
        """ not really needed. """
        return True

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
        return ret

    def run( self, slhafile ): ## , lhefile=None, unlink=True ):
        """
        Retrieve cross sections

        :param slhafile: SLHA file
        :returns: List of cross sections to be added
        """
        channels = self.findOpenChannels ( slhafile )
        xsecs = []
        for channel in channels:
            # obtain xsecs for all masses, but for the given channel
            for sqrts in self.sqrtses:
                xsecall,order,comment = self.getXSecsFor ( channel[0], channel[1], sqrts, "" )
                ## interpolate for the mass that we are looking for
                masses = ( channel[2], channel[3] )
                xsec = self.interpolate ( masses[0], xsecall )
                xsecs.append ( xsec )
        return xsecs

    def findOpenChannels ( self, slhafile ):
        slhadata = pyslha.readSLHAFile ( slhafile )
        masses = slhadata.blocks["MASS"]
        # print ( "findOpenChannels" )
        channels = []
        # productions of same-sign-pid pairs when the particle is within reach
        samesignmodes = ( 1000021, )
        # production of opposite-sign-pid pairs when the particle is within reach
        oppositesignmodes = ( 1000006, 1000005, 1000011, 1000013, 1000015 )

        # associate production
        associateproduction = ( ( 1000001, 1000021 ))
        ## production modes to add that needs to different particles
        ## to be unfrozen
        associateproductions = { ( 1000001, 1000021 ): ( 1000001, 1000021 ) }

        for pid,mass in masses.items():
            if pid < 999999:
                continue
            if mass > 5000:
                continue

            if pid in samesignmodes:
                channels.append ( (pid,pid, mass, mass ) )
            if pid in oppositesignmodes:
                channels.append ( (-pid,pid, mass, mass ) )
            for jpid, jmass in masses.items():
                if pid == jpid:
                    continue
                if (pid,jpid) in associateproductions:
                    channels.append ( (pid,jpid, mass, jmass ) )

        return channels

    def interpolate ( self, mass, xsecs ):
        """ interpolate between masses """
        if mass in xsecs:
            return xsecs[mass]
        if mass < min(xsecs.keys()):
            logger.info ( "mass %d<%d too low to interpolate, leave it as is."  % ( mass, min(xsecs.keys() ) ) )
            return None
        if mass > max(xsecs.keys()):
            logger.info ( "mass %d>%d too high to interpolate, leave it as is." % ( mass, max(xsecs.keys() ) ) )
            return None
        from scipy.interpolate import interp1d
        return interp1d ( list(xsecs.keys()), list(xsecs.values()) )( mass )

    def getXSecsFrom ( self, path, pb = True, columns={"mass":0,"xsec":1 } ):
        """ retrieve xsecs from filename
        :param pb: xsecs given in pb
        :param indices: the indices of the columns in the table, for mass and xsec
        """
        ret = {}
        if not os.path.exists ( path ):
            logger.info ( "could not find %s" % path )
            return ret
        logger.info ( "getting xsecs from %s" % path )
        f = open ( path, "rt" )
        lines=f.readlines()
        f.close()
        for line in lines:
            if line.find("#")>-1:
                line = line[:line.find("#")]
            if "mass [GeV]" in line: ## skip
                continue
            tokens = line.split ()
            if len(tokens)<2:
                continue
            mass = float(tokens[ columns["mass"] ])
            xsec = float(tokens[ columns["xsec"] ].replace("GeV","") )
            if not pb:
                xsec = xsec / 1000.
            ret[ mass ] = xsec
        return ret

    def getXSecsFor ( self, pid1, pid2, sqrts, ewk ):
        """ get the xsec dictionary for pid1/pid2, sqrts
        :param ewk: specify the ewkino process (hino, or wino)
        """
        filename=None
        order = 0
        pb = True
        columns = { "mass": 0, "xsec": 1 }
        isEWK=False
        comment="refxsec [pb]"
        if pid1 in [ 1000021 ] and pid2 == pid1:
            filename = "xsecgluino%d.txt" % sqrts
            columns["xsec"]=2
            isEWK=False
            order = 2 # 4
        if pid1 in [ -1000024 ] and pid2 in [ 1000023 ]:
            filename = "xsecN2C1m%d.txt" % sqrts
            order = 2
            isEWK=True
            pb = False
        if pid1 in [ 1000023 ] and pid2 in [ 1000024 ]:
            filename = "xsecN2C1p%d.txt" % sqrts
            order = 2
            pb = False
            isEWK=True
        if pid1 in [ 1000023 ] and pid2 in [ 1000023 ]:
            filename = "xsecN2N1p%d.txt" % sqrts
            order = 2
            pb = False
            isEWK=True
        if pid1 in [ 1000024 ] and pid2 in [ 1000025 ]:
            filename = "xsecN2C1p%d.txt" % sqrts
            order = 2
            pb = False
            isEWK=True
        if pid1 in [ -1000024 ] and pid2 in [ 1000025 ]:
            filename = "xsecN2C1m%d.txt" % sqrts
            order = 2
            isEWK=True
            pb = False
        if pid1 in [ -1000005, -1000006, -2000006 ] and pid2 == -pid1:
            ## left handed slep- slep+ production.
            filename = "xsecstop%d.txt" % sqrts
            order = 2 #3
            columns["xsec"]=2
            pb = True
        if pid1 in [ -1000024 ] and pid2 == -pid1:
            ## left handed slep- slep+ production.
            filename = "xsecC1C1%d.txt" % sqrts
            order = 2 #3
            pb = False
        if pid1 in [ -1000011, -1000013, -1000015 ] and pid2 == -pid1:
            ## left handed slep- slep+ production.
            filename = "xsecslepLslepL%d.txt" % sqrts
            order = 2 #3
        if pid1 in [ -2000011, -2000013, -2000015 ] and pid2 == -pid1:
            filename = "xsecslepRslepR%d.txt" % sqrts
            order = 2 # 3
        if filename == None:
            logger.info ( "could not identify filename for xsecs" )
            logger.info ( "seems like we dont have ref xsecs for the pids %d/%d?" % ( pid1, pid2 ) )
            sys.exit()
        if ewk == "hino":
            filename = filename.replace(".txt","hino.txt" )
        if isEWK:
            comment = " (%s)" % ewk
        path = os.path.join ( self.shareDir, filename )
        if not os.path.exists ( path ):
            logger.info ( "%s missing" % path )
            sys.exit()
        xsecs = self.getXSecsFrom ( path, pb, columns )
        return xsecs,order,comment

if __name__ == "__main__":
    setLogLevel ( "debug" )
    tool = ReferenceXSecWrapper()
    logger.info("installed: " + str(tool.installDirectory()))
    logger.info("check: " + wrapperBase.ok(tool.checkInstallation()))
    slhafile = "inputFiles/slha/simplyGluino.slha"
    # slhafile = "./test.slha"
    slhapath = os.path.join ( installation.installDirectory(), slhafile )
    logger.info ( "slhafile: " + slhapath )
    output = tool.run(slhapath )
    logger.info ( "done: %s" % output )
