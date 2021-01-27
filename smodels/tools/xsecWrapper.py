#!/usr/bin/env python3

"""
.. module:: xsecWrapper
   :synopsis: Wrapper for interfaceing the xsec code written by Andy Buckley et al [*] 
              with our xsecComputer
              [*] https://epjc.epj.org/articles/epjc/abs/2020/12/10052_2020_Article_8635/10052_2020_Article_8635.html

.. moduleauthor:: Wolfgang Waltenberger <wolfgang.waltenberger@gmail.com>

"""

from smodels.tools.wrapperBase import WrapperBase
from smodels.tools import wrapperBase
from smodels.tools.physicsUnits import fb, TeV
from smodels.tools.smodelsLogging import logger
from smodels.theory import crossSection
from smodels.theory.crossSection import NLO, XSection
from smodels import installation
from smodels.theory.exceptions import SModelSTheoryError as SModelSError
import sys, os

try:
    import xsec
except ImportError as e:
    logger.error ( "python library 'xsec' not installed. Wanna try e.g. pip install xsec?" )
    sys.exit()

class XsecWrapper(WrapperBase):
    """
    This is the wrapper around the xsec tool by Andy Buckley et al.
    strong production at NLL for MSSM

    """
    def __init__( self ):
        """ initialize the wrapper
        """
        WrapperBase.__init__(self)
        self.name = "Xsec"
        self.executablePath = os.path.abspath ( "./xsecWrapper.py" )
        ## initialize xsec
        
        xsec.init(data_dir="gp_dir",use_cache=True,cache_dir='cache_dir')
        xsec.set_energy(13000)
        self.set_processes( "debug" )
        self.download()


    def set_processes ( self, mode : str ):
        """ set the processes. mode can be either debug, most, full """
        processes = [ 
           (-2000006, 2000006), (-2000005, 2000005), (-2000004, 1000001),
           (-2000004, 1000002), (-2000004, 1000003), (-2000004, 1000004),
           (-2000004, 2000001), (-2000004, 2000002), (-2000004, 2000003),
           (-2000004, 2000004), (-2000003, 1000001), (-2000003, 1000002),
           (-2000003, 1000003), (-2000003, 2000001),
           # (-2000003, 1000004), (-2000002, 1000003), (-2000002, 1000004)
           # (-2000001, 1000002), (-2000001, 1000003), (-2000001, 1000004)
           (-2000003, 2000002), (-2000003, 2000003), (-2000002, 1000001),
           (-2000002, 1000002),
           (-2000002, 2000001), (-2000002, 2000002), (-2000001, 1000001),
           (-2000001, 2000001), (-1000006, 1000006), (-1000005, 1000005),
           (-1000004, 1000001), (-1000004, 1000002), (-1000004, 1000003),
           (-1000004, 1000004), (-1000003, 1000001), (-1000003, 1000002),
           (-1000003, 1000003), (-1000002, 1000001), (-1000002, 1000002),
           (-1000001, 1000001), (1000001, 1000001), (1000001, 1000002),
           (1000001, 1000003), (1000001, 1000004), (1000001, 1000021),
           (1000001, 2000001), (1000001, 2000002), (1000001, 2000003),
           (1000001, 2000004), (1000002, 1000002), (1000002, 1000003),
           (1000002, 1000004), (1000002, 1000021), (1000002, 2000001),
           (1000002, 2000002), (1000002, 2000003), (1000002, 2000004),
           (1000003, 1000003), (1000003, 1000004), (1000003, 1000021),
           (1000003, 2000001), (1000003, 2000002), (1000003, 2000003),
           (1000003, 2000004), (1000004, 1000004), (1000004, 1000021),
           (1000004, 2000001), (1000004, 2000002), (1000004, 2000003),
           (1000004, 2000004), (1000021, 1000021), (1000021, 2000001),
           (1000021, 2000002), (1000021, 2000003), (1000021, 2000004),
           (2000001, 2000001), (2000001, 2000002), (2000001, 2000003),
           (2000001, 2000004), (2000002, 2000002), (2000002, 2000003),
           (2000002, 2000004), (2000003, 2000003), (2000003, 2000004),
           (2000004, 2000004)]
        if mode == "most":
            processes = [ 
               (-2000006, 2000006), (-2000005, 2000005), (-2000004, 2000004), 
               (-2000003, 2000003), (-2000002, 2000002), (-2000001, 2000001), 
               (-1000006, 1000006), (-1000005, 1000005), (-1000004, 1000004), 
               (-1000003, 1000003), (-1000002, 1000002), (-1000001, 1000001), 
               (1000001, 1000001), (1000001, 1000002), (1000001, 1000021), 
               (1000002, 1000002), (1000002, 1000021),
               (1000003, 1000003), (1000003, 1000004),
               (1000004, 1000004), (1000004, 1000021),
               (1000004, 2000003),
               (1000021, 1000021), (1000021, 2000001),
               (1000021, 2000002), (1000021, 2000003), (1000021, 2000004),
               (2000001, 2000001), (2000001, 2000002), 
               (2000002, 2000002), (2000003, 2000003), (2000003, 2000004),
               (2000004, 2000004)]
        if mode == "debug":
            processes = [ ( 1000001, 1000001 ), ( 1000021, 1000021 ) ]
        self.processes = processes
        xsec.load_processes ( processes )

    def download ( self ):
        """ download the data files """
        
        if os.path.isdir('./gp_dir')==False:
            cmd = "xsec-download-gprocs -g gp_dir -t all"
            print("Xsec trained models not found. Downloading them...")
            import subprocess
            subprocess.getoutput ( cmd )
            print("Done")

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
        Regress for the given slhafile

        :param slhafile: input SLHA file
        :returns: List of cross sections

        """
        xseclist = crossSection.XSectionList()
        xsec.import_slha ( slhafile )
      
         
        try:
            ret = xsec.eval_xsection( verbose=0, check_consistency = True )
            centrals = ret[0]*10e-3 ## central values, from fb to pb
            lower_uncertainty_list=ret[1].tolist()*10e-3
            upper_uncertainty_list=ret[2].tolist()*10e-3
            for pids,central in zip(self.processes,centrals):
                mxsec = XSection()
                mxsec.info.sqrts = 13
                mxsec.info.order = NLO
                mxsec.info.label = "13 TeV (NLO)"
                mxsec.pid = pids
                mxsec.value = central * pb
                xseclist.add ( mxsec )
    
        except:
            #If inputs inconsistent with trained parameter space, return None.
            xseclist,lower_uncertainty_list,upper_uncertainty_list=None
         
         
        return xseclist,lower_uncertainty_list,upper_uncertainty_list
        
if __name__ == "__main__":
    import os
    tool = XsecWrapper()
    print("[xsecWrapper] installed: " + str(tool.installDirectory()))
    print("[xsecWrapper] check: " + wrapperBase.ok(tool.checkInstallation()))
    slhafile = os.environ["HOME"]+"/git/smodels/inputFiles/slha/simplyGluino.slha"
    slhapath = os.path.join ( installation.installDirectory(), slhafile )
    xsec = tool.run( slhapath )
    print ( xsec[0] )
    isok = abs ( xsec[0].value.asNumber ( pb ) - 4.19e-06  ) < 1e-5
    print("[xsecWrapper] run: " + wrapperBase.ok (isok) )
