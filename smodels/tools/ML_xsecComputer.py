#!/usr/bin/env python3

"""
.. module:: ML_xsecComputer
   :synopsis: Computation of reference ("theory") production cross sections using
   trained Machine Learning algorithms

.. moduleauthor:: Guillaume Chalons <guillaume.chalons@orange.fr>

"""
from __future__ import print_function
from smodels import installation
from smodels.tools import toolBox, runtime
from smodels.tools.physicsUnits import pb, TeV, GeV
from smodels.theory import crossSection
from smodels.theory.crossSection import LO, NLO, NLL
from smodels.tools.smodelsLogging import logger, setLogLevel
from smodels.theory.exceptions import SModelSTheoryError as SModelSError
#NEW FOR ML Computation of xsections
from sklearn.externals import joblib

import os, copy
import pyslha
import numpy as np


try:
    import cStringIO as io
except ImportError as e:
    import io
import sys

class ML_XSecComputer:
    """ cross section computer class, what else? """
    def __init__ ( self, model_name):
        """
        :name of the model from which to get the xsecs (only IDM at present)
        """
        self.model_name = self._checkModel ( model_name )
        
    def _checkSLHA ( self, slhafile ):
        if not os.path.isfile(slhafile):
            logger.error("SLHA file %s not found.", slhafile)
            raise SModelSError()
        try:
            f=pyslha.readSLHAFile(slhafile)
        except pyslha.ParseError as e:
            logger.error("File cannot be parsed as SLHA file: %s" % e )
            raise SModelSError()

    def _checkSqrts ( self, sqrts ):
        if type(sqrts)==type(float) or type(sqrts)==type(int):
            logger.warning("sqrt(s) given as scalar, will add TeV as unit." )
            sqrts=float(sqrts)*TeV
        return sqrts

    def _checkModel(self, model_name):
        models=["IDM"]
        if not model_name in models:
            logger.error("There is no ML computer for this model, only IDM supported.")
            raise SModelSError()
        return

    def read_extract_slha(self, slhafile):
        self._checkSLHA ( slhafile )
        #reading the file
        d = pyslha.read(slhafile)
        #Retrieving the relevant parameters
        MH0 = d.blocks['MASS'][35]
        MA0 = d.blocks['MASS'][36]
        MHC = d.blocks['MASS'][37]
        lamL = d.blocks['FRBLOCK'][5]
        lam2 = d.blocks['FRBLOCK'][6]

        input_ML = np.array([MH0,MA0,MHC,lam2,lamL]).reshape(1, -1)
        
        
        return input_ML

    def compute ( self, sqrts, inputFiles, model_name ):
        """
        Use the predict function of the trained ML model to get the cross section

        :param sqrts: sqrt{s} to get the cross section, given as a unum (e.g. 7.*TeV)
        :param slhafile: SLHA file
        :param lhefile: LHE file. If None, do not write pythia output to file. If
        
        :returns: XSectionList object

        """
        sqrts = self._checkSqrts( sqrts )
        
        
        #Loading the ML model
        IDM_NN_model_13TeV = joblib.load('smodels-database/ML_xsections_models/IDM_NN_pipeline_13TeV.gz')
        
        #Since the training has been done on transformed data
        #Need to load the transformation to "untransform" the output 
        # of the ML_algorithm to get the true cross sections
        qt_transform = joblib.load('smodels-database/ML_xsections_models/normalizer_IDM_13TeV.save')
        
        #Getting the cross sections
        # The results are stored in xsecs, an array of 6 values : -37 37, 35 37, 36 37, -37 35, -37 36, 35 36
        for inputFile in inputFiles:
            in_param = self.read_extract_slha(inputFile)
            self.xsecs = qt_transform.inverse_transform(IDM_NN_model_13TeV.predict(in_param))
        return self.xsecs


    def computeForOneFile (self, sqrtses, inputfile, model_name) :
        #compute the cross sections for one file.
        #:params sqrtses: list of sqrt{s} to retrieve the cross section from the ML computer
        logger.info("Computing SLHA cross section from %s." % inputFile )
        print()
        print( "     Cross sections:" )
        print( "=======================" )
        for s in sqrtses:
            xsec_ML = self.compute(s,inputfile, model_name)
    #def computeForOneFile ( self, sqrtses, inputFile, unlink,
                            #lOfromSLHA, tofile, pythiacard=None ):
        #"""
        #compute the cross sections for one file.
        #:param sqrtses: list of sqrt{s} tu run pythia, as a unum (e.g. 7*TeV)
            
        #"""
        #if tofile:
            #logger.info("Computing SLHA cross section from %s, adding to "
                        #"SLHA file." % inputFile )
            #for s in sqrtses:
                #ss = s*TeV 
                #self.compute( ss, inputFile, unlink= unlink, 
                              #loFromSlha= lOfromSLHA, pythiacard=pythiacard )
                #if tofile == "all":
                    #comment = str(self.nevents)+" evts, pythia%d [pb]"%\
                                              #self.pythiaVersion
                    #self.addXSecToFile(self.loXsecs, inputFile, comment )
                #comment = str(self.nevents)+" events, [pb], pythia%d for LO"%\
                                              #self.pythiaVersion
                #self.addXSecToFile( self.xsecs, inputFile, comment)
        #else:
            #logger.info("Computing SLHA cross section from %s." % inputFile )
            #print()
            #print( "     Cross sections:" )
            #print( "=======================" )
            #for s in sqrtses:
                #ss = s*TeV 
                #self.compute( ss, inputFile, unlink=unlink, loFromSlha=lOfromSLHA )
                #for xsec in self.xsecs: 
                    #print( "%s %20s:  %.3e pb" % \
                            #( xsec.info.label,xsec.pid,xsec.value/pb ) )
            #print()

    #def computeForBunch ( self, sqrtses, inputFiles, unlink, tofile):
        #""" compute xsecs for a bunch of slha files """
        ## computer = XSecComputer( order, nevents, pythiaVersion )
        #for inputFile in inputFiles:
            #logger.debug ( "computing xsec for %s" % inputFile )
            #self.computeForOneFile ( sqrtses, inputFile, unlink, tofile )

    def addXSecToFile( self, xsecs, slhafile, comment=None, complain=True):
        """
        Write cross sections to an SLHA file.
        
        :param xsecs: a XSectionList object containing the cross sections
        :param slhafile: target file for writing the cross sections in SLHA format
        :param comment: optional comment to be added to each cross section block
        :param complain: complain if there are already cross sections in file
        
        """
        
        if not os.path.isfile(slhafile):
            logger.error("SLHA file not found.")
            raise SModelSError()
        if len(xsecs) == 0:
            logger.warning("No cross sections available.")
            return False
        # Check if file already contain cross section blocks
        xSectionList = crossSection.getXsecFromSLHAFile(slhafile)
        if xSectionList and complain:
            logger.info("SLHA file already contains XSECTION blocks. Adding "
                           "only missing cross sections.")

        # Write cross sections to file, if they do not overlap any cross section in
        # the file
        outfile = open(slhafile, 'a')
        for xsec in xsecs:
            writeXsec = True
            for oldxsec in xSectionList:
                if oldxsec.info == xsec.info and set(oldxsec.pid) == set(xsec.pid):
                    writeXsec = False
                    break
            if writeXsec:
                outfile.write( self.xsecToBlock(xsec, (2212, 2212), comment) + "\n")
        outfile.close()

        return True

    def xsecToBlock( self, xsec, inPDGs=(2212, 2212), comment=None, xsecUnit = pb):
        """
        Generate a string for a XSECTION block in the SLHA format from a XSection
        object.

        :param inPDGs: defines the PDGs of the incoming states
                       (default = 2212,2212)

        :param comment: is added at the end of the header as a comment
        :param xsecUnit: unit of cross sections to be written (default is pb). 
                         Must be a Unum unit.

        """
        if type(xsec) != type(crossSection.XSection()):
            logger.error("Wrong input")
            raise SModelSError()
        # Sqrt(s) in GeV
        header = "XSECTION  " + str(xsec.info.sqrts / GeV)
        for pdg in inPDGs:
            # PDGs of incoming states
            header += " " + str(pdg)
        # Number of outgoing states
        header += " " + str(len(xsec.pid))
        for pid in xsec.pid:
            # PDGs of outgoing states
            header += " " + str(pid)
        if comment:
            header += " # " + str(comment)  # Comment
        entry = "  0  " + str(xsec.info.order) + "  0  0  0  0  " + \
                str( "%16.8E" % (xsec.value / xsecUnit) ) + " SModelSv" + \
                     installation.version()

        return "\n" + header + "\n" + entry


class ArgsStandardizer:
    """ simple class to collect all argument manipulators """
    
    def getModelName ( self, args ):
        """get the name of the model for which to compute the cross sections """
        model_name = args.model
        logger.info("Computing cross sections for the %s model." % model_name)
        return model_name

    def getInputFiles ( self, args ):
        """ geth the names of the slha files to run over """
        inputPath  = args.filename.strip()
        if not os.path.exists( inputPath ):
            logger.error( "Path %s does not exist." % inputPath )
            sys.exit(1)
        inputFiles = []
        if os.path.isfile ( inputPath ):
            inputFiles = [ inputPath ]
        else:
            files = os.listdir ( inputPath )
            for f in files:
                inputFiles.append ( os.path.join ( inputPath, f ) )
        return inputFiles

    def checkAllowedSqrtses ( self, sqrtses ):
        """ check if the sqrtses are 'allowed' """
        allowedsqrtses=[8, 13]
        for sqrts in sqrtses:
            if not sqrts in allowedsqrtses:
                logger.error("Cannot compute ML LO xsecs for sqrts = %d "
                        "TeV! Available are: %s TeV." % (sqrts, allowedsqrtses ))
                sys.exit(-2)

    def queryCrossSections ( self, filename ):
        if os.path.isdir ( filename ):
            logger.error ( "Cannot query cross sections for a directory." )
            sys.exit(-1)
        xsecsInfile = crossSection.getXsecFromSLHAFile(filename)
        if xsecsInfile:
            print ( "1" )
        else:
            print ( "0" )

    def getSqrtses ( self, args ):
        """ extract the sqrtses from argument list """
        sqrtses = [item for sublist in args.sqrts for item in sublist]
        if len(sqrtses) == 0:
            sqrtses = [8,13]
        sqrtses.sort()
        sqrtses = set(sqrtses)
        return sqrtses
    
    def checkNCPUs ( self, ncpus, inputFiles ):
        if ncpus < -1 or ncpus == 0:
            logger.error ( "Weird number of CPUs given: %d" % ncpus )
            sys.exit()
        if ncpus == -1:
            ncpus = runtime.nCPUs()
        ncpus = min ( len(inputFiles), ncpus )
        if ncpus == 1:
            logger.info ( "We run on a single cpu" )
        else:
            logger.info ( "We run on %d cpus" % ncpus )
        return ncpus


def main(args):
    canonizer = ArgsStandardizer()
    setLogLevel ( args.verbosity )
    if args.query:
        return canonizer.queryCrossSections ( args.filename )
    if args.colors:
        from smodels.tools.colors import colors
        colors.on = True
    sqrtses = canonizer.getSqrtses ( args )
    canonizer.checkAllowedSqrtses ( sqrtses )
    inputFiles = canonizer.getInputFiles ( args )
    logger.info("Input files %s : " % inputFiles)
    ncpus = canonizer.checkNCPUs ( args.ncpus, inputFiles )    
    model = canonizer.getModelName( args )
    
    children = []
    for i in range(ncpus):
        pid = os.fork()
        chunk = inputFiles [ i::ncpus ]
        if pid < 0:
            logger.error ( "fork did not succeed! Pid=%d" % pid ) 
            sys.exit()
        if pid == 0:
            logger.debug ( "chunk #%d: pid %d (parent %d)." % 
                       ( i, os.getpid(), os.getppid() ) )
            logger.debug ( " `-> %s" % " ".join ( chunk ) )
            computer = ML_XSecComputer(model)
            computer.compute(sqrtses,chunk,model)
#            computer.computeForBunch (  sqrtses, chunk, not args.keep,
#                                args.LOfromSLHA, toFile, pythiacard=pythiacard )
            logger.info(computer.xsecs)
            os._exit ( 0 )
        if pid > 0:
            children.append ( pid )
    for child in children:
        r = os.waitpid ( child, 0 )
        logger.debug ( "child %d terminated: %s" % (child,r) )
    logger.debug ( "all children terminated." )
