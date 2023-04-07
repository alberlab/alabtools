import numpy as np
import sys
import os
sys.path.append(os.path.abspath('..'))
from alabtools.imaging import CtFile
from alabtools.utils import Genome, Index


class Phaser(object):
    """
    A class that describes a general Phasing object.
    
    Works on CtFiles.
    
    Attributes
    ---------
    ct: CtFile instance
    
    """
    
    def __init__(self, ct, *args):
        
        if isinstance(ct, CtFile):
            self.ct = ct
        elif isinstance(ct, str) and ct[-3:] == '.ct':
            self.ct = CtFile(ct, 'r')
        else:
            raise ValueError("ct not in supported format.")
        
        self.phase = np.zeros((self.ct.ncell, self.ct.ndomain, self.ct.nspot_max))
    
    def phasing(self):
        
        assert self.ct.ncopy_max == 1, "Data is already phased."
        
        for cellID in self.ct.cell_labels:
            for chrom in self.ct.genome.chroms:
                
                crd = self.ct.coordinates[self.ct.cell_labels==cellID,
                                          self.ct.index.chromstr==chrom,
                                          0, :, :]  # np.array(ndomain_chrom, nspot_max, 3)
                
                dom = Index(chrom = self.ct.index.chromstr[self.ct.index.chromstr == chrom],
                            start = self.ct.index.start[self.ct.index.chromstr == chrom],
                            end = self.ct.index.end[self.ct.index.chromstr == chrom],
                            genome=self.ct.genome)
                
                cellnum = self.ct.get_cellnum(cellID)
                chrnum = self.ct.genome.getchrnum(chrom)
                
                phs = self._apply_chrom_phasing(crd, dom, cellnum, chrnum)
                                
                self.phase[cellnum, self.ct.index.chromstr == chrom, :] = phs
    
    
    # parallelize phasing
    def parallel_phasing(self, nproc=4):                     0, :, :]
                

    
    def _apply_chrom_phasing(self, *args):
        pass
    