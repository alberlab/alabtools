# ihm format writer
# A simple support for generating genome structure models in mmCIF firmat
from __future__ import division, print_function
KEY = 'KEY'
TEXT = 'TEXT'
import struct
import json
import os
datafile = os.path.join(os.path.dirname( os.path.abspath(__file__) ),
                        "config/ihm_template.json")  
with open(datafile) as f:
    Framework = json.loads(f.read())

class ihmWriter(object):
    def __init__(self, ihmfile, inargs=None):
        self.ihmfile = open(ihmfile, 'w')
        
        for info, value in inargs:
            self.info_writer(info, value)
        self.flush()
        from functools import partial
        for category in Framework.keys():
            write_category_function = partial(self._category_writer, category=category)
            write_category_function.__doc__ = "Write {} into loop\nKeys = [{}]\nTypes=[{}]".format(
                                                category, 
                                                ", ".join(Framework[category]['key']),
                                                ", ".join(Framework[category]['type']))
            write_category_function.__name__ =  "write_"+category
            setattr(self, "write_"+category, write_category_function)
            
    def flush(self):
        self.ihmfile.flush()
        
    def info_writer(self, info, value):
        if value.find(' ') >= 0:
            value = repr(value)
        self.ihmfile.write("{} {}\n".format(info, value))
        
    #-----------------handle loops---------------------
    def _check_record(self, rec, vtypes):
        if (len(rec) != len(vtypes)):
            raise RuntimeError("Missing data: {} values given, {} required.\n".format(len(rec),len(vtypes)))
        processed = []
        for x, t in zip(rec, vtypes):
            if t == TEXT:
                if str(x)[0] == str(x)[-1] == "'":
                    processed.append(str(x))
                else:
                    processed.append("'{}'".format(x))
            else:
                processed.append(str(x))
        return processed
    
    def _loop_writer(self, category, keys, data):
        self.ihmfile.write("#\nloop_\n")
        for key in keys:
            self.ihmfile.write("_{}.{}\n".format(category, key))
            
        for line in data:
            self.ihmfile.write(" ".join(line))
            self.ihmfile.write("\n")
            
        self.ihmfile.write("#\n")
        self.flush()
    
    def _category_writer(self, data, category):
        keys = Framework[category]['key']
        vtypes = Framework[category]['type']
        processed = []
        for d in data:
            processed.append(self._check_record(d, vtypes))
        
        self._loop_writer(category, keys, processed)
    #------------------------------------------------
    
    def close(self):
        self.ihmfile.close()
    def __del__(self):
        self.ihmfile.close()
#===============

class DCDWriter(object):
    """Utility class to write model coordinates to a binary DCD file.
       See :class:`Ensemble` and :class:`Model`. Since mmCIF is a text-based
       format, it is not efficient to store entire ensembles in this format.
       Instead, representative models should be deposited as mmCIF and
       the :class:`Ensemble` then linked to an external file containing
       only model coordinates. One such format is CHARMM/NAMD's DCD, which
       is written out by this class. The DCD files simply contain the xyz
       coordinates of all :class:`Atom` and :class:`Sphere` objects in each
       :class:`Model`. (Note that no other data is stored, such as sphere
       radii or restraint parameters.)
       :param file fh: The filelike object to write the coordinates to. This
              should be open in binary mode and should be a seekable object.
    """
    def __init__(self, fh):
        self.fh = fh
        self.nframes = 0

    def add_model(self, model):
        """Add the coordinates for the given :class:`Model` to the file as
           a new frame. All models in the file should have the same number of
           atoms and/or spheres, in the same order.
           :param model: Model with coordinates to write to the file.
           :type model: :class:`Model`
        """
        x = []
        y = []
        z = []
        for a in itertools.chain(model.get_atoms(), model.get_spheres()):
            x.append(a.x)
            y.append(a.y)
            z.append(a.z)
        self._write_frame(x, y, z)

    def _write_frame(self, x, y, z):
        self.nframes += 1
        if self.nframes == 1:
            self.ncoord = len(x)
            remarks = [
              b'Produced by python-ihm, https://github.com/ihmwg/python-ihm',
              b'This file is designed to be used in combination with an '
              b'mmCIF file',
              b'See PDB-Dev at https://pdb-dev.wwpdb.org/ for more details']
            self._write_header(self.ncoord, remarks)
        else:
            if len(x) != self.ncoord:
                raise ValueError("Frame size mismatch - frames contain %d "
                        "coordinates but attempting to write a frame "
                        "containing %d coordinates" % (self.ncoord, len(x)))
            # Update number of frames
            self.fh.seek(self._pos_nframes)
            self.fh.write(struct.pack('i', self.nframes))
            self.fh.seek(0, 2) # Move back to end of file

        # Write coordinates
        frame_size = struct.pack('i', struct.calcsize("%df" % self.ncoord))
        for coord in x, y, z:
            self.fh.write(frame_size)
            self.fh.write(struct.pack("%df" % self.ncoord, *coord))
            self.fh.write(frame_size)

    def _write_header(self, natoms, remarks):
        self.fh.write(struct.pack('i', 84) + b'CORD')
        self._pos_nframes = self.fh.tell()
        self.fh.write(struct.pack('i', self.nframes))
        self.fh.write(struct.pack('i', 0)) # istart
        self.fh.write(struct.pack('i', 0)) # nsavc
        self.fh.write(struct.pack('5i', 0, 0, 0, 0, 0))
        self.fh.write(struct.pack('i', 0)) # number of fixed atoms
        self.fh.write(struct.pack('d', 0.)) # delta
        self.fh.write(struct.pack('10i', 0, 0, 0, 0, 0, 0, 0, 0, 0, 84))
        remark_size = struct.calcsize('i') + 80 * len(remarks)
        self.fh.write(struct.pack('i', remark_size))
        self.fh.write(struct.pack('i', len(remarks)))
        for r in remarks:
            self.fh.write(r.ljust(80)[:80])
        self.fh.write(struct.pack('i', remark_size))
        self.fh.write(struct.pack('i', struct.calcsize('i')))
        self.fh.write(struct.pack('i', natoms)) # total number of atoms
        self.fh.write(struct.pack('i', struct.calcsize('i')))
