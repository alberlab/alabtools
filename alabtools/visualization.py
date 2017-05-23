import colorsys
import numpy as np

def dump_pdb(crd, radii, index, output_file=None):
    '''
    Returns a string with the pdb representation of a structure.
    Chain id is set to the chromosome copy number.
    Resid is set to the chromosome number.
    Resname is set to C<X> where <X> is the chromosome number.
    Occupancy is set to the bead radius.
    
    Parameters
    ----------
        crd : 2D array
            Coordinates of the beads (num_bead x 3)
        radii : array
            The radii of the beads
        index: alabtools.Index
            Index for the beads
        output_file : str
            If specified, the pdb file will be written to disk at this path

    Returns
    -------
        str : a string with the file content
    ''' 
    vstr = ''
    for i, (x,y,z) in enumerate(crd):
        vstr += '{:6s}{:5d} {:^4s}{:1s}{:3d}'.format('HETATM', i+1, 'BDX',  ' ', index.chrom[i])
        vstr += ' {:1d}{:>4s}{:1s}   '.format(index.copy[i] % 10, 'C%d' % index.chrom[i], ' ')
        vstr += '{:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}'.format(x/10, y/10, z/10, radii[i]/10, 0, 0, 0)
        vstr += '\n'

    if output_file is not None:
        with open(output_file, 'w') as f:
            f.write(vstr)

    return vstr

def dump_tcl(pdbfname, index=None, output_file=None):
    '''
    Dumps a tcl file to be used with vmd to visualize a pdb file.
    Use as vmd -e myfile.tcl

    Parameters
    ----------
        pdbfname : str
            Path to the pdb file to be opened
        index: alabtools.Index
            If specified, the function will generate consecutive beads bonds
            according to the chromosome information in the index
        output_file : str
            If specified, the result will be written to disk at this path

    Returns
    -------
        str : a string with the file content
    '''
    bondlist = ''
    if index is not None:
        for i in range(1, len(index)):
            if (index.chrom[i] == index.chrom[i-1] and
                index.copy[i] == index.copy[i-1]):
                bondlist += ' {%d %d}' % (i-1, i)

    # Generate colors
    h = np.arange(25.0)/24
    color = {}
    for i in range(23):
        color[i] = colorsys.hsv_to_rgb(h[i],1,1)

    vstr = ''
    # Set colors
    vstr += 'color change rgb 0 255 255 255\n'
    vstr += 'color change rgb 60 0 0 0\n'
    
    vstr += 'color Display Background 0\n' 
    vstr += 'color Display FPS 60\n'
    vstr += 'color Axes Labels 60\n'
    for i in range(23):
        cn = i+1
        vstr += 'color change rgb %d %f %f %f\n' % (cn, color[i][0], color[i][1], color[i][2])
    # Load coordinates
    vstr += 'mol new %s type pdb\n' % pdbfname
    # Set radiuses
    vstr += 'mol delrep 0 top\n'
    vstr += 'set sel [atomselect top all]\n'
    vstr += '$sel set radius [$sel get occupancy]\n'
    # Create representation
    vstr += 'mol representation VDW 1.000000 12.000000\n'
    vstr += 'mol color ResName\n'
    vstr += 'mol selection {all}\n'
    vstr += 'mol material Diffuse\n'
    vstr += 'mol addrep top\n'
    
    if len(bondlist) > 0:
        vstr += 'topo setbondlist { %s }\n' % bondlist
        vstr += 'mol representation Bonds 5.000000 12.000000\n'
        vstr += 'mol selection {all}\n'
        vstr += 'mol material Diffuse\n'
        vstr += 'mol addrep top\n'

    if output_file is not None:
        with open(output_file, 'w') as f:
            f.write(vstr)
            
    return vstr

