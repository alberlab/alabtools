# Contains functions to read FOF-CT (4DN Fish Omics Format - Chromatin Tracing) files.

import warnings
import os
import sys
import numpy as np
from alabtools.utils import Genome, Index

def read(filename):
    """
    Read the FOF-CT (4DN Fish Omics Format - Chromatin Tracing) csv file.
    
    Get the genome assembly, column names and data from the FOF-CT file.
    
    Args:
        filename (str): path to the csv file
    
    Returns:
        assembly (list of str): assembly names
        cols (np.array of str): column names
        data (np.array of str): data
    """

    # OPEN THE FILE AND READ THE LINES
    with open(filename, 'r') as csv:
        lines = csv.readlines()

    # SEPARATE HEADER AND DATA
    i_header_stop = 0  # index of the last line of the header
    for line in lines:
        if line[0] == '#' or line[1] == '#' or 'Trace_ID' in line:
            # header lines start with # or ## or contain Trace_ID
            i_header_stop += 1
        else:
            break  # header ends
    header_lines = lines[:i_header_stop]
    data_lines = lines[i_header_stop:]
    
    # CHECK THAT HEADER AND DATA ARE NOT EMPTY    
    if len(header_lines) == 0:
        raise ValueError('The header is empty.')
    if len(data_lines) == 0:
        raise ValueError('The data is empty.')

    # CLEAN THE HEADER
    header = []
    for line in header_lines:
        # Remove the characters: #, ##, ", \n, parentheses, commas, spaces
        # at the beginning and end of the line
        line = line.strip('#" ,\n()[]{}')
        header.append(line)

    # READ ASSEMBLY AND COLUMN NAMES FROM HEADER
    assembly = None
    cols = None
    for line in header:
        line = line.replace(' ', '')  # remove all spaces
        line = line.strip('( )')  # remove the parentheses at the beginning and end of the line
        if 'genome_assembly' in line or 'assembly' in line:  # read the genome assembly
            # Some datasets have multiple assemblies separated by /
            line = line.split('=')[1]  # split the line at the equal sign and take the second part
            assembly = line.split('/')  # take both assemblies if / is present
        if 'Trace_ID' in line:  # read the column names
            if '=' in line:  # = may or may not be present
                line = line.split('=')[1]  # if it is, take the right part
            cols = line.split(',')  # split the line at the commas
            cols = np.array(cols)
    if assembly is None:
        raise ValueError('The genome assembly is not specified in the header.')
    if cols is None:
        raise ValueError('The column names are not specified in the header.')
    
    # CHECK IF COLUMN NAMES ARE CORRECT
    required_cols = ['X', 'Y', 'Z', 'Chrom', 'Chrom_Start', 'Chrom_End', 'Trace_ID']
    for col in required_cols:
        if col not in cols:
            raise ValueError('The column {} is not present in the header.'.format(col))

    # CONVERT DATA TO NUMPY ARRAY
    data = []
    for line in data_lines:
        line = line.strip('#" ,\n()[]{}')  # remove special characters at the beginning and end
        line = line.replace(' ', '')  # remove all spaces
        values = line.split(',')  # split the line at the commas
        data.append(values)
    data = np.array(data)
    data[data == ''] = np.nan  # replace empty values with NaN

    return assembly, cols, data


def get_domains(data, cols):
    """
    Get the domains from the FOF-CT data.
    The domains are found as unique ordered tuples (chr, start, end).
    The domains are then ordered by chromosome first, then by start position.
    
    Args:
        data (np.array of str): data (from read_fofct)
        cols (np.array of str): column names (from read_fofct)
    
    Returns:
        chrstr (np.array of str): chromosome strings
        start (np.array of int): start positions
        end (np.array of int): end positions
    """
    
    # GET THE SPOTS' CHROMOSOME, START AND END POSITIONS
    spots_chrstr = data[:, cols == 'Chrom']
    spots_start = data[:, cols == 'Chrom_Start']
    spots_end = data[:, cols == 'Chrom_End']
    
    # IDENTIFY DOMAINS (UNSORTED)
    # The domains are found as unique ordered tuples (chr, start, end)
    domains = np.unique(np.hstack((spots_chrstr, spots_start, spots_end)), axis=0)

    # SORT DOMAINS BY CHROMOSOME
    chrstr, start, end = domains.T
    chrint = []  # convert the chromosome string to an integer (e.g. 'chr1' -> 1)
    for c in chrstr:
        c = c.split('chr')[1]  # remove the 'chr' part
        if c.isdigit():  # if it's a number
            c = int(c)
        elif c == 'X':
            c = 23  # if it's mouse this should be 20, but it doesn't matter
        elif c == 'Y':
            c = 24
        elif c == 'M':
            c = 25
        else:
            c = 26  # This is to deal with other chr labels (e.g. 'chr1_random')
        c = int(c)
        chrint.append(c)
    chrint = np.array(chrint)
    domains = domains[np.argsort(chrint)]  # Sort the domains by chrint
    
    # SORT DOMAINS BY START POSITION WITHIN EACH CHROMOSOME
    chrstr, start, end = domains.T
    start = start.astype(int)  # cast to int to avoid problems with sorting
    for c in np.unique(chrstr):
        idx = chrstr == c
        domains[idx] = domains[idx][np.argsort(start[idx])]

    # convert the domains to numpy arrays
    chrstr, start, end = domains.T
    start = start.astype(int)
    end = end.astype(int)
    
    return chrstr, start, end

def get_hashmaps(data, cols):
    """Creates hashmaps that map cellIDs, chroms and traceIDs to integers,
    i.e. the position of the cell/chrom/trace in the final arrays.
    
    Also returns the maximum copy number and the maximum number of spots.
    
    The hashmap for cellIDs is a 1-level dictionary.
    Example:
        cell_labels = {'cell1': 0, 'cell2': 1, 'cell3': 2}
    Usage:
        cellID = 'cell1'
        cellnum = cell_labels[cellID]
    
    The hashmap for the traces is a 3-level dictionary.
    Example:
        trace_hashmap = {'cell1': {'chr1': {'trace1': 0, 'trace2': 1},
                                   'chr2': {'trace1': 0},
                                   'chr3': {'trace1': 0, 'trace2': 1, 'trace3': 2}
                                  },
                         'cell2': {'chr1': {'trace1': 0, 'trace2': 1},
                                   'chr2': {'trace1': 0, 'trace2': 1, 'trace3': 2}
                                  }
                        }
    Usage:
        cellID, chrom, traceID = 'cell1', 'chr1', 'trace1'
        tracenum = trace_hashmap[cellID][chrom][traceID]
    
    Args:
        data (np.array of str): data
        cols (np.array of str): column names
    
    Returns:
        cell_labels (dict): cell hashmap
        trace_hashmap (dict): trace hashmap
        ncell (int): number of cells
        ncopy_max (int): maximum copy number of a trace.
        nspot_max (int): maximum number of spots in a trace.
    """
    
    # Extract the columns from the data
    cellIDs = data[:, cols == 'Cell_ID']
    chroms = data[:, cols == 'Chrom']
    traceIDs = data[:, cols == 'Trace_ID']
    
    # Initialize the cell hashmap
    cell_hashmap = {}
    ncell = 0
    # Initialize the trace hashmap
    trace_hashmap = {}
    ncopy_max = 1
    # Initialize the spot_count
    # We create a parallel hashmap to keep track of the number of spots in each trace
    # The difference between the hashmaps:
    #    - trace_hashmap keeps track of whether a trace is present or not, without counting
    #    - spotcount keeps track of the number of times each trace appears
    spotcount = {}
    nspot_max = 1
    
    # Loop through the spots
    for cellID, chrom, traceID in zip(cellIDs, chroms, traceIDs):
        # Add the cellID to the hashmap if it's not there yet
        if cellID not in cell_hashmap:
            cell_hashmap[cellID] = ncell
            ncell += 1
        # Add the traceID to the hashmap if it's not there yet
        # Case 1: cellID is not yet in the hashmap
        if cellID not in trace_hashmap:
            # Add the traceID to the hashmap
            trace_hashmap[cellID] = {chrom: {traceID: 0}}
            # Add the spot count to the hashmap
            spotcount[cellID] = {chrom: {traceID: 1}}
            continue
        # Case 2: cellID is in the hashmap but chrom is not
        if chrom not in trace_hashmap[cellID]:
            # Add the traceID to the hashmap
            trace_hashmap[cellID][chrom] = {traceID: 0}
            # Add the spot count to the hashmap
            spotcount[cellID][chrom] = {traceID: 1}
            continue
        # Case 3: cellID and chrom are in the hashmap, but traceID is not
        if traceID not in trace_hashmap[cellID][chrom]:
            # Get the maximum trace ID for this cell/chrom
            max_ID = max(trace_hashmap[cellID][chrom].values())
            # Add the trace ID to the hashmap with the next available ID
            trace_hashmap[cellID][chrom][traceID] = max_ID + 1
            # Update the maximum copy number if possible
            if max_ID + 1 > ncopy_max:
                ncopy_max = max_ID + 1
            # Add the spot count to the hashmap
            spotcount[cellID][chrom][traceID] = 1
            continue
        # Case 4: cellID, chrom and traceID are in the hashmap
        # Update the spot count
        spotcount[cellID][chrom][traceID] += 1
        # Update the maximum spot count if possible
        if spotcount[cellID][chrom][traceID] > nspot_max:
            nspot_max = spotcount[cellID][chrom][traceID]
    
    # Free up memory
    del spotcount
    
    return cell_hashmap, trace_hashmap, ncell, ncopy_max, nspot_max

def extract_genome_index(assembly, chromstr, start, end):
    """Extracts the genome and index objects from the FOF-CT data.

    Args:
        assembly (list of str): genome assembly names
        chromstr (np.array of str): chromosome strings
        start (np.array of int): start positions
        end (np.array of int): end positions

    Returns:
        genome (Genome): genome object
        index (Index): index object
    """
    
    # Add lower case version of assembly
    # Could be necessary for genome filenames in alabtools/genomes
    assembly = assembly + [a.lower() for a in assembly]
    
    # Initialize the Genome object
    genome = None
    # Loop through the assembly names to fine the assembly file
    for ass in assembly:
        # Path to the assembly file
        genome_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/genomes/'
        assembly_file = os.path.join(genome_dir, ass + '.info')
        # Check if assembly_file is present
        if not os.path.isfile(assembly_file):
            continue
        sys.stdout.write('Assembly {} found in alabtools/genomes. Using this.\n'.format(ass))
        # Create the Genome object
        # Note: if some chromosomes in np.unique(chrstr) are not in the assembly file, they will be ignored.
        genome = Genome(ass, usechr=np.unique(chromstr))
        break            
    if genome is None:
        raise ValueError('Assembly {} not found in alabtools/genomes. Need to include it.'.format(ass))
            
    # Create the Index object
    index = Index(chrom=chromstr, start=start, end=end, genome=genome)
    
    return genome, index

def extract_data(data, cols,
                 cell_hashmap, index_hashmap, trace_hashmap,
                 ncell, ndomain, ncopy_max, nspot_max):
    
    """Extracts the data from the FoF-CT data.

    Args:
        data (np.array of str): data
        cols (np.array of str): column names
        cell_hashmap (dict):
                hashmap to convert cellID to cell number
        index_hashmap (dict):
                hashmap to convert domain (chrom, start, end)
                to domain number
        trace_hashmap (dict):
                hashmap to convert traceID (along with cell/chrom)
                to trace number
        ncell (int): number of cells
        ndomain (int): number of domains
        ncopy_max (int): maximum number of copies (traces)
        nspot_max (int): maximum number of spots

    Returns:
        cell_labels (np.array(ncell), dtype='S10'): cell labels
        coordinates (np.array(ncell, ndomain, ncopy_max, nspot_max, 3, dtype=float64)):
                coordinates of the spots
        intensity (np.array(ncell, ndomain, ncopy_max, nspot_max, dtype=float64)):
                intensity of the spots
        nspot (np.array(ncell, ndomain, ncopy_max, dtype=int64)):
                number of spots in each trace
        ncopy (np.array(ncell, ndomain, dtype=int64)):
                number of copies (traces) in each domain
    """
    
    # Initialize data
    cell_labels = np.array(list(cell_hashmap.keys())).astype('S10')
    coordinates = np.full((ncell, ndomain, ncopy_max, nspot_max, 3),
                         np.nan,
                         dtype=np.float64)
    intensity = np.full((ncell, ndomain, ncopy_max, nspot_max),
                         np.nan,
                         dtype=np.float64)
    nspot = np.zeros((ncell, ndomain, ncopy_max),
                     dtype=np.int64)
    ncopy = np.zeros((ncell, ndomain),
                     dtype=np.int64)
    
    # Loop through the rows of the FOF-CT data
    for row in data:
        # Unpack the row into variables using the column names
        x = row[cols == 'X'][0].astype(np.float64)
        y = row[cols == 'Y'][0].astype(np.float64)
        z = row[cols == 'Z'][0].astype(np.float64)
        chrstr = row[cols == 'Chrom'][0]
        start = row[cols == 'Chrom_Start'][0].astype(np.int64)
        end = row[cols == 'Chrom_End'][0].astype(np.int64)
        traceID = row[cols == 'Trace_ID'][0]
        if 'Cell_ID' in cols:
            cellID = row[cols == 'Cell_ID'][0]
        else:
            # if datasets where only one chromosome is imaged TraceID = CellID
            cellID = np.copy(traceID)
        if 'Intensity' in cols:
            # if the spot intensity is present, use it
            lum = row[cols == 'Intensity'][0].astype(np.float64)
        else:
            lum = np.nan
        
        # get indices from the hashmaps
        cellnum = cell_hashmap[cellID]
        domnum = index_hashmap[(chrstr, start, end)]
        tracenum = trace_hashmap[cellID][chrstr][traceID]
        
        # get the spot number as the next available spot number
        # (first NaN value in coordinates)
        nan_idx = np.where(np.isnan(coordinates[cellnum, domnum, tracenum, :, 0]))[0]
        spotnum = np.min(nan_idx)
        
        # fill in the data
        coordinates[cellnum, domnum, tracenum, spotnum, :] = [x, y, z]
        intensity[cellnum, domnum, tracenum, spotnum] = lum
        # TO_DECIDE: I can also compute these two as in set_manually in the CtFile class
        nspot[cellnum, domnum, tracenum] += 1
        ncopy[cellnum, domnum] = len(trace_hashmap[cellID][chrstr])
        
    return cell_labels, coordinates, intensity, nspot, ncopy

def process(filename, in_assembly=None):
    """
    Read the FOF-CT (4DN Fish Omics Format - Chromatin Tracing) csv file.
    
    Get the genome assembly, column names and data from the FOF-CT file.
    
    Args:
        filename (str): path to the csv file
        in_assembly (str): genome assembly name (e.g. hg38)
    
    Returns:
        genome (Genome): genome object
        index (Index): index object
        cell_labels (np.array(ncell), dtype='S10'): cell labels
        coordinates (np.array(ncell, ndomain, ncopy_max, nspot_max, 3, dtype=float64)):
                coordinates of the spots
        intensity (np.array(ncell, ndomain, ncopy_max, nspot_max, dtype=float64)):
                intensity of the spots
        nspot (np.array(ncell, ndomain, ncopy_max, dtype=int64)):
                number of spots in each trace
        ncopy (np.array(ncell, ndomain, dtype=int64)):
                number of copies (traces) in each domain
    """
    
    # READ THE FILE
    assembly, cols, data = read(filename)
    
    # GET THE DOMAINS
    chromstr, start, end = get_domains(data, cols)
    
    # GET THE HASHMAPS
    cell_hashmap, trace_hashmap, ncell, ncopy_max, nspot_max = get_hashmaps(data, cols)
    
    # GET THE GENOME AND INDEX
    genome, index = extract_genome_index(assembly, chromstr, start, end)
    ndomain = len(index)
    index_hashmap = index.get_index_hashmap()
    
    # EXTRACT THE DATA
    cell_labels, coordinates, intensity, nspot, ncopy = extract_data(data, cols,
                                                                     cell_hashmap, index_hashmap, trace_hashmap,
                                                                     ncell, ndomain, ncopy_max, nspot_max)
    
    return genome, index, cell_labels, coordinates, intensity, nspot, ncopy, ncell, ndomain, ncopy_max, nspot_max
