# Contains functions to read FOF-CT (4DN Fish Omics Format - Chromatin Tracing) files.

import warnings
import os
import sys
import numpy as np
from alabtools.utils import Genome, Index


# Functions to read the FOF-CT file

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
    header_lines, data_lines = separate_header_data(lines)
    
    # CHECK THAT HEADER AND DATA ARE NOT EMPTY    
    if len(header_lines) == 0:
        raise ValueError('The header is empty.')
    if len(data_lines) == 0:
        raise ValueError('The data is empty.')

    # CLEAN THE HEADER
    header = clean_header(header_lines)

    # READ ASSEMBLY AND COLUMN NAMES FROM HEADER
    assembly = extract_assembly(header)
    cols = extract_cols(header)
    
    # CHECK IF ASSEMBLY AND COLUMNS ARE CORRECT
    if assembly is None:
        raise ValueError('The genome assembly is not specified in the header.')
    if cols is None:
        raise ValueError('The column names are not specified in the header.')
    required_cols = ['X', 'Y', 'Z', 'Chrom', 'Chrom_Start', 'Chrom_End', 'Trace_ID', 'Spot_ID']
    for col in required_cols:
        if col not in cols:
            raise ValueError('The column {} is not present in the header.'.format(col))

    # CONVERT DATA TO NUMPY ARRAY
    data = convert_data_to_array(data_lines)

    return assembly, cols, data

def separate_header_data(lines):
    """Separate the header and data from the FOF-CT file,
    given the lines of the file.

    Args:
        lines (list of str): lines of the FOF-CT file
    Returns:
        header (list of str): header lines
        data (list of str): data lines
    """
    # Initialize the index where the header ends
    i_header_stop = 0
    for line in lines:
        # header lines start with # or ## or contain Trace_ID
        if line[0] == '#' or line[1] == '#' or 'Trace_ID' in line:
            i_header_stop += 1
        else:
            break  # header ends
    header = lines[:i_header_stop]
    data = lines[i_header_stop:]
    return header, data

def clean_header(header):
    """Clean the header lines of the FOF-CT file.

    Args:
        header (list of str): header lines
    Returns:
        header_clean (list of str): cleaned header lines
    """
    header_clean = []
    for line in header:
        line = line.strip('#" ,\n()[]{}')
        header_clean.append(line)
    return header_clean

def extract_assembly(header):
    """Extract the genome assembly from the header lines of the FOF-CT file.

    Args:
        header (list of str): header lines (cleaned)
    Returns:
        assembly (list of str): assembly names
    """
    assembly_keys = ['genome_assembly', 'assembly']  # possible keys for the assembly
    assembly = None
    for line in header:
        line = line.replace(' ', '')
        for assembly_key in assembly_keys:
            if assembly_key not in line:
                continue
            # Some datasets have multiple assemblies separated by /
            line = line.split('=')[1]
            assembly = line.split('/')  # take both assemblies if / is present
            # stop looping through the assembly keys if the assembly is found
            break
        if assembly is not None:
            # stop looping through the header lines if the assembly is found
            break
    return assembly

def extract_cols(header):
    """Extract the column names from the header lines of the FOF-CT file.

    Args:
        header (list of str): header lines (cleaned)
    Returns:
        cols (np.array of str): column names
    """
    col_keys = ['Spot_ID, Trace_ID', 'Cell_ID']  # possible keys for the columns
    cols = None
    for line in header:
        line = line.replace(' ', '')
        for col_key in col_keys:
            if col_key not in line:
                continue
            # if the column key is in the line, extract the column names
            if '=' in line:  # = may or may not be present
                line = line.split('=')[1]  # if it is, take the right part
                line = line.replace(' ', '')
                line = line.strip('#" ,\n()[]{}')
            cols = line.split(',')
            cols = np.array(cols)
            # stop looping through the column keys if the columns are found
            break
        if cols is not None:
            # stop looping through the header lines if the columns are found
            break
    return cols

def convert_data_to_array(data):
    """Convert the data from the FOF-CT file to a numpy array.

    Args:
        data (list of str): data lines
    Returns:
        data_arr (np.array of str): data
    """
    data_arr = []
    for line in data:
        line = line.strip('#" ,\n()[]{}')
        line = line.replace(' ', '')
        values = line.split(',')
        data_arr.append(values)
    data_arr = np.array(data_arr)
    data_arr[data_arr == ''] = np.nan
    return data_arr


# Functions to process the genome and index

def get_domains(data, cols):
    """
    Get the domains from the FOF-CT data.
    The domains are found as unique ordered tuples (chr, start, end).
    The domains are then ordered by chromosome first, then by start position.
    
    Args:
        data (np.array of str): data (from read_fofct)
        cols (np.array of str): column names (from read_fofct)
    
    Returns:
        chromstr (np.array of str): chromosome strings
        start (np.array of int): start positions
        end (np.array of int): end positions
    """
    
    # GET THE SPOTS' CHROMOSOME, START AND END POSITIONS
    spots_chromstr = data[:, cols == 'Chrom']
    spots_start = data[:, cols == 'Chrom_Start']
    spots_end = data[:, cols == 'Chrom_End']
    
    # IDENTIFY DOMAINS (UNSORTED)
    # The domains are found as unique ordered tuples (chr, start, end)
    domains = np.unique(np.hstack((spots_chromstr, spots_start, spots_end)), axis=0)

    # SORT DOMAINS BY CHROMOSOME
    chromstr, start, end = domains.T
    chromint = chrstr_to_int(chromstr)
    domains = domains[np.argsort(chromint)]  # Sort the domains by chromint
    
    # SORT DOMAINS BY START POSITION WITHIN EACH CHROMOSOME
    chromstr, start, end = domains.T
    start = start.astype(int)  # cast to int to avoid problems with sorting
    for c in np.unique(chromstr):
        idx = chromstr == c
        domains[idx] = domains[idx][np.argsort(start[idx])]

    # convert the domains to numpy arrays
    chromstr, start, end = domains.T
    start = start.astype(int)
    end = end.astype(int)
    
    return chromstr, start, end

def chrstr_to_int(chromstr):
    """Converts chromosome strings to integers,
    e.g. 'chr1' -> 1, 'chrX' -> 23.

    Args:
        chrstr (np.array of str): chromosome strings
    Returns:
        chrint (np.array of int): chromosome integers
    """
    chromint = []
    for c in chromstr:
        c = c.split('chr')[1]
        if c.isdigit():
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
        chromint.append(c)
    chromint = np.array(chromint)
    return chromint

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


# Functions to process the data

def get_hashmaps(data, cols):
    """Creates hashmaps that map cellIDs, chroms, traceIDs and spotIDs to integers,
    i.e. the position of the cell/chrom/trace/spot in the final arrays.
    
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
    
    The hashmap for the spots is a 4-level dictionary.
    Example:
        spot_hashmap = {'cell1': {('chr1', 1000, 2000)': {'trace1': {'spot1': 0, 'spot2': 1},
                                                          'trace2': {'spot1': 0, 'spot2': 1, 'spot3': 2}
                                                         },
                                  ('chr2', 3000, 4000)': {'trace1': {'spot1': 0, 'spot2': 1, 'spot3': 2},
                                                          'trace2': {'spot1': 0, 'spot2': 1}
                                                         }
                                  },
                        'cell2': {('chr1', 2000, 3000)': {'trace1': {'spot1': 0, 'spot2': 1},
                                                          'trace2': {'spot1': 0}
                                                         },
                                  ('chr2', 5000, 6000)': {'trace1': {'spot1': 0, 'spot2': 1}
                                                         }
                                  }
                        }
    Usage:
        cellID, domain, traceID, spotID = 'cell1', ('chr1', 1000, 2000), 'trace1', 'spot1'
        spotnum = spot_hashmap[cellID][domain][traceID][spotID]
    
    
    Args:
        data (np.array of str): data
        cols (np.array of str): column names
    
    Returns:
        cell_labels (dict): cell hashmap
        trace_hashmap (dict): trace hashmap
        spot_hashmap (dict): spot hashmap
        ncell (int): number of cells
        ncopy_max (int): maximum copy number of a trace.
        nspot_max (int): maximum number of spots in a trace.
    """
    
    # Extract the columns from the data
    _, _, _, chroms, starts, ends, spotIDs, traceIDs, cellIDs, _ = unpack_data(data, cols)
    
    # Initialize the hashmaps and the related variables
    cell_hashmap = {}
    ncell = 0
    trace_hashmap = {}
    ncopy_max = 1
    spot_hashmap = {}
    nspot_max = 1
    
    # Fill the hashmaps by looping through the spots
    for cellID, chrom, start, end, traceID, spotID in zip(cellIDs, chroms, starts, ends, traceIDs, spotIDs):
        domain = (chrom, start, end)
        cell_hashmap, ncell = process_cell_hashmap(cellID, cell_hashmap, ncell)
        trace_hashmap, ncopy_max = process_trace_hashmap(cellID, chrom, traceID, trace_hashmap, ncopy_max)
        spot_hashmap, nspot_max = process_spot_hashmap(cellID, domain, traceID, spotID, spot_hashmap, nspot_max)
    
    return cell_hashmap, trace_hashmap, spot_hashmap, ncell, ncopy_max, nspot_max

def process_cell_hashmap(cellID, cell_hashmap, ncell):
    """Adds a cellID to the cell hashmap if it is not present.
    If it is already present, does nothing.
    Also increments the number of cells if necessary.
    """
    if cellID not in cell_hashmap:
            cell_hashmap[cellID] = ncell
            ncell += 1
    return cell_hashmap, ncell

def process_trace_hashmap(cellID, chrom, traceID, trace_hashmap, ncopy_max):
    """Adds a traceID to the trace hashmap if it is not present.
    If it is already present, does nothing.
    Also increments the maximum copy number if necessary.
    """
    # Case 1: cellID is not yet in the hashmap
    if cellID not in trace_hashmap:
        trace_hashmap[cellID] = {chrom: {traceID: 0}}
    # Case 2: cellID is in the hashmap but chrom is not
    elif chrom not in trace_hashmap[cellID]:
        trace_hashmap[cellID][chrom] = {traceID: 0}
    # Case 3: cellID and chrom are in the hashmap, but traceID is not
    elif traceID not in trace_hashmap[cellID][chrom]:
        # Add the traceID to the hashmap with the next available ID
        total_traces = len(trace_hashmap[cellID][chrom])
        trace_hashmap[cellID][chrom][traceID] = total_traces
        # Update the maximum copy number if necessary
        if total_traces + 1 > ncopy_max:
            ncopy_max = total_traces + 1
    # Case 4: cellID, chrom and traceID are in the hashmap
    else:
        pass
    return trace_hashmap, ncopy_max

def process_spot_hashmap(cellID, domain, traceID, spotID, spot_hashmap, nspot_max):
    """Adds a spotID to the spot hashmap if it is not present.
    If the spotID is already present it raises an error.
    Also increments the maximum number of spots if necessary.
    """
    # Case 1: cellID is not yet in the hashmap
    if cellID not in spot_hashmap:
        spot_hashmap[cellID] = {domain: {traceID: {spotID: 0}}}
    # Case 2: cellID is in the hashmap, but domain is not
    elif domain not in spot_hashmap[cellID]:
        spot_hashmap[cellID][domain] = {traceID: {spotID: 0}}
    # Case 3: cellID and domain are in the hashmap, but traceID is not
    elif traceID not in spot_hashmap[cellID][domain]:
        spot_hashmap[cellID][domain][traceID] = {spotID: 0}
    # Case 4: cellID, domain and traceID are in the hashmap, but spotID is not
    elif spotID not in spot_hashmap[cellID][domain][traceID]:
        # Add the spotID to the hashmap with the next available ID
        total_spots = len(spot_hashmap[cellID][domain][traceID])
        spot_hashmap[cellID][domain][traceID][spotID] = total_spots
        # Update the maximum number of spots if necessary
        if total_spots + 1 > nspot_max:
            nspot_max = total_spots + 1
    # Case 5: cellID, domain, traceID and spotID are in the hashmap
    else:
        raise ValueError('Spot ID is present twice in the data!')
    return spot_hashmap, nspot_max

def extract_data(data, cols,
                 cell_hashmap, index_hashmap, trace_hashmap, spot_hashmap,
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
        spot_hashmap (dict):
                hashmap to convert spotID (along with cell/chrom/trace)
                to spot number
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
    
    # Unpack the data into arrays
    xs, ys, zs, chromstrs, starts, ends, spotIDs, traceIDs, cellIDs, lums = unpack_data(data, cols)
    
    # Loop through the spots and fill in the data
    for x, y, z, chrstr, start, end, spotID, traceID, cellID, lum in zip(xs, ys, zs,
                                                                         chromstrs, starts, ends,
                                                                         spotIDs, traceIDs, cellIDs,
                                                                         lums):
        
        # get indices from the hashmaps
        cellnum = cell_hashmap[cellID]
        domnum = index_hashmap[(chrstr, start, end)]
        tracenum = trace_hashmap[cellID][chrstr][traceID]
        spotnum = spot_hashmap[cellID][(chrstr, start, end)][traceID][spotID]
        
        # fill in the data
        coordinates[cellnum, domnum, tracenum, spotnum, :] = [x, y, z]
        intensity[cellnum, domnum, tracenum, spotnum] = lum
        # TO DECIDE: I can also compute these two as in set_manually in the CtFile class
        nspot[cellnum, domnum, tracenum] += 1
        ncopy[cellnum, domnum] = len(trace_hashmap[cellID][chrstr])
        
    return cell_labels, coordinates, intensity, nspot, ncopy

def unpack_data(data, cols):
    """Unpacks the data from the FOF-CT data into separate arrays.

    Args:
        data (np.array of str): data
        cols (np.array of str): column names

    Returns:
        xs (np.array of float64): x coordinates
        ys (np.array of float64): y coordinates
        zs (np.array of float64): z coordinates
        chromstrs (np.array of str): chromosome strings
        starts (np.array of int64): start positions
        ends (np.array of int64): end positions
        spotIDs (np.array of str): spot IDs
        traceIDs (np.array of str): trace IDs
        cellIDs (np.array of str): cell IDs
        lums (np.array of float64): spot intensities
    """
    xs = data[:, cols == 'X'].astype(np.float64).squeeze()
    ys = data[:, cols == 'Y'].astype(np.float64).squeeze()
    zs = data[:, cols == 'Z'].astype(np.float64).squeeze()
    chromstrs = data[:, cols == 'Chrom'].squeeze()
    starts = data[:, cols == 'Chrom_Start'].astype(np.int64).squeeze()
    ends = data[:, cols == 'Chrom_End'].astype(np.int64).squeeze()
    spotIDs = data[:, cols == 'Spot_ID'].squeeze()
    traceIDs = data[:, cols == 'Trace_ID'].squeeze()
    if 'Cell_ID' in cols:
        cellIDs = data[:, cols == 'Cell_ID'].squeeze()
    else:
        # if datasets has only one chromosome TraceID = CellID
        cellIDs = np.copy(traceIDs)
        warnings.warn('Cell_ID not found in FOF-CT file. Assuming Cell_ID = Trace_ID.', UserWarning)
    if 'Intensity' in cols:
        lums = data[:, cols == 'Intensity'].astype(np.float64).squeeze()
    else:
        lums = np.full(len(xs), np.nan)
    return xs, ys, zs, chromstrs, starts, ends, spotIDs, traceIDs, cellIDs, lums
    


# Final function to process the FOF-CT file into the desired format

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
    cell_hashmap, trace_hashmap, spot_hashmap, ncell, ncopy_max, nspot_max = get_hashmaps(data, cols)
    
    # GET THE GENOME AND INDEX
    genome, index = extract_genome_index(assembly, chromstr, start, end)
    ndomain = len(index)
    index_hashmap = index.get_index_hashmap()
    
    # EXTRACT THE DATA
    cell_labels, coordinates, intensity, nspot, ncopy = extract_data(data, cols,
                                                                     cell_hashmap, index_hashmap, trace_hashmap, spot_hashmap,
                                                                     ncell, ndomain, ncopy_max, nspot_max)
    
    # FREE UP MEMORY
    del cols, data, chromstr, start, end, cell_hashmap, trace_hashmap, spot_hashmap, index_hashmap
    
    return genome, index, cell_labels, coordinates, intensity, nspot, ncopy, ncell, ndomain, ncopy_max, nspot_max
