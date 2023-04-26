# Contains functions to read FOF-CT (4DN Fish Omics Format - Chromatin Tracing) files.

import warnings
import numpy as np

def read_fofct(fofct_file):
    """
    Read the FOF-CT (4DN Fish Omics Format - Chromatin Tracing) csv file.
    
    Get the genome assembly, column names and data from the FOF-CT file.
    
    Args:
        focal_file (str): path to the csv file
    
    Returns:
        assembly (list of str): assembly names
        col_names (np.array of str): column names
        data (np.array of str): data
    """

    # READ THE FILE
    csv = open(fofct_file, 'r')
    lines = csv.readlines()
    csv.close()

    # SEPARATE HEADER AND DATA
    i_header_stop = 0  # index of the last line of the header
    for line in lines:
        if line[0] == '#' or line[1] == '#':  # header lines can start with #, ## or "#
            i_header_stop += 1
        elif 'Trace_ID' in line:
            # This is to deal with files that just have a column_names line like
            #  Spot_ID,Trace_ID,X,Y,Z,Chrom,Chrom_Start,Chrom_End
            i_header_stop += 1
        else:
            break  # header ends
    header_lines = lines[:i_header_stop]
    data_lines = lines[i_header_stop:]
            
    if len(header_lines) == 0:
        raise ValueError('The header is empty.')
    if len(data_lines) == 0:
        raise ValueError('The data is empty.')

    # CLEAN THE HEADER
    header = []
    for line in header_lines:
        while line[0] == '#' or line[0] == '"':
            line = line[1:]  # remove special characters at the beginning of the line (e.g. #, ##, ")
        if line[-1] == '\n':
            line = line[:-1]  # if the line ends with a line break, remove it
        while line[-1] == ',':
            line = line[:-1]  # if the line ends with commas, remove all of them
        while line[-1] == '"':
            line = line[:-1]  # if the line ends with quotes, remove all of them
        header.append(line)

    # READ ASSEMBLY AND COLUMN NAMES FROM HEADER
    assembly = None
    col_names = None
    for line in header:
        line = line.replace(' ', '')  # remove all spaces
        if 'genome_assembly' in line:  # read the genome assembly
            # Some datasets have multiple assemblies separated by /
            line = line.split('=')[1]  # split the line at the equal sign and take the second part
            assembly = line.split('/')  # take both assemblies if / is present
        if 'columns' in line:
            line = line.split('=')[1]  # split the line at the equal sign and take the second part
            if line[0] == '(':
                line = line[1:]  # if there is a '(' at the beginning, remove it
            if line[-1] == ')':
                line = line[:-1]  # if there is a ')' at the end, remove it
            col_names = line.split(',')  # split the line at the commas
            col_names = np.array(col_names)
        if 'Trace_ID' in line and '=' not in line:
            # This is to deal with files that just have a column_names line like
            #   Spot_ID,Trace_ID,X,Y,Z,Chrom,Chrom_Start,Chrom_End
            # so they are missing the 'columns=' part
            if line[0] == '(':
                line = line[1:]  # if there is a '(' at the beginning, remove it
            if line[-1] == ')':
                line = line[:-1]  # if there is a ')' at the end, remove it
            col_names = line.split(',')  # split the line at the commas
            col_names = np.array(col_names)
    if assembly is None:
        raise ValueError('The genome assembly is not specified in the header.')
    if col_names is None:
        raise ValueError('The column names are not specified in the header.')
    
    # CHECK IF COLUMN NAMES ARE CORRECT
    if 'X' not in col_names or 'Y' not in col_names or 'Z' not in col_names \
        or 'Chrom' not in col_names or 'Chrom_Start' not in col_names or 'Chrom_End' not in col_names \
            or 'Trace_ID' not in col_names:
                raise ValueError('FOF-CT file is not in the correct format.')

    # CONVERT DATA TO NUMPY ARRAY
    data = []
    for line in data_lines:
        if line[-1] == '\n':
            line = line[:-1]  # if the line ends with a line break, remove it
        line = line.replace(' ', '')  # remove all spaces
        values = line.split(',')  # split the line at the commas
        data.append(values)
    data = np.array(data)
    data[data == ''] = np.nan  # replace empty values with NaN

    return assembly, col_names, data


def get_domains_fofct(data, col_names):
    """
    Get the domains from the FOF-CT data.
    The domains are found as unique ordered tuples (chr, start, end).
    The domains are then ordered by chromosome first, then by start position.
    
    Args:
        data (np.array of str): data (from read_fofct)
        col_names (np.array of str): column names (from read_fofct)
    
    Returns:
        chrstr (np.array of str): chromosome strings
        start (np.array of int): start positions
        end (np.array of int): end positions
    """
    
    # GET THE SPOTS' CHROMOSOME, START AND END POSITIONS
    spots_chrstr = data[:, col_names == 'Chrom']
    spots_start = data[:, col_names == 'Chrom_Start']
    spots_end = data[:, col_names == 'Chrom_End']
    
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

def get_processed_data_fofct(data, col_names, genome, index):
    """
    Gets the processed data from the FOF-CT file (a table of strings, numpy object array),
    i.e. converts it to the numpy arrays of interest.
    
    HDF5 only supports homogeneous arrays (i.e. hyperrectangulars), but coord and nspot
    are heterogeneous (e.g. multiple spots per domain). We solve this issue by max-padding.
    
    Args:
        data (np.array of str): data (from read_fofct)
        col_names (np.array of str): column names (from read_fofct)
        genome (alabtools.utils.Genome)
        index (alautils.utils.Index)
        
    @param data: np.array of str (data)
            col_names: np.array of str (column names)
    @return: 
    """
    
    # READ ALL ARRAYS FROM DATA
    x = data[:, col_names == 'X'].transpose()[0].astype(np.float64)
    y = data[:, col_names == 'Y'].transpose()[0].astype(np.float64)
    z = data[:, col_names == 'Z'].transpose()[0].astype(np.float64)
    chrstr = data[:, col_names == 'Chrom'].transpose()[0]
    start = data[:, col_names == 'Chrom_Start'].transpose()[0].astype(np.int64)
    end = data[:, col_names == 'Chrom_End'].transpose()[0].astype(np.int64)
    traceID = data[:, col_names == 'Trace_ID'].transpose()[0]
    if 'Cell_ID' in col_names:
        cellID = data[:, col_names == 'Cell_ID'].transpose()[0]
    else:
        # if Cell_ID is not present, assume Cell_ID = Trace_ID
        # these should be the cases where imaging is performed only on one chromosome,
        # and thus there is no need to distinguish between chromosomes in the same cell.
        cellID = np.copy(traceID)
        warnings.warn('Cell_ID not found in FOF-CT file. Assuming Cell_ID = Trace_ID.', UserWarning)
    
    # GET ATTRIBUTES
    ncell = len(np.unique(cellID))
    nspot_tot = np.sum(~np.isnan(x))
    ntrace_tot = len(np.unique(traceID))
    cell_labels = np.unique(cellID)
    
    # CREATE COORD, NSPOT, NCOPY ARRAYS
    coord = []  # coordinates of spots (cell -> domain -> copy -> spot -> coord)
    nspot = []  # number of spots (cell -> domain -> copy -> nspot)
    ncopy = []  # number of copies (cell -> domain -> ncopy)
    nspot_max = 0  # maximum number of spots in a copy (for max-padding)
    ncopy_max = 0  # maximum number of copies in a domain (for max-padding)
    # Loop over: cell -> domain -> copy -> spot -> coord (x, y, z)
    for cell in cell_labels:
        # get cell data
        x_cell = x[cellID == cell]
        y_cell = y[cellID == cell]
        z_cell = z[cellID == cell]
        chrstr_cell = chrstr[cellID == cell]
        start_cell = start[cellID == cell]
        end_cell = end[cellID == cell]
        traceID_cell = traceID[cellID == cell]
        coord_cell = []  # coordinates for a cell
        nspot_cell = []  # number of spots for a cell
        ncopy_cell = []  # number of copies for a cell
        for dom in zip(index.chromstr, index.start, index.end):
            if dom[0] not in genome.chroms:
                warnings.warn("{} not present in Genome. Removed.".format(dom[0]), UserWarning)
                continue
            # get domain data (within cell)
            x_dom = x_cell[(chrstr_cell == dom[0]) & (start_cell == dom[1]) & (end_cell == dom[2])]
            y_dom = y_cell[(chrstr_cell == dom[0]) & (start_cell == dom[1]) & (end_cell == dom[2])]
            z_dom = z_cell[(chrstr_cell == dom[0]) & (start_cell == dom[1]) & (end_cell == dom[2])]
            traceID_dom = traceID_cell[(chrstr_cell == dom[0]) & (start_cell == dom[1]) & (end_cell == dom[2])]
            coord_dom = []  # coordinates for a domain in a cell
            nspot_dom = []  # number of spots for a domain in a cell
            ncopy_dom = len(np.unique(traceID_dom))  # number of copies of a domain in a cell
            if ncopy_dom > ncopy_max:
                ncopy_max = ncopy_dom  # update ncopy_max
            for trc in np.unique(traceID_dom):  # Here trace and copy are used as synonyms
                # get trace data (within cell/domain)
                x_trc = x_dom[traceID_dom == trc]
                y_trc = y_dom[traceID_dom == trc]
                z_trc = z_dom[traceID_dom == trc]
                # I checked that some data have NaNs in x, y, z. To be safe here I remove them
                x_trc_nonan = x_trc[~np.isnan(x_trc) & ~np.isnan(y_trc) & ~np.isnan(z_trc)]
                y_trc_nonan = y_trc[~np.isnan(x_trc) & ~np.isnan(y_trc) & ~np.isnan(z_trc)]
                z_trc_nonan = z_trc[~np.isnan(x_trc) & ~np.isnan(y_trc) & ~np.isnan(z_trc)]
                coord_trc = []  # coordinates for a copy in a domain in a cell
                nspot_trc = len(x_trc_nonan)
                if nspot_trc > nspot_max:
                    nspot_max = nspot_trc  # update nspot_max
                for xx, yy, zz in zip(x_trc_nonan, y_trc_nonan, z_trc_nonan):  # loop over spots
                    # append spot to trace
                    coord_trc.append([xx, yy, zz])
                # append trace to domain
                coord_dom.append(coord_trc)
                nspot_dom.append(nspot_trc)
            # append domain to cell
            coord_cell.append(coord_dom)
            nspot_cell.append(nspot_dom)
            ncopy_cell.append(ncopy_dom)
        # append cell to arrays
        coord.append(coord_cell)
        nspot.append(nspot_cell)
        ncopy.append(ncopy_cell)
    
    # CREATE HOMOGENEOUS ARRAYS (coord and nspot)
    ncopy = np.array(ncopy).astype(np.int64)  # already homogeneous
    ndomain = len(index)
    coord_homo = np.nan * np.zeros((ncell, ndomain, ncopy_max, nspot_max, 3), dtype=np.float64)
    nspot_homo = np.zeros((ncell, ndomain, ncopy_max), dtype=np.int64)
    for cell in range(ncell):  # Loop to go heterogeneous -> homogeneous
        for dom in range(ndomain):
            for trc in range(ncopy[cell, dom]):
                nspot_homo[cell, dom, trc] = nspot[cell][dom][trc]
                for spot in range(nspot[cell][dom][trc]):
                    for i in range(3):
                        coord_homo[cell, dom, trc, spot, i] = coord[cell][dom][trc][spot][i]
    # final cast
    coord = coord_homo
    nspot = nspot_homo.astype(np.int64)
    
    return ncell, nspot_tot, ntrace_tot, nspot_max, ncopy_max, cell_labels, coord, nspot, ncopy
