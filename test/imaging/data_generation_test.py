# This file contains the data generation for the tests of the imaging module.

import os
import numpy as np
import pickle as pkl


def write_fofct_file(filename, data):
    """Write a FoF-CT file (csv format) from the data.
    
    Saves the file in the current directory with the name 'test_fofct.csv'.

    Args:
        data (dict): Dictionary with the data to write.

    Returns:
        None
    """
    
    # create the FoF-CT file
    fofct_file = open(filename, 'w')
    
    # write the header
    fofct_file.write('#FOF-CT_version=v0.1,\n')
    fofct_file.write('#genome_assembly=hg38,\n')
    fofct_file.write('##XYZ_unit=micron,\n')
    fofct_file.write('"#experimenter_name: Francesco Musella,,,\n')
    fofct_file.write('###lab_name: Frank Alber,\n')
    fofct_file.write('#""description: test FOFCT file for CtFile reading,\n')
    fofct_file.write('##columns=(Spot_ID,Trace_ID,X,Y,Z,' +
                        'Chrom,Chrom_Start,Chrom_End,Cell_ID,' +
                        'Extra_Cell_ROI_ID,Additional_Feature)\n')
    
    # write the data
    spotID = 1
    for cellnum in range(data['ncell']):
        for domainnum in range(data['ndomain']):
            chrom = data['chromstr'][domainnum]
            chrom_start = data['start'][domainnum]
            chrom_end = data['end'][domainnum]
            for copynum in range(data['ncopy'][cellnum, domainnum]):
                for spotnum in range(data['nspot'][cellnum, domainnum, copynum]):
                    x = data['coordinates'][cellnum, domainnum, copynum, spotnum, 0]
                    y = data['coordinates'][cellnum, domainnum, copynum, spotnum, 1]
                    z = data['coordinates'][cellnum, domainnum, copynum, spotnum, 2]
                    if np.isnan(x) or np.isnan(y) or np.isnan(z):
                        raise ValueError('Coordinates must not be NaN.')
                    fofct_file.write('{},'.format(spotID))
                    fofct_file.write('{}_{}_{},'.format(cellnum, chrom, copynum))
                    fofct_file.write('{},{},{},'.format(x, y, z))
                    fofct_file.write('{},{},{},'.format(chrom, chrom_start, chrom_end))
                    fofct_file.write('{},'.format(cellnum))
                    fofct_file.write('{},'.format(cellnum**2))  # useless
                    fofct_file.write('{}'.format(cellnum**3))  # useless
                    fofct_file.write('\n')
                    spotID += 1
    
    # close the file
    fofct_file.close()
    
    return None
    

def generate_ct_data():
    """Generates random chromating tracing data for testing.
    
    For the coordinates generation, spots of different alleles are highly separated.
    In particular, they are generated in 3D boxes with distances much larger than the box size.
    
    The data consists of:
        - attributes (int): ncell, ndomain, ncopy_max, nspot_max, nspot_tot, ntrace_tot
        - genome (alabtools.Genome)
        - index (alabtools.Index)
        - ncopy (np.ndarray): ncell x ndomain
        - nspot (np.ndarray): ncell x ndomain x ncopy_max
        - coordinates (np.ndarray): ncell x ndomain x ncopy_max x nspot_max x 3

    Returns:
        data (dict): Dictionary with the data.
    """
    
    # set seed for reproducibility
    np.random.seed(0)
    
    # set genome assembly
    assembly = 'hg38'
    chroms = ['chr1', 'chr2', 'chrX']
    
    # set the attributes
    ncell = 4
    ndomain = 50
    ncopy_max = 2
    nspot_max = 4
    
    # create the index
    # first partition the domains among the chromosomes
    sizes = np.random.rand(len(chroms))  # list of # of domains for each chromosome
    sizes = ndomain * sizes / np.sum(sizes)
    sizes = np.round(sizes).astype(int)
    sizes[0] += ndomain - np.sum(sizes)  # adjust the sizes to have ndomain
    if np.any(sizes <= 0):
        raise ValueError('One of the sizes is <= 0. Try again.')
    chromstr = []
    start = []
    end = []
    for size, chrom in zip(sizes, chroms):
        for i in range(size):
            chromstr.append(chrom)
            # generate start[i] between end[-1]+1000 and end[i-1]+10000
            if i == 0:  # if i=0 there is end[-1], so start from 0
                start.append(np.random.randint(0, 10000))
            else:
                start.append(end[-1] + np.random.randint(1000, 10000))
            # generate end[i] between start[-1]+100 and start[i]+1000
            end.append(start[-1] + np.random.randint(100, 1000))
    chromstr, start, end = np.array(chromstr), np.array(start), np.array(end)
    
    # create ncopy
    # If I allows for ncopy to be 0, then there is the risk that,
    # for small data, a domain has no copy in any cell, and thus
    # that domain is missed in the test.
    # For now I am not allowing for ncopy to be 0, but this feature
    # can be added in the future.
    ncopy = np.random.randint(1, ncopy_max + 1, size=(ncell, ndomain))
    ncopy[:, chromstr == 'chrX'] = 1  # force ncopy=1 for chrX
    ncopy[:, chromstr == 'chrY'] = 1  # force ncopy=1 for chr2
    
    # create nspot
    nspot = np.zeros((ncell, ndomain, ncopy_max), dtype=int)
    for cellnum in range(ncell):
        for domainnum in range(ndomain):
            for copynum in range(ncopy[cellnum, domainnum]):
                # Same as for ncopy, I am not allowing for nspot to be 0.
                # If I allows for nspot to be 0, then there is the risk that,
                # for small data, a copy has no spot in any cell, and thus
                # that copy is missed in the test.
                nspot[cellnum, domainnum, copynum] = np.random.randint(1, nspot_max + 1)

    # create coordinates
    # create boxes to confine the coordinates of each copy, making them highly separated
    boxsize = 500  # length of the box side
    separation = 5000  # minimum distance between centroids of boxes
    centers = np.linspace(0, separation * (ncopy_max - 1), ncopy_max)  # same for x, y, and z
    # initialize coordinates
    coordinates = np.full((ncell, ndomain, ncopy_max, nspot_max, 3), np.nan)
    # loop over cells, domains, copies, and spots to generate coordinates
    for cellnum in range(ncell):
        for domainnum in range(ndomain):
            for copynum in range(ncopy[cellnum, domainnum]):
                for spotnum in range(nspot[cellnum, domainnum, copynum]):
                    x, y, z = centers[copynum] + boxsize * (np.random.rand(3) - 0.5)
                    coordinates[cellnum, domainnum, copynum, spotnum, :] = [x, y, z]
    
    # compute total number of spots
    nspot_tot = np.sum(nspot)
    
    # compute total number of traces
    ntrace_tot = 0
    # a trace is a chromosome copy in a cell
    # so we have to loop over chromosomes and check how many copies are in each cell
    for chrom in chroms:
        ncopy_chrom = ncopy[:, np.array(chromstr) == chrom]  # ncell x ndomain_chrom
        ncopy_max_chrom = np.max(ncopy_chrom, axis=1)  # ncell
        ntrace_tot += np.sum(ncopy_max_chrom)
    
    # return everything as a dictionary
    data = {
        'assembly': assembly,
        'chromstr': chromstr,
        'start': start,
        'end': end,
        'ncell': ncell,
        'ndomain': ndomain,
        'ncopy_max': ncopy_max,
        'nspot_max': nspot_max,
        'nspot_tot': nspot_tot,
        'ntrace_tot': ntrace_tot,
        'ncopy': ncopy,
        'nspot': nspot,
        'coordinates': coordinates
    }
    
    return data

if __name__ == '__main__':
    
    # set file directory as the working directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # generate data
    data = generate_ct_data()
    
    # create the FoF-CT file
    fofct_filename = 'test_fofct.csv'
    write_fofct_file(fofct_filename, data)
    
    # save data as a pickle file
    pkl_filename = 'test_data.pkl'
    with open(pkl_filename, 'wb') as f:
        pkl.dump(data, f)
    
