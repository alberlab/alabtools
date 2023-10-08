import os
import sys
import tempfile
import warnings
from functools import partial
import numpy as np
from scipy.spatial import distance
import pickle
import alphashape
import trimesh
import mrcfile
from .ctfile import CtFile
from .utils_imaging import flatten_coordinates
from alabtools.parallel import Controller


class CtEnvelope(object):
    """A class to represent the envelops fitted from a CtFile.
    
    Saved and loaded as a pickle file with the extension .ctenv.
    
    Attributes:
        filename (str): Name of the file (should end with .ctenv)
        mode (str): Mode of the file. Can be 'r' or 'w'.
        fitted (bool): True if the envelope has been fitted.
        ct_fit (str): Name of the CtFile used to fit the envelope.
        ncell (int): Number of cells.
    
    Datasets:
        cell_labels = np.array(ncell, dtype='S10')
        alpha = np.array(ncell, dtype=float)
        mesh = list(ncell, dtype=trimesh.Trimesh)
        volume = np.array(ncell, dtype=float)
    """
    
    def __init__(self, filename=None):
        
        if filename is not None:
            assert isinstance(filename, str), "filename must be a string."
            assert filename.endswith('.ctenv'), "filename must end with .ctenv."
            assert os.path.isfile(filename), "File {} not found.".format(filename)
            self.load(filename)
        
        else:
            self.fitted = False
            self.ct_fit = None
            self.ncell = None
            self.cell_labels = None
            self.alpha = None
            self.mesh = None
            self.volume = None
    
    def load(self, filename):
        """Loads a CtEnvelope from a pickle file.
        """
        
        try:
            with open(filename, 'rb') as f:
                loaded_ctenv = pickle.load(f)
        except:
            raise IOError("File {} could not be opened with pickle.".format(filename))
        
        assert hasattr(loaded_ctenv, 'fitted'), "Loaded CtEnvelope has no attribute 'fitted'."
        assert isinstance(loaded_ctenv.fitted, bool), "Loaded CtEnvelope.fitted must be a boolean."
        assert hasattr(loaded_ctenv, 'ct_fit'), "Loaded CtEnvelope has no attribute 'ct_fit'."
        assert hasattr(loaded_ctenv, 'ncell'), "Loaded CtEnvelope has no attribute 'ncell'."
        assert hasattr(loaded_ctenv, 'cell_labels'), "Loaded CtEnvelope has no attribute 'cell_labels'."
        assert hasattr(loaded_ctenv, 'alpha'), "Loaded CtEnvelope has no attribute 'alpha'."
        assert hasattr(loaded_ctenv, 'mesh'), "Loaded CtEnvelope has no attribute 'mesh'."
        assert hasattr(loaded_ctenv, 'volume'), "Loaded CtEnvelope has no attribute 'volume'."
        
        if loaded_ctenv.fitted == False:
            warnings.warn('Loaded CtEnvelope has not been fitted.')
        
        # update the attributes of the current object
        # (every object has a __dict__ attribute, which is a dictionary of its attributes.
        # In this way, we can update the attributes of the current object with the attributes
        # of the loaded object)
        self.__dict__.update(loaded_ctenv.__dict__)
    
    def save(self, filename):
        """Saves a CtEnvelope to a pickle file.
        """
        assert isinstance(filename, str), "filename must be a string."
        assert filename.endswith('.ctenv'), "filename must end with .ctenv."
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
    
    def sort_cells(self, order):
        """Orders the CtEnvelope according to the input order.

        Args:
            order (np.array(ncell, dtype=int)): New order of the cells.
                Must be a permutation of the numbers from 0 to ncell-1.
        """
        
        # assert the input order
        assert self.fitted == True, "CtEnvelope has not been fitted."
        assert len(order) == self.ncell,\
            "The length of the input order must be equal to the number of cells."
        assert np.array_equal(np.sort(order), np.arange(self.ncell)),\
            "The input order must be a permutation of the numbers from 0 to ncell-1."
        # sort the datasets
        self.cell_labels = self.cell_labels[order]
        self.alpha = self.alpha[order]
        self.mesh = [self.mesh[i] for i in order]
        self.volume = self.volume[order]
    
    def pop_cells(self, indices):
        """Removes the cells with the input indices.
        
        Args:
            indices (np.array(n, dtype=int)): Indices of the cells to remove.
                Must be a subset of the numbers from 0 to ncell-1.
        """
        
        # assert the input indices
        assert self.fitted == True, "CtEnvelope has not been fitted."
        assert len(indices) <= self.ncell,\
            "The length of the input indices must be less than or equal to the number of cells."
        for i in indices:
            assert i in np.arange(self.ncell),\
                "The input indices must be a subset of the numbers from 0 to ncell-1."
        # remove the cells
        self.cell_labels = np.delete(self.cell_labels, indices)
        self.alpha = np.delete(self.alpha, indices)
        self.mesh = [self.mesh[i] for i in range(self.ncell) if i not in indices]
        self.volume = np.delete(self.volume, indices)
        self.ncell -= len(indices)
    
    def run_alphashape(self, cfg):
        """Runs the alpha-shape algorithm.
        
        Updates the attributes of the current object.
        
        The configuration specifies the CtFile to use, the parameters of the alpha-shape
        and the controller to use.

        Args:
            cfg (dict or json): Configuration file.

        Returns:
            None
        """
        
        # get ct_name from cfg
        try:
            ct_name = cfg['ct_name']
        except KeyError:
            "ct_name not found in cfg."
        
        # open ct file and read the cell labels
        ct = CtFile(ct_name, 'r')
        cell_labels = ct.cell_labels
        ncell = ct.ncell
        ct.close()
        
        # create a temporary directory to store nodes' results
        temp_dir = tempfile.mkdtemp(dir=os.getcwd())
        sys.stdout.write("Temporary directory for nodes' results: {}\n".format(temp_dir))
        
        # create a Controller
        controller = Controller(cfg)
        
        # set the parallel and reduce tasks
        parallel_task = partial(self.parallel_alphashape,
                                cfg=cfg,
                                temp_dir=temp_dir)
        reduce_task = partial(self.reduce_alphashape,
                              cfg=cfg,
                              temp_dir=temp_dir)

        # run the parallel and reduce tasks
        alpha, mesh, volume = controller.map_reduce(parallel_task,
                                                    reduce_task,
                                                    args=cell_labels)
        
        # Delete the temporary directory and its contents
        os.system('rm -r {}'.format(temp_dir))
        
        # Update the attributes of the current object
        self.fitted = True
        self.ct_fit = ct_name
        self.ncell = ncell
        self.cell_labels = cell_labels
        self.alpha = alpha
        self.mesh = mesh
        self.volume = volume
    
    @staticmethod
    def reduce_alphashape(out_names, cfg, temp_dir):
        
        # get ct_name from cfg
        try:
            ct_name = cfg['ct_name']
        except KeyError:
            "ct_name not found in cfg."
        
        # open ct file and read the cell labels and the number of cells
        ct = CtFile(ct_name, 'r')
        ncell = ct.ncell
        cell_labels = ct.cell_labels
        ct.close()
        
        # check that the output size is correct
        assert len(out_names) == ncell, "Number of output files does not match number of cells."
        
        # initialize the alpha, mesh and volume lists
        alpha = list()
        mesh = list()
        volume = list()
        
        # Loop over the cells (in the same order as in the ct file)
        for cellID in cell_labels:
            # try to open the output file associated to cellID
            try:
                out_name = os.path.join(temp_dir, cellID + '.pkl')
                with open(out_name, 'rb') as f:
                    out = pickle.load(f)
                    alpha_cell, mesh_cell = out['alpha'], out['mesh']
            except IOError:
                raise IOError("File {} not found.".format(out_name))
            # append the alpha value and the mesh to the lists
            alpha.append(alpha_cell)
            mesh.append(mesh_cell)
            volume.append(mesh_cell.volume)
        
        # convert the lists to numpy arrays
        alpha = np.array(alpha)
        volume = np.array(volume)
                
        return alpha, mesh, volume

    @staticmethod
    def parallel_alphashape(cellID, cfg, temp_dir):
        """Parallel function to fit an alpha-shape to the coordinates of a cell.
        Data are saved in a pickle file in the temporary directory,
        and contain the alpha value used to fit the alpha-shape and the mesh.

        Args:
            cellID (str): ID of the cell.
            cfg (dict or json): Configuration of the parallel computation.
            temp_dir (str): Path to the temporary directory to store nodes' results.

        Returns:
            out_name (_type_): Name of the output file in the temporary directory.
        """
        
        # get data from the configuration
        try:
            ct_name = cfg['ct_name']
        except KeyError:
            raise KeyError("ct_name not found in cfg.")
        
        try:
            alpha = cfg['fit parameters']['alpha']
        except KeyError:
            raise KeyError("alpha not found in cfg['fit parameters'].")
        assert isinstance(alpha, float), "alpha must be a float."
        assert alpha > 0, "alpha must be > 0."
        
        try:
            force = cfg['fit parameters']['force']
        except KeyError:
            raise KeyError("force not found in cfg['fit parameters'].")
        assert isinstance(force, bool), "force must be a bool."
        
        do_remove_isolated = False
        if 'thresh' in cfg['fit parameters'] and 'min_neigh' in cfg['fit parameters']:
            do_remove_isolated = True
            thresh = cfg['fit parameters']['thresh']
            min_neigh = cfg['fit parameters']['min_neigh']
            assert isinstance(thresh, float), "thresh must be a float."
            assert thresh > 0, "thresh must be > 0."
            assert isinstance(min_neigh, int), "min_neigh must be an int."
            assert min_neigh > 0, "min_neigh must be > 0."
           
        # load the CtFile and get the coordinates of the cell
        ct = CtFile(ct_name, 'r')
        # reading in this way does not load the whole coordinates in memory, but only the cell of interest
        crd = ct['coordinates'][ct.get_cellnum(cellID), :, :, :, :]  # ndomain, ncopy_max, nspot_max, 3
        ct.close()
        
        # flatten the coordinates
        crd_flat, idx = flatten_coordinates(crd)
        # remove the nan coordinates
        points = crd_flat[~np.isnan(crd_flat).any(axis=1)]
        del crd_flat
        del idx
        # remove the isolated points
        if do_remove_isolated:
            points = remove_isolated(points, thresh, min_neigh)
        
        # fit the alpha-shape
        alpha, mesh = fit_alphashape(cellID, points, alpha, force)
        del points
        
        # save the alpha-shape as a pickle file in the temporary directory
        out_name = os.path.join(temp_dir, '{}.pkl'.format(cellID))
        with open(out_name, 'wb') as f:
            pickle.dump({'alpha': alpha, 'mesh': mesh}, f)
        del mesh
        
        return out_name
    
    def remove_outliers(self, ct):
        """Given a CtFile, remove the outliers from its data.
        
        The outliers are defined as points that - in each respective cell - are outside the mesh.

        Args:
            ct (CtFile): CtFile to clean.
        
        Returns:
            ct_clean (CtFile): CtFile without outliers.
        """
        
        # check that the alpha-shape is fitted
        assert self.fitted, "The alpha-shape must be fitted before removing outliers."
        # check that the cell labels match
        assert np.array_equal(self.cell_labels, ct.cell_labels),\
            "The cell labels of the CtFile do not match the cell labels of the CtEnvelope."
        
        # initialize the cleaned coordinates
        coordinates_clean = np.full(ct.coordinates.shape, np.nan)
        
        # loop over the cells
        for cellnum in range(self.cell_labels):
            
            # get the alpha-shape mesh
            mesh = self.mesh[cellnum]
            
            # get the cell coordinates
            crd = ct.coordinates[cellnum, :, :, :, :]
            # flatten the coordinates of the CtFile
            crd_flat, idx_flat = flatten_coordinates(crd)
            # remove the nan coordinates
            crd_flat_nonan = crd_flat[~np.isnan(crd_flat).any(axis=1)]
            idx_flat_nonan = idx_flat[~np.isnan(crd_flat).any(axis=1)]
            # remove the points outside the alpha-shape
            inside = mesh.contains(crd_flat_nonan)
            crd_flat_nonan_noout = crd_flat_nonan[inside]
            idx_flat_nonan_noout = idx_flat_nonan[inside]
                              
            # loop over the points and fill the cleaned coordinates
            # with the points inside the alpha-shape
            for w, ijk in enumerate(idx_flat_nonan_noout):
                coordinates_clean[tuple((cellnum, *ijk))] = crd_flat_nonan_noout[w]
        
        # create the cleaned CtFile
        ct_clean = CtFile(ct.name.replace('.ct', '_clean.ct'), 'w')
        # set the cleaned coordinates
        ct_clean.set_manually(coordinates_clean, ct.genome, ct.index, ct.cell_labels)
        # sort and trim
        ct_clean.sort_copies()
        ct_clean.sort_spots()
        ct_clean.trim()
        
        return ct_clean
    
    def run_mrc(self, cfg):
        """Compures the MRC files for each cell."""
        
        # Check that the required keys are in the configuration
        required_keys = ['ct_name', 'mrc parameters', 'parallel']
        for key in required_keys:
            assert key in cfg, "Key {} not found in the cfg.".format(key)
        required_mrc_keys = ['resolution', 'border', 'surface_thickness']
        for key in required_mrc_keys:
            assert key in cfg['mrc parameters'], "Key {} not found in the cfg['mrc parameters'].".format(key)

        # open ct file and read the cell labels
        with CtFile(cfg['ct_name'], 'r') as ct:
            cell_labels = ct.cell_labels
        
        # create a directory to store mrc files
        mrc_dir = os.path.join(os.getcwd(), 'mrc')
        os.makedirs(mrc_dir, exist_ok=True)
        sys.stdout.write("Directory for MRC files: {}\n".format(mrc_dir))
        
        # Save the mesh list as a pickle file in the mrc directory
        with open(os.path.join(mrc_dir, 'mesh.pkl'), 'wb') as f:
            pickle.dump(self.mesh, f)
        
        # create a Controller
        controller = Controller(cfg)
        
        # set the parallel and reduce tasks
        parallel_task = partial(self.parallel_mrc, cfg=cfg, mrc_dir=mrc_dir)
        reduce_task = self.reduce_mrc

        # run the parallel and reduce tasks
        origins, low_edges = controller.map_reduce(parallel_task, reduce_task, args=cell_labels)
        
        # Delete the mesh pickle file
        os.system('rm -r {}'.format(os.path.join(mrc_dir, 'mesh.pkl')))
        
        # Write a MRC metadata file as a pickled dictionary
        mrc_meta = {'resolution': cfg['mrc parameters']['resolution'],
                    'cell_labels': cell_labels,
                    'origins': origins,
                    'low_edges': low_edges}
        with open(os.path.join(mrc_dir, 'mrc_meta.pkl'), 'wb') as f:
            pickle.dump(mrc_meta, f)
        del origins, low_edges, mrc_meta
    
    @staticmethod
    def reduce_mrc(parallel_outs):
        # Get the cell IDs, cell numbers and origins from the parallel outputs
        cellnums, origins, lower_edges = zip(*parallel_outs)
        cellnums = np.array(cellnums).astype(int)
        origins = np.array(origins).astype(int)
        lower_edges = np.array(lower_edges).astype(float)
        # Sort origins according to the cell numbers
        sort_idx = np.argsort(cellnums)
        origins = origins[sort_idx]
        lower_edges = lower_edges[sort_idx]
        return origins, lower_edges
    
    @staticmethod
    def parallel_mrc(cellID, cfg, mrc_dir):
        # Read the cell number and the XYZ range from the CtFile
        with CtFile(cfg['ct_name'], 'r') as ct:
            cellnum = ct.get_cellnum(cellID)
            xyz_range = get_xyz_range(ct, cellnum)
        # Read the mesh from the pickle file
        with open(os.path.join(mrc_dir, 'mesh.pkl'), 'rb') as f:
            mesh = pickle.load(f)[cellnum]
        # Read the MRC parameters from the configuration
        resolution = cfg['mrc parameters']['resolution']  # in same physical units as data (e.g. nm)
        border = cfg['mrc parameters']['border']  # in voxels (white space around the cell)
        surface_thickness = cfg['mrc parameters']['surface_thickness']  # in same physical units as data (e.g. nm)
        # If resolution is a number, we use the same resolution for all the axes
        if isinstance(resolution, (int, float)):
            resolution = np.array([resolution, resolution, resolution])
        # Create the grid edges (physical units)
        grid_edg = create_grid_edges(xyz_range, resolution, border)
        # Set the origin of the MRC file (in voxels)
        origin = - np.round(grid_edg[:, 0] / resolution).astype(int)
        # Create the XYZ grid of points
        xyz_grid, grid_shape = create_xyz_grid(grid_edg, resolution)
        # Use mesh to compute containment matrix
        contained = mesh.contains(xyz_grid)
        contained = np.reshape(contained, grid_shape).astype(int)
        # Write the MRC file
        write_mrc(os.path.join(mrc_dir, '{}.mrc'.format(cellnum)), contained, tuple(origin), tuple(resolution))
        del contained
        # Use mesh to compute the surface of the cell
        dists = trimesh.proximity.signed_distance(mesh, xyz_grid)
        dists = np.reshape(dists, grid_shape)
        on_surface = (np.abs(dists) < surface_thickness).astype(int)
        write_mrc(os.path.join(mrc_dir, '{}.surface.mrc'.format(cellnum)), on_surface, tuple(origin), tuple(resolution))
        del on_surface, dists, xyz_grid
        # Return the origin of the MRC file and the lower left corner of the grid
        return cellnum, origin, grid_edg[:, 0]
            

def fit_alphashape(cellID, points, alpha, force=False):
    """
    Fits an alpha-shape to contain all the input points.
    If force is True, the alpha-shape is fitted with the input alpha value.
    Otherwise, the alpha value is found by a search algorithm starting from the input one
    and decreasing it by delta_alpha until the alpha-shape is closed.
    If at the end the alpha-shape is not closed, an error is raised.
    
    Args:
        points (numpy.ndarray([n_points, dim])): coordinates of the points to fit the alpha-shape.
        alpha (float): alpha value to be use (force=True) or initial alpha value (force=False).
        force (bool): if True, the alpha-shape is fitted with the input alpha value.
                    if False, the alpha value is found by a search algorithm starting from the input one.
    
    Returns:
        alpha (float): alpha value used to fit the alpha-shape.
        mesh (trimesh.Trimesh): alpha-shape fitted to the input points.
    """
    # The alphashape code doesn't give closed shapes if the input points are not float64
    points = points.astype(np.float64)
    # If force, we only use the input alpha value
    if force:
        alpha_shape = alphashape.alphashape(points, alpha)
        mesh = trimesh.Trimesh(vertices=alpha_shape.vertices, faces=alpha_shape.faces, process=True)
        if not mesh.is_watertight:
            raise ValueError('Cell {}: The alpha-shape is not closed.\
                Try increasing the alpha value or change force to False.'.format(cellID))
        return alpha, mesh
    # If not force, we find the alpha value by a search algorithm,
    # where we start with the input alpha and - if the shape is not closed - we halve it.
    max_iter, counter = 20, 0
    while True:
        counter += 1
        if counter > max_iter:
            raise ValueError('Cell {}: Alpha-shape not closed, could not find a suitable alpha value after {} iterations.'.format(cellID, max_iter))
        alpha_shape = alphashape.alphashape(points, alpha)
        mesh = trimesh.Trimesh(vertices=alpha_shape.vertices, faces=alpha_shape.faces, process=True)
        if mesh.is_watertight:
            break
        alpha = alpha / 2
    return alpha, mesh

def remove_isolated(points, thresh, min_neigh):
    """    
    Removes the isolated points from the input points (likely outliers).
    
    For each point, counts the number of points in a neighborhood of fixed radius.
    Then, removes the points with a number of neighbors below a threshold.
    
    Args:
        points (numpy.ndarray([n_points, dim])): coordinates of the points to fit the alpha-shape.
        thresh (float): radius of the sphere where we look for other points.
        min_neigh (int): minimum number of points required to be in the sphere.
    
    Returns:
        points_noout (numpy.ndarray([n_points_noout, dim])): coordinates of the points without outliers.
    """
    # Compute the condensed distance matrix (n * (n-1) / 2)
    # By default, scipy casts the input array to float64,
    # so this might require a lot of memory for large arrays.
    dcmat = distance.pdist(points)
    # Compute the condensed proximity matrix (boolean)
    pcmat = dcmat <= thresh
    del dcmat
    # Expand the proximity matrix to a square matrix
    pmat = distance.squareform(pcmat)
    del pcmat
    # Compute the number of neighbors for each locus
    neighs = np.nansum(pmat, axis=0)  # np.array(n_points)
    del pmat
    # Take only the points with enough neighbors
    points_noout = points[neighs >= min_neigh]
    del neighs
    return points_noout

def get_xyz_range(ct, cellnum):
    """Get the range of the X, Y and Z coordinates of a cell
    from the CtFile object."""
    xyz_range = np.zeros((3, 2)).astype(float)
    for i in range(3):
        x_i = ct['coordinates'][cellnum, :, :, :, i]
        xyz_range[i, 0] = np.nanmin(x_i)
        xyz_range[i, 1] = np.nanmax(x_i)
        del x_i
    return xyz_range

def create_grid_edges(xyz_range, res, border):
    """Create the edges of the grid that will be used to compute the MRC."""
    grid_edg = np.zeros((3, 2)).astype(float)
    for i in range(3):
        grid_edg[i, 0] = res[i] * xyz_range[i, 0] // res[i] - border * res[i]
        grid_edg[i, 1] = res[i] * xyz_range[i, 1] // res[i] + border * res[i]
    return grid_edg

def create_xyz_grid(grid_edg, res):
    """Create the 3D grid of points that will be used to compute the MRC."""
    x_grid = np.arange(grid_edg[0, 0], grid_edg[0, 1], res[0])
    y_grid = np.arange(grid_edg[1, 0], grid_edg[1, 1], res[1])
    z_grid = np.arange(grid_edg[2, 0], grid_edg[2, 1], res[2])
    xyz_grid = list()
    for x in x_grid:
        for y in y_grid:
            for z in z_grid:
                xyz_grid.append(np.array([x, y, z]))
    xyz_grid = np.array(xyz_grid)
    shape = (len(x_grid), len(y_grid), len(z_grid))
    return xyz_grid, shape

def write_mrc(filename, data, origin=(0, 0, 0), voxel_size=(1, 1, 1)):
    """Write a MRC file from a numpy array.
    
    Args:
        filename (str): name of the file to be written.
        data (np.array(shape=(n_x_grid, n_y_grid, n_z_grid))): grid of values (0 or 1)
    """
    # Swap the axes to match the MRC format
    data = np.swapaxes(data, 0, 2)
    # Ensure the data is in int8 format as we'll use MODE 0
    data = data.astype(np.int8)
    # Create a new MRC file and save the data
    with mrcfile.new(filename, overwrite=True) as mrc:
        mrc.set_data(data)
        mrc.nstart = origin
        mrc.voxel_size = voxel_size
