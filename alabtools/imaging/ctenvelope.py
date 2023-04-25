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
    
    def __init__(self, filename, mode='r'):
        
        # assert the input filename
        assert isinstance(filename, str), "The input filename must be a string."
        assert filename.endswith('.ctenv'), "The input filename must end with .ctenv."
        
        # assert the input mode
        assert mode in ['r', 'w'], "The input mode must be 'r' or 'w'."
        
        # set the filename and mode attributes
        self.filename = filename
        self.mode = mode
        
        if mode == 'r':
            self.load()
        
        if mode == 'w':
            self.fitted = False
            self.ct_fit = None
            self.ncell = None
            self.cell_labels = None
            self.alpha = None
            self.mesh = None
            self.volume = None
    
    def load(self):
        """Loads a CtEnvelope from a pickle file.
        """
        
        with open(self.filename, 'rb') as f:
            loaded_ctenv = pickle.load(f)
        
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
    
    def save(self):
        """Saves a CtEnvelope to a pickle file.
        """
        with open(self.filename, 'wb') as f:
            pickle.dump(self, f)
    
    def run(self, cfg):
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
        
        # open ct file
        ct = CtFile(ct_name, 'r')
        
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
                                                    args=ct.cell_labels)
        
        # Delete the temporary directory and its contents
        os.system('rm -r {}'.format(temp_dir))
        
        # Update the attributes of the current object
        self.fitted = True
        self.ct_fit = ct_name
        self.ncell = ct.ncell
        self.cell_labels = ct.cell_labels
        self.alpha = alpha
        self.mesh = mesh
        self.volume = volume
        
        # close the ct file
        ct.close()
                
        return None
    
    @staticmethod
    def reduce_alphashape(out_names, cfg, temp_dir):
        
        # get ct_name from cfg
        try:
            ct_name = cfg['ct_name']
        except KeyError:
            "ct_name not found in cfg."
        
        # open ct file
        ct = CtFile(ct_name, 'r')
        
        # check that the output size is correct
        assert len(out_names) == ct.ncell, "Number of output files does not match number of cells."
        
        # initialize the alpha, mesh and volume lists
        alpha = list()
        mesh = list()
        volume = list()
        
        # Loop over the cells (in the same order as in the ct file)
        for cellID in ct.cell_labels:
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
        
        # close the ct file
        ct.close()
                
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
        
        if not force:
            try:
                delta_alpha = cfg['fit parameters']['delta_alpha']
            except KeyError:
                raise KeyError("delta_alpha not found in cfg['fit parameters'] (required: force is False).")
            assert isinstance(delta_alpha, float), "delta alpha must be a float."
            assert delta_alpha > 0, "delta alpha must be > 0."
        
        try:
            thresh = cfg['fit parameters']['thresh']  # threshold for outlier detection
        except KeyError:
            raise KeyError("thresh not found in cfg['fit parameters'].")
        
        try:
            min_neigh = cfg['fit parameters']['min_neigh']  # min neigh for outlier detection
        except KeyError:
            raise KeyError("min_neigh not found in cfg['fit parameters'].")
        
        # load the CtFile
        ct = CtFile(ct_name, 'r')
        
        # get the cell coordinates
        crd = ct.coordinates[ct.get_cellnum(cellID),
                            :, :, :, :]  # np.array(ndomain, ncopy_max, nspot_max, 3)
        # flatten the coordinates
        crd_flat, _ = flatten_coordinates(crd)
        # remove the nan coordinates
        crd_flat_nonan = crd_flat[~np.isnan(crd_flat).any(axis=1)]
        # remove the isolated points
        crd_flat_nonan_noiso= remove_isolated(crd_flat_nonan, thresh, min_neigh)
        
        # fit the alpha-shape
        alpha, mesh = fit_alphashape(crd_flat_nonan_noiso, alpha, delta_alpha, force)
        
        # save the alpha-shape as a pickle file in the temporary directory
        out_name = os.path.join(temp_dir, '{}.pkl'.format(cellID))
        with open(out_name, 'wb') as f:
            pickle.dump({'alpha': alpha, 'mesh': mesh}, f)
        
        # close the CtFile
        ct.close()
        
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
        ct_clean.set_manually(coordinates_clean, ct.genome, ct.index, ct.cell_labels)
        
        return ct_clean
            

def fit_alphashape(points, alpha, delta_alpha, force=False):
    """
    Fits an alpha-shape to contain all the input points.
    If force is True, the alpha-shape is fitted with the input alpha value.
    Otherwise, the alpha value is found by a search algorithm starting from the input one
    and decreasing it by delta_alpha until the alpha-shape is closed.
    If at the end the alpha-shape is not closed, an error is raised.
    
    Args:
        points (numpy.ndarray([n_points, dim])): coordinates of the points to fit the alpha-shape.
        alpha (float): alpha value to be use (force=True) or initial alpha value (force=False).
        delta_alpha (float): alpha value decrease amount (if force=False).
        force (bool): if True, the alpha-shape is fitted with the input alpha value.
                    if False, the alpha value is found by a search algorithm starting from the input one.
    
    Returns:
        alpha (float): alpha value used to fit the alpha-shape.
        mesh (trimesh.Trimesh): alpha-shape fitted to the input points.
    """
    
    # If the input alpha value is provided, use it.
    if force:
        alpha_shape = alphashape.alphashape(points, alpha)
        mesh = trimesh.Trimesh(vertices=alpha_shape.vertices, faces=alpha_shape.faces, process=True)
        if not mesh.is_watertight:
            raise ValueError('The alpha-shape is not closed. Try increasing the alpha value or change force to False.')
        return alpha, mesh
    
    # Otherwise, find the alpha value by a search algorithm.
    while True:
        alpha_shape = alphashape.alphashape(points, alpha)
        mesh = trimesh.Trimesh(vertices=alpha_shape.vertices, faces=alpha_shape.faces, process=True)
        if mesh.is_watertight:  # check if the mesh is closed
            break
        alpha = alpha - delta_alpha
        if alpha <= 0:
            raise ValueError('alpha <= 0 reached but the alpha-shape is not closed.')
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
    
    # Compute the distance matrix
    dmat = distance.cdist(points, points, 'euclidean')
    # set the diagonal to nan
    np.fill_diagonal(dmat, np.nan)
    # Compute the proximity matrix
    pmat = np.zeros(dmat.shape)
    pmat[dmat <= thresh] = 1
    # Compute the number of neighbors for each locus
    neighs = np.nansum(pmat, axis=0)  # np.array(n_points)
    # Take only the points with enough neighbors
    points_noout = points[neighs >= min_neigh]
    return points_noout
