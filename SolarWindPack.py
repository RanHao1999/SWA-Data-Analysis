# In this script, we store the classes and functions for the SolarWindPack.
# Author = Hao Ran

import numpy as np
import copy
from Funcs import *

import numpy as np
import copy

class SolarWindParticle:
    """
    This is an universal class for handling solar wind particle species.
    We can set time, magnetic field, VDFs, and measurement grids for different species.
    """

    def __init__(self, species, time=None, magfield=None, vdf=None, grid=None, coord_type="Spherical"):
        self.species = species.lower()
        self.time = time
        self.magfield = magfield

        # Set coord type.
        if coord_type not in ("Spherical", "3D Field-Aligned", "2D Field-Aligned"):
            raise ValueError("coord_type should only be 'Spherical' or '3D Field-Aligned', or '2D Field-Aligned'")
        self.coord_type = coord_type

        # set coordinate grid.
        if grid is None:
            self.grid = None
        else:
            self.set_grid(grid, coord_type)

        self.vdf = vdf
    
    # Functions for setting some general properties.
    def set_time(self, time):
        self.time = time

    def set_magfield(self, magfield):
        self.magfield = magfield

    def set_grid(self, grid, coord_type=None):
        """
        Set or update the measurement grid for the particel species.
        """
        if coord_type:
            if coord_type not in ("Spherical", "3D Field-Aligned", "2D Field-Aligned"):
                raise ValueError("coord_type should only be 'Spherical' or '3D Field-Aligned', or '2D Field-Aligned'")
            self.coord_type = coord_type
        
        if self.coord_type == "Spherical":
            self.grid = {
                "elevation": grid[0], 
                "azimuth": grid[1],
                "velocity": grid[2]
            }
        elif self.coord_type == "3D Field-Aligned":
            self.grid = {
                "Vpara": grid[0],
                "Vperp1": grid[1],
                "Vperp2": grid[2]
            }
        elif self.coord_type == "2D Field-Aligned":
            self.grid = {
                "Vpara": grid[0],
                "Vperp": grid[1]
            }

    def set_vdf(self, vdf, component=None):
        """
        Set a VDF for the species.
        If components are specified, store VDFs for each component.
        Otherwise, store as a single VDF.
        """
        allowed_components = {
            'proton': ['core', 'beam'],
            'alpha': ['core', 'beam'],
            'electron': ['core', 'strahl', 'halo']
        }

        if component is None:
            self.vdf = np.array(vdf)
        else:
            if self.species not in allowed_components:
                raise ValueError(f"Species '{self.species}' is not supported.")
            if component not in allowed_components[self.species]:
                raise ValueError(f"Invalid component: {component}. Use {allowed_components[self.species]}")
            
            if self.vdf is None or not isinstance(self.vdf, dict):
                self.vdf = {comp: None for comp in allowed_components[self.species]}
            
            self.vdf[component] = np.array(vdf)
    
    def get_vdf(self, component=None):
        """
        Get a VDF for the species or its components.
        If no component is specified, return the sum of all components if they exist.
        """
        allowed_components = {
            'proton': ["core", "beam"],
            'alpha': ["core", "beam"],
            'electron': ["core", "strahl", "halo"]
        }

        if isinstance(self.vdf, dict):
            if component:
                if self.species in allowed_components:
                    if component not in allowed_components[self.species]:
                        raise ValueError(f"Invalid component '{component}' for {self.species}. "
                                         f"Allowed components are {allowed_components[self.species]}.")
                else:
                    raise ValueError(f"Components are not defined for species '{self.species}'.")

                if component not in self.vdf:
                    raise ValueError(f"Component '{component}' not found for {self.species}.")
                
                if self.vdf[component] is None:
                    raise ValueError(f"VDF for component '{component}' of {self.species} is not set.")
                
                return self.vdf[component]
            else:
                combined_vdf = None
                for comp_vdf in self.vdf.values():
                    if comp_vdf is not None:
                        if combined_vdf is None:
                            combined_vdf = comp_vdf.copy()
                        else:
                            combined_vdf += comp_vdf
                if combined_vdf is None:
                    raise ValueError(f"No VDF components set for {self.species}.")
                return combined_vdf
        else:
            if component is not None:
                raise ValueError(f"Invalid component '{component}' for {self.species}. This species does not have sub-components.")
            return self.vdf
        
    def copy(self):
        return copy.deepcopy(self)


    def __repr__(self):
        """
        Provide a summary of the SolarWindParticle object, showing species, time, magnetic field, and the status of VDFs.
        """        
        def get_status(vdf):
            if vdf is None:
                return "not set"
            if isinstance(vdf, dict):
                return ", ".join([f"{comp}: {'set' if v is not None else 'not set'}" for comp, v in vdf.items()])
            return "single VDF set"

        vdf_status = get_status(self.vdf)
        grid_status = (f"Coordinate Grids Set in {self.coord_type}" if self.grid 
                       else "Coordinate Grids not set")
        
        return (f"Species = {self.species}, "
                f"Time = {self.time}, "
                f"Magnetic Field = {self.magfield}, "
                f"VDF = {vdf_status}, "
                f"{grid_status}")



def transferToFieldAligned(SW_spherical, transfer_Matrix, VPbulk_SRF=np.array([0.0, 0.0, 0.0]), coord_type="3D Field-Aligned"):
    """
    Transfer the VDFs from spherical to Catesian coordinate in Proton rest frame. (Para, Perp1, Perp2)
    V_bulk: if not specified, then do not move the frame.
    """

    def GetVelocityVector(theta, phi, velocity):
        phi_grid, theta_grid, velocity_grid = np.meshgrid(np.radians(phi), np.radians(theta), velocity, indexing='ij')
        vx = velocity_grid * np.cos(theta_grid) * np.cos(phi_grid)
        vy = velocity_grid * np.cos(theta_grid) * np.sin(phi_grid)
        vz = velocity_grid * np.sin(theta_grid)
        return vx, vy, vz

    def Transfer2SRF(transferrMatrixx, Vx, Vy, Vz):
        Vx_SRF = transferrMatrixx[0, 0] * Vx + transferrMatrixx[0, 1] * Vy + transferrMatrixx[0, 2] * Vz
        Vy_SRF = transferrMatrixx[1, 0] * Vx + transferrMatrixx[1, 1] * Vy + transferrMatrixx[1, 2] * Vz
        Vz_SRF = transferrMatrixx[2, 0] * Vx + transferrMatrixx[2, 1] * Vy + transferrMatrixx[2, 2] * Vz
        return Vx_SRF, Vy_SRF, Vz_SRF   
    
    # Get parameters from the input SW object.
    theta = SW_spherical.grid['elevation']
    phi = SW_spherical.grid['azimuth']
    velocity = SW_spherical.grid['velocity']
    magF_SRF = SW_spherical.magfield    

    vx, vy, vz = GetVelocityVector(theta, phi, velocity)

    # Transfer the VDFs to SRF coordinates.
    vx_SRF, vy_SRF, vz_SRF = Transfer2SRF(transfer_Matrix, vx, vy, vz)

    # To proton-rest frame.
    vx_SRF -= VPbulk_SRF[0]
    vy_SRF -= VPbulk_SRF[1]
    vz_SRF -= VPbulk_SRF[2]

    # transfer to field-aligned coordinates.
    (Nx, Ny, Nz, Px, Py, Pz, Qx, Qy, Qz) = fieldAlignedCoordinates(magF_SRF[0], magF_SRF[1], magF_SRF[2])
    Vpara, Vperp1, Vperp2 = rotateVectorIntoFieldAligned(vx_SRF, vy_SRF, vz_SRF, Nx, Ny, Nz, Px, Py, Pz, Qx, Qy, Qz)

    SolarWind_Cartesian = SW_spherical.copy()
    SolarWind_Cartesian.set_grid([Vpara, Vperp1, Vperp2], coord_type=coord_type)
    return SolarWind_Cartesian

def FieldAligned3D_To_2D(SW_FieldAligned_3D, component=None, ParaBins=100, PerpBins=50):
    # Turn a 3D field-aligned VDF to 2D field-aligned VDF.
    # Useful when want to plot or calculate 2D moments.
    # Please note that the 2D VDF is not the same as the original 3D VDF, many points are missed.
    # Caution when use this for further processing!

    # Get grids from the input SW object.
    Vpara = SW_FieldAligned_3D.grid['Vpara']
    Vperp1 = SW_FieldAligned_3D.grid['Vperp1']
    Vperp2 = SW_FieldAligned_3D.grid['Vperp2']
    Vperp = np.sqrt(Vperp1**2 + Vperp2**2)

    # VDF from the input SW object.
    VDF = SW_FieldAligned_3D.get_vdf(component)

    Vpara_flat = Vpara.flatten()
    Vperp_flat = Vperp.flatten()
    VDF_flat = VDF.flatten()

    vpar_min, vpar_max = Vpara.min(), Vpara.max()
    vperp_min, vperp_max = Vperp.min(), Vperp.max()

    # 2D on a 257*129 grid.
    # User can adjust this grid size.
    # Will return a 256*128 grid.
    vpar_bins = np.linspace(vpar_min, vpar_max, ParaBins)
    vperp_bins = np.linspace(vperp_min, vperp_max, PerpBins)
    vpar_grid, vperp_grid = np.meshgrid(vpar_bins, vperp_bins, indexing='ij')

    # Regrid
    H, Xedges, Yedges = np.histogram2d(Vpara_flat, Vperp_flat, bins=(vpar_bins, vperp_bins), weights=VDF_flat)
    counts, _, _ =np.histogram2d(Vpara_flat, Vperp_flat, bins=(vpar_bins, vperp_bins))
    H_avg = np.divide(H, counts, out=np.zeros_like(H), where = counts != 0)

    # New SW_Particle object.
    SW_FieldAligned_2D = SW_FieldAligned_3D.copy()
    SW_FieldAligned_2D.set_grid([(Xedges[1:] + Xedges[:-1]) / 2, (Yedges[1:] + Yedges[:-1]) / 2], coord_type="2D Field-Aligned")
    SW_FieldAligned_2D.set_vdf(H_avg)

    return SW_FieldAligned_2D


# Following are some functions for calculating moments of the VDFs.
# In spherical.
def cal_density_Spherical(SW_spherical, component=None):

    int_kernal = SW_spherical.get_vdf(component)
    azimuth = SW_spherical.grid['azimuth']
    elevation = SW_spherical.grid['elevation']
    velocity = SW_spherical.grid['velocity']

    subazi, subele, subspeed = np.meshgrid(azimuth, elevation, velocity, indexing='ij')
    # With a Jacobian matrix v^2 cos(elevation)
    int_kernal = int_kernal * (subspeed ** 2) * np.cos(np.radians(subele))
    int_ele = np.trapz(int_kernal, x=np.radians(elevation), axis=1)
    int_ele_energy = -np.trapz(int_ele, x=velocity, axis=1)
    int_all = np.trapz(int_ele_energy, x=np.radians(azimuth), axis=0)

    return float(int_all)

def cal_bulk_velocity_Spherical(SW_spherical, component=None):

    int_kernal = SW_spherical.get_vdf(component)
    azimuth = SW_spherical.grid['azimuth']
    elevation = SW_spherical.grid['elevation']
    velocity = SW_spherical.grid['velocity']

    # Convert angles to radians for calculations
    # Note the result is in SRF coordiantes
    subazi, subele, subspeed = np.meshgrid(azimuth, elevation, velocity, indexing='ij')

    # Calculate vector components for bulk velocity
    jacobian = subspeed**2 * np.cos(np.radians(subele))

    vx_kernal = -int_kernal * subspeed * np.cos(np.radians(subele)) * np.cos(np.radians(subazi)) * jacobian
    vy_kernal = int_kernal * subspeed * np.cos(np.radians(subele)) * np.sin(np.radians(subazi)) * jacobian
    vz_kernal = -int_kernal * subspeed * np.sin(np.radians(subele)) * jacobian

    # Integrate along each dimension
    vx_ele = np.trapz(vx_kernal, x=np.radians(elevation), axis=1)
    vy_ele = np.trapz(vy_kernal, x=np.radians(elevation), axis=1)
    vz_ele = np.trapz(vz_kernal, x=np.radians(elevation), axis=1)

    vx_ele_energy = -np.trapz(vx_ele, x=velocity, axis=1)
    vy_ele_energy = -np.trapz(vy_ele, x=velocity, axis=1)
    vz_ele_energy = -np.trapz(vz_ele, x=velocity, axis=1)

    vx_all = np.trapz(vx_ele_energy, x=np.radians(azimuth), axis=0)
    vy_all = np.trapz(vy_ele_energy, x=np.radians(azimuth), axis=0)
    vz_all = np.trapz(vz_ele_energy, x=np.radians(azimuth), axis=0)

    # Divide by the density to get the bulk velocity vector
    density = cal_density_Spherical(SW_spherical, component)
    vx_bulk = vx_all / density
    vy_bulk = vy_all / density
    vz_bulk = vz_all / density

    # Return the bulk velocity as a vector
    return np.array([vx_bulk, vy_bulk, vz_bulk])

def cal_pressure_tensor_Spherical(SW_spherical, component=None):
    mass_proton = 1.6726219e-27 # kg
    mass_alpha = 6.6464731e-27 # kg
    mass_electron = 9.10938356e-31 # kg

    mass_dict = {
        'proton': mass_proton,
        'alpha': mass_alpha,
        'electron': mass_electron
    }

    # Necessary parameters.
    mass = mass_dict[SW_spherical.species]
    vx_bulk, vy_bulk, vz_bulk = cal_bulk_velocity_Spherical(SW_spherical, component)
    int_kernal = SW_spherical.get_vdf(component)
    azimuth = SW_spherical.grid['azimuth']
    elevation = SW_spherical.grid['elevation']
    velocity = SW_spherical.grid['velocity'] 

    subazi, subele, subspeed = np.meshgrid(azimuth, elevation, velocity, indexing='ij')

    vx = -subspeed * np.cos(np.radians(subele)) * np.cos(np.radians(subazi))
    vy = subspeed * np.cos(np.radians(subele)) * np.sin(np.radians(subazi))
    vz = -subspeed * np.sin(np.radians(subele))

    dvx = vx - vx_bulk
    dvy = vy - vy_bulk
    dvz = vz - vz_bulk

    jacobian = subspeed**2 * np.cos(np.radians(subele))

    Pxx_kernal = int_kernal * mass * dvx * dvx * jacobian
    Pxy_kernal = int_kernal * mass * dvx * dvy * jacobian
    Pxz_kernal = int_kernal * mass * dvx * dvz * jacobian
    Pyy_kernal = int_kernal * mass * dvy * dvy * jacobian
    Pyz_kernal = int_kernal * mass * dvy * dvz * jacobian
    Pzz_kernal = int_kernal * mass * dvz * dvz * jacobian


    def integrate_pressure(kernal):
        int_ele = np.trapz(kernal, x=np.radians(elevation), axis=1)
        int_energy = -np.trapz(int_ele, x=velocity, axis=1)
        return np.trapz(int_energy, x=np.radians(azimuth), axis=0)
    
    Pxx = integrate_pressure(Pxx_kernal)
    Pxy = integrate_pressure(Pxy_kernal)
    Pxz = integrate_pressure(Pxz_kernal)
    Pyy = integrate_pressure(Pyy_kernal)
    Pyz = integrate_pressure(Pyz_kernal)
    Pzz = integrate_pressure(Pzz_kernal)

    return np.array([[Pxx, Pxy, Pxz], [Pxy, Pyy, Pyz], [Pxz, Pyz, Pzz]])

def Temperature_para_perp(SW_Spherical, component=None):

    k_B = 1.38064852e-23 # J/K
    Density = cal_density_Spherical(SW_Spherical, component)
    B_field = SW_Spherical.magfield

    K2eV = {
        'proton': 8.617333262145e-5, 
        'alpha': 8.617333262145e-5, 
        'electron': 8.617333262145e-5
    }

    k2eV = K2eV[SW_Spherical.species]

    PTensor = cal_pressure_tensor_Spherical(SW_Spherical, component)
    TTensor = PTensor / Density / k_B

    b_hat = B_field / np.linalg.norm(B_field)

    # Tpara
    T_para = np.dot(b_hat, np.dot(TTensor, b_hat))
    T_trace = np.trace(TTensor)
    T_perp = (T_trace - T_para) / 2 # K
    # K to eV
    T_para = T_para * k2eV
    T_perp = T_perp * k2eV
    
    return T_para, T_perp


# In 2D field-aligned.
# Here, we calculate the density, bulk velocity, and temperature in 2D.
def cal_density_2D(SW_2D, component=None):
    Vpara = SW_2D.grid['Vpara']
    Vperp = SW_2D.grid['Vperp']

    Vpara_grid, Vperp_grid = np.meshgrid(Vpara, Vperp, indexing='ij')
    int_kernal = SW_2D.get_vdf(component) * 2 * np.pi * Vperp_grid
    int_para = np.trapz(int_kernal, x=Vpara, axis=0)
    int_perp = np.trapz(int_para, x=Vperp, axis=0)

    return int_perp

# More will be added.