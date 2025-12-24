import numpy as np
from pint import Quantity
import physics
from typing import Union
from pint.errors import UnitStrippedWarning
import warnings
from scipy import interpolate
import sys
from datetime import datetime
import os
from netCDF4 import Dataset as NCDF
import numba as nb
import math
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore', category=UnitStrippedWarning)

def _ensure_unit_aware_arrays(input_quantity : Union[int, float, list, np.ndarray, Quantity], output_unit : str) -> Quantity:
    if isinstance(input_quantity, int) or isinstance(input_quantity, float) or isinstance(input_quantity, list) or isinstance(input_quantity, np.ndarray):
        input_quantity = Quantity(np.array(input_quantity, ndmin = 1), output_unit)
    elif isinstance(input_quantity, Quantity):
        input_quantity = Quantity(np.array(input_quantity, ndmin = 1), input_quantity.units).to_base_units().to(output_unit)
    return input_quantity

def pressure_reference_levels(ref_z : Union[int, float, np.ndarray, Quantity], z : Union[int, float, np.ndarray, Quantity], p : Union[int, float, np.ndarray, Quantity]) -> tuple[Quantity, np.ndarray]:
    ref_z = _ensure_unit_aware_arrays(ref_z, "m")
    z = _ensure_unit_aware_arrays(z, "m")
    p = _ensure_unit_aware_arrays(p, "Pa")

    ref_p = []
    is_below = []

    for i in range(len(ref_z)):
        level = np.where(z >= ref_z[i])[0][0]
        ref_p.append(p[level].m)
        if (z[level] > ref_z[i]) and not (level == 0):
            level -= 1
            is_below.append(False)
        elif (z[level] == ref_z[i]):
            is_below.append(False)
        else:
            is_below.append(True)
        
        
    #print(ref_p)
    return Quantity(ref_p, "Pa"), np.array(is_below)

def hypsometric_P_to_Z(T_bar : Union[int, float, np.ndarray, Quantity], p_1 : Union[int, float, np.ndarray, Quantity], p_2 : Union[int, float, np.ndarray, Quantity], r_v_bar : Union[int, float, np.ndarray, Quantity], z_1 : Union[None, int, float, np.ndarray, Quantity] = None):
    '''
    Parameters
    ----------------------
    T_bar  -  int, float, np.ndarray, or pint.Quantity
        The layer average temperature of the layer.
        If provided as a pint.Quantity, may be in any units of temperature. If provided as an int or float, expects units of Kelvin.
    
    p_1  -  int, float, np.ndarray, or pint.Quantity
        The pressure of the initial pressure level.
        If provided as a pint.Quantity, may be in any units of pressure. If provided as an int or float, expects units of Pascals.
    
    p_2  -  int, float, np.ndarray, or pint.Quantity
        The pressure of the final pressure level.
        If provided as a pint.Quantity, may be in any units of pressure. If provided as an int or float, expects units of Pascals.
    
    r_v_bar  -  int, float, np.ndarray, or pint.Quantity
        The layer average mixing ratio of the layer.
        If provided as a pint.Quantity, may be in any units of mass ratio. If provided as an int or float, expects units of kilograms per kilogram or dimensionless.
    
    z_1  -  None, int, float, np.ndarray, or pint.Quantity (Optional, defaults to None)
        The height at p_1.
        If provided as a pint.Quantity, may be in any units of length. If provided as an int or float, expects units of meters.

    Returns
    ----------------------
    z_2 OR dz  -  pint.Quantity
        If z_1 was specified, returns the height z_2, in meters.     
        If z_1 was not specified, returns the height difference dz between the two pressure levels, in meters.
    '''

    T_bar = _ensure_unit_aware_arrays(T_bar, "K")
    p_1 = _ensure_unit_aware_arrays(p_1, "Pa")
    p_2 = _ensure_unit_aware_arrays(p_2, "Pa")
    r_v_bar = _ensure_unit_aware_arrays(r_v_bar, "kg kg^(-1)")
    
    if z_1 is not None:
        z_1 = _ensure_unit_aware_arrays(z_1, "m")
    
        return z_1 + ((physics.R_d * (T_bar * (1 + (0.61 * r_v_bar)))/physics.g_0) * np.log(p_1/p_2)) # returns z_2
    
    else:
        return (physics.R_d * (T_bar * (1 + (0.61 * r_v_bar)))/physics.g_0) * np.log(p_1/p_2) # returns dz

def hypsometric_Z_to_P(T_bar : Union[int, float, np.ndarray, Quantity], z_1 : Union[int, float, np.ndarray, Quantity], z_2 : Union[int, float, np.ndarray, Quantity], r_v_bar : Union[int, float, np.ndarray, Quantity], p_1 : Union[None, int, float, np.ndarray, Quantity] = None, downward : Union[bool, np.ndarray] = False, profile : bool = True) -> Quantity:
    '''
    Parameters
    ----------------------
    T_bar  -  int, float, np.ndarray, or pint.Quantity
        The layer average temperature of the layer.
        If provided as a pint.Quantity, may be in any units of temperature. If provided as an int or float, expects units of Kelvin.
    
    z_1  -  int, float, np.ndarray, or pint.Quantity
        The height of the initial height level.
        If provided as a pint.Quantity, may be in any units of length. If provided as an int or float, expects units of meters.
    
    z_2  -  int, float, np.ndarray, or pint.Quantity
        The height of the final height level.
        If provided as a pint.Quantity, may be in any units of length. If provided as an int or float, expects units of meters.
    
    r_v_bar  -  int, float, np.ndarray, or pint.Quantity
        The layer average mixing ratio of the layer.
        If provided as a pint.Quantity, may be in any units of mass ratio. If provided as an int or float, expects units of kilograms per kilogram or dimensionless.
    
    p_1  -  None, int, float, np.ndarray, or pint.Quantity (Optional, defaults to None)
        The pressure at z_1.
        If provided as a pint.Quantity, may be in any units of pressure. If provided as an int or float, expects units of Pascals.
    
    downward - bool or np.ndarray (Optional, defaults to False)
        Whether the given p_1 is the higher (z) pressure or the lower (z) pressure.
        Must be the same length as z_1.
    
    profile - bool (Optional, defaults to True)
        Whether the given data represents a single column.
        If true, will perform the calculation recursively on the previous p_2 (only uses p_1[0]).
        If false, considers all calculations separately, requiring that p_1 be the same size as z_1.

    Returns
    ----------------------
    p_2 OR p_1/p_2  -  pint.Quantity
        If p_1 was specified, returns the pressure p_2, in pascals.     
        If p_1 was not specified, returns the ratio of the initial to final pressure levels, as a dimensionless number.
    '''
    
    T_bar = _ensure_unit_aware_arrays(T_bar, "K")
    z_1 = _ensure_unit_aware_arrays(z_1, "m")
    z_2 = _ensure_unit_aware_arrays(z_2, "m")
    r_v_bar = _ensure_unit_aware_arrays(r_v_bar, "kg kg^(-1)")
    
    if p_1 is not None:
        p_1 = _ensure_unit_aware_arrays(p_1, "Pa")

        p_2 = np.array([])
        #print(z_1.m.size, downward)
        if (z_1.m.size == 1) and (type(downward) == bool):
            if downward:
                p_2 = np.append(p_2, (p_1*(np.e**(((z_2 - z_1) * physics.g_0)/(physics.R_d * (T_bar * (1 + (0.61 * r_v_bar))))))).m) # returns a p_2 that is below the original p_1
            else:
                p_2 = np.append(p_2, (p_1/(np.e**(((z_2 - z_1) * physics.g_0)/(physics.R_d * (T_bar * (1 + (0.61 * r_v_bar))))))).m) # returns p_2
        else:
            for i in range(len(z_1.m)):
                if profile:
                    if p_1.m.size != 1:
                        p_1 = p_1[0]
                    if i == 0:
                        if downward[i]:
                            p_2 = np.append(p_2, (p_1*(np.e**(((z_2[i] - z_1[i]) * physics.g_0)/(physics.R_d * (T_bar[i] * (1 + (0.61 * r_v_bar[i]))))))).m) # returns a p_2 that is below the original p_1
                        else:
                            p_2 = np.append(p_2, (p_1/(np.e**(((z_2[i] - z_1[i]) * physics.g_0)/(physics.R_d * (T_bar[i] * (1 + (0.61 * r_v_bar[i]))))))).m) # returns p_2
                    else:
                        if downward[i]:
                            p_2 = np.append(p_2, (Quantity(p_2[i-1], "Pa")*(np.e**(((z_2[i] - z_1[i]) * physics.g_0)/(physics.R_d * (T_bar[i] * (1 + (0.61 * r_v_bar[i]))))))).m) # returns a p_2 that is below the original p_1
                        else:
                            p_2 = np.append(p_2, (Quantity(p_2[i-1], "Pa")/(np.e**(((z_2[i] - z_1[i]) * physics.g_0)/(physics.R_d * (T_bar[i] * (1 + (0.61 * r_v_bar[i]))))))).m) # returns p_2

        return Quantity(p_2, "Pa")
    else:
        return 1/(np.e**(((z_2 - z_1) * physics.g_0)/(physics.R_d * (T_bar * (1 + (0.61 * r_v_bar)))))) # returns p_2/p_1, i.e. the ratio of the pressures at z_1 to z_2

def poissons_T_to_theta(T : Union[int, float, np.ndarray, Quantity], p : Union[int, float, np.ndarray, Quantity], r_v : Union[int, float, np.ndarray, Quantity]) -> Quantity:
    '''
    Parameters
    ----------------------
    T  -  int, float, np.ndarray, or pint.Quantity
        The temperature of the parcel to find the potential temperature of.
        If provided as a pint.Quantity, may be in any units of temperature. If provided as an int or float, expects units of Kelvin.
    
    p  -  int, float, np.ndarray, or pint.Quantity
        The pressure of the parcel to find the potential temperature of.
        If provided as a pint.Quantity, may be in any units of pressure. If provided as an int or float, expects units of Pascals.
    
    r_v  -  int, float, np.ndarray, or pint.Quantity
        The mixing ratio of the parcel to find the potential temperature of.
        If provided as a pint.Quantity, may be in any units of mass ratio. If provided as an int or float, expects units of kilograms per kilogram.
    
    Returns
    ----------------------
    theta  -  pint.Quantity
        The potential temperature of the given parcel, in Kelvin.
    '''

    T = _ensure_unit_aware_arrays(T, "K")
    p = _ensure_unit_aware_arrays(p, "Pa")
    r_v = _ensure_unit_aware_arrays(r_v, "kg kg^(-1)")
    
    return T * (physics.p_0/p)**((physics.R_d/physics.c_p)*(1 - 0.28 * r_v))

def poissons_dT_to_dtheta(dT : Union[int, float, np.ndarray, Quantity], p : Union[int, float, np.ndarray, Quantity], r_v : Union[int, float, np.ndarray, Quantity]) -> Quantity:
    '''
    Parameters
    ----------------------
    T  -  int, float, np.ndarray, or pint.Quantity
        The temperature of the parcel to find the potential temperature of.
        If provided as a pint.Quantity, may be in any units of temperature. If provided as an int or float, expects units of Kelvin.
    
    p  -  int, float, np.ndarray, or pint.Quantity
        The pressure of the parcel to find the potential temperature of.
        If provided as a pint.Quantity, may be in any units of pressure. If provided as an int or float, expects units of Pascals.
    
    r_v  -  int, float, np.ndarray, or pint.Quantity
        The mixing ratio of the parcel to find the potential temperature of.
        If provided as a pint.Quantity, may be in any units of mass ratio. If provided as an int or float, expects units of kilograms per kilogram.
    
    Returns
    ----------------------
    theta  -  pint.Quantity
        The potential temperature of the given parcel, in Kelvin.
    '''

    dT = _ensure_unit_aware_arrays(dT, "K s^(-1)")
    p = _ensure_unit_aware_arrays(p, "Pa")
    r_v = _ensure_unit_aware_arrays(r_v, "kg kg^(-1)")
    
    return dT * (physics.p_0/p)**((physics.R_d/physics.c_p)*(1 - 0.28 * r_v))

def poissons_theta_to_T(theta : Union[int, float, np.ndarray, Quantity], p : Union[int, float, np.ndarray, Quantity], r_v : Union[int, float, np.ndarray, Quantity]) -> Quantity:
    '''
    Parameters
    ----------------------
    theta  -  int, float, np.ndarray, or pint.Quantity
        The potential temperature of the parcel to find the temperature of.
        If provided as a pint.Quantity, may be in any units of temperature. If provided as an int or float, expects units of Kelvin.
    
    p  -  int, float, np.ndarray, or pint.Quantity
        The pressure of the parcel to find the temperature of.
        If provided as a pint.Quantity, may be in any units of pressure. If provided as an int or float, expects units of Pascals.
    
    r_v  -  int, float, np.ndarray, or pint.Quantity
        The mixing ratio of the parcel to find the temperature of.
        If provided as a pint.Quantity, may be in any units of mass ratio. If provided as an int or float, expects units of kilograms per kilogram.
    
    Returns
    ----------------------
    T  -  pint.Quantity
        The temperature of the given parcel at reference pressure p, in Kelvin.
    '''

    theta = _ensure_unit_aware_arrays(theta, "K")
    p = _ensure_unit_aware_arrays(p, "Pa")
    r_v = _ensure_unit_aware_arrays(r_v, "kg kg^(-1)")
    
    return theta / (physics.p_0/p)**((physics.R_d/physics.c_p)*(1 - 0.28 * r_v))

def calc_e_s(T : Union[int, float, np.ndarray, Quantity]) -> Quantity:
    '''
    Parameters
    ----------------------
    T  -  int, float, np.ndarray, or pint.Quantity
        The temperature at which to find the saturation vapor pressure.
        If provided as a pint.Quantity, may be in any units of temperature. If provided as an int or float, expects units of Kelvin.
    
    Returns
    ----------------------
    e_s  -  pint.Quantity
        The saturation vapor pressure for the given temperature, in Pascals.
    '''

    T = _ensure_unit_aware_arrays(T, "K")
    
    term_1 = physics.L_tp/physics.R_v
    term_2 = (1/physics.T_tp) - (1/T)
    term_3 = ((physics.c_pl - physics.c_pv)/physics.R_v)
    term_4 = physics.T_tp/T
    term_5 = np.log(T/physics.T_tp)
    return physics.e_s_tp * np.e**((term_1 * term_2) + (term_3 * (1 - term_4 - term_5)))


def r_v_from_e(e : Union[int, float, np.ndarray, Quantity], p : Union[int, float, np.ndarray, Quantity]) -> Quantity:
    e = _ensure_unit_aware_arrays(e, "Pa")
    p = _ensure_unit_aware_arrays(p, "Pa")

    r_v = ((physics.R_d/physics.R_v)*e)/(p - e)
    return r_v

def e_from_r_v(r_v : Union[int, float, np.ndarray, Quantity], p : Union[int, float, np.ndarray, Quantity]) -> Quantity:
    r_v = _ensure_unit_aware_arrays(r_v, "Kg Kg^(-1)")
    p = _ensure_unit_aware_arrays(p, "Pa")

    e = ((r_v * p)/((physics.R_d/physics.R_v) + r_v))
    return e

def r_v_from_T_d(T_d : Union[int, float, np.ndarray, Quantity], p : Union[int, float, np.ndarray, Quantity]) -> Quantity:
    T_d = _ensure_unit_aware_arrays(T_d, "K")
    p = _ensure_unit_aware_arrays(p, "Pa")
    
    e = calc_e_s(T_d)
    r_v = r_v_from_e(e, p)
    return r_v

def r_v_from_RH(T : Union[int, float, np.ndarray, Quantity], RH : Union[int, float, np.ndarray, Quantity]) -> Quantity:
    T = _ensure_unit_aware_arrays(T, "K")
    RH = _ensure_unit_aware_arrays(RH, "dimensionless")
    
    e_s = calc_e_s(T)
    r_v = RH * r_v_from_e(e_s)
    return r_v

def e_from_q_v(q_v : Union[int, float, np.ndarray, Quantity], p : Union[int, float, np.ndarray, Quantity]) -> Quantity:
    q_v = _ensure_unit_aware_arrays(q_v, "kg kg^(-1)")
    p = _ensure_unit_aware_arrays(p, "Pa")

    e = (q_v * p)/((physics.R_d/physics.R_v) + q_v * (1 - (physics.R_d/physics.R_v)))
    return e

def e_from_rho_v(rho_v : Union[int, float, np.ndarray, Quantity], T : Union[int, float, np.ndarray, Quantity]) -> Quantity:
    rho_v = _ensure_unit_aware_arrays(rho_v, "kg m^(-3)")
    T = _ensure_unit_aware_arrays(T, "K")

    e = rho_v * physics.R_v * T
    return e

def Coriolis_u(lat : float, v_ag : Union[int, float, np.ndarray, Quantity]) -> Quantity:
    v_ag = _ensure_unit_aware_arrays(v_ag, "m s^(-1)")
    
    f = 2 * physics.omega * np.sin(np.deg2rad(lat))
    return f * v_ag

def Coriolis_v(lat : float, u_ag : Union[int, float, np.ndarray, Quantity]) -> Quantity:
    u_ag = _ensure_unit_aware_arrays(u_ag, "m s^(-1)")
    
    f = 2 * physics.omega * np.sin(np.deg2rad(lat))
    return -f * u_ag


def psi_m_surface_layer(z : Union[int, float, np.ndarray, Quantity], L : Union[int, float, np.ndarray, Quantity]) -> Quantity:
    z = _ensure_unit_aware_arrays(z, "m")
    L = _ensure_unit_aware_arrays(L, "m")

    phi_m = phi_m_surface_layer(z, L)

    if (z/L) <= 0:
        psi_m = 2 * np.log((1 + phi_m**(-1))/2) + np.log((1 + phi_m**(-2))/2) - 2 * np.tan(phi_m**(-1)) + (np.pi/2)
    else:
        psi_m = -4.7 * (z/L)

    return psi_m

def psi_h_surface_layer(z : Union[int, float, np.ndarray, Quantity], L : Union[int, float, np.ndarray, Quantity]) -> Quantity:
    z = _ensure_unit_aware_arrays(z, "m")
    L = _ensure_unit_aware_arrays(L, "m")

    phi_h = phi_h_surface_layer(z, L)

    if (z/L) <= 0:
        psi_h = 2 * np.log((1 + 0.74 * phi_h**(-1))/2)
    else:
        psi_h = -6.35 * (z/L)

    return psi_h

def phi_m_surface_layer(z : Union[int, float, np.ndarray, Quantity], L : Union[int, float, np.ndarray, Quantity]) -> Quantity:
    z = _ensure_unit_aware_arrays(z, "m")
    L = _ensure_unit_aware_arrays(L, "m")

    psi_m = np.zeros(len(z))
    psi_m[(z/L) <= 0] = (1 - (15 * (z[(z/L) <= 0]/L[(z/L) <= 0])))**(-1/4)
    psi_m[(z/L) > 0] = 1 + (4.7 * (z[(z/L) > 0]/L[(z/L) > 0]))

    return psi_m

def phi_h_surface_layer(z : Union[int, float, np.ndarray, Quantity], L : Union[int, float, np.ndarray, Quantity]) -> Quantity:
    z = _ensure_unit_aware_arrays(z, "m")
    L = _ensure_unit_aware_arrays(L, "m")

    psi_h = np.zeros(len(z))
    psi_h[(z/L) <= 0] = 0.74 * (1 - (9 * (z[(z/L) <= 0]/L[(z/L) <= 0])))**(-1/2)
    psi_h[(z/L) > 0] = 0.74 + (4.7 * (z[(z/L) > 0]/L[(z/L) > 0]))

    return psi_h

def sfc_stars(V_z_s : Union[int, float, np.ndarray, Quantity], theta_bar : Union[int, float, np.ndarray, Quantity], theta_z_s : Union[int, float, np.ndarray, Quantity], T_g : Union[int, float, np.ndarray, Quantity], z : Union[int, float, np.ndarray, Quantity], z_0 : Union[int, float, np.ndarray, Quantity], psi_m : Union[int, float, np.ndarray, Quantity], psi_h : Union[int, float, np.ndarray, Quantity], rec : int = 0) -> tuple[Quantity]:
    V_z_s = _ensure_unit_aware_arrays(V_z_s, "m s^(-1)")
    theta_bar = _ensure_unit_aware_arrays(theta_bar, "K")
    theta_z_s = _ensure_unit_aware_arrays(theta_z_s, "K")
    T_g = _ensure_unit_aware_arrays(T_g, "K")
    z = _ensure_unit_aware_arrays(z, "m")
    z_0 = _ensure_unit_aware_arrays(z_0, "m")
    psi_m = _ensure_unit_aware_arrays(psi_m, "dimensionless")
    psi_h = _ensure_unit_aware_arrays(psi_h, "dimensionless")
    #print(theta_z_s, T_g)

    u_star_old = (physics.von_Karman * V_z_s)/(np.log(z/z_0)-psi_m)
    theta_star_old = (physics.von_Karman * (theta_z_s - T_g))/(0.74 * (np.log(z/z_0)-psi_h))

    if (u_star_old.m != 0) and (theta_star_old.m != 0):
        L_old = Monin_Obukhov_L(theta_bar, u_star_old, theta_star_old)
        psi_m_old = psi_m_surface_layer(z, L_old)
        psi_h_old = psi_h_surface_layer(z, L_old)

        u_star_new = u_star_surface_layer(V_z_s, z, z_0, psi_m_old)
        theta_star_new = theta_star_surface_layer(theta_z_s, T_g, z, z_0, psi_h_old)
        L_new = Monin_Obukhov_L(theta_bar, u_star_new, theta_star_new)
        psi_m_new = phi_m_surface_layer(z, L_new)
        psi_h_new = phi_h_surface_layer(z, L_new)
        flag = True
        storage1 = [u_star_new]
        storage2 = [theta_star_new]
        while flag:
            u_star_old = u_star_new
            theta_star_old = theta_star_new
            L_old = L_new
            psi_m_old = psi_m_new
            psi_h_old = psi_h_new

            u_star_new = u_star_surface_layer(V_z_s, z, z_0, psi_m_old)
            theta_star_new = theta_star_surface_layer(theta_z_s, T_g, z, z_0, psi_h_old)
            storage1.append(u_star_new)
            storage2.append(theta_star_new)
            if (np.abs(theta_star_new.m - theta_star_old.m) < 0.001) and (np.abs(u_star_old.m - u_star_old.m) < 0.001):
                flag = False
            elif len(storage1) >= 500:
                u_star_new = Quantity(np.min([x.m for x in storage1[-10:]]), storage1[-1].units)
                theta_star_new = Quantity(np.min([x.m for x in storage2[-10:]]), storage2[-1].units)
                flag = False
            L_new = Monin_Obukhov_L(theta_bar, u_star_new, theta_star_new)
            psi_m_new = phi_m_surface_layer(z, L_new)
            psi_h_new = phi_h_surface_layer(z, L_new)
            #print(len(storage1), u_star_old, theta_star_old)
        
        u_star = u_star_new
        theta_star = theta_star_new
    else:
        u_star = u_star_old
        theta_star = theta_star_old
        
    return (u_star, theta_star)

def Monin_Obukhov_L(theta_bar : Union[int, float, np.ndarray, Quantity], u_star : Union[int, float, np.ndarray, Quantity], theta_star : Union[int, float, np.ndarray, Quantity]) -> Quantity:
    theta_bar = _ensure_unit_aware_arrays(theta_bar, "K")
    u_star = _ensure_unit_aware_arrays(u_star, "m s^(-1)")
    theta_star = _ensure_unit_aware_arrays(theta_star, "K")

    if theta_star.m != 0:
        L = (theta_bar * u_star**2)/(physics.von_Karman * physics.g_0 * theta_star)
    else:
        L = Quantity(np.inf, "m")

    return L


def u_star_surface_layer(V_z_s : Union[int, float, np.ndarray, Quantity], z : Union[int, float, np.ndarray, Quantity], z_0 : Union[int, float, np.ndarray, Quantity], psi_m : Union[int, float, np.ndarray, Quantity]) -> Quantity:
    V_z_s = _ensure_unit_aware_arrays(V_z_s, "m s^(-1)")
    z = _ensure_unit_aware_arrays(z, "m")
    z_0 = _ensure_unit_aware_arrays(z_0, "m")
    psi_m = _ensure_unit_aware_arrays(psi_m, "dimensionless")

    u_star = (physics.von_Karman * V_z_s)/(np.log(z/z_0)-psi_m)

    return u_star

def theta_star_surface_layer(theta_z_s, theta_z_0, z, z_0, psi_h) -> Quantity:
    theta_z_s = _ensure_unit_aware_arrays(theta_z_s, "K")
    theta_z_0 = _ensure_unit_aware_arrays(theta_z_0, "K")
    z = _ensure_unit_aware_arrays(z, "m")
    z_0 = _ensure_unit_aware_arrays(z_0, "m")
    psi_h = _ensure_unit_aware_arrays(psi_h, "dimensionless")


    theta_star = (physics.von_Karman * (theta_z_s - theta_z_0))/(0.74 * (np.log(z/z_0)-psi_h))

    return theta_star


def fluxes_surface_layer(theta_bar_field : Union[int, float, np.ndarray, Quantity], T_g : Union[int, float, np.ndarray, Quantity], u_bar_field : Union[int, float, np.ndarray, Quantity], v_bar_field : Union[int, float, np.ndarray, Quantity], z_field : Union[int, float, np.ndarray, Quantity], z_s : Union[int, float, np.ndarray, Quantity], z_0 : Union[int, float, np.ndarray, Quantity], height_index : int, model_bottom : bool = False) -> tuple[Quantity]:
    theta_bar_field = _ensure_unit_aware_arrays(theta_bar_field, "K")
    T_g = _ensure_unit_aware_arrays(T_g, "K")
    u_bar_field = _ensure_unit_aware_arrays(u_bar_field, "m s^(-1)")
    v_bar_field = _ensure_unit_aware_arrays(v_bar_field, "m s^(-1)")
    z_field = _ensure_unit_aware_arrays(z_field, "m")
    z_s = _ensure_unit_aware_arrays(z_s, "m")
    z_0 = _ensure_unit_aware_arrays(z_0, "m")

    if model_bottom:
        theta_bar = theta_bar_field[1:][height_index[:-1]]
        u_bar = u_bar_field[1:][height_index[:-1]]
        v_bar = v_bar_field[1:][height_index[:-1]]
        z = z_field[1:][height_index[:-1]]
    else:
        theta_bar = (theta_bar_field[1:][height_index[1:]] + theta_bar_field[:-1][height_index[1:]]) / 2
        u_bar = (u_bar_field[1:][height_index[1:]] + u_bar_field[:-1][height_index[1:]]) / 2
        v_bar = (v_bar_field[1:][height_index[1:]] + v_bar_field[:-1][height_index[1:]]) / 2
        z = (z_field[1:][height_index[1:]] + z_field[:-1][height_index[1:]]) / 2

    V_z_s = np.sqrt(u_bar_field[0]**2 + v_bar_field[0]**2)
    theta_z_s = Quantity(interpolate.interp1d(z_field, theta_bar_field)(z_s), "K")
    T_g
    

    # Recursively find u* and theta*
    u_star, theta_star = sfc_stars(V_z_s, theta_bar, theta_z_s, T_g, z, z_0, 0, 0)

    mu = np.arctan2(v_bar_field[0], u_bar_field[0])

    uw = -u_star**2 * np.cos(mu)
    vw = -u_star**2 * np.sin(mu)
    thetaw = -u_star * theta_star

    #print(uw, vw, thetaw)
    #sys.exit()
    return uw, vw, thetaw

def stars_alt(theta_z : Union[int, float, np.ndarray, Quantity], theta_g : Union[int, float, np.ndarray, Quantity], r_v_z : Union[int, float, np.ndarray, Quantity], r_v_g : Union[int, float, np.ndarray, Quantity], z_s : Union[int, float, np.ndarray, Quantity], z_0 : Union[int, float, np.ndarray, Quantity], u_bar : Union[int, float, np.ndarray, Quantity], v_bar : Union[int, float, np.ndarray, Quantity]) -> Quantity:
    theta_z = _ensure_unit_aware_arrays(theta_z, "K")
    theta_g = _ensure_unit_aware_arrays(theta_g, "K")
    r_v_z = _ensure_unit_aware_arrays(r_v_z, "kg kg^(-1)")
    r_v_g = _ensure_unit_aware_arrays(r_v_g, "kg kg^(-1)")
    z_s = _ensure_unit_aware_arrays(z_s, "m")
    z_0 = _ensure_unit_aware_arrays(z_0, "m")
    u_bar = _ensure_unit_aware_arrays(u_bar, "m s^(-1)")
    v_bar = _ensure_unit_aware_arrays(v_bar, "m s^(-1)")

    drag = physics.von_Karman**2 / (np.log(z_s/z_0))**2
    drag_ratio = 0.74
    V = Quantity(np.array(np.sqrt(u_bar**2 + v_bar**2)), "m s^(-1)")

    theta_bar = (theta_g + theta_z)/2
    dthetatz = (theta_z - theta_g)/(z_s)
    dudz = (u_bar)/(z_s)
    dvdz = (v_bar)/(z_s)

    R_i = bulk_richardson_layer(theta_bar, dthetatz, dudz, dvdz)
    const_b = 9.4
    const_b_prime = 4.7
    const_c_star_momentum = 7.4
    const_c_star_heat = 5.3

    if R_i < 0:
        F_m = 1 - ((const_b * R_i)/(1 + const_c_star_momentum * drag * const_b * np.sqrt(z_s/z_0) * np.sqrt(np.abs(R_i))))
        F_h = 1 - ((const_b * R_i)/(1 + const_c_star_heat * drag * const_b * np.sqrt(z_s/z_0) * np.sqrt(np.abs(R_i))))
    else:
        F_m = 1/((1 + const_b_prime * R_i)**2)
        F_h = 1/((1 + const_b_prime * R_i)**2)
    
    
    u_star_squared = drag * V**2 * F_m
    u_star = np.sqrt(u_star_squared) * np.sign(F_m)
    if u_star.m == 0:
        theta_star = Quantity(0, "K")
        r_v_star = Quantity(0, "kg kg^(-1)")
    else:
        theta_star = ((drag/drag_ratio) * V * (theta_z - theta_g) * F_h)/u_star
        r_v_star = ((drag/drag_ratio) * V * (r_v_z - r_v_g) * F_h)/u_star
    theta_star = Quantity(theta_star, "K")

    return u_star, theta_star, r_v_star

def fluxes_surface_layer_alt(u_star : Union[int, float, np.ndarray, Quantity], theta_star : Union[int, float, np.ndarray, Quantity], r_v_star : Union[int, float, np.ndarray, Quantity], u_z_s : Union[int, float, np.ndarray, Quantity], v_z_s : Union[int, float, np.ndarray, Quantity]) -> tuple[Quantity]:
    u_star = _ensure_unit_aware_arrays(u_star, "m s^(-1)")
    theta_star = _ensure_unit_aware_arrays(theta_star, "K")
    r_v_star = _ensure_unit_aware_arrays(r_v_star, "K")
    u_z_s = _ensure_unit_aware_arrays(u_z_s, "m s^(-1)")
    v_z_s = _ensure_unit_aware_arrays(v_z_s, "m s^(-1)")

    mu = np.arctan2(v_z_s, u_z_s)

    uw = -u_star**2 * np.cos(mu)
    vw = -u_star**2 * np.sin(mu)
    thetaw = -u_star * theta_star
    r_vw = -u_star * r_v_star

    return uw, vw, thetaw, r_vw

def bulk_richardson_layer(theta_bar : Union[int, float, np.ndarray, Quantity], dthetadz : Union[int, float, np.ndarray, Quantity], dudz : Union[int, float, np.ndarray, Quantity], dvdz : Union[int, float, np.ndarray, Quantity]) -> Quantity:
    theta_bar = _ensure_unit_aware_arrays(theta_bar, "K")
    dthetadz = _ensure_unit_aware_arrays(dthetadz, "K m^(-1)")
    dudz = _ensure_unit_aware_arrays(dudz, "s^(-1)")
    dvdz = _ensure_unit_aware_arrays(dvdz, "s^(-1)")

    R_i = ((physics.g_0 * dthetadz))/(theta_bar * (dudz**2 + dvdz**2 + Quantity(1e-6, "s^(-2)")))

    return R_i

def boundary_layer_height_tendency(u_star : Union[int, float, np.ndarray, Quantity], theta_star : Union[int, float, np.ndarray, Quantity], theta_field : Union[int, float, np.ndarray, Quantity], z_field : Union[int, float, np.ndarray, Quantity], theta_g : Union[int, float, np.ndarray, Quantity], z_s : Union[int, float, np.ndarray, Quantity], z_i : Union[int, float, np.ndarray, Quantity], z_0 : Union[int, float, np.ndarray, Quantity], lat : Union[int, float, np.ndarray, Quantity]) -> Quantity:
    u_star = _ensure_unit_aware_arrays(u_star, "m s^(-1)")
    theta_star = _ensure_unit_aware_arrays(theta_star, "K")
    theta_field = _ensure_unit_aware_arrays(theta_field, "K")
    theta_g = _ensure_unit_aware_arrays(theta_g, "K")
    z_s = _ensure_unit_aware_arrays(z_s, "m")
    z_i = _ensure_unit_aware_arrays(z_i, "m")
    z_0 = _ensure_unit_aware_arrays(z_0, "m")

    theta_z_s = Quantity(interpolate.interp1d(z_field, theta_field)(z_s), "K")
    theta_above_z_i = Quantity(interpolate.interp1d(z_field, theta_field)(z_i + Quantity(25, "m")), "K")
    theta_below_z_i = Quantity(interpolate.interp1d(z_field, theta_field)(z_i - Quantity(25, "m")), "K")
    dtheta_idz = (theta_above_z_i - theta_below_z_i)/Quantity(50, "m")

    f = 2 * physics.omega * np.sin(np.deg2rad(lat))

    w_star = wstar(theta_z_s, u_star, theta_star, z_s)
    dz_idt = (1.8 * (w_star**3 + 1.1 * u_star**3 - 3.3 * u_star**2 * f * z_i))/(physics.g_0 * (z_i**2/theta_z_s) * dtheta_idz + 9 * w_star**2 + 7.2 * u_star**2)

    #if theta_star.m <= 0:
    #    w_star = wstar(theta_z_s, u_star, theta_star, z_s)
    #    dz_idt = (1.8 * (w_star**3 + 1.1 * u_star**3 - 3.3 * u_star**2 * f * z_i))/(physics.g_0 * (z_i**2/theta_z_s) * dtheta_idz + 9 * w_star**2 + 7.2 * u_star**2)
    #else:
    #    dz_idt = 0.06 * ((u_star**2)/(z_i * f)) * (1 - ((3.3 * z_i * f)/u_star)**3)
    return dz_idt

def wstar(theta_z : Union[int, float, np.ndarray, Quantity], u_star : Union[int, float, np.ndarray, Quantity], theta_star : Union[int, float, np.ndarray, Quantity], z : Union[int, float, np.ndarray, Quantity]) -> Quantity:
    theta_z = _ensure_unit_aware_arrays(theta_z, "K")
    u_star = _ensure_unit_aware_arrays(u_star, "m s^(-1)")
    theta_star = _ensure_unit_aware_arrays(theta_star, "K")
    z = _ensure_unit_aware_arrays(z, "m")
    
    if theta_star <= 0:
        w_star = (-(physics.g_0/theta_z) * u_star * theta_star * z)**(1/3)
    else:
        w_star = Quantity(0, "m s^(-1)")
    
    return w_star

def eddy_momentum_diffusivity(theta_field : Union[int, float, np.ndarray, Quantity], theta_g : Union[int, float, np.ndarray, Quantity], z_field : Union[int, float, np.ndarray, Quantity], z_s : Union[int, float, np.ndarray, Quantity], z_i : Union[int, float, np.ndarray, Quantity], u_bar_field : Union[int, float, np.ndarray, Quantity], v_bar_field : Union[int, float, np.ndarray, Quantity], z_0 : Union[int, float, np.ndarray, Quantity], L : Union[int, float, np.ndarray, Quantity], u_star : Union[int, float, np.ndarray, Quantity], height_index : int) -> Quantity:
    theta_field = _ensure_unit_aware_arrays(theta_field, "K")
    theta_g = _ensure_unit_aware_arrays(theta_g, "K")
    L = _ensure_unit_aware_arrays(L, "m")
    u_star = _ensure_unit_aware_arrays(u_star, "m s^(-1)")
    z_field = _ensure_unit_aware_arrays(z_field, "m")
    z_s = _ensure_unit_aware_arrays(z_s, "m")
    z_i = _ensure_unit_aware_arrays(z_i, "m")
    z_0 = _ensure_unit_aware_arrays(z_0, "m")
    u_bar_field = _ensure_unit_aware_arrays(u_bar_field, "m s^(-1)")
    v_bar_field = _ensure_unit_aware_arrays(v_bar_field, "m s^(-1)")

    dtheta = (theta_field[1:][height_index[1:]] - theta_field[:-1][height_index[1:]])
    theta_bar = (theta_field[1:][height_index[1:]] + theta_field[:-1][height_index[1:]])/2
    du = (u_bar_field[1:][height_index[1:]] - u_bar_field[:-1][height_index[1:]])
    u_bar = (u_bar_field[1:][height_index[1:]] + u_bar_field[:-1][height_index[1:]])/2
    dv = (v_bar_field[1:][height_index[1:]] - v_bar_field[:-1][height_index[1:]])
    v_bar = (v_bar_field[1:][height_index[1:]] + v_bar_field[:-1][height_index[1:]])/2
    dz = (z_field[1:][height_index[1:]] - z_field[:-1][height_index[1:]])
    z_bar = (z_field[1:][height_index[1:]] + z_field[:-1][height_index[1:]]) / 2
    dthetadz = dtheta/dz
    dudz = du/dz
    dvdz = dv/dz
    
    unstable = ((z_bar/L) <= 0) & ~(L.m == np.inf)

    K_m = np.zeros(dthetadz.size)
    if L < 0:
        dKmdz = (-15 * physics.von_Karman * u_star * z_s)/(4 * L * (1 - 15 * ((z_s)/L)))
        K_Ms = ((physics.von_Karman * u_star * z_s)/((1 - 15 * (z_s/L))**(-1/4)))
        K_m[unstable] = Quantity(0.001, "m^(2) s^(-1)") + ((z_i - z_bar[unstable])**2/(z_i - z_s)**2) * (K_Ms - Quantity(0.001, "m^(2) s^(-1)") + (z_bar[unstable] - z_s) * (dKmdz + 2 * ((K_Ms - Quantity(0.001, "m^(2) s^(-1)"))/(z_i - z_s))))
        #print(L, (z_i - z_bar[unstable])**2/(z_i - z_s)**2, dKmdz)
    else:
        K_m[unstable] = Quantity(0, "m^(2) s^(-1)")

    R_ic = 0.115 * ((dz[~unstable]).to("m").m)**0.175
    R_i = bulk_richardson_layer(theta_bar[~unstable], dthetadz[~unstable], dudz[~unstable], dvdz[~unstable])
    l = np.zeros(len(R_i))

    high = z_bar[~unstable] > Quantity(200, "m")
    low_Ri = R_i <= R_ic
    l[(low_Ri) & (~high)] = physics.von_Karman * z_bar[~unstable][(low_Ri) & (~high)]
    l[(low_Ri) & (high)] = Quantity(np.ones(len(l[(low_Ri) & (high)])) * 70, "m")
    K_m[~unstable][low_Ri] = 1.1 * ((R_ic[low_Ri] - R_i[low_Ri])/(R_ic[low_Ri])) * l[low_Ri]**2 * np.sqrt(dudz[~unstable][low_Ri]**2 + dvdz[~unstable][low_Ri]**2)
    K_m[~unstable][~low_Ri] = Quantity(np.zeros(len(K_m[~unstable][~low_Ri])), "m^(2) s^(-1)")
    #if dthetadz <= Quantity(0, "K m^(-1)"):
    #    K_m = np.sqrt(dudz**2 + dvdz**2) * physics.von_Karman**2 * z_bar*2 * (1/(phi_m**2))
    #    
        # This seems to almost always be a complex number..?
        #K_m_z_s = (physics.von_Karman * u_star * z_s)/phi_m
        #dkmdz = 0.25 * physics.von_Karman * u_star * ((4 * z_s**3 - 75 * ((z_s**4)/L))/(z_s**4 - 15 * ((z_s**5)/L))**(3/4))
        #K_m = physics.k_m_z_i + (((z_i - z)**2)/((z_i - z_s)**2))*(K_m_z_s - physics.k_m_z_i + (z - z_s) * (dkmdz + 2 * ((K_m_z_s - physics.k_m_z_i)/(z_i - z_s))))

    return K_m

def eddy_heat_diffusivity(theta_field : Union[int, float, np.ndarray, Quantity], theta_g : Union[int, float, np.ndarray, Quantity], z_field : Union[int, float, np.ndarray, Quantity], z_s : Union[int, float, np.ndarray, Quantity], z_i : Union[int, float, np.ndarray, Quantity], u_bar_field : Union[int, float, np.ndarray, Quantity], v_bar_field : Union[int, float, np.ndarray, Quantity], z_0 : Union[int, float, np.ndarray, Quantity], L : Union[int, float, np.ndarray, Quantity], u_star : Union[int, float, np.ndarray, Quantity], height_index : int) -> Quantity:
    theta_field = _ensure_unit_aware_arrays(theta_field, "K")
    theta_g = _ensure_unit_aware_arrays(theta_g, "K")
    L = _ensure_unit_aware_arrays(L, "m")
    u_star = _ensure_unit_aware_arrays(u_star, "m s^(-1)")
    z_field = _ensure_unit_aware_arrays(z_field, "m")
    z_s = _ensure_unit_aware_arrays(z_s, "m")
    z_i = _ensure_unit_aware_arrays(z_i, "m")
    z_0 = _ensure_unit_aware_arrays(z_0, "m")
    u_bar_field = _ensure_unit_aware_arrays(u_bar_field, "m s^(-1)")
    v_bar_field = _ensure_unit_aware_arrays(v_bar_field, "m s^(-1)")

    dtheta = (theta_field[1:][height_index[1:]] - theta_field[:-1][height_index[1:]])
    theta_bar = (theta_field[1:][height_index[1:]] + theta_field[:-1][height_index[1:]])/2
    du = (u_bar_field[1:][height_index[1:]] - u_bar_field[:-1][height_index[1:]])
    u_bar = (u_bar_field[1:][height_index[1:]] + u_bar_field[:-1][height_index[1:]])/2
    dv = (v_bar_field[1:][height_index[1:]] - v_bar_field[:-1][height_index[1:]])
    v_bar = (v_bar_field[1:][height_index[1:]] + v_bar_field[:-1][height_index[1:]])/2
    dz = (z_field[1:][height_index[1:]] - z_field[:-1][height_index[1:]])
    z_bar = (z_field[1:][height_index[1:]] + z_field[:-1][height_index[1:]]) / 2
    dthetadz = dtheta/dz
    dudz = du/dz
    dvdz = dv/dz

    unstable = ((z_bar/L) <= 0) & ~(L.m == np.inf)

    K_h = np.zeros(dthetadz.size)
    if L < 0:
        dKhdz = (-9 * physics.von_Karman * u_star * z_s)/(1.48 * L * (1 - 9 * ((z_s)/L)))
        K_Hs = ((physics.von_Karman * u_star * z_s)/((1 - 9 * (z_s/L))**(-1/2)))
        K_h[unstable] = Quantity(0.001, "m^(2) s^(-1)") + ((z_i - z_bar[unstable])**2/(z_i - z_s)**2) * (K_Hs - Quantity(0.001, "m^(2) s^(-1)") + (z_bar[unstable] - z_s) * (dKhdz + 2 * ((K_Hs - Quantity(0.001, "m^(2) s^(-1)"))/(z_i - z_s))))
    else:
        K_h[unstable] = Quantity(0, "m^(2) s^(-1)")

    R_ic = 0.115 * ((dz[~unstable]).to("m").m)**0.175
    R_i = bulk_richardson_layer(theta_bar[~unstable], dthetadz[~unstable], dudz[~unstable], dvdz[~unstable])
    l = np.zeros(len(R_i))

    high = z_bar[~unstable] > Quantity(200, "m")
    low_Ri = R_i <= R_ic
    l[(low_Ri) & (~high)] = physics.von_Karman * z_bar[~unstable][(low_Ri) & (~high)]
    l[(low_Ri) & (high)] = Quantity(np.ones(len(l[(low_Ri) & (high)])) * 70, "m")
    K_h[~unstable][low_Ri] = 1.1 * ((R_ic[low_Ri] - R_i[low_Ri])/(R_ic[low_Ri])) * l[low_Ri]**2 * np.sqrt(dudz[~unstable][low_Ri]**2 + dvdz[~unstable][low_Ri]**2)
    K_h[~unstable][~low_Ri] = Quantity(np.zeros(len(K_h[~unstable][~low_Ri])), "m^(2) s^(-1)")

    return K_h

def ground_temperature_tendency(T_1 : Union[int, float, np.ndarray, Quantity], u_star : Union[int, float, np.ndarray, Quantity], theta_star : Union[int, float, np.ndarray, Quantity], T_g : Union[int, float, np.ndarray, Quantity], p_1 : Union[int, float, np.ndarray, Quantity], air_emissivity  : Union[int, float, np.ndarray, Quantity], soil_emissivity : Union[int, float, np.ndarray, Quantity], soil_density : Union[int, float, np.ndarray, Quantity], soil_heat_capacity : Union[int, float, np.ndarray, Quantity], soil_depth : Union[int, float, np.ndarray, Quantity], shortwave_reduction : float, current_model_time : datetime, lat : float) -> Quantity:
    T_1 = _ensure_unit_aware_arrays(T_1, "K")
    u_star = _ensure_unit_aware_arrays(u_star, "m s^(-1)")
    theta_star = _ensure_unit_aware_arrays(theta_star, "K")
    T_g = _ensure_unit_aware_arrays(T_g, "K")
    p_1 = _ensure_unit_aware_arrays(p_1, "Pa")
    air_emissivity = _ensure_unit_aware_arrays(air_emissivity, "dimensionless")
    soil_emissivity = _ensure_unit_aware_arrays(soil_emissivity, "dimensionless")
    soil_density = _ensure_unit_aware_arrays(soil_density, "kg m^(3)")
    soil_heat_capacity = _ensure_unit_aware_arrays(soil_heat_capacity, "J kg^(-1) K^(-1)")
    soil_depth = _ensure_unit_aware_arrays(soil_depth, "m")

    sensible_heat_flux = sensible_heat(p_1, T_1, u_star, theta_star)
    incoming_shortwave = ground_incoming_shortwave(current_model_time, lat, shortwave_reduction)
    outgoing_longwave = ground_outgoing_longwave(T_g, soil_emissivity)
    absorbed_longwave = soil_emissivity * air_emissivity * physics.Stefan_Boltzmann * T_1**4

    dTgdt = (1/(soil_density * soil_heat_capacity * soil_depth)) * (- sensible_heat_flux + incoming_shortwave + absorbed_longwave - outgoing_longwave)
    return dTgdt

def sensible_heat(p_1, T_1, u_star, theta_star) -> Quantity:
    p_1 = _ensure_unit_aware_arrays(p_1, "Pa").to_base_units()
    T_1 = _ensure_unit_aware_arrays(T_1, "K")
    theta_star = _ensure_unit_aware_arrays(theta_star, "K")
    u_star = _ensure_unit_aware_arrays(u_star, "m s^(-1)")

    H = - (p_1 / (physics.R_d * T_1)) * physics.c_p * u_star * theta_star
    return H

def ground_incoming_shortwave(ToA_incoming_exitance : np.ndarray, current_model_time : datetime, lat : float, shortwave_reduction : float) -> Quantity:
    local_hour = current_model_time.hour + current_model_time.minute / 60 + current_model_time.second / 3600
    solar_declination = -23.44 * np.cos((np.deg2rad(360)/365) * (int(current_model_time.strftime("%j")) + local_hour/24) + 10) # Very simple approximation
    solar_hour_angle = (local_hour - 12) * (360/24)
    solar_zenith_angle = np.arccos(np.sin(np.deg2rad(lat)) * np.sin(np.deg2rad(solar_declination)) + np.cos(np.deg2rad(lat)) * np.cos(np.deg2rad(solar_declination)) * np.cos(np.deg2rad(solar_hour_angle)))
    if Quantity(-90, "degree") <= np.rad2deg(solar_zenith_angle) <= Quantity(90, "degree"):
        incoming_shortwave = ToA_incoming_exitance * (0.6 * np.cos(solar_zenith_angle) + 0.2 * np.cos(solar_zenith_angle)**2)  * (1 - shortwave_reduction)
    else:
        incoming_shortwave = Quantity(np.zeros_like(ToA_incoming_exitance), "W m^(-1)")
    return incoming_shortwave

def ground_outgoing_longwave(T_g : Union[int, float, np.ndarray, Quantity], soil_emissivity : Union[int, float, np.ndarray, Quantity]) -> Quantity:
    T_g = _ensure_unit_aware_arrays(T_g, "K")
    soil_emissivity = _ensure_unit_aware_arrays(soil_emissivity, "dimensionless")

    LW_out = soil_emissivity * physics.Stefan_Boltzmann * T_g**4
    return LW_out

def longwave_radiative_transfer(T : Union[int, float, np.ndarray, Quantity], air_emissivity : float, soil_emissivity : float) -> Quantity:
    T = _ensure_unit_aware_arrays(T, "K")

    F_net = np.zeros(T.size)
    F_net[0] =  - physics.Stefan_Boltzmann * soil_emissivity * T[0]**4 + physics.Stefan_Boltzmann * soil_emissivity * air_emissivity * T[1]**4
    F_net[1] = - 2 * physics.Stefan_Boltzmann.m * air_emissivity * T[1].m**4 + physics.Stefan_Boltzmann * soil_emissivity * air_emissivity * T[0]**4 + physics.Stefan_Boltzmann * air_emissivity**2 * T[2]**4
    F_net[2:-1] = - 2 * physics.Stefan_Boltzmann.m * air_emissivity * T[2:-1].m**4 + physics.Stefan_Boltzmann * air_emissivity**2 * (np.roll(T, 1)[2:-1]**4 + np.roll(T, -1)[2:-1]**4)
    F_net[-1] = Quantity(0, "W m^(-2)")

    return F_net

def longwave_radiative_transfer_effective_layer(T : Union[int, float, np.ndarray, Quantity], air_emissivity : float, soil_emissivity : float, layer_ground_dist : np.ndarray, effective_radiative_layer_dist : np.ndarray) -> Quantity:
    T = _ensure_unit_aware_arrays(T, "K")

    F_net = np.zeros(T.size)
    Temps2d = np.tile(T, (T.size, 1))
    h = int(np.nanmax(effective_radiative_layer_dist))
    Temps2d = np.pad(Temps2d, [(0,0), (0,h+1)], mode = "edge")
        
    air_contrib = np.nansum(Temps2d**4 * (1 - air_emissivity)**effective_radiative_layer_dist, axis = 1)
    
    ground_contrib = T[0]**4 * (1 - air_emissivity)**(layer_ground_dist[:-h-1] - 1)
    F_net[0] =  - physics.Stefan_Boltzmann.m * soil_emissivity * T[0].m**4 + physics.Stefan_Boltzmann.m * soil_emissivity * air_emissivity * air_contrib[0].m
    F_net[1:] = - 2 * physics.Stefan_Boltzmann.m * air_emissivity * T[1:].m**4 + physics.Stefan_Boltzmann.m * air_emissivity**2 * air_contrib[1:].m + physics.Stefan_Boltzmann.m * soil_emissivity * air_emissivity * ground_contrib[1:].m
    F_net = Quantity(F_net, "W m^(-2)")

    return F_net

def get_emissivity(min_wavenumber : Union[int, float, np.ndarray, Quantity], max_wavenumber : Union[int, float, np.ndarray, Quantity], gasses = list[str]) -> list[np.ndarray]:
    ret = []
    dr = os.path.dirname(os.path.realpath(__file__))
    Data = NCDF(f"{dr}/radiation-data/radiation.nc", "r", format='NETCDF4')
    wavenumber = Quantity(Data["wavenumber"][:], "m^(-1)")
    selection = (wavenumber >= min_wavenumber) & (wavenumber <= max_wavenumber)

    for gas in gasses:
        ret.append(Data[f"{gas.upper()}_absorbtivity_smoothed"][selection])

    return ret, wavenumber[selection]

def get_absorbtion_coefficient(min_wavenumber : Union[int, float, np.ndarray, Quantity], max_wavenumber : Union[int, float, np.ndarray, Quantity], gasses = list[str]) -> list[np.ndarray]:
    ret = []
    dr = os.path.dirname(os.path.realpath(__file__))
    Data = NCDF(f"{dr}/radiation-data/radiation.nc", "r", format='NETCDF4')
    wavenumber = Quantity(Data["wavenumber"][:], "m^(-1)")
    selection = (wavenumber >= min_wavenumber) & (wavenumber <= max_wavenumber)

    for gas in gasses:
        ret.append(Data[f"{gas.upper()}_absorbtion_coefficient"][selection])

    return ret, wavenumber[selection]

def weighted_absorbtion_coefficients(e : Union[int, float, np.ndarray, Quantity], p : Union[int, float, np.ndarray, Quantity], absorbtion_coefficient_CO2 : np.ndarray, absorbtion_coefficient_H2O : np.ndarray, absorbtion_coefficient_O2 : np.ndarray, absorbtion_coefficient_N2 : np.ndarray) -> np.ndarray:
    e = _ensure_unit_aware_arrays(e, "Pa")
    p = _ensure_unit_aware_arrays(p, "Pa")

    p_2d = np.tile(p, (absorbtion_coefficient_N2.size, 1))
    #p_2d = (p_2d[:,1:] + p_2d[:,:-1]) / 2
    e_2d = np.tile(e, (absorbtion_coefficient_N2.size, 1))
    #e_2d = (e_2d[:,1:] + e_2d[:,:-1]) / 2

    absorbtion_coefficient_N2_2d = np.tile(absorbtion_coefficient_N2, (e.size, 1)).transpose()
    absorbtion_coefficient_O2_2d = np.tile(absorbtion_coefficient_O2, (e.size, 1)).transpose()
    absorbtion_coefficient_CO2_2d = np.tile(absorbtion_coefficient_CO2, (e.size, 1)).transpose()
    absorbtion_coefficient_H2O_2d = np.tile(absorbtion_coefficient_H2O, (e.size, 1)).transpose()

    tau = (((p_2d - e_2d)/p_2d) * 0.000426 * absorbtion_coefficient_CO2_2d) + (((p_2d - e_2d)/p_2d) * 0.78084 * absorbtion_coefficient_N2_2d) + (((p_2d - e_2d)/p_2d) * 0.20948 * absorbtion_coefficient_O2_2d) + ((e_2d/p_2d) * absorbtion_coefficient_H2O_2d)

    return tau.m

def ozone_absorbtion_coefficients(ozone_layer_average_ozone : float, ozone_layer_average_water_vapor : float, absorbtion_coefficient_CO2 : np.ndarray, absorbtion_coefficient_H2O : np.ndarray, absorbtion_coefficient_O2 : np.ndarray, absorbtion_coefficient_N2 : np.ndarray, absorbtion_coefficient_O3 : np.ndarray) -> np.ndarray:

    tau = 0.000426 * absorbtion_coefficient_CO2 + 0.78084 * absorbtion_coefficient_N2 + 0.20948 * absorbtion_coefficient_O2 + ozone_layer_average_water_vapor * absorbtion_coefficient_H2O + ozone_layer_average_ozone * absorbtion_coefficient_O3

    return tau

def point_to_point_exitance(T : Union[int, float, np.ndarray, Quantity], layer_depth : Union[int, float, np.ndarray, Quantity], ground_emissivity : float, absorbtion_coefficients : Union[int, float, np.ndarray, Quantity], wavenumber : Union[int, float, np.ndarray, Quantity], M_in : np.ndarray, ozone_absorbtion_coeff : np.ndarray, ozone_layer_temperature : float, ozone_layer_depth : float) -> Quantity:
    #T = _ensure_unit_aware_arrays(T, "K")
    #layer_depth = _ensure_unit_aware_arrays(layer_depth, "m")
    #absorbtion_coefficients = _ensure_unit_aware_arrays(absorbtion_coefficients, "dimensionless")

    M = spectral_exitance(T, wavenumber)

    #M_path_upwards = np.zeros((M.shape[0], M.shape[1]-1))
    #M_path_downwards = np.zeros((M.shape[0], M.shape[1]-1))
    M_net = np.zeros_like(M)
    transmittance = np.exp(-absorbtion_coefficients*(layer_depth[np.newaxis,:]))
    ozone_transmittance = np.exp(-ozone_absorbtion_coeff*ozone_layer_depth)
    ozone_emission = spectral_exitance(np.array(ozone_layer_temperature, ndmin = 1), wavenumber)
    absobrtivity = 1 - transmittance

    up = M[:,0] * ground_emissivity
    M_net[:,0] = -M[:,0] * ground_emissivity
    for i in range(1, layer_depth.size - 1):
        transmitted = up * transmittance[:,i]
        produced = M[:,i] * absobrtivity[:,i]
        absorbed = up * absobrtivity[:,i]
        up = transmitted + produced
        M_net[:,i] = absorbed - produced
    


    down = M_in[:,0] * ozone_transmittance + ozone_emission[:,0] * (1 - ozone_transmittance)
    M_net[:,-1] = down * absobrtivity[:,-1]
    down = down * transmittance[:,-1] + M[:,-1] * absobrtivity[:,-1]
    for i in range(layer_depth.size-1, 0, -1):
        transmitted = down * transmittance[:,i]
        produced = M[:,i] * absobrtivity[:,i]
        absorbed = down * absobrtivity[:,i]
        down = transmitted + produced
        M_net[:,i] = M_net[:,i] + (absorbed - produced)

    absorbed = down * ground_emissivity
    M_net[:,0] = M_net[:,0] + (absorbed)

    return Quantity(M_net, "W m^(-1)")

def weighted_absorbtivity_spectra(e : Union[int, float, np.ndarray, Quantity], p : Union[int, float, np.ndarray, Quantity], emissivity_CO2 : np.ndarray, emissivity_H2O : np.ndarray, emissivity_O2 : np.ndarray, emissivity_N2 : np.ndarray) -> Quantity:
    e = _ensure_unit_aware_arrays(e, "Pa")
    p = _ensure_unit_aware_arrays(p, "Pa")

    p_2d = np.tile(p, (emissivity_N2.size, 1))
    e_2d = np.tile(e, (emissivity_N2.size, 1))
    emissivity_N2_2d = np.tile(emissivity_N2, (e.size, 1)).transpose()
    emissivity_O2_2d = np.tile(emissivity_O2, (e.size, 1)).transpose()
    emissivity_CO2_2d = np.tile(emissivity_CO2, (e.size, 1)).transpose()
    emissivity_H2O_2d = np.tile(emissivity_H2O, (e.size, 1)).transpose()

    absorbtivity_spectra = 1 - np.e**(-(-((((p_2d - e_2d)/p_2d) * 0.000426)[:,:] * np.log(1-emissivity_CO2_2d[:,:]+0.000000001))-((((p_2d - e_2d)/p_2d) * 0.78084)[:,:] * np.log(1-emissivity_N2_2d[:,:]+0.000000001))-((((p_2d - e_2d)/p_2d) * 0.20948)[:,:] * np.log(1-emissivity_O2_2d[:,:]+0.000000001))-((e_2d/p_2d)[:,:] * np.log(1-emissivity_H2O_2d[:,:]+0.000000001))))

    return absorbtivity_spectra

#def weighted_absorbtivity_spectra(e : Union[int, float, np.ndarray, Quantity], p : Union[int, float, np.ndarray, Quantity], emissivity_CO2 : np.ndarray, emissivity_H2O : np.ndarray, emissivity_O2 : np.ndarray, emissivity_N2 : np.ndarray) -> Quantity:
#    e = _ensure_unit_aware_arrays(e, "Pa")
#    p = _ensure_unit_aware_arrays(p, "Pa")
#
#    absorbtivity_spectra = (((p - e)/p) * 0.78084) * emissivity_N2 + (((p - e)/p) * 0.20948) * emissivity_O2 + (((p - e)/p) * 0.000426) * emissivity_CO2 + (e/p) * emissivity_H2O
#
#    return absorbtivity_spectra

@nb.njit(cache=True,fastmath=True,parallel=True)
def planck(T : np.ndarray, wavenumber : np.ndarray, const_1 : float, const_2 : float) -> np.ndarray:
    num_layers = T.size
    num_wn = wavenumber.size
    B = np.empty((num_wn, num_layers))

    for i in nb.prange(num_wn):
        wn = wavenumber[i]
        wn_cubed = wn*wn*wn
        for j in range(num_layers):
            exponent = (const_1 * wn)/(T[j])
            B[i,j] = (const_2 * wn_cubed) / (np.exp(exponent) - 1)
    
    return B

def spectral_exitance(T : Union[int, float, np.ndarray], wavenumber : Union[int, float, np.ndarray]) -> Quantity:
    #T = _ensure_unit_aware_arrays(T, "K")
    #wavenumber = _ensure_unit_aware_arrays(wavenumber, "m^(-1)")

    B = planck(T, wavenumber, ((physics.Planck * physics.c)/physics.Boltzmann).m, (2 * physics.Planck * physics.c**2).m)
    M = B * np.pi

    return M

def filtered_exitance(M : Union[int, float, np.ndarray, Quantity], absorbtivity_spectra : np.ndarray) -> Quantity:
    M = _ensure_unit_aware_arrays(M, "W m^(-1)")

    absorbtivity_spectra2d = np.tile(absorbtivity_spectra, (M.shape[1], 1)).transpose()

    M_prime = M * absorbtivity_spectra2d

    return M_prime

def irradiance(M_prime : Union[int, float, np.ndarray, Quantity]) -> Quantity:
    M_prime = _ensure_unit_aware_arrays(M_prime, "W m^(-1)")

    I = np.sum(M_prime, axis = 0) * Quantity(10, "cm^(-1)").to("m^(-1)")

    return I

def net_radiant_longwave_power(M_prime : Union[int, float, np.ndarray, Quantity], T_g : Union[int, float, np.ndarray, Quantity], ground_emissivity : float, filters : list[np.ndarray], wavenumber : Union[int, float, np.ndarray, Quantity], air_emissivity : Union[int, float, np.ndarray, Quantity]) -> Quantity:
    M_prime = _ensure_unit_aware_arrays(M_prime, "W m^(-1)")
    T_g = _ensure_unit_aware_arrays(T_g, "K")
    wavenumber = _ensure_unit_aware_arrays(wavenumber, "m^(-1)")

    lower_filter_inner, upper_filter_inner, upper_absorbtive_filter, lower_absorbtive_filter = filters

    weighted_absorbtivity = np.expand_dims(weighted_absorbtivity, axis = 0)
    weighted_absorbtivity = np.tile(weighted_absorbtivity, (lower_filter_inner.shape[0], 1, 1))

    weighted_absorbtivity = np.expand_dims(weighted_absorbtivity, axis = 3)
    weighted_absorbtivity = np.tile(weighted_absorbtivity, (1, 1, 1, ()-1))

    weighted_absorbtivity_product_upper = np.where(upper_filter_inner[:,:,:,-1], np.nanprod(1 - np.where(upper_absorbtive_filter[:,:,:,:], weighted_absorbtivity, np.nan), axis = 3), np.nan)
    weighted_absorbtivity_product_lower = np.where(lower_filter_inner[:,:,:,-1], np.nanprod(1 - np.where(lower_absorbtive_filter[:,:,:,:], weighted_absorbtivity, np.nan), axis = 3), np.nan)

    M_prime = np.expand_dims(M_prime, axis = 0)
    M_prime = np.tile(M_prime, (lower_filter_inner.shape[0]-1, 1, 1))
    M_prime_upper = np.where(upper_filter_inner[1:,:,:,-1], M_prime, np.nan)
    M_prime_lower = np.where(lower_filter_inner[1:,:,:,-1], M_prime, np.nan)

    M_g = spectral_exitance(T_g, wavenumber)
    M_g = np.expand_dims(M_g, axis = 1)
    M_g = np.tile(M_g, (1, lower_filter_inner.shape[2]))
    M_g_prime = ground_emissivity * M_g

    M_prime_prime = np.nansum(weighted_absorbtivity_product_upper[1:,:,:] * M_prime_upper, axis = 0) + np.nansum(weighted_absorbtivity_product_lower[1:,:,:] * M_prime_lower, axis = 0)
    M_g_prime_prime = np.nansum(weighted_absorbtivity_product_upper[:weighted_absorbtivity_product_upper.shape[2],:,0].transpose() * M_g_prime, axis = 0)


    I_prime_air = np.sum(weighted_absorbtivity[0,:,:,-1] * M_prime_prime, axis = 0) * Quantity(1, "cm^(-1)").to_base_units()

    I_prime_g = np.sum(weighted_absorbtivity[0,:,:,-1] * M_g_prime_prime, axis = 0) * Quantity(1, "cm^(-1)").to_base_units()

    P = np.zeros(weighted_absorbtivity.shape[2])

    P[0] = (- irradiance(M_g_prime[:,0]) + ground_emissivity * I_prime_air[1:])# * Quantity(1, "m^2")
    P[1:] = (- 2 * irradiance(M_prime[1:]) + I_prime_air[1:] + I_prime_g[1:])# * Quantity(1, "m^2")

    del weighted_absorbtivity_product_lower
    del weighted_absorbtivity_product_upper
    del M_prime_lower
    del M_prime_upper
    del M_prime_prime
    del M_g_prime
    del M_g_prime_prime

    return P

def net_irradience_simplified(T : Union[int, float, np.ndarray, Quantity], e : Union[int, float, np.ndarray, Quantity], p : Union[int, float, np.ndarray, Quantity], ground_emissivity : float, epsilon_a : np.ndarray, wavenumber : Union[int, float, np.ndarray, Quantity]) -> Quantity:
    T = _ensure_unit_aware_arrays(T, "K")
    e = _ensure_unit_aware_arrays(e, "Pa")
    p = _ensure_unit_aware_arrays(p, "Pa")
    wavenumber = _ensure_unit_aware_arrays(wavenumber, "m^(-1)")


    M = spectral_exitance(T, wavenumber)
    #print(M.shape)
    #print(epsilon_a.shape)
    M_prime = np.zeros(M.shape)
    M_prime[:,0] = ground_emissivity * M[:,0].m
    M_prime[:,1:] = epsilon_a[:,1:] * M[:,1:].m
    M_prime = Quantity(M_prime, M.units)

    I = np.zeros(M.shape[1])
    I[0] = - irradiance(M_prime[:,0]).m + ground_emissivity * irradiance(M_prime[:,1]).m
    I[1] = - 2 * irradiance(M_prime[:,1]).m + irradiance(epsilon_a[:,1] * M_prime[:,0].m).m + irradiance(M_prime[:,2].m).m
    I[2:-1] = - 2 * irradiance(M_prime[:,2:-1]).m + irradiance(np.roll(M_prime[:,2:-1].m, (0, 1)) + np.roll(M_prime[:,2:-1].m, (0, -1))).m
    I[-1] = - irradiance(M_prime[:,-1]).m + irradiance(M_prime[:,-2]).m

    I = Quantity(I, "W m^(-2)")

    return I

if __name__ == "__main__":

    pass