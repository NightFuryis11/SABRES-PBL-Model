from pint import Quantity

von_Karman = Quantity(0.35, "dimensionless")                            # von Karman constant
g_0 = Quantity(9.81, "m s^(-2)")                                        # Standard gravity
c_p = Quantity(1005.7, "J K^(-1) kg^(-1)")                              # Specific heat of air at constant pressure
e_s_tp = Quantity(661, "Pa")                                            # Saturation vapor pressure at the triple point of water
R_v = Quantity(461.5, "J K^(-1) kg^(-1)")                               # Water vapor gas constant
R_d = Quantity(287.5, "J K^(-1) kg^(-1)")                               # Dry air gas constant
T_tp = Quantity(273.16, "K")                                            # Temperature at the triple point of water
L_tp = Quantity(2.501*10**6, "J kg^(-1)")                               # Latent heat of vaporization at the triple point of water
c_pv = Quantity(1864, "J K^(-1) kg^(-1)")                               # Specific heat of water vapor at constant pressure
c_pl = Quantity(4184, "J K^(-1) kg^(-1)")                               # Specific heat of liquid water at constant pressure
p_0 = Quantity(100000, "Pa")                                            # Standard reference pressure
omega = Quantity(-7.292*10**(-5), "s^(-1)")                             # Planetary rotation rate
k_m_z_i = Quantity(0.001, "m^(2) s^(-1)")                               # Eddy momentum diffusivity at the top of the boundary layer
k_h_z_i = Quantity(0.001, "m^(2) s^(-1)")                               # Eddy heat diffusivity at the top of the boundary layer
S_0 = Quantity(342.0, "W m^(-2)")                                       # Average incoming solar shortwave radiation at the top of the atmosphere
Stefan_Boltzmann = Quantity(5.670 * 10**(-8), "W m^(-2) K^(-4)")        # Stefan-Boltzman radiative constant