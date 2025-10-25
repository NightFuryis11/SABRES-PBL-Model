import numpy as np
from netCDF4 import Dataset as NCDF
from netCDF4 import date2num, num2date
import os
from typing import Union
from datetime import datetime, timedelta
from pytz import timezone
import grid
import ingest
import dynamics
import physics
from pint import Quantity
import matplotlib.pyplot as plt
from timezonefinder import TimezoneFinder
import sys
import pandas as pd
from time import sleep
import shutil
from scipy import interpolate

def get_config(filepath : str = "./pbl.config") -> Union[dict, None]:
    if os.path.exists(filepath):
        cfg_dict = {}
        with open(filepath, "r") as F:
            for line in F.readlines():
                if line =="\n":
                    pass
                elif line.startswith("#") or line.startswith("\\"):
                    pass
                else:
                    splitline = line.split("=")
                    val, unit, datatype = [x.strip() for x in splitline[1].split(",")]
                    if datatype == "int":
                        cfg_dict[f"{splitline[0].strip()}"] = int(val)
                    elif datatype == "float":
                        cfg_dict[f"{splitline[0].strip()}"] = float(val)
                    elif datatype == "bool":
                        cfg_dict[f"{splitline[0].strip()}"] = bool(int(val))
                    
                    if not unit == "":
                        cfg_dict[f"{splitline[0].strip()}"] = Quantity(cfg_dict[f"{splitline[0].strip()}"], unit)
        return cfg_dict
    else:
        return None

def plot_timeseries(filename : str, title : str = "Latest_Output", vertical_line : list = [-1]) -> None:
    Dataset = NCDF(filename, "r", format = "NETCDF4")

    z = Dataset["height"]
    dz = Dataset["height_change"]
    mdz = Dataset["midpoint_height_change"]
    p = Dataset["pressure"]
    dpdt = Dataset["dpdt"]
    T = Dataset["temperature"]
    theta = Dataset["potential_temperature"]
    T_g = Dataset["surface_temperature"]
    dTdt = Dataset["dTdt"]
    r_v = Dataset["mixing_ratio"]
    dr_vdt = Dataset["dr_vdt"]
    u_g = Dataset["u-component_of_geostrophic_wind"]
    v_g = Dataset["v-component_of_geostrophic_wind"]
    u_bar = Dataset["u-component_of_mean_wind"]
    v_bar = Dataset["v-component_of_mean_wind"]
    dudt = Dataset["dudt"]
    dvdt = Dataset["dvdt"]
    dthetadt = Dataset["dthetadt"]
    z_s = Dataset["surface_layer_height"]
    z_i = Dataset["pbl_top_height"]
    dz_idt = Dataset["dz_idt"]
    uw = Dataset["u-momentum_flux"]
    vw = Dataset["v-momentum_flux"]
    thetaw = Dataset["potential_temperature_flux"]
    K_m = Dataset["eddy_momentum_diffusivity"]
    K_h = Dataset["eddy_heat_diffusivity"]
    puwpz = Dataset["differential_u-momentum_flux"]
    pvwpz = Dataset["differential_v-momentum_flux"]
    pthetawpz = Dataset["differential_potential_temperature_flux"]
    data_time = Dataset["time"]
    model_time = Dataset["model_time"]
    shortwave_reduction = Dataset["shortwave_reduction"]
    SW_in = Dataset["incoming_solar_radiation"]
    data_tz = Dataset["timezone"][0]

    if "lat" in Dataset.variables.keys():
        lat = Dataset["lat"]
    else:
        lat = config["default_lat"]
    
    if "lon" in Dataset.variables.keys():
        lon = Dataset["lon"]
    else:
        lon = config["default_lon"]

    local_time = np.array([num2date(x, data_time.units, only_use_cftime_datetimes=False).replace(tzinfo = timezone("UTC")).astimezone(timezone(data_tz)).strftime("%Y-%m-%d %H:%M:%S") for x in data_time[:].data])

    
    #theta[0,:] = [280, 282, 284, 286]


    fig = plt.figure(1, figsize = (20, 7))
    #ax1 = fig.add_subplot(4, 3, 1)
    #ax2 = fig.add_subplot(4, 3, 2)
    #ax3 = fig.add_subplot(4, 3, 3)
    #ax4 = fig.add_subplot(4, 3, (4, 6))
    #ax5 = fig.add_subplot(4, 3, (7, 9))
    #ax6 = fig.add_subplot(4, 3, (10, 12))

    ax1 = fig.add_subplot(3, 3, 3)
    ax2 = fig.add_subplot(3, 3, 6)
    ax3 = fig.add_subplot(3, 3, 9)

    ax4 = fig.add_subplot(3, 3, 1)
    ax5 = fig.add_subplot(3, 3, 4)
    ax6 = fig.add_subplot(3, 3, 7)

    ax7 = fig.add_subplot(3, 3, 2)
    ax8 = fig.add_subplot(3, 3, 5)
    ax9 = fig.add_subplot(3, 3, 8)

    #print(Dataset["differential_v-momentum_flux"][:,:])
    #print(Dataset["differential_potential_temperature_flux"][:,:])

    for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]:
        ax.plot(local_time, z_s[:].data, color = "grey", linestyle = "dotted", linewidth = 0.5, label = "Surface Layer Height", zorder = 3)
        ax.plot(local_time, z_i[:].data, color = "black", linestyle = "dashed", linewidth = 0.5, label = "Boundary Layer Height", zorder = 4)
        for val in vertical_line:
            if val != -1:
                ax.plot(np.ones(z[:].data.shape) * data_time[val].data, z[:].data, color = "cyan", linewidth = 0.5, zorder = 5)
        
    
    surface_potential_temperature = dynamics.poissons_T_to_theta(T_g[:], p[:,0], r_v[:,0])

    #ax3.plot(Dataset["time"][:].data, surface_potential_temperature, color = "lime")
    #ax3.plot(Dataset["time"][:].data, Dataset["potential_temperature"][:,1].data, color = "orange")

    theta_series = np.insert(theta[:,1:], 0, surface_potential_temperature.m, axis = 1)

    u_bar_plot = ax1.pcolormesh(local_time, z[:].data, np.swapaxes(u_bar[:,:], 0, 1), shading='nearest', vmin = -np.nanmax([np.abs(np.nanmin(u_bar[:,:])), np.abs(np.nanmax(u_bar[:,:]))]), vmax = np.nanmax([np.abs(np.nanmin(u_bar[:,:])), np.abs(np.nanmax(u_bar[:,:]))]), cmap = "PuOr", zorder = 1)
    v_bar_plot = ax2.pcolormesh(local_time, z[:].data, np.swapaxes(v_bar[:,:], 0, 1), shading='nearest', vmin = -np.nanmax([np.abs(np.nanmin(v_bar[:,:])), np.abs(np.nanmax(v_bar[:,:]))]), vmax = np.nanmax([np.abs(np.nanmin(v_bar[:,:])), np.abs(np.nanmax(v_bar[:,:]))]), cmap = "PuOr", zorder = 1)
    theta_plot = ax3.pcolormesh(local_time, z[:].data, np.swapaxes(theta_series, 0, 1), shading='nearest', vmin = np.nanmin(theta_series), vmax = np.nanmax(theta_series), cmap = "Reds", zorder = 1)

    uw_plot = ax4.pcolormesh(local_time, z[:].data, np.swapaxes(uw[1:,:], 0, 1), shading="flat", vmin = -np.nanmax([np.abs(np.nanmin(uw[1:,:])), np.abs(np.nanmax(uw[1:,:]))]), vmax = np.nanmax([np.abs(np.nanmin(uw[1:,:])), np.abs(np.nanmax(uw[1:,:]))]), cmap = "PuOr", zorder = 1)
    vw_plot = ax5.pcolormesh(local_time, z[:].data, np.swapaxes(vw[1::,:], 0, 1), shading="flat", vmin = -np.nanmax([np.abs(np.nanmin(vw[1:,:])), np.abs(np.nanmax(vw[1:,:]))]), vmax = np.nanmax([np.abs(np.nanmin(vw[1:,:])), np.abs(np.nanmax(vw[1:,:]))]), cmap = "PuOr", zorder = 1)
    thetaw_plot = ax6.pcolormesh(local_time, z[:].data, np.swapaxes(thetaw[1:,:], 0, 1), shading="flat", vmin = -np.nanmax([np.abs(np.nanmin(thetaw[1:,:])), np.abs(np.nanmax(thetaw[1:,:]))]), vmax = np.nanmax([np.abs(np.nanmin(thetaw[1:,:])), np.abs(np.nanmax(thetaw[1:,:]))]), cmap = "bwr", zorder = 1)

    puwpz_plot = ax7.pcolormesh(local_time, z[:].data, np.swapaxes(puwpz[:,::], 0, 1), shading='nearest', vmin = -np.nanmax([np.abs(np.nanmin(puwpz[:,:])), np.abs(np.nanmax(puwpz[:,:]))]), vmax = np.nanmax([np.abs(np.nanmin(puwpz[:,:])), np.abs(np.nanmax(puwpz[:,:]))]), cmap = "PuOr", zorder = 1)
    pvwpz_plot = ax8.pcolormesh(local_time, z[:].data, np.swapaxes(pvwpz[:,:], 0, 1), shading='nearest', vmin = -np.nanmax([np.abs(np.nanmin(pvwpz[:,:])), np.abs(np.nanmax(pvwpz[:,:]))]), vmax = np.nanmax([np.abs(np.nanmin(pvwpz[:,:])), np.abs(np.nanmax(pvwpz[:,:]))]), cmap = "PuOr", zorder = 1)
    pthetawpz_plot = ax9.pcolormesh(local_time, z[:].data, np.swapaxes(pthetawpz[:,:], 0, 1), shading='nearest', vmin = -np.nanmax([np.abs(np.nanmin(pthetawpz[:,:])), np.abs(np.nanmax(pthetawpz[:,:]))]), vmax = np.nanmax([np.abs(np.nanmin(pthetawpz[:,:])), np.abs(np.nanmax(pthetawpz[:,:]))]), cmap = "bwr", zorder = 1)

    tick_ids = np.round(np.linspace(0, len(local_time) - 1, 8)).astype(int)
    
    ax1.set_xticks(local_time[tick_ids])
    ax1.set_xticklabels([])
    ax1.set_title("Output")
    ax2.set_xticks(local_time[tick_ids])
    ax2.set_xticklabels([])
    ax3.set_xticks(local_time[tick_ids])
    ax3.set_xticklabels(local_time[tick_ids], rotation = 80)

    ax4.set_xticks(local_time[tick_ids])
    ax4.set_xticklabels([])
    ax4.set_title("Fluxes")
    ax4.set_ylabel("z")
    ax5.set_xticks(local_time[tick_ids])
    ax5.set_xticklabels([])
    ax5.set_ylabel("z")
    ax6.set_xticks(local_time[tick_ids])
    ax6.set_xticklabels(local_time[tick_ids], rotation = 80)
    ax6.set_ylabel("z")

    ax7.set_xticks(local_time[tick_ids])
    ax7.set_xticklabels([])
    ax7.set_title("Differential Fluxes")
    ax8.set_xticks(local_time[tick_ids])
    ax8.set_xticklabels([])
    ax9.set_xticks(local_time[tick_ids])
    ax9.set_xticklabels(local_time[tick_ids], rotation = 80)

    cbar1 = fig.colorbar(u_bar_plot, pad = 0.01)
    cbar1.set_label(r"$\overline{u}$")
    cbar2 = fig.colorbar(v_bar_plot, pad = 0.01)
    cbar2.set_label(r"$\overline{v}$")
    cbar3 = fig.colorbar(theta_plot, pad = 0.01)
    cbar3.set_label(r"$\overline{\theta}$")

    cbar4 = fig.colorbar(uw_plot, pad = 0.01)
    cbar4.set_label(r"$u'w'$")
    cbar5 = fig.colorbar(vw_plot, pad = 0.01)
    cbar5.set_label(r"$v'w'$")
    cbar6 = fig.colorbar(thetaw_plot, pad = 0.01)
    cbar6.set_label(r"$\theta' w'$")

    cbar7 = fig.colorbar(puwpz_plot, pad = 0.01)
    cbar7.set_label(r"$\frac{\partial \overline{u'w'}}{\partial z}$")
    cbar8 = fig.colorbar(pvwpz_plot, pad = 0.01)
    cbar8.set_label(r"$\frac{\partial \overline{v'w'}}{\partial z}$")
    cbar9 = fig.colorbar(pthetawpz_plot, pad = 0.01)
    cbar9.set_label(r"$\frac{\partial \overline{\theta' w'}}{\partial z}$")

    fig.suptitle(f'PBL Model Output from {local_time[0]} to {local_time[-1]}')
    fig.savefig(f"{title}.png", dpi = 150, bbox_inches = "tight")
    plt.close()
    Dataset.close()

def plot_radiation(filename : str, title : str = "Latest_Output_Radiation"):
    Dataset = NCDF(filename, "r", format = "NETCDF4")

    z = dynamics._ensure_unit_aware_arrays(Dataset["height"][:], "m")
    theta = dynamics._ensure_unit_aware_arrays(Dataset["potential_temperature"][:,:], "K")
    u_bar = dynamics._ensure_unit_aware_arrays(Dataset["u-component_of_mean_wind"][:,:], "m s^(-1)")
    v_bar = dynamics._ensure_unit_aware_arrays(Dataset["v-component_of_mean_wind"][:,:], "m s^(-1)")
    T_g = dynamics._ensure_unit_aware_arrays(Dataset["surface_temperature"][:], "K")
    z_s = dynamics._ensure_unit_aware_arrays(Dataset["surface_layer_height"][:], "m")
    z_i = dynamics._ensure_unit_aware_arrays(Dataset["pbl_top_height"][:], "m")
    p = dynamics._ensure_unit_aware_arrays(Dataset["pressure"][:,:], "Pa")
    r_v = dynamics._ensure_unit_aware_arrays(Dataset["mixing_ratio"][:,:], "kg kg^(-1)")
    SW = dynamics._ensure_unit_aware_arrays(Dataset["incoming_solar_radiation"][:], "W m^(-2)")
    u_star = dynamics._ensure_unit_aware_arrays(Dataset["u_star"][:], "m s^(-1)")
    theta_star = dynamics._ensure_unit_aware_arrays(Dataset["theta_star"][:], "K")
    data_time = Dataset["time"]
    data_tz = Dataset["timezone"][:]
    z_0 = Quantity(0.1, "m")
    lat = Quantity(33.16, "degree")
    model_time = Dataset["model_time"]
    rho = Quantity(1.2, "kg m^(3)")
    # p[this_ind,1].to_base_units() / (physics.R_d * T_1)
    # rho

    local_time = np.array([num2date(x, data_time.units, only_use_cftime_datetimes=False).replace(tzinfo = timezone("UTC")).astimezone(timezone(data_tz)).strftime("%Y-%m-%d %H:%M:%S") for x in data_time[:].data])

    T_1 = dynamics.poissons_theta_to_T(theta[:,1], p[:,1], r_v[:,1])
    H = - (p[:,1].to_base_units() / (physics.R_d * T_1)) * physics.c_p * u_star[:] * theta_star[:]
    #print((p[:,1].to_base_units() / (physics.R_d * T_1[:])), u_star[:], theta_star[:])
    SW_in = SW[:]
    LW_in = physics.Stefan_Boltzmann * T_1**4
    LW_out = dynamics.ground_outgoing_longwave(T_g[:], 0.9)

    theta_bar = (dynamics.poissons_T_to_theta(T_g[:], p[:,0], r_v[:,0]) + theta[:,1])/2
    dthetatz = (theta[:,1] - dynamics.poissons_T_to_theta(T_g[:], p[:,0], r_v[:,0]))/(z_s[:])
    dudz = (u_bar[:,1])/(z_s[:])
    dvdz = (v_bar[:,1])/(z_s[:])

    R_i = dynamics.bulk_richardson_layer(theta_bar, dthetatz, dudz, dvdz)
    #print(R_i)
    F_m = 1 - ((9.4 * R_i)/(1 + 7.4 * (physics.von_Karman**2 / (np.log(z_s[:]/z_0))**2) * 9.4 * np.sqrt(z_s[:]/z_0) * np.sqrt(np.abs(R_i))))
    #print(F_m)
    u_star_squaredn = (physics.von_Karman**2 / (np.log(z_s[:]/z_0))**2) * (np.sqrt(u_bar[:,1]**2 + v_bar[:,1]**2))**2 * F_m
    #print(u_star_squaredn)
    #du_starn = np.sqrt(u_star_squaredn)
    #print((physics.von_Karman**2 / (np.log(z_s[:]/z_0))**2))
    #print(u_bar[:,1], v_bar[:,1])
    #print((np.sqrt(u_bar[:,1]**2 + v_bar[:,1]**2))**2)


    fig = plt.figure(2, figsize = (12, 8))
    ax = fig.add_subplot(111)
    ax2 = ax.twinx()
    ax.plot(model_time[:]/3600, H[:], color = "green", label = r"$H$")
    ax.plot(model_time[:]/3600, SW_in[:], color = "yellow", label = r"$F^\downarrow_{SW}$")
    ax.plot(model_time[:]/3600, LW_in[:], color = "red", label = r"$F^\downarrow_{LW}$")
    ax.plot(model_time[:]/3600, LW_out[:], color = "orange", label = r"$F^\uparrow_{LW}$")
    ax.plot(model_time[:]/3600, -H[:] + SW_in[:] + LW_in[:] - LW_out[:], color = "black", label = "Sum")
    ax.set_ylabel(r"Surface Energy Contribution ($\frac{W}{m^{-2}}$)")
    ax.set_xlabel("Hours Since Model Initialization")

    ax2.plot(model_time[:]/3600, T_1[:], color = "cyan", label = r"$T_1$", linestyle = "dashed")
    ax2.plot(model_time[:]/3600, T_g[:], color = "blue", label = r"$T_g$", linestyle = "dashed")
    ax2.set_ylabel(r"Temperature ($K$)")

    fig.legend()
    fig.suptitle(f'PBL Model SFC Energy Budget from {local_time[0]} to {local_time[-1]}')
    fig.savefig(f"{title}.png", dpi = 150, bbox_inches = "tight")
    plt.close()
    Dataset.close()

def progress(total_time : float, timestep : float, current_step : int) -> None:
    sys.stdout.write("\r")

    num_steps = np.ceil(total_time.to("s").m / timestep.to("s").m)
    fraction = current_step / num_steps


    sys.stdout.write("[%-40s] %s" % ('='*int(40*fraction), f"{current_step:>4}/{int(num_steps):<4}"))

def run_model(config : dict, prepared_filename : str, **kwargs) -> str:
    if "progress_tracker" in kwargs.keys():
        progress_tracker = kwargs['progress_tracker']
    else:
        progress_tracker = True
    
    if "fudge_time" in kwargs.keys():
        fudge_time = timedelta(seconds = kwargs["fudge_time"].to("s").m)
    else:
        fudge_time = timedelta(seconds = 0)

    runtime = datetime.now(timezone("UTC"))

    output_filename = prepared_filename.replace("INPUT", "OUTPUT")
    if os.path.isfile(output_filename):
        output_filename = output_filename.split(".")[0] + "__" + runtime.strftime("%H-%H-%S") + ".nc"
    shutil.copy(prepared_filename, output_filename)

    ground_energy_enabled = config["ground_energy_enabled"]
    surface_layer_enabled = config["surface_layer_enabled"]
    boundary_layer_enabled = config["boundary_layer_enabled"]
    free_atmosphere_distinction = config["free_atmosphere_distinction"]
    model_top_enabled = config["model_top_enabled"]

    soil_emissivity = config["soil_emissivity"]
    soil_density = config["soil_density"]
    soil_heat_capacity = config["soil_heat_capacity"]
    soil_depth = config["soil_depth"]

    Dataset = NCDF(output_filename, "r+", format = "NETCDF4")

    z = Dataset["height"]
    dz = Dataset["height_change"]
    mdz = Dataset["midpoint_height_change"]
    p = Dataset["pressure"]
    dpdt = Dataset["dpdt"]
    T = Dataset["temperature"]
    theta = Dataset["potential_temperature"]
    T_g = Dataset["surface_temperature"]
    dTdt = Dataset["dTdt"]
    r_v = Dataset["mixing_ratio"]
    dr_vdt = Dataset["dr_vdt"]
    u_g = Dataset["u-component_of_geostrophic_wind"]
    v_g = Dataset["v-component_of_geostrophic_wind"]
    u_bar = Dataset["u-component_of_mean_wind"]
    v_bar = Dataset["v-component_of_mean_wind"]
    dudt = Dataset["dudt"]
    dvdt = Dataset["dvdt"]
    dthetadt = Dataset["dthetadt"]
    z_s = Dataset["surface_layer_height"]
    z_i = Dataset["pbl_top_height"]
    dz_idt = Dataset["dz_idt"]
    uw = Dataset["u-momentum_flux"]
    vw = Dataset["v-momentum_flux"]
    thetaw = Dataset["potential_temperature_flux"]
    K_m = Dataset["eddy_momentum_diffusivity"]
    K_h = Dataset["eddy_heat_diffusivity"]
    puwpz = Dataset["differential_u-momentum_flux"]
    pvwpz = Dataset["differential_v-momentum_flux"]
    pthetawpz = Dataset["differential_potential_temperature_flux"]
    time = Dataset["time"]
    model_time = Dataset["model_time"]
    shortwave_reduction = Dataset["shortwave_reduction"]
    SW_in = Dataset["incoming_solar_radiation"]
    u_star = Dataset["u_star"]
    theta_star = Dataset["theta_star"]
    data_tz = Dataset["timezone"][0]

    if "lat" in Dataset.variables.keys():
        lat = Dataset["lat"]
    else:
        lat = config["default_lat"]
    
    if "lon" in Dataset.variables.keys():
        lon = Dataset["lon"]
    else:
        lon = config["default_lon"]

    elapsed_time = Quantity(0, "s")
    current_time_index = np.count_nonzero(~np.isnan(theta[:,1])) - 1
    current_model_state_time_utc = num2date(time[current_time_index], time.units, only_use_cftime_datetimes=False).replace(tzinfo = timezone("UTC")) + fudge_time
    current_model_state_time_local = current_model_state_time_utc.astimezone(timezone(data_tz))
    timestep = config["time_step"]
    total_time = config["total_time"]
    z_0 = config["roughness_length"]

    if progress_tracker:
        progress(total_time, timestep.to("s"), current_time_index)

    #print(thetaw[current_time_index,:])
    #print(theta[current_time_index,:])
    
    Dataset.close()

    while elapsed_time < total_time:
        Dataset = NCDF(output_filename, "r+", format = "NETCDF4")

        z = Dataset["height"]
        dz = Dataset["height_change"]
        mdz = Dataset["midpoint_height_change"]
        p = Dataset["pressure"]
        dpdt = Dataset["dpdt"]
        T = Dataset["temperature"]
        theta = Dataset["potential_temperature"]
        T_g = Dataset["surface_temperature"]
        dTdt = Dataset["dTdt"]
        r_v = Dataset["mixing_ratio"]
        dr_vdt = Dataset["dr_vdt"]
        u_g = Dataset["u-component_of_geostrophic_wind"]
        v_g = Dataset["v-component_of_geostrophic_wind"]
        u_bar = Dataset["u-component_of_mean_wind"]
        v_bar = Dataset["v-component_of_mean_wind"]
        dudt = Dataset["dudt"]
        dvdt = Dataset["dvdt"]
        dthetadt = Dataset["dthetadt"]
        z_s = Dataset["surface_layer_height"]
        z_i = Dataset["pbl_top_height"]
        dz_idt = Dataset["dz_idt"]
        uw = Dataset["u-momentum_flux"]
        vw = Dataset["v-momentum_flux"]
        thetaw = Dataset["potential_temperature_flux"]
        K_m = Dataset["eddy_momentum_diffusivity"]
        K_h = Dataset["eddy_heat_diffusivity"]
        puwpz = Dataset["differential_u-momentum_flux"]
        pvwpz = Dataset["differential_v-momentum_flux"]
        pthetawpz = Dataset["differential_potential_temperature_flux"]
        time = Dataset["time"]
        model_time = Dataset["model_time"]
        shortwave_reduction = Dataset["shortwave_reduction"]
        SW_in = Dataset["incoming_solar_radiation"]
        u_star = Dataset["u_star"]
        theta_star = Dataset["theta_star"]
        L = Dataset["monin_obukhov_length_scale"]
        data_tz = Dataset["timezone"][0]

        elapsed_time += timestep.to("s")
        current_model_state_time_utc += timedelta(seconds = timestep.to("s").m)
        current_model_state_time_local = current_model_state_time_utc.astimezone(timezone(data_tz))
        current_time_index += 1
        model_time[current_time_index] = model_time[current_time_index-1] + timestep.to("s")
        time[current_time_index] = date2num(current_model_state_time_utc, time.units)
        model_bottom = (z == z[0])
        surface_layer = (z > z[0]) & (z <= z_s[current_time_index-1])
        boundary_layer = (z > z_s[current_time_index-1]) & (z <= z_i[current_time_index-1]) & (z < z[-1])
        free_atmosphere = (z > z_i[current_time_index-1]) & (z < z[-1])
        model_top = (z == z[-1])

        # Calculate surface layer-based stars
        theta_z_s = Quantity(interpolate.interp1d(z[:].data, theta[current_time_index-1,:].data)(z_s[current_time_index-1].data), "K")
        u_z_s = Quantity(interpolate.interp1d(z[:].data, u_bar[current_time_index-1,:].data)(z_s[current_time_index-1].data), "m s^(-1)")
        v_z_s = Quantity(interpolate.interp1d(z[:].data, v_bar[current_time_index-1,:].data)(z_s[current_time_index-1].data), "m s^(-1)")
        u_star[current_time_index], theta_star[current_time_index] = dynamics.stars_alt(theta_z_s, dynamics.poissons_T_to_theta(T_g[current_time_index-1], p[current_time_index-1,0], r_v[current_time_index-1,0]), z_s[current_time_index-1], z_0, u_z_s, v_z_s)
        L[current_time_index] = dynamics.Monin_Obukhov_L((theta_z_s + dynamics.poissons_T_to_theta(T_g[current_time_index-1], p[current_time_index-1,0], r_v[current_time_index-1,0]))/2, u_star[current_time_index], theta_star[current_time_index])

        # Calculate fluxes for the current time step
        if surface_layer_enabled:
            uw[current_time_index,surface_layer[1:]], vw[current_time_index,surface_layer[1:]], thetaw[current_time_index,surface_layer[1:]] = dynamics.fluxes_surface_layer_alt(u_star[current_time_index], theta_star[current_time_index], u_z_s, v_z_s)
        else:
            uw[current_time_index,model_bottom[:-1]] = 0
            vw[current_time_index,model_bottom[:-1]] = 0
            thetaw[current_time_index,model_bottom[:-1]] = 0
            uw[current_time_index,surface_layer[1:]] = 0
            vw[current_time_index,surface_layer[1:]] = 0
            thetaw[current_time_index,surface_layer[1:]] = 0

        if free_atmosphere_distinction:
            if boundary_layer_enabled:
                K_m[current_time_index,boundary_layer[1:]] = dynamics.eddy_momentum_diffusivity(theta[current_time_index-1,:], dynamics.poissons_T_to_theta(T_g[current_time_index-1], p[current_time_index-1,0], r_v[current_time_index-1,0]), z[:], z_s[current_time_index-1], z_i[current_time_index-1], u_bar[current_time_index-1,:], v_bar[current_time_index-1,:], z_0, L[current_time_index], u_star[current_time_index], boundary_layer)
                K_h[current_time_index,boundary_layer[1:]] = dynamics.eddy_heat_diffusivity(theta[current_time_index-1,:], dynamics.poissons_T_to_theta(T_g[current_time_index-1], p[current_time_index-1,0], r_v[current_time_index-1,0]), z[:], z_s[current_time_index-1], z_i[current_time_index-1], u_bar[current_time_index-1,:], v_bar[current_time_index-1,:], z_0, L[current_time_index], u_star[current_time_index], boundary_layer)
                uw[current_time_index,boundary_layer[1:]] = - K_m[current_time_index,boundary_layer[1:]] * (Quantity(u_bar[current_time_index-1,1:][boundary_layer[1:]] - u_bar[current_time_index-1,:-1][boundary_layer[1:]], "m s^(-1)")/Quantity(z[1:][boundary_layer[1:]] - z[:-1][boundary_layer[1:]], "m"))
                vw[current_time_index,boundary_layer[1:]] = - K_m[current_time_index,boundary_layer[1:]] * (Quantity(v_bar[current_time_index-1,1:][boundary_layer[1:]] - v_bar[current_time_index-1,:-1][boundary_layer[1:]], "m s^(-1)")/Quantity(z[1:][boundary_layer[1:]] - z[:-1][boundary_layer[1:]], "m"))
                thetaw[current_time_index,boundary_layer[1:]] = - K_h[current_time_index,boundary_layer[1:]] * (Quantity(theta[current_time_index-1,1:][boundary_layer[1:]] - theta[current_time_index-1,:-1][boundary_layer[1:]], "K")/Quantity(z[1:][boundary_layer[1:]] - z[:-1][boundary_layer[1:]], "m"))
            else:
                uw[current_time_index,boundary_layer[1:]] = 0
                vw[current_time_index,boundary_layer[1:]] = 0
                thetaw[current_time_index,boundary_layer[1:]] = 0
            uw[current_time_index,free_atmosphere[1:]] = 0
            vw[current_time_index,free_atmosphere[1:]] = 0
            thetaw[current_time_index,free_atmosphere[1:]] = 0
        else:
            if boundary_layer_enabled:
                K_m[current_time_index,(boundary_layer[1:] | free_atmosphere[1:])] = dynamics.eddy_momentum_diffusivity(theta[current_time_index-1,:], dynamics.poissons_T_to_theta(T_g[current_time_index-1], p[current_time_index-1,0], r_v[current_time_index-1,0]), z[:], z_s[current_time_index-1], z_i[current_time_index-1], u_bar[current_time_index-1,:], v_bar[current_time_index-1,:], z_0, L[current_time_index], u_star[current_time_index], (boundary_layer | free_atmosphere))
                K_h[current_time_index,(boundary_layer[1:] | free_atmosphere[1:])] = dynamics.eddy_heat_diffusivity(theta[current_time_index-1,:], dynamics.poissons_T_to_theta(T_g[current_time_index-1], p[current_time_index-1,0], r_v[current_time_index-1,0]), z[:], z_s[current_time_index-1], z_i[current_time_index-1], u_bar[current_time_index-1,:], v_bar[current_time_index-1,:], z_0, L[current_time_index], u_star[current_time_index], (boundary_layer | free_atmosphere))
                uw[current_time_index,(boundary_layer[1:] | free_atmosphere[1:])] = - K_m[current_time_index,(boundary_layer[1:] | free_atmosphere[1:])] * (Quantity(u_bar[current_time_index-1,1:][(boundary_layer | free_atmosphere)[1:]] - u_bar[current_time_index-1,:-1][(boundary_layer | free_atmosphere)[1:]], "m s^(-1)")/Quantity(z[1:][(boundary_layer | free_atmosphere)[1:]] - z[:-1][(boundary_layer | free_atmosphere)[1:]], "m"))
                vw[current_time_index,(boundary_layer[1:] | free_atmosphere[1:])] = - K_m[current_time_index,(boundary_layer[1:] | free_atmosphere[1:])] * (Quantity(v_bar[current_time_index-1,1:][(boundary_layer | free_atmosphere)[1:]] - v_bar[current_time_index-1,:-1][(boundary_layer | free_atmosphere)[1:]], "m s^(-1)")/Quantity(z[1:][(boundary_layer | free_atmosphere)[1:]] - z[:-1][(boundary_layer | free_atmosphere)[1:]], "m"))
                thetaw[current_time_index,(boundary_layer[1:] | free_atmosphere[1:])] = - K_h[current_time_index,(boundary_layer[1:] | free_atmosphere[1:])] * (Quantity(theta[current_time_index-1,1:][(boundary_layer | free_atmosphere)[1:]] - theta[current_time_index-1,:-1][(boundary_layer | free_atmosphere)[1:]], "K")/Quantity(z[1:][(boundary_layer | free_atmosphere)[1:]] - z[:-1][(boundary_layer | free_atmosphere)[1:]], "m"))
            else:
                uw[current_time_index,(boundary_layer[1:] | free_atmosphere[1:])] = 0
                vw[current_time_index,(boundary_layer[1:] | free_atmosphere[1:])] = 0
                thetaw[current_time_index,(boundary_layer[1:] | free_atmosphere[1:])] = 0
        
        if model_top_enabled:
            uw[current_time_index,model_top[1:]] = 0
            vw[current_time_index,model_top[1:]] = 0
            thetaw[current_time_index,model_top[1:]] = 0
        else:
            uw[current_time_index,model_top[1:]] = 0
            vw[current_time_index,model_top[1:]] = 0
            thetaw[current_time_index,model_top[1:]] = 0


        # Calculate differential fluxes
        # At the ground
        puwpz[current_time_index,model_bottom] = Quantity(0, uw.units) / Quantity(mdz[model_top[1:]], mdz.units)
        pvwpz[current_time_index,model_bottom] = Quantity(0, vw.units) / Quantity(mdz[model_top[1:]], mdz.units)
        pthetawpz[current_time_index,model_bottom] = Quantity(0, thetaw.units) / Quantity(mdz[model_top[1:]], mdz.units)
        
        # At the model top
        puwpz[current_time_index,model_top] = Quantity(- uw[current_time_index,model_top[1:]], uw.units) / Quantity(mdz[model_top[1:]]/2, mdz.units) #Quantity(0, uw.units) / Quantity(mdz[model_top[1:]], mdz.units)
        pvwpz[current_time_index,model_top] = Quantity(- vw[current_time_index,model_top[1:]], vw.units) / Quantity(mdz[model_top[1:]]/2, mdz.units) #Quantity(0, vw.units) / Quantity(mdz[model_top[1:]], mdz.units)
        pthetawpz[current_time_index,model_top] = Quantity(- thetaw[current_time_index,model_top[1:]], thetaw.units) / Quantity(mdz[model_top[1:]]/2, mdz.units) #Quantity(0, thetaw.units) / Quantity(mdz[model_top[1:]], mdz.units)
        
        # Everywhere else
        puwpz[current_time_index,1:-1] = Quantity(uw[current_time_index,1:] - uw[current_time_index,:-1], uw.units) / Quantity(mdz[1:], mdz.units)
        pvwpz[current_time_index,1:-1] = Quantity(vw[current_time_index,1:] - vw[current_time_index,:-1], vw.units) / Quantity(mdz[1:], mdz.units)
        pthetawpz[current_time_index,1:-1] = Quantity(thetaw[current_time_index,1:] - thetaw[current_time_index,:-1], thetaw.units) / Quantity(mdz[1:], mdz.units)

        
        # Calculate tendencies with respect to time for the current time step
        # At the ground
        SW_in[current_time_index] = dynamics.ground_incoming_shortwave(current_model_state_time_local, lat, shortwave_reduction[current_time_index-1])
        dudt[current_time_index,model_bottom] = - (Quantity(u_bar[current_time_index-1,model_bottom], u_bar.units)/timestep.to("s")) # Momentum AT the ground should be zero, so simply get rid of it. dynamics.Coriolis_u(lat, v_bar[current_time_index-1,j], v_g[current_time_index-1,j]) - Quantity((puwpz[current_time_index,j]), puwpz.units) #
        dvdt[current_time_index,model_bottom] = - (Quantity(v_bar[current_time_index-1,model_bottom], v_bar.units)/timestep.to("s")) # dynamics.Coriolis_v(lat, u_bar[current_time_index-1,j], u_g[current_time_index-1,j]) - Quantity((pvwpz[current_time_index,j]), pvwpz.units) #
        if ground_energy_enabled:
            dTdt[current_time_index] = dynamics.ground_temperature_tendency(theta[current_time_index-1,1], u_star[current_time_index], theta_star[current_time_index], z[:], T_g[current_time_index-1], z_s[current_time_index-1], z_0, p[current_time_index-1,:], r_v[current_time_index-1,0], soil_emissivity, soil_density, soil_heat_capacity, soil_depth, shortwave_reduction[current_time_index-1], current_model_state_time_local, lat)
        else:
            dTdt[current_time_index] = Quantity(0, "K s^(-1)")
        dthetadt[current_time_index,model_bottom] = Quantity(0, "K s^(-1)") #- Quantity((pthetawpz[current_time_index,model_bottom]), pthetawpz.units)
        dpdt[current_time_index,model_bottom] = Quantity(0, "Pa s^(-1)")
        dr_vdt[current_time_index,model_bottom] = Quantity(0, "kg kg^(-1)")

        # Within the surface layer
        dudt[current_time_index,surface_layer] = dynamics.Coriolis_u(lat, v_bar[current_time_index-1,surface_layer], v_g[current_time_index-1,surface_layer]) - Quantity((puwpz[current_time_index,surface_layer]), puwpz.units)
        dvdt[current_time_index,surface_layer] = dynamics.Coriolis_v(lat, u_bar[current_time_index-1,surface_layer], u_g[current_time_index-1,surface_layer]) - Quantity((pvwpz[current_time_index,surface_layer]), pvwpz.units)
        dthetadt[current_time_index,surface_layer] = - Quantity((pthetawpz[current_time_index,surface_layer]), pthetawpz.units)
        if ground_energy_enabled:
            dthetadt[current_time_index,1] = dthetadt[current_time_index,1]# + dynamics.ground_outgoing_longwave(T_g[current_time_index-1], soil_emissivity) * (1/((p[current_time_index,1]/(physics.R_d * T[current_time_index,1])) * physics.c_p * (z[2] - z[1])))
        dpdt[current_time_index,model_bottom] = Quantity(0, "K s^(-1)")
        dr_vdt[current_time_index,model_bottom] = Quantity(0, "kg kg^(-1)")

        # Within the boundary layer
        dudt[current_time_index,boundary_layer] = dynamics.Coriolis_u(lat, v_bar[current_time_index-1,boundary_layer], v_g[current_time_index-1,boundary_layer]) - Quantity((puwpz[current_time_index,boundary_layer]), puwpz.units)
        dvdt[current_time_index,boundary_layer] = dynamics.Coriolis_v(lat, u_bar[current_time_index-1,boundary_layer], u_g[current_time_index-1,boundary_layer]) - Quantity((pvwpz[current_time_index,boundary_layer]), pvwpz.units)
        dthetadt[current_time_index,boundary_layer] = - Quantity((pthetawpz[current_time_index,boundary_layer]), pthetawpz.units)
        dpdt[current_time_index,model_bottom] = Quantity(0, "K s^(-1)")
        dr_vdt[current_time_index,model_bottom] = Quantity(0, "kg kg^(-1)")

        # Within the free atmosphere
        dudt[current_time_index,free_atmosphere] = dynamics.Coriolis_u(lat, v_bar[current_time_index-1,free_atmosphere], v_g[current_time_index-1,free_atmosphere]) - Quantity((puwpz[current_time_index,free_atmosphere]), puwpz.units)
        dvdt[current_time_index,free_atmosphere] = dynamics.Coriolis_v(lat, u_bar[current_time_index-1,free_atmosphere], u_g[current_time_index-1,free_atmosphere]) - Quantity((pvwpz[current_time_index,free_atmosphere]), pvwpz.units)
        dthetadt[current_time_index,free_atmosphere] = - Quantity((pthetawpz[current_time_index,free_atmosphere]), pthetawpz.units)
        dpdt[current_time_index,model_bottom] = Quantity(0, "K s^(-1)")
        dr_vdt[current_time_index,model_bottom] = Quantity(0, "kg kg^(-1)")

        # At the top of the model
        dudt[current_time_index,model_top] = dynamics.Coriolis_u(lat, v_bar[current_time_index-1,model_top], v_g[current_time_index-1,model_top]) - Quantity(puwpz[current_time_index,model_top], puwpz.units)
        dvdt[current_time_index,model_top] = dynamics.Coriolis_v(lat, u_bar[current_time_index-1,model_top], u_g[current_time_index-1,model_top]) - Quantity(pvwpz[current_time_index,model_top], pvwpz.units)
        dthetadt[current_time_index,model_top] = - Quantity(pthetawpz[current_time_index,model_top], pthetawpz.units)
        dpdt[current_time_index,model_bottom] = Quantity(0, "Pa s^(-1)")
        dr_vdt[current_time_index,model_bottom] = Quantity(0, "kg kg^(-1)")


        # Calculate the new primary model values
        shortwave_reduction[current_time_index] = shortwave_reduction[current_time_index-1]
        u_bar[current_time_index,:] = Quantity(u_bar[current_time_index-1,:], u_bar.units) + timestep.to("s") * Quantity(dudt[current_time_index,:], dudt.units)
        v_bar[current_time_index,:] = Quantity(v_bar[current_time_index-1,:], v_bar.units) + timestep.to("s") * Quantity(dvdt[current_time_index,:], dvdt.units)
        theta[current_time_index,:] = Quantity(theta[current_time_index-1,:], theta.units) + timestep.to("s") * Quantity(dthetadt[current_time_index,:], dthetadt.units)
        p[current_time_index,:] = Quantity(p[current_time_index-1,:], p.units) + timestep.to("s") * Quantity(dpdt[current_time_index,:], dpdt.units)
        T_g[current_time_index] = Quantity(T_g[current_time_index-1], T_g.units) + timestep.to("s") * Quantity(dTdt[current_time_index], dTdt.units)
        r_v[current_time_index,:] = Quantity(r_v[current_time_index-1,:], r_v.units) + timestep.to("s") * Quantity(dr_vdt[current_time_index,:], dr_vdt.units)

        # Post-timestep calculations
        # If enabled, allow for the geostrophic wind to vary
        if not config["constant_geostrophic_flow"]:
            pass
        else:
            u_g[current_time_index,:] = u_g[0,:]
            v_g[current_time_index,:] = v_g[0,:]

        # Calculate new PBL (and surface layer, if enabled) depth
        if not config["constant_boundary_layer_depth"]:
            dz_idt[current_time_index] = dynamics.boundary_layer_height_tendency(u_star[current_time_index], theta_star[current_time_index], theta[current_time_index-1,:], z[:], dynamics.poissons_T_to_theta(T_g[current_time_index-1], p[current_time_index-1,0], r_v[current_time_index-1,0]), z_s[current_time_index-1], z_i[current_time_index-1], z_0, lat)
            z_i[current_time_index] = z_i[current_time_index-1] + dz_idt[current_time_index] * timestep.to("s")
        else:
            z_i[current_time_index] = z_i[current_time_index-1]
        
        # Clip boundary layer height to within the model
        if z_i[current_time_index] > (z[-1] - Quantity(25, "m")):
            z_i[current_time_index] = (z[-1] - Quantity(25, "m"))
        elif z_i[current_time_index] < Quantity(25, "m"):
            z_i[current_time_index] = Quantity(25, "m")

        if not config["constant_surface_layer_depth"]:
            z_s[current_time_index] = z_i[current_time_index] / 10
        else:
            z_s[current_time_index] = z_s[0]

        if z_s[current_time_index] > (z[-1] - Quantity(25, "m")):
            z_s[current_time_index] = (z[-1] - Quantity(25, "m"))
        elif z_s[current_time_index] < Quantity(25, "m"):
            z_s[current_time_index] = Quantity(25, "m")

        if progress_tracker:
            progress(total_time, timestep.to("s"), current_time_index)
    
        Dataset.close()
    duration = (datetime.now(timezone("UTC")) - runtime).total_seconds()
    print(f"\n   Model run completed in {int(duration // 3600):02d}:{int((duration % 3600) // 60):02d}:{int(duration % 60):02d}.")
    return output_filename


if __name__ == "__main__":
    config = get_config()

    #dummy_datetime = datetime(2025, 10, 16, 0).replace(tzinfo = timezone("UTC"))
    #dummy_inputs = {"z" : [0, 25, 50, 75], "u_g" : 5, "v_g" : 5, "T" : [280, 282, 284, 286], "p" : [1000, 995, 990, 985], "u_bar" : [0, 2, 4, 6], "v_bar" : [0, 2, 4, 6]}
    #filename = ingest.ingest_1d(config, dummy_datetime, dummy_inputs)

    #input_filename = ingest.ingest_file(config, "https://www.nsstc.uah.edu/~nair/files/ats681/pbl1D_init.csv", input_data_time = datetime(2025, 7, 16, 1, 45).replace(tzinfo = timezone("UTC")), column_names = ['z', 'p', 't'])
    
    #input_filename = ingest.ingest_previous_run("PBL_MODEL_OUTPUT_2025-10-23_08-17-32.nc", "continue", time = Quantity(3.5, "hr"))

    sounding_datetime = datetime(2025, 10, 25, 12).replace(tzinfo = timezone("UTC"))
    sounding_station = "BMX"
    input_filename = ingest.ingest_sounding(config, sounding_datetime, sounding_station)

    #input_filename = "PBL_MODEL_INPUT_2025-10-25_11-05-37.nc"

    output_filename = run_model(config, input_filename)

    plot_timeseries(output_filename)
    plot_radiation(output_filename)
    #plot_timeseries("PBL_MODEL_OUTPUT_2025-10-25_15-08-54.nc")
    #plot_radiation("PBL_MODEL_OUTPUT_2025-10-25_15-08-54.nc")