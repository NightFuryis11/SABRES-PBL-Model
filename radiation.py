import hapi
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset as NCDF
from pint import Quantity
import sys


def create_radiation_database(download_new_data : bool = False) -> None:
    hapi.db_begin("local_radiation_data")

    if download_new_data:
        hapi.fetch("CO2", 2, 1, 1, 100000)
        hapi.fetch("N2", 22, 1, 1, 100000)
        hapi.fetch("H2O", 1, 1, 1, 100000)
        hapi.fetch("O2", 7, 1, 1, 100000)
        hapi.fetch("O3", 3, 1, 1, 100000)

    nu = np.arange(1, 100000.0001, 0.01)

    _, abscoef_CO2 = hapi.absorptionCoefficient_Lorentz(SourceTables = ["CO2"], OmegaGrid = nu, HITRAN_units = False)
    _, abscoef_N2 = hapi.absorptionCoefficient_Lorentz(SourceTables = ["N2"], OmegaGrid = nu, HITRAN_units = False)
    _, abscoef_H2O = hapi.absorptionCoefficient_Lorentz(SourceTables = ["H2O"], OmegaGrid = nu, HITRAN_units = False)
    _, abscoef_O2 = hapi.absorptionCoefficient_Lorentz(SourceTables = ["O2"], OmegaGrid = nu, HITRAN_units = False)
    _, abscoef_O3 = hapi.absorptionCoefficient_Lorentz(SourceTables = ["O3"], OmegaGrid = nu, HITRAN_units = False)

    #_, absorbtion_CO2 = hapi.absorptionSpectrum(nu, abscoef_CO2, Environment = {"l":5000})
    #_, absorbtion_N2 = hapi.absorptionSpectrum(nu, abscoef_N2, Environment = {"l":5000})
    #_, absorbtion_H2O = hapi.absorptionSpectrum(nu, abscoef_H2O, Environment = {"l":5000})
    #_, absorbtion_O2 = hapi.absorptionSpectrum(nu, abscoef_O2, Environment = {"l":5000})
    #_, absorbtion_O2 = hapi.absorptionSpectrum(nu, abscoef_O2, Environment = {"l":5000})

    window_size = 1000
    weights = np.ones(window_size)/window_size

    wavelength = 1/nu

    DS = NCDF("SABRES-PBL-Model/radiation-data/radiation.nc", "w", format='NETCDF4')
    DS.title = "SSABLeModS Radiation Database"
    DS.description = "The absorbtivity spectra for the gasses used within the SSABLeModS PBL Model, retrieved via the HITRAN API (HAPI) version 1."
    DS.version = "1.0.0"

    nu_dim = DS.createDimension("wavenum", nu.size)


    nu_var = DS.createVariable("wavenumber", np.float64, ("wavenum"))
    nu_var[:] = Quantity(nu[:], "cm^(-1)").to("m^(-1)").m
    nu_var.units = "m^(-1)"

    abscoef_CO2_var = DS.createVariable("CO2_absorbtion_coefficient", np.float64, ("wavenum"))
    abscoef_CO2_var[:] = abscoef_CO2[:]
    abscoef_CO2_var.units = "dimensionless"

    abscoef_N2_var = DS.createVariable("N2_absorbtion_coefficient", np.float64, ("wavenum"))
    abscoef_N2_var[:] = abscoef_N2[:]
    abscoef_N2_var.units = "dimensionless"

    abscoef_H2O_var = DS.createVariable("H2O_absorbtion_coefficient", np.float64, ("wavenum"))
    abscoef_H2O_var[:] = abscoef_H2O[:]
    abscoef_H2O_var.units = "dimensionless"

    abscoef_O2_var = DS.createVariable("O2_absorbtion_coefficient", np.float64, ("wavenum"))
    abscoef_O2_var[:] = abscoef_O2[:]
    abscoef_O2_var.units = "dimensionless"

    abscoef_O3_var = DS.createVariable("O3_absorbtion_coefficient", np.float64, ("wavenum"))
    abscoef_O3_var[:] = abscoef_O3[:]
    abscoef_O3_var.units = "dimensionless"

    abscoef_CO2_smoo_var = DS.createVariable("CO2_absorbtion_coefficient_smoothed", np.float64, ("wavenum"))
    abscoef_CO2_smoo_var[:] = np.convolve(abscoef_CO2, weights, mode = "same")[:]
    abscoef_CO2_smoo_var.units = "dimensionless"

    abscoef_N2_smoo_var = DS.createVariable("N2_absorbtion_coefficient_smoothed", np.float64, ("wavenum"))
    abscoef_N2_smoo_var[:] = np.convolve(abscoef_N2, weights, mode = "same")[:]
    abscoef_N2_smoo_var.units = "dimensionless"

    abscoef_H2O_smoo_var = DS.createVariable("H2O_absorbtion_coefficient_smoothed", np.float64, ("wavenum"))
    abscoef_H2O_smoo_var[:] = np.convolve(abscoef_H2O, weights, mode = "same")[:]
    abscoef_H2O_smoo_var.units = "dimensionless"

    abscoef_O2_smoo_var = DS.createVariable("O2_absorbtion_coefficient_smoothed", np.float64, ("wavenum"))
    abscoef_O2_smoo_var[:] = np.convolve(abscoef_O2, weights, mode = "same")[:]
    abscoef_O2_smoo_var.units = "dimensionless"

    abscoef_O3_smoo_var = DS.createVariable("O3_absorbtion_coefficient_smoothed", np.float64, ("wavenum"))
    abscoef_O3_smoo_var[:] = np.convolve(abscoef_O3, weights, mode = "same")[:]
    abscoef_O3_smoo_var.units = "dimensionless"

    #abs_CO2_var = DS.createVariable("CO2_absorbtivity", np.float64, ("wavenum"))
    #abs_CO2_var[:] = absorbtion_CO2[:]
    #abs_CO2_var.units = "dimensionless"

    #abs_N2_var = DS.createVariable("N2_absorbtivity", np.float64, ("wavenum"))
    #abs_N2_var[:] = absorbtion_N2[:]
    #abs_N2_var.units = "dimensionless"

    #abs_H2O_var = DS.createVariable("H2O_absorbtivity", np.float64, ("wavenum"))
    #abs_H2O_var[:] = absorbtion_H2O[:]
    #abs_H2O_var.units = "dimensionless"

    #abs_O2_var = DS.createVariable("O2_absorbtivity", np.float64, ("wavenum"))
    #abs_O2_var[:] = absorbtion_O2[:]
    #abs_O2_var.units = "dimensionless"

    #abs_CO2_smoo_var = DS.createVariable("CO2_absorbtivity_smoothed", np.float64, ("wavenum"))
    #abs_CO2_smoo_var[:] = np.convolve(absorbtion_CO2, weights, mode = "same")[:]
    #abs_CO2_smoo_var.units = "dimensionless"

    #abs_N2_smoo_var = DS.createVariable("N2_absorbtivity_smoothed", np.float64, ("wavenum"))
    #abs_N2_smoo_var[:] = np.convolve(absorbtion_N2, weights, mode = "same")[:]
    #abs_N2_smoo_var.units = "dimensionless"

    #abs_H2O_smoo_var = DS.createVariable("H2O_absorbtivity_smoothed", np.float64, ("wavenum"))
    #abs_H2O_smoo_var[:] = np.convolve(absorbtion_H2O, weights, mode = "same")[:]
    #abs_H2O_smoo_var.units = "dimensionless"

    #abs_O2_smoo_var = DS.createVariable("O2_absorbtivity_smoothed", np.float64, ("wavenum"))
    #abs_O2_smoo_var[:] = np.convolve(absorbtion_O2, weights, mode = "same")[:]
    #abs_O2_smoo_var.units = "dimensionless"

    #wavelength_var = DS.createVariable("wavelength", np.float64, ("wavenum"))
    #wavelength_var[:] = (1/Quantity(nu[:], "cm^(-1)").to("m^(-1)")).m
    #wavelength_var.units = "m"

    DS.close()