from netCDF4 import Dataset as NCDF
from pint import Quantity
import dynamics
import physics
from typing import Union
import numpy as np
from scipy import interpolate
from datetime import datetime
import grid
from pytz import timezone
import shutil
import os
import pandas as pd
from siphon.simplewebservice.wyoming import WyomingUpperAir
import sys
from timezonefinder import TimezoneFinder
from netCDF4 import date2num, num2date


DS = NCDF("PBL_MODEL_OUTPUT_2025-10-25_16-55-29.nc", "r", format = "NETCDF4")

time = DS["time"]
tz = DS["timezone"][0]

local_time = np.array([num2date(x, time.units, only_use_cftime_datetimes=False).replace(tzinfo = timezone("UTC")).astimezone(timezone(tz)) for x in time[:].data])

print(local_time[0].strftime("%m-%d %H:%M:%S"))