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
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import subprocess

def make_gif(filename : str):
    Dataset = NCDF(filename, "r", format = "NETCDF4")

    T = Dataset["temperature"]
    theta = Dataset["potential_temperature"]
    dTdt = Dataset["dTdt"]
    dthetadt = Dataset["dthetadt"]
    z_i = Dataset["pbl_top_height"]
    z_s = Dataset["surface_layer_height"]
    #LW = Dataset["net_longwave_radiation"]

    z = Dataset["height"]
    model_time = Dataset["model_time"]
    if os.path.exists("SABRES-PBL-Model/gif-frames/"):
        shutil.rmtree("SABRES-PBL-Model/gif-frames/")
    
    os.mkdir("SABRES-PBL-Model/gif-frames/")

    for i in [x.astype(np.int32) for x in np.floor(np.linspace(0, len(model_time), 50, endpoint = False))]:
        fig = plt.figure(1, figsize = (12, 12))
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax4 = fig.add_subplot(223)
        ax5 = fig.add_subplot(224)

        for ax in [ax4, ax5]:
            ax.plot(np.zeros(z[:].size), z[:], color = "grey", linestyle = "dashed")

        ax1.plot(T[i,:], z[:], color = "orange", label = r"$T$")
        ax1.set_xlim(np.nanmin(T[:,:]), np.nanmax(T[:,:]))
        ax1.set_title("Temperature")
        ax1.set_xlabel(r"$T$ (K)")
        ax1.set_ylabel(r"$z$ (m)")
        ax2.plot(theta[i,:], z[:], color = "red", label = r"$\theta$")
        ax2.set_xlabel(r"$\theta$ (K)")
        ax2.set_xlim(np.nanmin(theta[:,:]), np.nanmax(theta[:,:]))
        ax2.set_title("Potential Temperature")
        #ax3.plot(LW[i,:], z[:], color = "salmon", label = r"$F^{net}_{LW}$")
        #ax3.set_xlabel(r"$F^{net}_{LW}$ $\left(\dfrac{W}{m^2}\right)$")
        #ax3.set_xlim(np.nanmin(LW[:,:]), np.nanmax(LW[:,:]))
        #ax3.set_title("Net Longwave Radiation")
        ax4.plot(dTdt[i,:], z[:], color = "green", label = r"$\dfrac{\partial T}{\partial t}$")
        ax4.set_xlim(np.nanmin(dTdt[:,:]), np.nanmax(dTdt[:,:]))
        ax4.set_title("Temperature Tendency")
        ax4.set_xlabel(r"$\dfrac{\partial T}{\partial t}$ $\left(\dfrac{K}{s}\right)$")
        ax4.set_ylabel(r"$z$ (m)")
        ax5.plot(dthetadt[i,:], z[:], color = "blue", label = r"$\dfrac{\partial \theta}{\partial t}$")
        ax5.set_xlim(np.nanmin(dthetadt[:,:]), np.nanmax(dthetadt[:,:]))
        ax5.set_title("Potential Temperature Tendency")
        ax5.set_xlabel(r"$\dfrac{\partial \theta}{\partial t}$ $\left(\dfrac{K}{s}\right)$")

        for ax in [ax1, ax2, ax4, ax5]:
            x_min, x_max = ax.get_xlim()
            ax.hlines([z_s[i], z_i[i]], xmin = x_min, xmax = x_max, colors = ["thistle", "powderblue"], linestyles = ["dotted", "dashed"])

        fig.legend()
        fig.suptitle(f"Frame {i}")
        fig.savefig(f"SABRES-PBL-Model/gif-frames/frame{i:06d}.png", dpi = 150, bbox_inches = "tight")
        plt.close()

    img_dir = "SABRES-PBL-Model/gif-frames/"
    images = []
    for file_name in sorted(os.listdir(img_dir)):
        if file_name.endswith('.png'):
            file_path = os.path.join(img_dir, file_name)
            images.append(imageio.imread(file_path))

    # Make it pause at the end so that the viewers can ponder
    for _ in range(10):
        images.append(imageio.imread(file_path))

    imageio.mimsave('SABRES-PBL-Model/evolution.gif', images, duration = 5, loop = 65535)

if __name__ == "__main__":
    make_gif("SABRES-PBL-Model\PBL_MODEL_OUTPUT_2025-11-22_23-25-00.nc")