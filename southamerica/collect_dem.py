import os
import sys
import shutil
import gdal
import scipy.interpolate
import pyproj
import numpy as np


import download_dem

def get_dem_bounds(lonlat_pts):
    minlat = np.min(lonlat_pts[:,1])
    minlon = np.min(lonlat_pts[:,0])
    maxlat = np.max(lonlat_pts[:,1])
    maxlon = np.max(lonlat_pts[:,0])
    latrange = maxlat - minlat
    lonrange = maxlon - minlon
    bounds = (
        minlat - latrange * 0.1,
        minlon - lonrange * 0.1,
        maxlat + latrange * 0.1,
        maxlon + lonrange * 0.1
    )
    return bounds

def get_pt_elevations(lonlat_pts, zoom, n_interp = 100):
    bounds = get_dem_bounds(lonlat_pts)
    LON, LAT, DEM = get_dem(zoom, bounds, n_interp)
    return scipy.interpolate.griddata(
        (LON, LAT), DEM, (lonlat_pts[:,0], lonlat_pts[:,1])
    )

def get_dem(zoom, bounds, n_width, dest_dir = 'dem_download'):
    print('downloading dem data for bounds = ' + str(bounds) + ' and zoom = ' + str(zoom))
    api_key = open('mapboxapikey').read()
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)
    os.makedirs(dest_dir)
    dest = os.path.join(dest_dir, 'raw_merc.tif')
    download_dem.download(dest, download_dem.tiles(zoom, *bounds), api_key, verbose = False)
    filebase, fileext = os.path.splitext(dest)
    dataset_merc = gdal.Open(dest)
    filename_latlon = os.path.join(dest_dir, 'latlon.tif')
    dataset_latlon = gdal.Warp(filename_latlon, dataset_merc, dstSRS = 'EPSG:4326')
    dem = dataset_latlon.ReadAsArray().astype(np.float64)
    width = dataset_latlon.RasterXSize
    height = dataset_latlon.RasterYSize
    gt = dataset_latlon.GetGeoTransform()
    xs = np.linspace(0, width - 1, width)
    ys = np.linspace(0, height - 1, height)
    X, Y = np.meshgrid(xs, ys)
    lon = gt[0] + X * gt[1] + Y * gt[2]
    lat = gt[3] + X * gt[4] + Y * gt[5]
    assert(gt[2] == 0)
    assert(gt[4] == 0)
    minlat, minlon = bounds[0], bounds[1]
    maxlat, maxlon = bounds[2], bounds[3]
    expand = 0
    LON, LAT = np.meshgrid(
        np.linspace(minlon - expand, maxlon + expand, n_width), 
        np.linspace(minlat - expand, maxlat + expand, n_width)
    )
    DEM = scipy.interpolate.griddata(
        (lon.flatten(), lat.flatten()), dem.flatten(),
        (LON, LAT)
    )
    return LON.flatten(), LAT.flatten(), DEM.flatten()

def project(inx, iny, dem, proj_name, inverse = False):
    wgs84 = pyproj.Proj('+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs')
    if proj_name == 'ellps':
        proj = pyproj.Proj('+proj=geocent +datum=WGS84 +units=m +no_defs')
    elif proj_name.startswith('utm'):
        zone = proj_name[3:]
        print(zone)
        proj = pyproj.Proj("+proj=utm +zone=" + zone + ", +north +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
    if inverse:
        x,y,z = pyproj.transform(proj, wgs84, inx, iny, dem)
    else:
        x,y,z = pyproj.transform(wgs84, proj, inx, iny, dem)
    projected_pts = np.vstack((x,y,z)).T.copy()
    return projected_pts