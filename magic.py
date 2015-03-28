# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 13:36:50 2015

@author: mcgibbon
"""
import netCDF4 as nc4
import os
import re
import pytz
import numpy as np
import numexpr as ne
import matplotlib
from datetime import datetime, timedelta
from scipy.interpolate import interp1d
import datetime as datetime_mod
import calendar
from scipy.io import loadmat
from twinetcdf import find_nearest_index

R_d = 287.04 # gas constant in J/(K kg)
g = 9.80655 # m/s^2
R_earth = 6378100. # radius of Earth in metres
Omega = 7.292*10**(-5) # rotation of Earth in rad/s

def avg_datetime(datetimes):
    timestamps = []
    for dt in datetimes:
        timestamps.append(calendar.timegm(dt.timetuple()))
    return datetime.utcfromtimestamp(np.mean(timestamps)).replace(tzinfo=datetimes[0].tzinfo)
    total = sum(dt.hour * 3600 + dt.minute * 60 + dt.second for dt in datetimes)
    avg = total / len(datetimes)
    minutes, seconds = divmod(int(avg), 60)
    hours, minutes = divmod(minutes, 60)
    mean = datetime.combine(datetime_mod.date(1900, 1, 1), datetime_mod.time(hours, minutes, seconds))
    return mean.replace(tzinfo=datetimes[0].tzinfo)

def get_it_min_max(time, datetime_start, datetime_end):
    if datetime_start is None:
        it_min = 0
    else:
        it_min = find_nearest_index(datetime_start, time)
    if datetime_end is None:
        it_max = len(time)
    else:
        it_max = find_nearest_index(datetime_end, time) + 1
    return it_min, it_max
    
def get_day_in_year(dt):
    '''Takes in a datetime, and returns the day within the year as a float.
    '''
    year_start = datetime(dt.year,1,1,tzinfo=dt.tzinfo)
    diff = dt - year_start
    return diff.total_seconds() / 86400.
    
def datetime_to_SAM_time(time):
    '''datetime_to_SAM_time(array([datetime, ...])) --> float
       Takes in a timezone-aware datetime array and returns an array of floats
       indicating the time passed in days since January 1, 0:00 of that year.
    '''
    SAM_time = np.zeros(time.shape,dtype=np.float)
    for i in range(len(time)):
        SAM_time[i] = get_day_in_year(time[i])
    return SAM_time

def relative_humidity_to_specific_humidity(RH, T, p):
    '''Takes in an array of relative humidity as a ratio,
       temperature in degrees Kelvin, and pressure in hPa.
       returns an array of specific humidity in kg/kg.
    '''
    es = 611.2*np.exp(17.67*(T[:]-273.15)/(T[:]-29.5))
    qs = 0.622*es/(p[:]*100.-es)
    return RH * qs
    #e_s = 6.11*np.exp(np.log(10) * (7.5 * T[:])/(237.7 + T[:]))
    #w_s = 0.622*e_s/p[:]
    #w = RH * w_s # note RH is a ratio, not a percentage!
    #return w / (1 + w)
    
class NetCDF(object):
    
    def __init__(self, filename):
        if not self.filename_pattern:
            raise NotImplementedError('Must implement filename_pattern')
        self._filename = filename
        self._netcdf = nc4.Dataset(filename, 'r')
        try:
            self._file_time = self._get_time_from_filename(filename)
        except ValueError:
            self._file_time = None
        self._set_time()
        
    def _set_time(self):
        raise NotImplementedError()
                
    def time(self, datetime_start=None, datetime_end=None):
        try:
            it_min, it_max = get_it_min_max(self._time, datetime_start, datetime_end)
            return self._time[it_min:it_max]
        except NameError:
            raise NotImplementedError('self._time must be set by __init__ for subclasses of NetCDF')
            
    def file_time(self):
        if self._file_time is None:
            raise ValueError('No time found from filename {} using pattern'.format(self.filename, self.filename_pattern.pattern))
        else:
            return self._file_time
        
    def _get_time_from_filename(self, filename):
        if not isinstance(filename, basestring):
            raise TypeError('filename must be string')
        try:
            match = self.filename_pattern.search(filename)
        except NameError:
            raise NotImplementedError('must define filename_pattern for {}'.format(repr(self.type)))
        if match:
            g = match.groups()
            if len(g) > 3:
                t = datetime(int(g[0]),int(g[1]),int(g[2]),int(g[3]),int(g[4]),int(g[5]))
            else:
                t = datetime(int(g[0]),int(g[1]),int(g[2]))
            return t.replace(tzinfo=pytz.utc)
        else:
            raise ValueError('Could not determine time from filename {}'.format(filename))
            
    def get(self, varname):
        return self._netcdf.variables[varname]
        
    def netcdf(self):
        return self._netcdf
        
    def plot(self, ax, varname, title=None, xlim=None, ylim=None):
        var = self.get(varname)
        if len(var.shape) != 1: # not a 1D variable
            raise ValueError('varname must refer to a 1D variable')
        ax.plot(var, self.get('alt'))
        try:
            xlabel = var.long_name + ' ' + var.units
        except KeyError: # TODO: this may be the wrong error type to check
            xlabel = varname
        if title is not None:
            ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Height (m)')
        if ylim is not None:
            ax.set_ylim(ylim)
        if xlim is not None:
            ax.set_xlim(xlim)
            
    def pcolormesh(self, fig, ax, varname, title=None, cbar_label=None,
                   xlim=None, ylim=None, clim=None):
        datenums = matplotlib.dates.date2num(self.time())
        z = np.repeat(self.get('z')[:][None,:], len(datenums), axis=0)
        data = self.get(varname)
        if len(data.shape) != 2:
            raise ValueError('varname must refer to a 2D variable')
        im = ax.pcolormesh(np.repeat(datenums[:,None],z.shape[1], axis=1), z, data)
        ax.xaxis_date()
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%b %d\n%H:%M'))
        ax.set_xlabel('Time (UTC)')
        ax.set_ylabel('Height (m)')
        if title is not None:
            ax.set_title(title)
        cbar=fig.colorbar(im)
        if cbar_label is not None:
            cbar.ax.get_yaxis().labelpad = 15
            cbar.ax.set_ylabel(cbar_label, rotation=270)
        if ylim is not None:
            ax.set_ylim(ylim)
        if clim is not None:
            #cbar.set_clim(-0.00008,0.00008)
            im.set_clim(vmin=clim[0],vmax=clim[1])
        elif ylim is not None:
            iz_min = max([find_nearest_index(ylim[0], z[i,:])
                            for i in range(z.shape[0])])
            iz_max = min([find_nearest_index(ylim[1], z[i,:])
                            for i in range(z.shape[0])])
            if z[0,-1] < z[0,0]:
                # note that the 0 index in z is at TOA
                im.set_clim(vmin=data[:,iz_max:iz_min].min(),
                            vmax=data[:,iz_max:iz_min].max())
            else:
                im.set_clim(vmin=data[:,iz_min:iz_max].min(),
                            vmax=data[:,iz_min:iz_max].max())
        if xlim is None:
            ax.set_xlim((datenums[0],datenums[-1]))
        else:
            ax.set_xlim((xlim[0], xlim[1]))
        

class NetCDFSet(object):
    element_type = NetCDF
    
    def __init__(self, foldername=None):
        self._datasets = set([])
        if foldername is not None:
            self.add_folder(foldername)
            
    def add_folder(self, foldername):
        '''add_folder(string) --> None
           Takes in a folder containing NetCDF files, and adds files matching
           the element_type.filename_pattern to this NetCDFSet.
        '''
        files = os.listdir(foldername)
        for filename in files:
            try:
                match = self.element_type.filename_pattern.search(filename)
            except NameError:
                raise NotImplementedError('must define filename_pattern for {}'.format(repr(self.element_type)))
            if match:
                self.add_element(self.element_type(os.path.join(foldername, filename)))
          
    def add_element(self, element):
        '''add_element(NetCDF) --> None
           Takes in a NetCDF object and adds it to the NetCDFSet.
        '''
        if not isinstance(element, self.element_type):
            raise TypeError('element must be of type {}'.format(repr(self.element_type)))
        self._datasets.add(element)

    def items(self):
        '''items() --> [NetCDF, ...]
           Returns NetCDF elements of the NetCDFSet, sorted by their file_time()
           method.
        '''
        return sorted(self._datasets, key=lambda x:x.file_time())
        
    def save(self, filename, clobber=False, time_name='number_of_forecast_times'):
        '''save(filename) --> None
           Concatenates the elements of the NetCDFSet and saves them into a
           single NetCDF file at the given filename.
           time_name should be the NetCDF variable name of the time variable
           in each of the NetCDF elements.
        '''
        if len(self._datasets) == 0:
            raise ValueError('No datasets to save.')
        # load data
        ingrps = [item.netcdf() for item in self.items()]
        outgrp = nc4.Dataset(filename, 'w', clobber=clobber)
        # initialize non-time dimensionsspyder find and replace
        for dimension in ingrps[0].dimensions.iteritems():
            if dimension[0] != time_name:
                outgrp.createDimension(dimension[0], len(dimension[1]))
        # initialize new time dimension
        num_timesteps = 0
        for grp in ingrps:
            num_timesteps += len(grp.dimensions[time_name])
        outgrp.createDimension(time_name, num_timesteps)
        # initialize variables
        for v_name, var in ingrps[0].variables.iteritems():
            out_var = outgrp.createVariable(v_name, var.datatype, var.dimensions)
            out_var.setncatts(var.__dict__)
        # set global attributes
        outgrp.setncatts(ingrps[0].__dict__)
        # iterate and merge
        i=0
        for grp in ingrps:
            try:
                steps = len(grp.dimensions[time_name])
            except KeyError:
                raise ValueError('time_name {} does not exist in NetCDF'.format(time_name))
            for v_name, var in grp.variables.iteritems():
                if time_name in var.dimensions:
                    if len(var.dimensions) > 1:
                        outgrp.variables[v_name][i:i+steps,:] = var[:]
                    elif len(var.dimensions) == 1:
                        outgrp.variables[v_name][i:i+steps] = var[:]
                elif i==0:
                    outgrp.variables[v_name][:] = var[:]
            grp.close()
            i = i + steps
        outgrp.close()

class ARMData(NetCDF):
    filename_pattern = re.compile(r'.+?.(\d{4})(\d\d)(\d\d).+?\.cdf')

    def _set_time(self):
        base_time = datetime.utcfromtimestamp(self._netcdf.variables['base_time'].getValue())
        self._time = np.array([(base_time + timedelta(seconds=offset)
                      ).replace(tzinfo=pytz.UTC)
                     for offset in self._netcdf.variables['time_offset'][:]])
                         
class LWPData(ARMData):
    filename_pattern = re.compile(r'magmwrlosM1\.b1\.(\d{4})(\d\d)(\d\d).+?\.cdf')
    
class RadFluxData(ARMData):
    filename_pattern = re.compile(r'magprpradM1\.a1\.(\d{4})(\d\d)(\d\d).+?\.cdf')
    
    
class SoundingData(NetCDF):
    filename_pattern = re.compile(r'magsondewnpnM1\.b1\.(\d{4})(\d\d)(\d\d)\.(\d\d)(\d\d)(\d\d)\.custom\.cdf')

    def _set_time(self):
        pass
    
    def time(self):
        raise NotImplementedError()
    
    

class SoundingDataSet(NetCDFSet):
    element_type = SoundingData
    z = np.concatenate(
        (np.arange(1.,2000.,10.),
        np.arange(2000.,4000.,25.),
        np.arange(4000.,25000.,200.),)  
    )
    z_bounds = np.zeros(
        shape=(z.shape[0]+1,),
        dtype=z.dtype,
    )
    z_bounds[0] = z[0]
    z_bounds[-1] = z[-1]
    z_bounds[1:-1] = 0.5*(z[:-1] + z[1:])
    
    def __init__(self, foldername=None):
        super(SoundingDataSet, self).__init__(foldername)
        self._init_data()
    
    def add_element(self, element):
        '''add_element(NetCDF) --> None
           Takes in a SoundingData object and adds it to the SoundingDataSet.
        '''
        super(SoundingDataSet, self).add_element(element)
        self._init_data()
        
    def _init_data(self):
        self._data = {}
        self._data['time'] = np.array([s.file_time() for s in self.items()])
            
    def _init_var(self, varname):
        datasets = self.items()
        if varname == 'time' or len(datasets[0].get(varname).shape) != 1:
            return # time axis is handled separately
        data = np.zeros(
            shape=(len(self._data['time']), len(self.z)),
            dtype=datasets[0].get(varname).dtype,
        )
        for it, s in enumerate(datasets):
            iz_swap = 281
            # exponential kernel with sigma of 25m up to 4km
            z = s.get('alt')[:]
            var = s.get(varname)[:][:,None]
            kernel = np.exp(-(z[:,None]-self.z[:iz_swap])**2/(2*(20.)**2))
            kernel = kernel / np.sum(kernel, axis=0)[None,:]
            data[it,:iz_swap] = ne.evaluate('sum(kernel*var, axis=0)',
            # switch to a sigma of 200m above this
                {'var':var, 'kernel':kernel})
            kernel = np.exp(-(z[:,None]-self.z[iz_swap:])**2/(2*(200.)**2))
            kernel = kernel / np.sum(kernel, axis=0)[None,:]
            data[it,iz_swap:] = ne.evaluate('sum(kernel*var, axis=0)',
                {'var':var, 'kernel':kernel})
            #for iz in range(len(self.z)):
            #    z_curr = s.get('alt')
            #    data[it,iz] = np.nanmean(var[(z_curr >= self.z_bounds[iz]) &
            #                                (z_curr < self.z_bounds[iz+1])]
            #                            )
        self._data[varname] = data
        
    def get(self, varname, datetime_start=None, datetime_end=None):
        if varname not in self._data.keys():
            self._init_var(varname)
        it_min, it_max = get_it_min_max(self._data['time'], datetime_start, datetime_end)
        return self._data[varname][it_min:it_max]
        
    def save(self, filename, clobber=False, time_name='number_of_forecast_times'):
        '''save(filename) --> None
           Concatenates the elements of the NetCDFSet and saves them into a
           single NetCDF file at the given filename.
           time_name should be the NetCDF variable name of the time variable
           in each of the NetCDF elements.
        '''
        raise NotImplementedError('Cannot save SoundingSet objects')

def get_gauss_weight(lon_grid, lat_grid, lon, lat, sigma):
    cutoff_sigma = 3. # number of sigmas to stop grid
    horizontal_resolution = abs(lat_grid[1]-lat_grid[0])
    if abs(lon_grid[1]-lon_grid[0]) != horizontal_resolution:
        raise ValueError('longitude and latitude must have same grid spacing, instead is {:.2f} and {:.2f}'.format(
            abs(lon_grid[1]-lon_grid[0]),abs(lat_grid[1]-lat_grid[0])))
    ilon = find_nearest_index(lon, lon_grid)
    ilat = find_nearest_index(lat, lat_grid)
    sigma_ind = sigma / horizontal_resolution
    lind = max(0, ilon - int(cutoff_sigma*sigma_ind)) # lower horizontal index
    rind = min(ilon + int(cutoff_sigma*sigma_ind), len(lon_grid))+1
    # remember lat decreases with index
    bind = max(0, ilat - int(cutoff_sigma*sigma_ind)) # lower vertical index
    tind = min(ilat + int(cutoff_sigma*sigma_ind), len(lat_grid))+1
    weight_lon = np.exp(-0.5*((lon_grid[lind:rind] - lon)/sigma)**2)
    weight_lat = np.exp(-0.5*((lat_grid[bind:tind] - lat)/sigma)**2)
    weight = weight_lat[:,None]*weight_lon[None,:]
    return weight / ne.evaluate('sum(weight)'), lind, rind, bind, tind
        
class SurfaceData(NetCDF):
    filename_pattern = re.compile(r'ecmwf_oper_(\d{4})(\d\d)(\d\d)_magic\.sfc\.nc')
    horizontal_resolution = 0.5 # degrees per horizontal index
            
    def _set_time(self):
        time_offsets = self._netcdf.variables['forecast_verification_time']
        initial_time = datetime.strptime(str(getattr(self._netcdf,'forecast_verification_date')), '%Y%m%d')
        time = np.zeros(time_offsets.shape,dtype=datetime)
        for i in range(len(time)):
            assert i % 24 == int(time_offsets[i]) # make sure our times are accurate
            time[i] = (initial_time + timedelta(hours=i)).replace(tzinfo=pytz.UTC)
        self._time = time
        if abs(self.get('grid_latitude')[1] -self.get('grid_latitude')[0]) != self.horizontal_resolution:
            raise ValueError('incorrect horizontal resolution in data, is {:.2f} and should be {:.2f}'.format(
                self.get('grid_latitude')[1] -self.get('grid_latitude')[0],
                self.horizontal_resolution))
             
    def file_time(self):
        if self._file_time is None:
            raise ValueError('No time found from filename')
        else:
            return self._file_time
            
    def plot_following(self, ax, ship, varname, time=None, ylim=None):
        raise NotImplementedError
        
    def get_filtered(self, varname, it, lon, lat, sigma):
        '''Takes in a variable name, time index, position, and horizontal sigma in degrees.
           Returns the vertical profile of the variable at that position, filtered
           by a horizontal gaussian with the given sigma.
        '''
        weight, lind, rind, bind, tind = get_gauss_weight(self.get('grid_longitude'), self.get('grid_latitude'), lon, lat,
                                  sigma)
                                  
        tmp = ne.evaluate('sum(var*weight, axis=1)',
                                {
                                    'var':self.get(varname)[it,bind:tind,lind:rind],
                                    'weight':weight[:,:],
                                })
        return ne.evaluate('sum(var, axis=0)',
                           {
                               'var':tmp,
                            })
        
    def get_filtered_ship(self, varname, ship, sigma,
                                  datetime_start=None, datetime_end=None,
                                  lat_delta=0., lon_delta=0.):
        '''Takes in a variable name, ShipData, and horizontal sigma in degrees.
           Returns the vertical profile of the variable following the ship,
           filtered by a horizontal gaussian with the given sigma.
        '''
        it_min, it_max = get_it_min_max(self.time(), datetime_start, datetime_end)
        var = np.zeros((it_max-it_min,))
        time = self.time()[it_min:it_max]
        lon = ship.get_interp('lon', time) + lon_delta
        lat = ship.get_interp('lat', time) + lat_delta
        for it in range(it_min, it_max):
            i = it - it_min
            var[i] = self.get_filtered(varname, it, lon[i],lat[i], sigma) 
        return var
    
class SurfaceDataSet(NetCDFSet):
    element_type = SurfaceData
    
class AlongTrackData(NetCDF):
    filename_pattern = re.compile(r'magecmwf2dX1.00.(\d{4})(\d\d)(\d\d).000000.raw.magic_leg_15.nc')
        
    def _set_time(self):
        times = []
        time_offsets = self._netcdf.variables['utc']
        for i in range(len(time_offsets)):
            initial_time = datetime.strptime(str(int(self._netcdf.variables['verification_date'][i])), '%Y%m%d')
            times.append((initial_time + timedelta(hours=time_offsets[i])
                  ).replace(tzinfo=pytz.UTC))
        self._time = np.array(times)
        
    def get_profile(self, varname, datetime_start=None, datetime_end=None):
        it_min, it_max = get_it_min_max(self.time(), datetime_start, datetime_end)
        return self.get(varname)[it_min:it_max,:]
    
class GriddedData(NetCDF):
    filename_pattern = re.compile(r'ecmwf_oper_(\d{4})(\d\d)(\d\d)_magic\.var\.nc')
    horizontal_resolution = 0.5 # degrees per horizontal index
    
    def _set_time(self):
        time_offsets = self._netcdf.variables['forecast_verification_time']
        initial_time = datetime.strptime(str(getattr(self._netcdf,'forecast_verification_date')), '%Y%m%d')
        time = np.zeros(time_offsets.shape,dtype=datetime)
        for i in range(len(time)):
            assert i % 24 == int(time_offsets[i]) # make sure our times are accurate
            time[i] = (initial_time + timedelta(hours=i)).replace(tzinfo=pytz.UTC)
        self._time = time
        if abs(self.get('grid_latitude')[1] -self.get('grid_latitude')[0]) != self.horizontal_resolution:
            raise ValueError('incorrect horizontal resolution in data, is {:.2f} and should be {:.2f}'.format(
                self.get('grid_latitude')[1] -self.get('grid_latitude')[0],
                self.horizontal_resolution))
        
    def get_profile_filtered(self, varname, it, lon, lat, sigma):
        '''Takes in a variable name, time index, position, and horizontal sigma in degrees.
           Returns the vertical profile of the variable at that position, filtered
           by a horizontal gaussian with the given sigma.
        '''
        weight, lind, rind, bind, tind = get_gauss_weight(self.get('grid_longitude'), self.get('grid_latitude'), lon, lat,
                                  sigma)
        tmp = ne.evaluate('sum(var*weight, axis=2)',
                                {
                                    'var':self.get(varname)[it,:,bind:tind,lind:rind],
                                    'weight':weight[None,:,:],
                                })
        result = ne.evaluate('sum(var, axis=1)',
                           {
                               'var':tmp,
                            })
        return result
        #return ne.evaluate('sum(sum(var*weight, axis=1), axis=2)',
        #                   {
        #                       'var':self.get(varname)[it,:,bind:tind,lind:rind],
        #                       'weight':weight[None,:,:],
        #                    })
    def get_profile_filtered_ship(self, varname, ship, sigma,
                                  datetime_start=None, datetime_end=None,
                                  lat_delta=0., lon_delta=0.):
        '''Takes in a variable name, ShipData, and horizontal sigma in degrees.
           Returns the vertical profile of the variable following the ship,
           filtered by a horizontal gaussian with the given sigma.
        '''
        it_min, it_max = get_it_min_max(self.time(), datetime_start, datetime_end)
        profile = np.zeros((it_max-it_min,) + self.get(varname).shape[1:2])
        time = self.time()[it_min:it_max]
        lon = ship.get_interp('lon', time) + lon_delta
        lat = ship.get_interp('lat', time) + lat_delta
        for it in range(it_min, it_max):
            i = it - it_min
            profile[i,:] = self.get_profile_filtered(varname, it, lon[i],lat[i], sigma) 
        return profile
        
    def get_gradient_filtered_ship(self, varname, ship, direction, sigma,
                                   datetime_start=None, datetime_end=None):
        '''direction should be one of 'lat', 'lon', 'y', or 'x'.
        '''
        z = self.get_profile_filtered_ship(
                    'height_above_reference_ellipsoid', ship, sigma,
                                        datetime_start, datetime_end)
        it_min, it_max = get_it_min_max(self.time(), datetime_start, datetime_end)
        time = self.time()[it_min:it_max]
        if direction in ('lat', 'y'):
            dy = R_earth*self.horizontal_resolution*np.pi/180.
            top = self.get_profile_filtered_ship(varname, ship, sigma,
                                        datetime_start, datetime_end,
                                        lat_delta=self.horizontal_resolution)
            top_z = self.get_profile_filtered_ship(
                    'height_above_reference_ellipsoid', ship, sigma,
                                        datetime_start, datetime_end,
                                        lat_delta=self.horizontal_resolution)
            bottom = self.get_profile_filtered_ship(varname, ship, sigma,
                                        datetime_start, datetime_end,
                                        lat_delta=-1.*self.horizontal_resolution)
            bottom_z = self.get_profile_filtered_ship(
                    'height_above_reference_ellipsoid', ship, sigma,
                                        datetime_start, datetime_end,
                                        lat_delta=-1*self.horizontal_resolution)
            z = self.get_profile_filtered_ship(
                    'height_above_reference_ellipsoid', ship, sigma,
                                        datetime_start, datetime_end)
            return_var = np.zeros_like(top)
            for it in range(return_var.shape[0]):
                var_top = interp1d(top_z[it,:], top[it,:], bounds_error=False)(z[it,:])
                var_top[np.isnan(var_top)] = top[it,:][np.isnan(var_top)]
                var_bottom = interp1d(bottom_z[it,:], bottom[it,:], bounds_error=False)(z[it,:])
                var_bottom[np.isnan(var_bottom)] = bottom[it,:][np.isnan(var_bottom)]
                return_var[it, :] = var_top - var_bottom
            return return_var/(2.*dy)
        elif direction in ('lon', 'x'):
            lat = ship.get_interp('lat', time)
            dx = R_earth*np.cos(np.pi/180.*lat)*self.horizontal_resolution*np.pi/180.
            right = self.get_profile_filtered_ship(varname, ship, sigma,
                                        datetime_start, datetime_end,
                                        lon_delta=self.horizontal_resolution)
            right_z = self.get_profile_filtered_ship(
                    'height_above_reference_ellipsoid', ship, sigma,
                                        datetime_start, datetime_end,
                                        lon_delta=self.horizontal_resolution)
            left = self.get_profile_filtered_ship(varname, ship, sigma,
                                        datetime_start, datetime_end,
                                        lon_delta=-1.*self.horizontal_resolution)
            left_z = self.get_profile_filtered_ship(
                    'height_above_reference_ellipsoid', ship, sigma,
                                        datetime_start, datetime_end,
                                        lon_delta=-1*self.horizontal_resolution)
            return_var = np.zeros_like(left)
            for it in range(return_var.shape[0]):
                # NaN's appear if we are interpolating out of our grid
                # for such values, just leave them as they are. This only happens
                # at the surface and at the TOA.
                var_right = interp1d(right_z[it,:], right[it,:], bounds_error=False)(z[it,:])
                var_right[np.isnan(var_right)] = right[it,:][np.isnan(var_right)]
                var_left = interp1d(left_z[it,:], left[it,:], bounds_error=False)(z[it,:])
                var_left[np.isnan(var_left)] = left[it,:][np.isnan(var_left)]
                return_var[it, :] = var_right - var_left
            return return_var/(2.*dx[:,None])
        else:
            raise ValueError("direction must be one of 'lat', 'lon', 'y', or 'x'")

class GriddedDataSet(NetCDFSet):
    element_type = GriddedData

class TextData(object):
    
    def __init__(self, filename):
        self._data = {}
        self._interp = {}
        data = np.loadtxt(filename)
        self._length = data.shape[0]
        for i in range(data.shape[1]):
            self._data[self._rownames[i]] = data[:,i]
        if set(('year', 'month', 'day')).issubset(set(self._rownames)):
            dt_args = (self._data['year'], self._data['month'], self._data['day'])
            if set(('hour', 'minute', 'second')).issubset(set(self._rownames)):
                dt_args = dt_args + (self._data['hour'], self._data['minute'], self._data['second'])
            time = np.zeros(self._length, dtype=datetime)
            for i in range(self._length):
                time[i] = datetime(*[int(arg[i]) for arg in dt_args]).replace(tzinfo=pytz.UTC)
            self._data['time'] = time
            
    def get(self, varname, datetime_start=None, datetime_end=None):
        it_min, it_max = get_it_min_max(self._data['time'], datetime_start, datetime_end)
        return self._data[varname][it_min:it_max]
        
    def get_interp(self, varname, time):
        '''Time can be a single datetime or an array of datetimes. An array is
           returned.
        '''
        if varname not in self._data.keys():
            raise ValueError('variable {} not in data'.format(varname))
        elif 'time' not in self._data.keys():
            raise ValueError('cannot interpolate data with no time axis')
        elif varname not in self._interp.keys():
            self._interp[varname] = interp1d([calendar.timegm(dt.timetuple()) for dt in self._data['time']], self._data[varname])
        if isinstance(time, datetime):
            return self._interp[varname](calendar.timegm(time.timetuple()))
        else:
            return self._interp[varname]([calendar.timegm(dt.timetuple()) for dt in time])
        
    def get_interp_value(self, varname, time, window_seconds=None):
        '''Time should be a single timezone-aware datetime. A single value is
           returned. If window_seconds is given, the average value within a
           window whose width is that many seconds is returned, so long as
           three points are found in that window.'''
        if varname not in self._data.keys():
            raise ValueError('variable {} not in data'.format(varname))
        elif 'time' not in self._data.keys():
            raise ValueError('cannot interpolate data with no time axis')
        if window_seconds is not None: # Return average value in window
            td = timedelta(seconds=window_seconds/2.)
            window = (self.get('time') > time - td) & (self.get('time') < time + td)
            if np.count_nonzero(window) > 3: # Make sure average is relevant
                return np.nanmean(self.get(varname)[window])
        # No window or average is not relevant, Return interpolated value.
        return self.get_interp(varname, time)
        
    def plot(self, ax, varname, time=None, time_format='%b %d\n%H:%M', xlim=None, ylim=None):
        '''Plots the given variable on an argument-defined axis. Interpolates
           to the time axis if given. Also sets x and y-axis labels.
        '''
        if time is None:
            time = self.get('time')
            data = self.get(varname)
        else:
            data.self.get_interp(varname, time)
        datenums = matplotlib.dates.date2num(time)
        ax.plot(datenums, data)
        ax.xaxis_date()
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter(time_format))
        ax.set_xlabel('Time (UTC)')
        try:
            ylabel = self.long_name[varname]
        except KeyError:
            ylabel = varname
        if varname in self.units.keys():
            ylabel += ' ' + self.units[varname]
        ax.set_ylabel(ylabel)
        if ylim:
            ax.set_ylim(ylim)
        if xlim:
            ax.set_xlim(xlim)

class HeatFluxData(TextData):
    
    def __init__(self, filename, window_seconds, datetime_start=None, datetime_end=None):
        '''
        Parameters
        ----------
        filename : string
            Matlab data file to open
        window_seconds : float
            Length of each averaging window
        datetime_start : datetime, optional
            Earliest time to include in stored data
        datetime_end : datetime, optional
            Latest time to include in stored data
        '''
        self._interp = {}
        self._data = {}
        if datetime_start is not None and datetime_end is not None and (datetime_end < datetime_start):
            raise ValueError('datetime_start must be before datetime_end')
        fluxdata = loadmat(filename)['flux']
        sensible_obs = np.array(-1*fluxdata['Hs'][0][0].flatten())
        latent_obs = np.array(-1*fluxdata['Hl'][0][0].flatten())
        flux_times = []
        for matlab_datenum in fluxdata['dt'][0][0]:
            flux_times.append((datetime.fromordinal(int(matlab_datenum[0])) + timedelta(days=matlab_datenum[0]%1) - timedelta(days = 366)).replace(tzinfo=pytz.UTC))
        flux_times = np.array(flux_times)
        #nanmean = lambda dat: np.nansum(dat) / np.sum(np.isfinite(dat))
        # Take time-window average of data
        if datetime_start is not None:
            dt = max(datetime_start, flux_times[0])
        else:
            dt = flux_times[0]
        if datetime_end is None:
            datetime_end = flux_times[-1]
        interp_times = np.empty((int((datetime_end - datetime_start).total_seconds()/window_seconds),), dtype=datetime)
        shf_interp, lhf_interp = np.zeros(interp_times.shape, dtype=sensible_obs.dtype), np.zeros(interp_times.shape, dtype=latent_obs.dtype)    
        i = 0
        while i < len(interp_times):# and dt < datetime_end
            max_time = dt + timedelta(seconds=window_seconds)
            time_window_flux = (flux_times < max_time) & (flux_times >= dt)
            shf_interp[i] = np.nanmean(sensible_obs[time_window_flux])
            lhf_interp[i] = np.nanmean(latent_obs[time_window_flux])
            interp_times[i] = avg_datetime((dt, max_time))
            i += 1
            dt = max_time
        valid = ~np.isnan(shf_interp) & ~np.isnan(lhf_interp)
        self._length = interp_times.shape[0]
        self._data['shf'] = shf_interp[valid]
        self._data['lhf'] = lhf_interp[valid]
        self._data['time'] = interp_times[valid]

class ShipData(TextData):
    
    _rownames = ['navg', 'year', 'month', 'day', 'hour', 'minute', 'second',
                 'lat', 'lon', 'sog', 'cog', 'pitch', 'pstd', 'roll', 'rstd',
                 'hdg']
    units = {
        'lat':'degrees N',
        'lon':'degrees E',
        'u':'m/s',
        'v':'m/s',
        'sog':'m/s',
        'cog':'degrees',
    }
    long_name = {
        'u': 'Eastward Wind',
        'v': 'Northward Wind',
        'lat': 'Latitude',
        'lon': 'Longitude',
        'sog': 'Speed over ground',
        'cog': 'Course over ground',
    }
    
    def __init__(self, filename):
        super(ShipData, self).__init__(filename)
        self._data['u'] = ne.evaluate('speed_ship*sin(angle_ship*2*pi/360.)',
                {'pi':np.pi, 'speed_ship':self._data['sog'], 'angle_ship':self._data['cog']})
        self._data['v'] = ne.evaluate('speed_ship*cos(angle_ship*2*pi/360.)',
                {'pi':np.pi, 'speed_ship':self._data['sog'], 'angle_ship':self._data['cog']})
        self._data['lon'] += 360.

class SSTData(TextData):
    
    _rownames = ['year', 'month', 'day', 'hour', 'minute', 'second', 'lat',
                 'lon', 'sst']
    units = {
        'lat':'degrees N',
        'lon':'degrees E',
        'sst':'K',
    }
    long_name = {
        'lat':'Latitude',
        'lon':'Longitude',
        'sst':'Sea Surface Temperature',
    }
    
    def __init__(self, filename):
        super(SSTData, self).__init__(filename)
        self._data['sst'] += 273.15
        
class SamOutputData(NetCDF):
    filename_pattern = re.compile(r'MAGIC_(\d{4})(\d\d)(\d\d).+?\.nc')
    
    def _set_time(self):
        year = self.file_time().year
        self._time = np.array([(datetime(year,1,1) + timedelta(days=float(t))
                      ).replace(tzinfo=pytz.UTC)
                     for t in self._netcdf.variables['time'][:]])

class LargeScaleForcingCalculator(object):
    
    def __init__(self, surface, gridded, ship):
        self.surface = surface
        self.gridded = gridded
        self.ship = ship
        
    def get_forcings(self, sigma_h=None,
                     datetime_start=None, datetime_end=None):
        time = self.gridded.time(datetime_start=datetime_start, datetime_end=datetime_end)
        lat, lon = self.ship.get_interp('lat',time), self.ship.get_interp('lon',time)
        T = self.gridded.get_profile_filtered_ship('air_temperature',
                    self.ship, sigma_h, datetime_start=datetime_start,
                    datetime_end=datetime_end)
        u = self.gridded.get_profile_filtered_ship('eastward_wind',
                    self.ship, sigma_h, datetime_start=datetime_start,
                    datetime_end=datetime_end)
        v = self.gridded.get_profile_filtered_ship('northward_wind',
                    self.ship, sigma_h, datetime_start=datetime_start,
                    datetime_end=datetime_end)
        assert u.shape[0] == time.shape[0]
        for it in range(len(time)):
            u[it,:] -= self.ship.get_interp_value('u', time[it], window_seconds=(time[1]-time[0]).total_seconds())
            v[it,:] -= self.ship.get_interp_value('v', time[it], window_seconds=(time[1]-time[0]).total_seconds())
        z = self.gridded.get_profile_filtered_ship('height_above_reference_ellipsoid',
                    self.ship, sigma_h, datetime_start=datetime_start,
                    datetime_end=datetime_end)
        tls = -1*(u*self.gridded.get_gradient_filtered_ship('air_temperature',
                    self.ship, 'lon', sigma_h, datetime_start=datetime_start,
                    datetime_end=datetime_end) +
                  v*self.gridded.get_gradient_filtered_ship('air_temperature',
                    self.ship, 'lat', sigma_h, datetime_start=datetime_start,
                    datetime_end=datetime_end))
        qls = -1*(u*self.gridded.get_gradient_filtered_ship('specific_humidity',
                    self.ship, 'lon', sigma_h, datetime_start=datetime_start,
                    datetime_end=datetime_end) +
                  v*self.gridded.get_gradient_filtered_ship('specific_humidity',
                    self.ship, 'lat', sigma_h, datetime_start=datetime_start,
                    datetime_end=datetime_end)) * 10**-3
        omega = self.gridded.get_profile_filtered_ship('vertical_air_velocity_expressed_as_tendency_of_pressure',
                    self.ship, sigma_h, datetime_start=datetime_start,
                    datetime_end=datetime_end)
        omega = omega - (omega[:,-1])[:,None] # removes effect of surface pressure change
        p = self.gridded.get_profile_filtered_ship('air_pressure',
                    self.ship, sigma_h, datetime_start=datetime_start,
                    datetime_end=datetime_end)
        q = self.gridded.get_profile_filtered_ship('specific_humidity',
                    self.ship, sigma_h, datetime_start=datetime_start,
                    datetime_end=datetime_end)
        rho = p/(R_d * T * (1+0.608*q))
        f = 2*Omega*np.sin(lat[:]*np.pi/180.)
        u_geo = -1./(rho * f[:,None]) * self.gridded.get_gradient_filtered_ship(
                    'air_pressure', self.ship, 'lat', sigma_h, datetime_start=datetime_start,
                    datetime_end=datetime_end)
        v_geo = 1./(rho * f[:,None]) * self.gridded.get_gradient_filtered_ship(
                    'air_pressure', self.ship, 'lon', sigma_h, datetime_start=datetime_start,
                    datetime_end=datetime_end)
        # note that surface pressure data is stored in gridded data
        p_surf = np.exp(self.gridded.get_profile_filtered_ship('logarithm_of_surface_pressure',
                    self.ship, sigma_h, datetime_start=datetime_start,
                    datetime_end=datetime_end)[:,0])
        #wls = -1*omega*(R_d * T * (1+0.608*q))/(p*g)
        wls = -1*omega/(rho*g)
        return {
            'p_surf':p_surf,
            'p':p,
            'rho':rho,
            'omega':omega,
            'q':q,
            'z':z,
            'T':T,
            'u':u,
            'v':v,
            'u_geo':u_geo,
            'v_geo':v_geo,
            'qls':qls,
            'tls':tls,
            'wls':wls,
            'lat':lat,
            'lon':lon,
            'time':time,
        }
        
class SamInputGenerator(object):
    time_between_soundings = 60*60*6 # in seconds
    
    def __init__(self, surface, gridded, sst, ship, heat_flux, sigma_h=2.,
                 sounding_folder=None, sounding_set=None,
                 datetime_start=None, datetime_end=None):
        if sounding_folder is not None and sounding_set is not None:
            raise ValueError('Cannot accept both sounding_folos.path.join(foldername,der and sounding_set kwargs.')
        elif sounding_folder is None and sounding_set is None:
            raise ValueError('Must give either sounding_folder or sounding_set kwarg.')
        elif sounding_folder is not None:
            self.sounding_set = SoundingDataSet(foldername=sounding_folder)
        else:
            self.sounding_set = sounding_set
        if not isinstance(surface, SurfaceData):
            raise TypeError('Invalid python object for surface data.')
        if not isinstance(gridded, GriddedData):
            raise TypeError('Invalid python object for gridded data.')
        self.sigma_h = sigma_h
        self.surface = surface
        self.gridded = gridded
        self.sst = sst
        self.ship = ship
        self.flux = heat_flux
        self._lsf = None
        self._datetime_start = datetime_start
        self._datetime_end = datetime_end
        
    def output_forcings(self, foldername):
        sfc = self.output_sfc_file(os.path.join(foldername, 'sfc'))
        lsf = self.output_lsf_file(os.path.join(foldername, 'lsf'))
        snd = self.output_snd_file(os.path.join(foldername, 'snd'))
        return sfc, lsf, snd
        
    def output_sfc_file(self, filename):
        time = self.sst.get('time', datetime_start=self._datetime_start,
                            datetime_end=self._datetime_end)
        sst = self.sst.get('sst', datetime_start=self._datetime_start,
                           datetime_end=self._datetime_end)
        # Use a window for H and LE because the data is much finer than for SST
        # fluxes have minute-scale data, SST has hour-scale data
        H = self.flux.get_interp('shf', time)
        LE = self.flux.get_interp('lhf', time)
        Tau = np.zeros_like(LE) # momentum flux... I think
        variables = (datetime_to_SAM_time(time),sst,H,LE,Tau)
        sfc_string = 'day sst(K) H(W/m2) LE(W/m2) TAU(m2/s2)\n'
        for i in range(time.shape[0]):
            sfc_string += '   '.join(['%.3f' for j in range(len(variables))]) % tuple([item[i] for item in variables])
            sfc_string += '\n'
        if filename is not None:
            with open(filename, 'w') as f:
                f.write(sfc_string)
        return {'time':time, 'sst':sst, 'H':H, 'LE':LE, 'Tau':Tau,}
        
    def output_lsf_file(self, filename):
        if self._lsf is None:
            calculator = LargeScaleForcingCalculator(self.surface, self.gridded, self.ship)
            lsf = calculator.get_forcings(sigma_h=self.sigma_h,
                                          datetime_start=self._datetime_start,
                                          datetime_end=self._datetime_end)
            # convert pressures from Pa to hPa
            lsf['p'] *= 0.01
            lsf['p_surf'] *= 0.01
            # save for future use
            self._lsf = lsf
        else:
            lsf = self._lsf
        vars_1d = (lsf['p_surf'],)
        vars_2d = (lsf['z'],lsf['p'],lsf['tls'],lsf['qls'],lsf['u_geo'],
                   lsf['v_geo'],lsf['wls'])
        time = lsf['time']
        header_string = 'z[m] p[mb] tpls[K/s] qls[kg/kg/s] uls vls wls[m/s]\n'
        oned_string = 'day,levels, pres0\n'
        lsf_string = self._get_SAM_MAGIC_generic(time, header_string, oned_string, vars_1d, vars_2d)
        if filename is not None:
            with open(filename, 'w') as f:
                f.write(lsf_string)
        return lsf
    
    def output_snd_file(self, filename):
        snd_string = ' z[m] p[mb] tp[K] q[g/kg] u[m/s] v[m/s]\n'
        return_data = []
        time = self.sounding_set.get('time', datetime_start=self._datetime_start,
                                     datetime_end=self._datetime_end)
        header_string = ' z[m] p[mb] tp[K] q[g/kg] u[m/s] v[m/s]\n'
        oned_string = 'day,levels,pres0\n'

        # note surface pressure is in the gridded dataset (don't ask me why)
        p_surf = np.zeros(shape=time.shape)
        soundings = self.sounding_set.items()
        ip = 0
        for i in range(len(soundings)):
            if soundings[i].file_time() < time[0] or soundings[i].file_time() > time[-1]:
                continue
            lat = self.ship.get_interp_value('lat', soundings[i].file_time())
            lon = self.ship.get_interp_value('lon', soundings[i].file_time())
            it = find_nearest_index(soundings[i].file_time(), self.gridded.time())
            p_surf[ip] = np.exp(self.gridded.get_profile_filtered(
                'logarithm_of_surface_pressure', it, lon, lat, self.sigma_h)[0])*0.01
            ip += 1
        vars_1d = (p_surf,)

        z = self.sounding_set.z
        # Sounding already stores pressure in hPa
        p = self.sounding_set.get('pres', datetime_start=self._datetime_start,
                                     datetime_end=self._datetime_end)
        T = self.sounding_set.get('tdry', datetime_start=self._datetime_start,
                                     datetime_end=self._datetime_end)[:] + 273.15 # Convert to Kelvin
        # Convert to potential temperature for sounding file
        theta = T*(1e3/p)**(1.-5./7.)
        # convert RH from percent to ratio
        RH = self.sounding_set.get('rh', datetime_start=self._datetime_start,
                                     datetime_end=self._datetime_end)*0.01
        # convert q from kg/kg to g/kg
        q = relative_humidity_to_specific_humidity(RH, T, p)*1000.
        # Correct q and theta within cloud
        # Set value in cloud to be last unsaturated value
        last_q = q[:,-1]
        last_theta = theta[:,-1]
        for i in range(q.shape[1]):
            in_cloud = RH[:,i] > 0.99
            q[in_cloud,i] = last_q[in_cloud]
            theta[in_cloud,i] = last_theta[in_cloud]
            # ~ is bitwise not operator
            last_q[~in_cloud] = q[~in_cloud,i]
            last_theta[~in_cloud] = theta[~in_cloud,i]
        u = self.sounding_set.get('u_wind', datetime_start=self._datetime_start,
                                     datetime_end=self._datetime_end)
        v = self.sounding_set.get('v_wind', datetime_start=self._datetime_start,
                                     datetime_end=self._datetime_end)
        vars_2d = (np.repeat(z[None,:], len(time), axis=0),p,theta,q,u,v,)
        snd_string = self._get_SAM_MAGIC_generic(time, header_string, oned_string,
                                                 vars_1d, vars_2d, invert_z=False)
        return_data = {
            'time':time,
            'z':z,
            'p':p,
            'p_surf':p_surf,
            'theta':theta,
            'T':T,
            'u':u,
            'v':v,
            'q':q,
            'RH':RH,
        }
        # write our sounding file
        if filename is not None:
            with open(filename, 'w') as f:
                f.write(snd_string)
        return return_data
        
    def _get_SAM_MAGIC_generic(self, time, header_string, oned_string, vars_1d,
                               vars_2d, invert_z=True):
        assert len(vars_2d) > 0        
        return_string = header_string
        for i in range(vars_2d[0].shape[0]):
            return_string += '%.3f   %d' % (get_day_in_year(time[i]), vars_2d[0].shape[1])
            if vars_1d:
                return_string += '   '
            return_string += '   '.join(['%.3f' for j in range(len(vars_1d))]) % tuple([item[i] for item in vars_1d])
            return_string += '   ' + oned_string
            if invert_z:
                j_iter = range(vars_2d[0].shape[1]-1,-1,-1)
            else:
                j_iter = range(vars_2d[0].shape[1])
            for j in j_iter:
                print_vars = []
                for item in vars_2d:
                    if item[i,j] > 0.1:
                        print_vars.append(item[i,j])
                    else:
                        print_vars.append(item[i,j])
                return_string += '   '.join(['%.3e' for k in range(len(vars_2d))]) % tuple(print_vars)
                return_string += '\n'
        return return_string
