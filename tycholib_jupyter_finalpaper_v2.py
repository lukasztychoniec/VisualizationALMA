
# coding: utf-8

# In[17]:


import numpy as np
import astropy.constants as const
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
from astropy.io import fits as pf
#import pyfits as pf
import astropy.units as u
import matplotlib as mpl
from astropy.wcs import WCS
from astropy.visualization import (LogStretch, ImageNormalize, SqrtStretch)
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredEllipse
from scipy.interpolate import interp1d
import glob
import math

import time
from scipy import ndimage
from matplotlib.patches import Rectangle, Ellipse



###########  functions ######################################################

def mom_quick(size, datain, idxvin, idxvout, chanwidth):
    tab = np.empty([size, size])
    #print ('size', size)
    #print (np.shape(datain))
    for m in range(size):
        for n in range(size):
            if np.isnan(datain[0, m, n]):
                continue
            if datain[0, m, n] == 0:
                continue
            rmsmom0 = 0
            if rmsmom0 > 0:
                rmsmeasure = np.std(datain[idxvrmsin:idxvrmsout, m, n])
                idxrmsmom0 = np.where(datain[idxvin:idxvout, m, n] > rmsmeasure * multi_rmsmom0)
                tab[m, n] = np.sum(datain[idxvin:idxvout, m, n][idxrmsmom0]) * chanwidth

            else:
                tab[m, n] = np.sum(datain[idxvin:idxvout, m, n]) * chanwidth
    return tab

def mom_quicker(size, datain, idxvin, idxvout, chanwidth, rms = 0.0):
    tab = np.empty([size, size])
    reduced_datain = datain[idxvin:idxvout,:,:]
    width = idxvout-idxvin
    
    if rms == 0:
        tab = np.nansum(reduced_datain,axis=0) * chanwidth
    else:
        for i in range(width):
            idx = np.where(reduced_datain[i,:,:]>rms)
            mask = np.zeros([size, size])
            mask[idx]=1
            tab = tab + reduced_datain[i,:,:]*mask * chanwidth

    return tab


def mom_quicker_array(size, datain, idxvin, idxvout, chanwidth, rms = 0.0):
    tab = np.zeros([size, size])
#    reduced_datain = datain[idxvin:idxvout,:,:]
#    width = idxvout-idxvin
    
    if rms == 0:
        for s in range(len(idxvin)):
            print (s,idxvin[s],idxvout[s])
            tab_temp = np.nansum(datain[int(idxvin[s]):int(idxvout[s]),:,:],axis=0) * chanwidth
            tab = tab+tab_temp
    else:
        for i in range(width):
            idx = np.where(reduced_datain[i,:,:]>rms)
            mask = np.zeros([size, size])
            mask[idx]=1
            tab = tab + reduced_datain[i,:,:]*mask * chanwidth

    return tab




def mom_quicker_conditioned(size, datain, idxvin, idxvout, chanwidth):
    tab = np.empty([size, size])
    reduced_datain = datain[idxvin:idxvout,:,:]
    width = idxvout-idxvin
    
    above_zero = np.where(reduced_datain[:,:,:]>0)
    print (np.shape(above_zero))
    tab = np.nansum(reduced_datain[above_zero],axis=0) * chanwidth


    return tab

def make_circle(x, y, xc, yc, r, pixval_arcsec):
    x = np.linspace(0, x-1, num=x)
    y = np.linspace(0, y-1, num=y)
    
    circle_tab = np.zeros([len(x), len(y)])
    a = xc
    b = yc
    #print (a, b)
    r = r / pixval_arcsec
    for xi in range(len(x)):
        for yi in range(len(y)):
            if (xi - b) ** 2 + (yi - a) ** 2 < r ** 2:
                circle_tab[xi, yi] = 1
            else:
                circle_tab[xi, yi] = 0
    return np.ma.make_mask(circle_tab)

def make_ellipse(x, y, xc, yc, bpa, bmaj, bmin, pixval_arcsec):
    angle = bpa
    D = bmaj / pixval_arcsec
    d = bmin / pixval_arcsec
    # yc=size/2.0
    # xc=size/2.0
    ellipse_tab = np.zeros([len(x), len(y)])
    for xi in range(len(x)):
        for yi in range(len(y)):
            cosa = math.cos(angle * np.pi / 180.0)
            sina = math.sin(angle * np.pi / 180.0)
            dd = d / 2 * d / 2
            DD = D / 2 * D / 2
            a = (cosa * (yi - yc) + sina * (xi - xc)) ** 2
            b = (sina * (yi - yc) - cosa * (xi - xc)) ** 2
            if (a / dd) + (b / DD) <= 1:
                ellipse_tab[yi, xi] = 1
            else:
                ellipse_tab[yi, xi] = 0
    return np.ma.make_mask(ellipse_tab)

def kms_to_ghz(v, restfreq):
    f = restfreq - (((v) * restfreq / const.c.to('km/s').value))
    f = f / 1e9
    return f

def tick_function(X,restfreq):
                V=restfreq-(((X)*restfreq/const.c.to('km/s').value))
                V=V/1e9
                return ["%.3f"% z for z in V]


def ghz_to_kms(X,restfreq):
    V = (restfreq - X * 1e9) * const.c.to('km/s').value / restfreq
    return ["%.3f" % z for z in V]
def ghz_to_kms_scalar(x,restfreq):
                V=(restfreq-x*1e9)*const.c.to('km/s').value/restfreq
                return V


def kms_to_ghz(X,restfreq):
    V = restfreq - (((X) * restfreq / const.c.to('km/s').value))
    return ["%.3f" % z for z in V]
def kms_to_ghz_scalar(v, restfreq):
    f = restfreq - (((v) * restfreq / const.c.to('km/s').value))
    f = f / 1e9
    return f


########################################################################################################################
#############################################tycho_func1           #####################################################
########################################################################################################################






def tycho_func1(ax, figtype, source, mol, prefix, vin, vout, regcolor,
                params = None,
                autocontours_check=False,
                customcontours_check=False,
                customcontours='',
                plotcolour = False,
                sourcemask=True,
                sourcemask_color = '',
                plotcont=True,
                plotcontour=True,
                maptype='plotreg',
                custommap = '',
                lowres_out=False,
                ticklabels=False,
                ticklabels_ra=True,
                ticklabels_dec=True,
                rmsprint = False,
                rmsmask = True,
                plotmom1 = False,
                rms_flux = 3,
                calculate_moment = True,
                velregime = None,
                showregion=False,
                yrmspos = 0.5,
                plot_scalebar=True,
                plot_contbeam=True,
                plot_linebeam=True,
                scalebar_size = 4000,
                cmap_vmax = 400.0):

#
#This function plots a moment map either as a contour or colormap
#It gets ax as an input and returns modified axis

########################################################################################################################
# in order to plot blueshifted emission on the redshifted side a parameter sourcemask_color is used
# if not given it is assumed to be the same as regcolor
    if sourcemask_color == '':
        sourcemask_color = str(regcolor)

# start measuring the duration of the function
    func1_start_time = time.time()
    #print("--- %s minutes --- " % ((time.time() - func1_start_time) / 60.0))

# opening the json catalogs with sources and figure parameters
    with open('json/sources.json', 'r') as f:
       sources = json.load(f)
    with open('json/figure1_params.json', 'r') as f:
       params = json.load(f)

# defining the location of the fits files, beam files and region files


# if HCN things are different because of the larger image size
    if mol == 'HCN':
        band = 'Band3'
        r1_red = pf.getdata(prefix + sources[source]['field'] + '/' + source + '_' + 'red' + '_B3.fits')
        r1_blue = pf.getdata(prefix + sources[source]['field'] + '/' + source + '_' + 'blue' + '_B3.fits')
        r1_noise = pf.getdata(prefix + sources[source]['field'] + '/' + sources[source]['field'] + '_' + 'noise' + '_B3.fits')
        contimage = prefix + sources[source]['field'] + '/cont_B3/' + sources[source]['field'] + '_' + 'cont' + '.fits'  ###### field + mol
        contimage_beam = prefix + sources[source]['field'] + '/cont_B3/' + sources[source]['field'] + '_' + 'cont' + '.fits.beam'  ###### field + mol

# all the other lines
    else:
        band = 'Band6'
        r1_red = pf.getdata(prefix + sources[source]['field'] + '/' + source + '_' + 'redEHV' + '_B6.fits')
        r1_blue = pf.getdata(prefix + sources[source]['field'] + '/' + source + '_' + 'blue' + '_B6.fits')
        r1_noise = pf.getdata(prefix + sources[source]['field'] + '/' + sources[source]['field'] + '_' + 'noise' + '_B6.fits')
        contimage = prefix + sources[source]['field'] + '/cont/' + sources[source]['field'] + '_' + 'cont' + '.fits'  ###### field + mol
        contimage_beam =  prefix + sources[source]['field'] + '/cont/' + sources[source]['field'] + '_' + 'cont' + '.beam'  ###### field + mol

    if source == 'Emb8N' and mol == 'H2CO':
        image = prefix + 'Emb8N_h2co_nocontsub.pbcor.fits'  ###### field + mol
    else:
        image = prefix + sources[source]['field'] + '/' + sources[source]['field'] + '_' + mol + '.fits'  ###### field + mol


# get the WCS and datain and header - obsolete because it is already done in the FitsHandler?
    #w = WCS(image)
    #w = w.dropaxis(3)  # Remove the Stokes axis
    #w = w.dropaxis(2)  # and spectral axis
    #datain, header = pf.getdata(image, header=True)
    #datain = datain[0, :, :, :]


    #continuum data, outside of fitshandler
    data_continuum, header_continuum = pf.getdata(contimage,header=True)
    data_continuum = data_continuum[0, 0, :, :]

    dataset_K, header, size, velocity, chanwidth, nchan, restfreq, pixval_arcsec,w = FitsHandler1(source, mol)

    datain = dataset_K
    veltab = velocity

########################################################################################################################
    #IMPORTING FLUX FILES
    prefix = '../../Projects/'
    if mol == 'HCN':
        fluximage = prefix + sources[source]['field']+'/'+sources[source]['field']+'_B3_flux.fits' ###### field + mol
    else:
        fluximage = prefix + sources[source]['field']+'/'+sources[source]['field']+'_B6_flux.fits' ###### field + mol
    fluximage = pf.getdata(fluximage)
    if len(np.shape(fluximage)) == 4:
        fluximage = fluximage[0,0,:,:]

    #####################
    # finding the maximum sensitivity pixel
    maxflux = np.where(fluximage == np.nanmax(fluximage))
    #print ('---- peak primary beam ',np.nanmax(fluximage))
    #print ('---- peak primary beam sensitivity at pixel',maxflux)
    #print ('---- size primary beam file',np.shape(fluximage))



    f = plt.figure()
    ax1 = f.add_subplot('111')
    x = np.linspace(0, size, size)
    pixval_x = header['CDELT1']
    pixval_arcsec = np.abs((pixval_x) * 3600.0)

    if mol != 'HCN':
        lam = 0.001300448783239
    else:
        lam = 0.003
    D = 12.0
    x_fwhm = 1.28 * (lam / D * 180.0 / np.pi * 3600.0) / pixval_arcsec
    y_fwhm = 1.28 * (lam / D * 180.0 / np.pi * 3600.0) / pixval_arcsec
    y = np.linspace(0, size, size)
    x, y = np.meshgrid(x, y)
    # print 'gauss value at center', (gauss2d(224,224,x_cen=224,y_cen=224,x_sigma=100.0/2.35482,y_sigma=100.0/2.35482))
    # print 'gauss value at 100,100', (gauss2d(100,100,x_cen=224,y_cen=224,x_sigma=100.0/2.35482,y_sigma=100.0/2.35482))
    im = ax1.imshow( gauss2d(x, y, x_cen=size / 2.0, y_cen=size / 2.0, x_sigma=x_fwhm / 2.35482, y_sigma=y_fwhm / 2.35482),
        cmap='viridis',vmin=0,vmax=1)
    cbar = f.colorbar(im, orientation='vertical')
    plt.savefig(source+'_'+mol+'_gauss.png')
    plt.close()


    flux_m = 800
    flux_n = 600

    if source == 'Emb8N' or source == 'S68N':
        if mol == 'CO':
            smooth = False
            velrms_var1 = [30, 80]
            flux_m = 800
            flux_n = 600
        if mol == 'SiO' :
            smooth = True
            velrms_var1 = [-38, -10]
            velrms_var2 = [-60, -42]
            flux_m = 800
            flux_n = 600
        if mol == 'H2CO':
            smooth = True
            velrms_var1 = [10, 35]
            velrms_var2 = [-38, -27]
            flux_m = 750
            flux_n = 750
        if mol == 'HCN':
            smooth = False
            velrms_var1 = [10,50]
            flux_m = 1600
            flux_n = 1600
    if source == 'SMM1a' or source == 'SMM1b' or source == 'SMM1d':
        if mol == 'CO':
            smooth = False
            velrms_var1 = [-70, -20]
            flux_m = 800
            flux_n = 600
        if mol == 'SiO':
            smooth = True
            velrms_var1 = [-30, -10]
            velrms_var2 = [-60, -42]
            flux_m = 800
            flux_n = 600
        if mol == 'H2CO':
            smooth = True
            velrms_var1 = [-25, -10]
            velrms_var2 = [-38, -27]
            flux_m = 750
            flux_n = 750
        if mol == 'HCN':
            smooth = False
            velrms_var1 = [10,35]
            flux_m = 1200
            flux_n = 1400

    ####plotting primary beam sensitivity


    idxvrms_var1_in = np.argmin(np.abs(veltab - (velrms_var1[0])))
    idxvrms_var1_out = np.argmin(np.abs(veltab - (velrms_var1[1])))


    rms_circle = cmask([flux_m,flux_n], 20, datain[0,:,:])
    idx_rms_circle = np.where(rms_circle)

    #rmscenter_pix_beam = np.nanstd(datain[idxvrms_var1_in:idxvrms_var1_out,idx_rms_circle[0],idx_rms_circle[1]])
    rmscenter_pix_beam = np.nanstd(datain[idxvrms_var1_in:idxvrms_var1_out,idx_rms_circle[0],idx_rms_circle[1]])
    rmscenter_pix_beam = rmscenter_pix_beam*fluximage[flux_m,flux_n]

    if mol == 'HCN':
        if source == 'S68N' or source == 'Emb8N':
           # print ('getting gauss approx instead of the fluxfile!')
            fluximage = gauss2d(x, y, x_cen=size / 2.0, y_cen=size / 2.0, x_sigma=x_fwhm / 2.35482, y_sigma=y_fwhm / 2.35482)
            rmscenter_pix_beam = rmscenter_pix_beam* gauss2d(flux_m,flux_n, x_cen=size / 2.0, y_cen=size / 2.0, x_sigma=x_fwhm / 2.35482, y_sigma=y_fwhm / 2.35482)


    meantab_rms = np.zeros(np.shape(veltab))
    for i in range(np.shape(veltab)[0]):
        meantab_rms[i] = np.nanmean(datain[i,idx_rms_circle[0],idx_rms_circle[1]])

    if smooth:
        idxvrms_var2_in = np.argmin(np.abs(veltab - (velrms_var2[0])))
        idxvrms_var2_out = np.argmin(np.abs(veltab - (velrms_var2[1])))



    rms_circle = cmask([flux_m, flux_n], 5, datain[0, :, :])
    idx_rms_circle = np.where(rms_circle)


    if smooth:
        rmscenter_pix_beam_lowres = np.nanstd(datain[idxvrms_var2_in:idxvrms_var2_out, idx_rms_circle[0], idx_rms_circle[1]])
        rmscenter_pix_beam_lowres = rmscenter_pix_beam_lowres/fluximage[flux_m,flux_n]
        rmscenter_pix_beam_lowres = rmscenter_pix_beam_lowres


        meantab_rms_lowres = np.zeros(np.shape(veltab))
        for i in range(np.shape(veltab)[0]):
            meantab_rms_lowres[i] = np.nanmean(datain[i, idx_rms_circle[0], idx_rms_circle[1]])

    f = plt.figure()
    ax1 = f.add_subplot('111')
    ax1.text(-20,5,rmscenter_pix_beam)
    ax1.plot(veltab,meantab_rms)
    plt.savefig(source+'_'+mol+'_rms_spectra.png')
    plt.close()

    #print ('rms within the circle of radius 20pix.....', rmscenter_pix_beam)
    if smooth:
        print ('rms lowres.....', rmscenter_pix_beam_lowres)





    circle_mask_tab = np.zeros(np.shape(datain[0,:,:]))
    circle_mask_tab[idx_rms_circle] = 1


    f = plt.figure()
    ax1 = f.add_subplot('111')
    im = ax1.imshow(fluximage,origin='lower',vmin=0,vmax=1)
    ax1.plot(flux_n,flux_m,'+',markersize=10)
    cbar = f.colorbar(im, orientation='vertical')

    plt.savefig(source+'_'+mol+'_rms_flux.png')
    plt.close()



    ############################################

    if smooth:
       if mol == 'SiO':
                vresred = 40.5
                vresblue = -40.5
       elif mol == 'H2CO':
                vresred = 36.0
                vresblue = -26
       else:
                vresred = 200
                vresblue = -200
    else:
        vresred = 200
        vresblue = -200






########################################################################################################################
    rmsfromimage = False
    #loading an image of CO
    if rmsfromimage:
        rmstab = pf.getdata(prefix + sources[source]['field'] + '/' + sources[source]['field'] + '_' + mol +'_rms.fits', header=False)

#func1 - define the purpose of the function and parameters used


# here we check if we deal with B3 data (which is larger ~3200 px) or B6 (1500 px)
# the B3 data have different region files associated with them

#!!! so it seems like those regions are not custom at all...
#!!! if anything unusual

    if source == 'Emb8N':
        beam_shift_x = 40
        beam_shift_y = 20
    else:
        beam_shift_x = 50
        beam_shift_y = 50



    xc_world = sources[source]['coords']['ra']
    yc_world = sources[source]['coords']['dec']


    if maptype == 'custom':
        xi_world, xo_world, yi_world, yo_world = custommap
    else:
        xi_world = sources[source][maptype]['xi']
        xo_world = sources[source][maptype]['xo']
        yi_world = sources[source][maptype]['yi']
        yo_world = sources[source][maptype]['yo']
    #if maptype == 'source':
    #    xi_world = sources[source]['plotreg']['xi']
    #    xo_world = sources[source]['plotreg']['xo']
    #    yi_world = sources[source]['plotreg']['yi']
    #    yo_world = sources[source]['plotreg']['yo']
    #elif maptype == 'field':
    #    xi_world = sources[source]['fieldreg']['xi']
    #    xo_world = sources[source]['fieldreg']['xo']
    #    yi_world = sources[source]['fieldreg']['yi']
    #    yo_world = sources[source]['fieldreg']['yo']


    xi_pix, yi_pix = w.all_world2pix(xi_world, yi_world, 0)  # Find pixel correspondig to ra,dec
    xo_pix, yo_pix = w.all_world2pix(xo_world, yo_world, 0)  # Find pixel correspondig to ra,dec



    if figtype == 'fig1_custom' and plotcontour == True:
        #if mol != 'HCN':
            ax.set_xlim(xi_pix, xo_pix)
            ax.set_ylim(yi_pix, yo_pix)
    if figtype == 'fig1' and plotcontour == True:
        #if mol != 'HCN':
            ax.set_xlim(xi_pix, xo_pix)
            ax.set_ylim(yi_pix, yo_pix)
    else:
        ax.set_xlim(xi_pix, xo_pix)
        ax.set_ylim(yi_pix, yo_pix)

    #xcen, ycen = w.all_world2pix(xc_world, yc_world, 0)

    #print(xc_world)

    xoff = 350
    yoff = 350


#    f = plt.figure(figsize=(6,6))

    cmap = mpl.cm.Greys
    cmap.set_under('white')
    cmap.set_over('black')
    cmap.set_bad('white')


    #print (source, band)
    #print (params[source])
    cont_vmin = params[source]['cont'][band]['vmin']
    cont_vmax = params[source]['cont'][band]['vmax']
    cont_stretch = params[source]['cont'][band]['stretch']

    #print(cont_vmin, cont_vmax, cont_stretch)

    if plotcont:
        norm = ImageNormalize (data_continuum, vmin = cont_vmin, vmax= cont_vmax, stretch=LogStretch(cont_stretch))



        beam_table_cont = \
            np.loadtxt(contimage_beam)
        pixval_x_cont, pixval_arcsec_cont, beam_cont_area_tab, beam_cont_a, beam_cont_b, beam_cont_pa\
            = BeamHandler(beam_table_cont, header_continuum)

        #careful - this works for offset, likely won't work for HCN
        #ax.imshow(data_continuum, norm = norm, vmin = cont_vmin, vmax= cont_vmax, cmap=cmap,transform=ax.get_transform(w))
        ax.imshow(data_continuum, norm = norm, vmin = cont_vmin, vmax= cont_vmax, cmap=cmap)

        rms_cont = np.nanstd(data_continuum)

        #ax.contourf(data_continuum, norm = norm, cmap=cmap, origin='lower',
        #                 extend='both')
        #ax.contour(data_continuum, levels=np.array([3, 6, 9, 12, 15, 20]) * rms_cont / 1.0, colors='k', origin='lower',
        #           linewidths=0.5)

        #ae = AnchoredEllipse(ax.transData, width=beam_cont_a/pixval_arcsec_cont, height=beam_cont_b/pixval_arcsec_cont, angle=beam_cont_pa,
        #                     loc='lower left', pad=0.5, borderpad=0.4,
        #                     frameon=False)

        ############# IF YOU WANT CONTINUUM BEAM
        if (figtype == 'fig1' and mol == 'H2CO') or (figtype == 'fig2' and velregime == 'EHV') or (figtype == 'fig1_custom')\
                or (figtype == 'fig1' and mol == 'HCN'):

            if plot_contbeam:

                beam1 = patches.Ellipse((xi_pix + beam_shift_x-20, yi_pix + beam_shift_y),width=beam_cont_a/pixval_arcsec_cont, height=beam_cont_b/pixval_arcsec_cont, angle=beam_cont_pa, facecolor='black',edgecolor='black')

            #print (ax.get_xlim()[0]+10,ax.get_ylim()[0]+10)
                ax.add_patch(beam1)


                #print ('Beam size cont: ', beam_cont_a, beam_cont_b)

        #ax.add_artist(ae)
        #print (pixval_x_cont, pixval_arcsec_cont, beam_cont_area_tab, beam_cont_a, beam_cont_b, beam_cont_pa)


#    ax.imshow(data_continuum,norm=LogNorm(vmin=0.0001,vmax=0.01),cmap='Greys',origin='lower')


    momtab = np.zeros([size, size])
    mom1tab = np.zeros([size, size])
    nocut_mom1tab = np.zeros([size, size])


    nocut_momtab = np.zeros([size, size])

    if lowres_out:
        if mol == 'SiO':
            if vout>39.5: vout=39.5
            if vin<-39.5: vin=-39.5
        if mol == 'H2CO':
                if vin < -24.5: vin = -24.5

    idxvin = np.argmin(np.abs(veltab - (vin)))
    idxvout = np.argmin(np.abs(veltab - (vout)))

    #print ('vin, vout:', vin, vout)


    vrms_in = sources[source][mol]['vrms_'+sourcemask_color][0]
    vrms_out = sources[source][mol]['vrms_'+sourcemask_color][1]
    #print ('vrms_in', vrms_in)
    #print ('vrms_out', vrms_out)


    idxvrmsin = np.argmin(np.abs(veltab - (vrms_in)))
    idxvrmsout = np.argmin(np.abs(veltab - (vrms_out)))

    noise_idx = np.where(r1_noise == 1)


    if showregion:
        ax.contour(r1_red, levels=[1],
                   colors='red', linewidths=1, transform=ax.get_transform(w))
        ax.contour(r1_blue, levels=[1],
               colors='blue', linewidths=1, transform=ax.get_transform(w))

    if calculate_moment:

        if figtype == 'fig2':
            print('Calculating moment map without threshold for %s %s' % (velregime, mol))
        if figtype == 'fig1':
            print('Calculating moment map without threshold for %s' % (mol))
        print("--- %s minutes --- " % ((time.time() - func1_start_time) / 60.0))


        if 'fitsfiles/mom0_'+source+'_'+mol+'_'+str(vin)+'_'+str(vout)+'_rms0'+'.fits' not in glob.glob('fitsfiles/*.fits'):
            print ('No moment map found - creating')
            for m in range(size):
                for n in range(size):
                    if np.isnan(datain[0, m, n]):
                        continue
                    if datain[0, m, n] == 0:
                        continue
                    nocut_momtab[m, n] = np.sum(datain[idxvin:idxvout, m, n]) * chanwidth
            pf.writeto('fitsfiles/mom0_'+source+'_'+mol+'_'+str(vin)+'_'+str(vout)+'_rms0'+'.fits',nocut_momtab)
        nocut_momtab = pf.getdata('fitsfiles/mom0_'+source+'_'+mol+'_'+str(vin)+'_'+str(vout)+'_rms0'+'.fits')

        if plotmom1:
            if 'fitsfiles/mom1_'+source+'_'+mol+'_'+str(vin)+'_'+str(vout)+'_rms0'+'.fits' not in glob.glob('fitsfiles/*.fits'):
                for m in range(size):
                    for n in range(size):
                        if np.isnan(datain[0, m, n]):
                            continue
                        if datain[0, m, n] == 0:
                            continue
                        nocut_momtab[m, n] = np.sum(datain[idxvin:idxvout, m, n]) * chanwidth
                        if plotmom1:
                            for k in range(len(datain[idxvin:idxvout, m, n])):
                                nocut_mom1tab[m, n] = datain[idxvin + k, m, n] * veltab[idxvin + k] * chanwidth + \
                                                      nocut_mom1tab[m, n]
                            nocut_mom1tab[m, n] = nocut_mom1tab[m, n] / nocut_momtab[m, n]
                    pf.writeto('fitsfiles/mom1_'+source+'_'+mol+'_'+str(vin)+'_'+str(vout)+'_rms0'+'.fits',nocut_mom1tab, clobber=True)

        if plotmom1:
            nocut_mom1tab = pf.getdata('fitsfiles/mom1_'+source+'_'+mol+'_'+str(vin)+'_'+str(vout)+'_rms0'+'.fits')
        #print (np.shape(nocut_momtab))
        #print (nocut_momtab)


        #print('Moment map maximum without threshold for %s' % (mol))
        #print(("--- %.2f K km s-1 --- "%(np.nanmax(nocut_momtab))))

    # print ('Done')
        if rmsmask:
            # print("--- %s minutes --- " % ((time.time() - func1_start_time) / 60.0))
          #print('Calculating moment map of ' + mol + ' with rms threshold of: ' + str(rms_flux) + ' sigma')
        # print(min(veltab), max(veltab))
        # print(idxvrmsin, idxvrmsout)
          if 'fitsfiles/mom0_' + source + '_' + mol + '_' + str(vin) + '_' + str(vout) + '_rms'+str(rms_flux)+ '.fits' not in glob.glob('fitsfiles/*.fits'):
            for m in range(size):
                for n in range(size):
                    if np.isnan(datain[0, m, n]):
                        continue
                    if datain[0, m, n] == 0:
                        continue
                    if sourcemask:
                        sourcemask_color = sourcemask_color
                        if sourcemask_color == 'red':
                            if r1_red[m, n] == 0:
                                continue
                        if sourcemask_color == 'blue':
                            if r1_blue[m, n] == 0:
                                continue

                    fluxscaling = fluximage[m, n]

                    #rmsval = np.nanstd(datain[idxvrmsin:idxvrmsout, m, n])
                    rmsval = rmscenter_pix_beam/fluxscaling

#                    print ("Compare rms measured on pixel vs. rms from flux file:")
                    #print (rmsval, rmsval_fluximage)


#                    if smooth:
#                        rmsval = rmscenter_pix_beam_lowres/fluxscaling


                        # idxrms_flux = np.where(datain[idxvin:idxvout, m, n] > rms_flux * rmstab[m,n])
                    idxrms_flux = np.where(datain[idxvin:idxvout, m, n] > rms_flux * rmsval)

                    momtab[m, n] = np.sum(datain[idxvin:idxvout, m, n][idxrms_flux]) * chanwidth

                    #else:
                    #    mom1tab[m, n] = 0.0

                # print("--- %s minutes --- " % ((time.time() - func1_start_time) / 60.0))
                # print ("Done")
            pf.writeto('fitsfiles/mom0_' + source + '_' + mol + '_' + str(vin) + '_' + str(vout) + '_rms'+str(rms_flux)+ '.fits',
                    momtab)
          if plotmom1:
              if 'fitsfiles/mom1_' + source + '_' + mol + '_' + str(vin) + '_' + str(vout) + '_rms' + str(
                            rms_flux) + '.fits' not in glob.glob('fitsfiles/*.fits'):
                        for m in range(size):
                            for n in range(size):
                                if np.isnan(datain[0, m, n]):
                                    continue
                                if datain[0, m, n] == 0:
                                    continue
                                if sourcemask:
                                    sourcemask_color = sourcemask_color
                                    if sourcemask_color == 'red':
                                        if r1_red[m, n] == 0:
                                            continue
                                    if sourcemask_color == 'blue':
                                        if r1_blue[m, n] == 0:
                                            continue

                                fluxscaling = fluximage[m, n]

                                # rmsval = np.nanstd(datain[idxvrmsin:idxvrmsout, m, n])
                                rmsval = rmscenter_pix_beam / fluxscaling

                                #                    print ("Compare rms measured on pixel vs. rms from flux file:")
                                # print (rmsval, rmsval_fluximage)


                                #                    if smooth:
                                #                        rmsval = rmscenter_pix_beam_lowres/fluxscaling


                                # idxrms_flux = np.where(datain[idxvin:idxvout, m, n] > rms_flux * rmstab[m,n])
                                idxrms_flux = np.where(datain[idxvin:idxvout, m, n] > rms_flux * rmsval)

                                momtab[m, n] = np.sum(datain[idxvin:idxvout, m, n][idxrms_flux]) * chanwidth

                                if plotmom1:
                                    # if mom0[m, n] > mom0rms:  # CAREFUL, THIS NUMBER IS HARDCODED
                                    for k in range(len(datain[idxvin:idxvout, m, n][idxrms_flux])):
                                        mom1tab[m, n] = datain[idxvin + k, m, n] * veltab[idxvin + k] * chanwidth + \
                                                        mom1tab[m, n]
                                    mom1tab[m, n] = mom1tab[m, n] / momtab[m, n]
                        pf.writeto('fitsfiles/mom1_' + source + '_' + mol + '_' + str(vin) + '_' + str(
                                            vout) + '_rms' + str(
                                            rms_flux) + '.fits', mom1tab, clobber=True)


                                    # else:
                                    #    mom1tab[m, n] = 0.0

                                    # print("--- %s minutes --- " % ((time.time() - func1_start_time) / 60.0))
                                    # print ("Done")
          momtab = pf.getdata(
                'fitsfiles/mom0_' + source + '_' + mol + '_' + str(vin) + '_' + str(vout) + '_rms'+str(rms_flux)+ '.fits')
          if plotmom1:
                mom1tab = pf.getdata(
                'fitsfiles/mom1_' + source + '_' + mol + '_' + str(vin) + '_' + str(vout) + '_rms'+str(rms_flux)+ '.fits')
        else:
            #('Using moment map calculated without threshold')
            momtab = nocut_momtab
            mom1tab = nocut_mom1tab

    # print (mom0[m,n],rmsval,chanwidth






        #print("--- %s minutes --- " % ((time.time() - func1_start_time) / 60.0))

        if mol in params[source]:
            if 'rms' in params[source][mol][sourcemask_color]:
                momrms = params[source][mol][sourcemask_color]['rms']
            else:
                momrms = np.std(momtab[noise_idx])
        else:
            momrms = np.std(momtab[noise_idx])


# if momrms == 0: continue

    nocut_rms = np.nanstd(nocut_momtab[noise_idx])

    momrms = nocut_rms

    #print ('-----momrms------: ',momrms)



    if plotmom1:
        reg_mom1 = np.where(momtab>3*momrms)
        reg_mom1_tab = np.zeros([size,size])
        reg_mom1_tab[reg_mom1] = 1

        mom1tab = mom1tab * reg_mom1_tab

        cmap = mpl.cm.seismic
        #cmap = mpl.cm.jet
        cmap.set_under('white', alpha=0)
        cmap.set_over('white')
        cmap.set_bad('white')
        im = ax.imshow(mom1tab, cmap=cmap, vmin=vin,vmax=vout)

        cbar  = plt.colorbar(im, orientation='vertical',ticks=[vin,vout+(vin-vout)/2.0,vout])
        cbar.set_label('v$_{\\rm LSR}$ [km s$^{-1}$]')


    if plotcontour:

        #print ('maxval before region', np.nanmax(momtab))


        if sourcemask:
            if sourcemask_color == 'red':
                momtab = momtab * r1_red
            elif sourcemask_color == 'blue':
                momtab = momtab * r1_blue
        maxval = np.nanmax(momtab)
        nocut_maxval = np.nanmax(nocut_momtab)

        #print ('maxval', maxval)
        #print ('maxval without threshold', nocut_momtab)


        #modifier for too high rms on the blueshifted Emb8N
        #if source == 'Emb8N':
        #    if sourcemask_color == 'blue':
        #        if mol != 'SiO':
        #            momrms = momrms/2.0


    #autocontours_check = True
    #autocontours = np.array([4, 5, 9, 15, 18, 30, 40, 50, 60, 80, 100])
        #autocontours = np.linspace(momrms*3,maxval,10)/momrms
        if 'levels_'+regcolor in sources[source][mol]:
            autocontours = np.array(sources[source][mol]['levels_'+regcolor])*momrms
        else:
            autocontours = np.array([3,6,9,15,20,40,60,80,100])*momrms
        #print (autocontours)

    #print(np.linspace(momrms,maxval,10)/momrms)

        if autocontours_check:
            contours = autocontours
        elif customcontours_check:
            print('custom contours')
            contours = customcontours
        else:
            if mol in params[source]:
                contours =  np.array(params[source][mol][regcolor]['contours'])
            else:
                contours = autocontours

    # axarr[plotidx].contour(momtab,levels=[5,8,10,20,80,120,150],colors=regcolor)
        if autocontours[0]<maxval:



            if regcolor == 'red':
                if figtype == 'fig1':
                    xrmspos = 0.05
                    if source == 'SMM1d': xrmspos = 0.75
                    yrmspos = 0.55
                if figtype == 'fig2':
                    xrmspos = 0.05
                    yrmspos = 0.55

                if rmsprint:
                    ax.text(xrmspos, yrmspos, r'' + str(round(momrms, 2)), color='#DF1420',
                                transform=ax.transAxes,
                                bbox={'facecolor': 'white', 'alpha': 0.99, 'edgecolor': 'white'})
                ax.contour(momtab, levels=contours,
                               colors='black',linewidths=1.0)
                ax.contour(momtab, levels=contours,
                               colors='#DF1420',linewidths=0.8)


            #fontsize=16
            elif regcolor == 'blue':

                if figtype == 'fig1' or figtype == 'fig1_custom':
                    xrmspos = 0.05
                    if source == 'SMM1d': xrmspos = 0.75
                    yrmspos = 0.65
                if figtype == 'fig2':
                    xrmspos = 0.05
                    yrmspos = 0.35


                if rmsprint:
                    ax.text(xrmspos, yrmspos, r'' + str(round(momrms, 2)), color='#2846EE',
                                transform=ax.transAxes,
                                bbox={'facecolor': 'white', 'alpha': 0.99, 'edgecolor': 'white'})
                ax.contour(momtab, levels=contours,
                               colors='black',linewidths=1.0)
                ax.contour(momtab, levels=contours,
                               colors='#2846EE',linewidths=0.8)

            elif regcolor == 'black':
                if rmsprint:
                    ax.text(0.05, 0.65, r'' + str(round(momrms, 2)), color='#DF1420',
                                transform=ax.transAxes,fontsize=16,
                                bbox={'facecolor': 'white', 'alpha': 0.99, 'edgecolor': 'white'})
                ax.contour(momtab, levels=contours,
                               colors='black',linewidths=0.7)

            else:
                if rmsprint:
                    ax.text(0.05, yrmspos, r'' + str(round(momrms, 2)), color=regcolor,
                            transform=ax.transAxes, fontsize=16,
                            bbox={'facecolor': 'white', 'alpha': 0.99, 'edgecolor': 'white'})
                ax.contour(momtab, levels=contours,
                               colors='black',linewidths=1.2)
                ax.contour(momtab, levels=contours,
                               colors=regcolor,linewidths=1.0)

        if mol == 'H2CO':
            mollabel = 'H$_2$CO'
        else:
            mollabel = mol

        if figtype == 'fig1':
            ax.text(0.09, 0.76, mollabel, color='black',
                            transform=ax.transAxes,fontsize=22,
                            bbox={'facecolor': 'white', 'alpha': 0.99, 'edgecolor': 'white'})
        if figtype == 'fig2':
            ax.text(0.07, 0.76, velregime, color='black',
                            transform=ax.transAxes,fontsize=22,
                            bbox={'facecolor': 'white', 'alpha': 0.99, 'edgecolor': 'white'})


        ############ if you want line beam

        if (figtype == 'fig1' and mol == 'H2CO') or (figtype == 'fig2' and velregime == 'EHV') or (figtype == 'fig1_custom')\
                or (figtype == 'fig1' and mol == 'HCN'):
            beam_table_line= \
                np.loadtxt(prefix + sources[source]['field'] + '/' + sources[source]['field'] + '_' + mol + '.beam')


            if mol == 'H2CO' and figtype == 'fig1_custom':
                pixval_x_line, pixval_arcsec_line, beam_line_area_tab, beam_line_a, beam_line_b, beam_line_pa \
                    = BeamHandler(beam_table_line, header, beam_type='max')
                beam_shift_x=40
                beam_shift_y=40
            else:
                pixval_x_line, pixval_arcsec_line, beam_line_area_tab, beam_line_a, beam_line_b, beam_line_pa \
                    = BeamHandler(beam_table_line, header, beam_type='min')


            if plot_linebeam:

                beam2 = patches.Ellipse((xi_pix + beam_shift_x, yi_pix + beam_shift_y), width=beam_line_a / pixval_arcsec_line,
                                height=beam_line_b /pixval_arcsec_line, angle=beam_line_pa, facecolor='red',
                                edgecolor='black')

                #print(ax.get_xlim()[0] + 10, ax.get_ylim()[0] + 10)
                ax.add_patch(beam2)

                #print ('beam size', mol ,beam_line_a, beam_line_b )

        if source == 'Emb8N':
            sourcelabel = 'Ser-emb 8 (N)'
        if source == 'S68N':
            sourcelabel = 'S68N'
        if source == 'SMM1a':
            sourcelabel = 'SMM1-a'
        if source == 'SMM1b':
            sourcelabel = 'SMM1-b'
        if source == 'SMM1d':
            sourcelabel = 'SMM1-d'

    #if figtype == 'fig1':
#
 #       if source == 'SMM1a':
  #          plt.subplots_adjust(top=0.98, bottom=0.06, left=0.07, right=0.95, hspace=0.0,
   #                         wspace=0.0)
    #    if source == 'Emb8N':
     #       plt.subplots_adjust(top=0.78, bottom=0.15, left=0.10, right=0.99, hspace=0.0,
      #                     wspace=0.0)
       # if source == 'S68N':
        #        plt.subplots_adjust(top=0.98, bottom=0.15, left=0.12, right=0.95, hspace=0.0,
        #                            wspace=0.0)
        #else:
        #    plt.subplots_adjust(top=0.98, bottom=0.06, left=0.07, right=0.95, hspace=0.0,
        #                    wspace=0.0)

    if plotcolour:


        if sourcemask:
            if sourcemask_color == 'red':
                momtab = momtab * r1_red
            elif sourcemask_color == 'blue':
                momtab = momtab * r1_blue

        # axarr[plotidx].contour(momtab,levels=[5,8,10,20,80,120,150],colors=regcolor)
        #print('plotting colours, whatsup?')
        if regcolor == 'red':
            cmap = plt.cm.Reds
            imnorm = ImageNormalize(momtab,vmin=3*momrms, vmax=30*momrms, stretch=LogStretch(1.5),clip=False)
            cmap.set_under(color='white', alpha=0.3)

            #ax.imshow(momtab, cmap=cmap, vmin=3*momrms, vmax=30*momrms, transform=ax.get_transform(w))
            ax.imshow(momtab, cmap=cmap, norm = imnorm)

        if regcolor == 'blue':
            cmap = plt.cm.Blues
            #cmap.set_under(3*momrms)
            imnorm = ImageNormalize(momtab,vmin=3*momrms, vmax=80*momrms, stretch=LogStretch(1.5),clip=False)
            cmap.set_under(color='white', alpha=0.3)

            ax.imshow(momtab, cmap=cmap, norm = imnorm)
        if regcolor == 'special':
            cmap = plt.cm.inferno
            #map.set_under(color='black', alpha=0.3)
            ax.imshow(momtab, cmap=cmap, transform=ax.get_transform(w))
 #           ax.contourf(momtab, levels=np.array([3,6,9,12,15,20,30,50])*momrms)

        if (figtype == 'fig1' and mol == 'H2CO') or (figtype == 'fig2' and velregime == 'EHV') or (
                figtype == 'fig1_custom') \
                    or (figtype == 'fig1' and mol == 'HCN'):
                beam_table_line = \
                    np.loadtxt(prefix + sources[source]['field'] + '/' + sources[source]['field'] + '_' + mol + '.beam')
                pixval_x_line, pixval_arcsec_line, beam_line_area_tab, beam_line_a, beam_line_b, beam_line_pa \
                    = BeamHandler(beam_table_line, header)

                if plot_linebeam:
                    beam2 = patches.Ellipse((xi_pix + beam_shift_x, yi_pix + beam_shift_y),
                                            width=beam_line_a / pixval_arcsec_line,
                                            height=beam_line_b / pixval_arcsec_line, angle=beam_line_pa,
                                            facecolor='red',
                                            edgecolor='black')

                    #print(ax.get_xlim()[0] + 10, ax.get_ylim()[0] + 10)
                    ax.add_patch(beam2)

                    #print('beam size', mol, beam_line_a, beam_line_b)


    if figtype == 'fig1':

        if mol == 'CO':
            if source == 'S68N':
                labelx=0.80
                labely=1.04
            elif source == 'Emb8N':
                labelx = 0.80
                labely = 1.04
            elif source ==  'SMM1a':
                labelx = 0.80
                labely = 1.04
            elif source == 'SMM1b':
                labelx = 0.72
                labely = 1.04
            else:
                labelx = 0.80
                labely = 1.04
            ax.text(labelx,labely,sourcelabel,
            transform=ax.transAxes,
            color='black',
            rotation=0,
            fontsize=32,
            bbox={'facecolor': 'white', 'alpha': 0.99, 'edgecolor': 'white'})

    if figtype == 'fig2':
            if velregime == 'fast':
                if source == 'Emb8N':
                    labelx = 0.25
                    labely = 1.04
                else:
                    labelx = 0.4
                    labely = 1.04
                #ax.text(labelx, labely, sourcelabel, transform=ax.transAxes, color='black', rotation=0,
                #        fontsize=32,
                #        bbox={'facecolor': 'white', 'alpha': 0.99, 'edgecolor': 'white'})

    # if regcolor == 'red':
    ##        axarr[plotidx].contourf(momtab,levels= momrms*np.array([4,5,9,15,18,30,40,50,60,80,100]),cmap='Reds')
    #    else:
    #        axarr[plotidx].contourf(momtab,levels= momrms*np.array([4,5,9,15,18,30,40,50,60,80,100]),cmap='Blues')

    #    axarr[plotidx].set_xlim([xcen-xoff,xcen+xoff])
    #    axarr[plotidx].set_ylim([ycen-yoff,ycen+yoff])



    #if source == 'Emb8N':
    #    if mol == 'H2CO':
    #        ax.set_title('H$_2$CO')
    #    else:
    #        ax.set_title(mol)

    #print ('size ratio:',(np.abs((xo_pix-xi_pix)/(yo_pix-yi_pix))))
    #print ('x size:', np.abs(xo_pix-xi_pix))
    #print ('y size:', np.abs(yo_pix-yi_pix))


    size_ratio = (np.abs((xo_pix-xi_pix)/(yo_pix-yi_pix)))
    x_size = np.abs(xo_pix-xi_pix)
    y_size = np.abs(yo_pix-yi_pix)


    coord_style = 'offset'

    #if coord_style == 'RaDec':
    ##    lon = ax.coords['ra']
    #    lat = ax.coords['dec']
    #    lon.display_minor_ticks(True)
    #    lat.display_minor_ticks(True)
    # if source == 'SMM1d':
    #    lon.set_axislabel('R.A.')
    #    lat.set_axislabel('Decl.', minpad=-1)
    #    lon.set_ticklabel_visible(True)
    #    lon.set_major_formatter('hh:mm:ss.s')
    #if ticklabels==False:
    #        lon.set_ticklabel_visible(False)
    #        lat.set_ticklabel_visible(False)
    #        lon.set_axislabel('')
    #        lat.set_axislabel('')
    #    if ticklabels_ra==False:
    #        lon.set_ticklabel_visible(False)
    #        lon.set_axislabel('')
    #    if ticklabels_dec==False:
    #        lat.set_ticklabel_visible(False)
    #        lat.set_axislabel("")
    #    lon.set_ticks(exclude_overlapping=True, size=6)
    #    lat.set_ticks(exclude_overlapping=True, size=6)
    #    ax.set_xticklabels(ax.get_xticklabels(),fontsize=4)


    #if coord_style == 'offset':
    print('offset')
    zoom_in_pixels_x = 10.0 / abs(pixval_arcsec)
    ax.set_xticks([xi_pix + x_size / 2.0 - zoom_in_pixels_x,\
                   xi_pix + x_size / 2.0 - 0.5 * zoom_in_pixels_x, \
                   xi_pix + x_size / 2.0,\
                   xi_pix + x_size / 2.0 + 0.5 * zoom_in_pixels_x, \
                   xi_pix+x_size/2.0+zoom_in_pixels_x])
    ticks_x =  np.array(ax.get_xticks())
    ax.set_xticklabels(np.round(ticks_x*abs(pixval_x)*3600.0-(xi_pix+x_size/2.0)*(abs(pixval_x)*3600),1))
    #print (ticks_x)
    ax.set_xlabel('$\Delta$ RA ["]')


    zoom_in_pixels_y = 4.0 / abs(pixval_arcsec)

    ax.set_yticks([yi_pix + y_size / 2.0 - zoom_in_pixels_y, \
               yi_pix + y_size / 2.0 - 0.5 * zoom_in_pixels_y, \
               yi_pix + y_size / 2.0,\
               yi_pix + y_size / 2.0 + 0.5 * zoom_in_pixels_y, \
               yi_pix + y_size / 2.0 + zoom_in_pixels_y])
    ticks_y = np.array(ax.get_yticks())
    ax.set_yticklabels(np.round(ticks_y * abs(pixval_x) * 3600.0 - (yi_pix + y_size / 2.0) * (abs(pixval_x) * 3600), 1))
    ax.set_ylabel('$\Delta$ Dec ["]')
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")





    #ax.coords.frame.set_linewidth(3)



    if (figtype == 'fig1' and mol == 'H2CO') or (figtype == 'fig2' and velregime == 'EHV'):
        if plot_scalebar == True:
            scalebar = AnchoredSizeBar(ax.transData,
                               scalebar_size/(pixval_arcsec*435), str(int(scalebar_size))+' au', 4,
                               pad=0.5,
                               color='black',
                               frameon=False,
                               size_vertical=1)
    #                           fontproperties=fontprops)
            ax.add_artist(scalebar)

    elif (figtype == 'fig1_custom'):
        if plot_scalebar == True:
            scalebar = AnchoredSizeBar(ax.transData,
                               scalebar_size/(pixval_arcsec*435), str(int(scalebar_size))+' au', 4,
                               pad=0.5,
                               color='black',
                               frameon=False,
                               size_vertical=1)
            ax.add_artist(scalebar)

    #mpl.rcParams['axes.linewidth'] = 5.0

    #rcParams["figure.figsize"] = [x_size/100.0*4.0,y_size/100.0]



    return f,ax,x_size/100.0,y_size/100.0







# ###### First big calculation: moments and max velocity per pixel.
#
# Let's review inputs:
# * vin, vout - from input json file we get proper values, and then depending on whether red or blue we get values. Note how it differs for EHV, where limits end for fast/slow boundaries
# * datain,header - image, if the values were initially in Jy/beam we take the converted table
# * region = red or blue, will have consequences in certain paths inside the function
# * rms = cutoff factor for max velocity.
# * rmsvel = table with in and out velocity for measuring rms
# * plot = True, if we want plotted output
# * vlsr = vlsr
# * r1 = r1, region file
# * sigma_cutoff = True, if we want to sum pixels only greater than rms*rmsvel value (i.e. to the max velocity)f
# * rmsmask = False - on top of that we can put a different mask on the max rms, but this would differ from Nienke's method where vmax was a cutoff
# * rms_flux =3 - cutoff value to this mask
# * mom1mask = True -we can put additional mask on the mom1 map based on the mom0 rms
#
# Calculated from: mom1cutoff * rms * chanwidth * 10 (as expected linewidth)
# * mom1cutoff=3 - cutoff to this mask
#
# **Joe's script takes peak value and then goes to the v_max, that might be safer than just picking the fastes value**

# In[18]:


def moment_map_exe(source,mol,regions,sources,values,tables,datain,header,plot=True):
    full=True

    if full:
        velregimes = ['slow','fast','EHV','total']
        #velregimes = ['EHV_hires']

        regions = regions
    else:
        velregimes = ['slow']
        regions= ['blue']
    prefix = '../../Projects/'

    if mol == 'HCN':
        r1_red = pf.getdata(prefix + sources[source]['field']+'/'+source+'_'+'red'+'_B3.fits')
        r1_blue = pf.getdata(prefix + sources[source]['field']+'/'+source+'_'+'blue'+'_B3.fits')
    else:
        r1_red = pf.getdata(prefix + sources[source]['field']+'/'+source+'_'+'red'+'_B6.fits')
        r1_blue = pf.getdata(prefix + sources[source]['field']+'/'+source+'_'+'blue'+'_B6.fits')

    for region in regions:
        if region == 'blue':
            region_color = region
            r1=r1_blue #make sure it works for ALMA
        elif region == 'red':
            region_color = region
            r1=r1_red #make sure it works for ALMA
        else:
            if region in ['H2CO_bullet','b3','b5']:
                region_color = 'blue'
            if region in ['r2','r4','r5']:
                region_color = 'red'
            if mol == 'HCN':
                r1 = pf.getdata(prefix + sources[source]['field'] + '/' + source + '_' + region + '_B3.fits')
            else:
                r1 = pf.getdata(prefix + sources[source]['field'] + '/' + source + '_' + region + '_B6.fits')

        for velregime in velregimes:
                if region_color == 'blue':
                    if velregime == 'total':
                        vin = sources[source]['velout'][0]
                        vout = sources[source]['velin'][0]
                    if velregime == 'EHV':
                        vin = sources[source]['velout'][0]
                        vout = sources[source]['ehvlim'][0]
                        if vout == -99:
                            vout = sources[source]['velout'][0]
                    if velregime == 'fast':
                        vin = sources[source]['ehvlim'][0]
                        vout = sources[source]['ihvlim'][0]
                        if vin == -99:
                            vin = sources[source]['velout'][0]
                    if velregime == 'EHV_hires':
                        vin = sources[source]['highres_SiO'][0]
                        vout = sources[source]['ehvlim'][0]
                        if sources[source]['ehvlim'][0] == -99:
                            vin = sources[source]['velout'][0]
                    elif velregime == 'slow':
                        vin = sources[source]['ihvlim'][0]
                        vout = sources[source]['velin'][0]

                elif region_color == 'red':
                    if velregime == 'total':
                        vin = sources[source]['velin'][1]
                        vout = sources[source]['velout'][1]
                    elif velregime == 'slow':
                        vin = sources[source]['velin'][1]
                        vout = sources[source]['ihvlim'][1]
                    elif velregime == 'fast':
                        vin = sources[source]['ihvlim'][1]
                        vout = sources[source]['ehvlim'][1]
                        if vout == -99:
                            vout = sources[source]['velout'][1]
                    elif velregime == 'EHV':
                        vin = sources[source]['ehvlim'][1]
                        vout = sources[source]['velout'][1]
                        if vin == -99:
                            vin = sources[source]['velout'][1]
                    elif velregime == 'EHV_hires':
                        vin = sources[source]['ehvlim'][1]
                        vout = sources[source]['highres_SiO'][1]
                        if vin == -99:
                                vin = sources[source]['velout'][1]

                if mol in sources[source]:
                    rmsvel = sources[source][mol]['vrms_' + region_color]
                else:
                    rmsvel = sources[source]['rms'+region_color]

                if region not in values[source]:
                    values[source].update({region:{}})

                if velregime not in values[source][region]:
                        values[source][region].update({velregime:{}})
                #if type(values[source][region][velregime][mol]['mom0mean']) == float :
                #    values[source][region][velregime][mol].update({'mom0mean': {}})

                if mol not in values[source][region][velregime]:
                        values[source][region][velregime].update({mol: {}})

                if 'mom0sum' not in values[source][region][velregime][mol]:
                    values[source][region][velregime][mol].update({'mom0sum':{},
                                                                   'mom0mean':{},
                                                                   'vmax_mean':{},
                                                                   'vmax_max':{},
                                                                   'rms':{}})
                values[source][region][velregime].update({'CO_SiO': {}})
                values[source][region][velregime]['CO_SiO'].update({'mom0mean': {}})
                values[source][region][velregime].update({'CO_H2CO': {}})
                values[source][region][velregime]['CO_H2CO'].update({'mom0mean': {}})
                values[source][region][velregime].update({'CO_HCN': {}})
                values[source][region][velregime]['CO_HCN'].update({'mom0mean': {}})


                values[source][region][velregime][mol]['mom0sum']['value'],\
                values[source][region][velregime][mol]['mom0sum']['rms'],\
                values[source][region][velregime][mol]['mom0sum']['flag'], \
                values[source][region][velregime][mol]['mom0mean']['value'],\
                values[source][region][velregime]['CO_SiO']['mom0mean']['value'],\
                values[source][region][velregime]['CO_H2CO']['mom0mean']['value'],\
                values[source][region][velregime]['CO_HCN']['mom0mean']['value'],\
                values[source][region][velregime][mol]['mom0mean']['rms'],\
                values[source][region][velregime][mol]['mom0mean']['flag'],\
                values[source][region][velregime][mol]['vmax_mean'],\
                values[source][region][velregime][mol]['vmax_max'],\
                values[source][region][velregime][mol]['rms'],\
                tables[source][region][velregime][mol]['rms'],                \
                tables[source][region][velregime][mol]['mom0'], \
                tables[source][region][velregime][mol]['mom1'],                \
                tables[source][region][velregime][mol]['maxval'],\
                tables[source][region][velregime][mol]['dettab']\
                 = moment_map(source,mol,vin,vout,datain,header,
                                                                              region,region_color,rms=3,rmsvel=rmsvel,plot=plot,
                                                                              r1=r1,sigma_cutoff=True,rmsmask=True,
                                                                              rms_flux=3,
                                                                              mom1mask = True, mom1cutoff = 3, velreg = velregime)


                if velregime == 'total':
                    if region == 'blue':
                        sources[source]['vellim'][0] = values[source][region][velregime][mol]['vmax_max']
                    if region == 'red':
                        sources[source]['vellim'][1] = values[source][region][velregime][mol]['vmax_max']

    return values, tables



def mass_main_exe(source,mol,tr,T,regions,sources,values,tables,datain,header,pixval_arcsec,plot=True):
    velregimes = ['slow','fast','EHV','total']
    #velregimes = ['EHV_hires']
    prefix = '../../Projects/'



    if mol == 'HCN':
        r1_red = pf.getdata(prefix + sources[source]['field']+'/'+source+'_'+'red'+'_B3.fits')
        r1_blue = pf.getdata(prefix + sources[source]['field']+'/'+source+'_'+'blue'+'_B3.fits')
    else:
        r1_red = pf.getdata(prefix + sources[source]['field']+'/'+source+'_'+'red'+'_B6.fits')
        r1_blue = pf.getdata(prefix + sources[source]['field']+'/'+source+'_'+'blue'+'_B6.fits')

    for region in regions:
        if region == 'blue':
            r1=r1_blue #make sure it works for ALMA
        elif region == 'red':
            r1=r1_red #make sure it works for ALMA
        else:
            if region == 'H2CO_bullet':
                region_color = 'blue'
            if mol == 'HCN':
                r1 = pf.getdata(prefix + sources[source]['field'] + '/' + source + '_' + region + '_B3.fits')
            else:
                r1 = pf.getdata(prefix + sources[source]['field'] + '/' + source + '_' + region + '_B6.fits')

        for velregime in velregimes:
            #print (source, region, velregime, mol)

                for par in ['Ntot_cm','Ntot','Nmax_cm','Nmol','Mmass']:
                    if par in values[source][region][velregime][mol]:
                        if type(values[source][region][velregime][mol][par]) == float:
                            values[source][region][velregime][mol].update({par: {}})
                    else:
                        values[source][region][velregime][mol].update({par:{}})


                values[source][region][velregime][mol]['Ntot_cm']['value'], \
                values[source][region][velregime][mol]['Ntot_cm']['rms'], \
                values[source][region][velregime][mol]['Ntot_cm']['flag'], \
                values[source][region][velregime][mol]['Ntot']['value'], \
                values[source][region][velregime][mol]['Ntot']['rms'], \
                values[source][region][velregime][mol]['Ntot']['flag'], \
                values[source][region][velregime][mol]['Nmax_cm']['value'], \
                values[source][region][velregime][mol]['Nmax_cm']['rms'], \
                values[source][region][velregime][mol]['Nmax_cm']['flag'], \
                values[source][region][velregime][mol]['Nmol']['value'], \
                values[source][region][velregime][mol]['Nmol']['rms'], \
                values[source][region][velregime][mol]['Nmol']['flag'], \
                values[source][region][velregime][mol]['Mmass']['value'], \
                values[source][region][velregime][mol]['Mmass']['rms'], \
                values[source][region][velregime][mol]['Mmass']['flag'], \
                        = mass_main(source,mol,tr,T,region,velregime,values,np.array(tables[source][region][velregime][mol]['mom0']),np.array(tables[source][region][velregime][mol]['dettab']),r1,pixval_arcsec,
                              plot=True, positive_mask=False)


                if velregime == 'total':
                    if region == 'blue':
                        sources[source]['vellim'][0] = values[source][region][velregime][mol]['vmax_max']
                    if region == 'red':
                        sources[source]['vellim'][1] = values[source][region][velregime][mol]['vmax_max']

    return values, tables
#def mass_main(source,mol,tr,T,region,velregime,mom0,r1,pixval_arcsec,plot=False,positive_mask=False):
#return  np.nanmean((Ntot*r1)[region_idx]), np.nansum(Ntot*r1),  np.nansum(N_mol*r1),  np.nansum(M_mol*r1),#Ntot.tolist(), N_mol.tolist(), M_mol.tolist()


# In[6]:


def custom_moment(source,mol,vin,vout,datain,header,rmsmom0=0,multi_rmsmom0=3,rmspec=0,rmsvel=[0,0],r1=[10,10],rmsreg=[10,10],
                  show_patches=True,ylim=[0,0],xlim=[0,0]):

    plt.rc('font', family='serif')

    size = np.shape(datain[0,:,0])[0]
    tab = np.zeros([size,size])
    nchan = np.shape(datain[:,0,0])[0]
    meantab = np.zeros([nchan,1])
    sumtab = np.zeros([nchan,1])

    with open('json/sources.json', 'r') as f:
        sources = json.load(f)

#        border[0] = sources[source]['vellim'][0]
#        border[1] = sources[source]['vellim'][1]
#        print (border)



    vlsr = sources[source]['vlsr']
#    ehvblue = sources[source]['ehvlim'][0]
#    ehvred = sources[source]['ehvlim'][1]


#####
    veltab, chanwidth, nchan, restfreq = SpecaxisHandler(datain,header,vlsr)


    idxvin = np.argmin(np.abs(veltab - (vin)))
    idxvout = np.argmin(np.abs(veltab - (vout)))
    idxvrmsin = np.argmin(np.abs(veltab - (rmsvel[0])))
    idxvrmsout = np.argmin(np.abs(veltab - (rmsvel[1])))

    idxres1 = np.argmin(np.abs(veltab - (-40.5)))
    idxres2 = np.argmin(np.abs(veltab - (40.5)))



    for m in range(size):
        for n in range(size):

            if np.isnan(datain[0,m,n]):
                continue
            if datain[0,m,n] == 0:
                continue
            if rmsmom0 > 0:
                rmsmeasure = np.std(datain[idxvrmsin:idxvrmsout,m,n])
                idxrmsmom0  = np.where(datain[idxvin:idxvout,m,n]>rmsmeasure*multi_rmsmom0)
                tab[m,n] = np.sum(datain[idxvin:idxvout,m,n][idxrmsmom0])*chanwidth
            else:
                tab[m,n] = np.sum(datain[idxvin:idxvout,m,n])*chanwidth


 #           else:
  #              tab[m,n] = np.sum(datain[idxvin:idxvout,m,n])*chanwidth


    region_idx = np.where(r1==1)



    for ix in range(len(veltab)):
        if r1 == [10,10]:
            meantab[ix] = np.nanmean(datain[ix,:,:])
        else:
            if rmspec > 0:
                idxrmspec = np.where(datain[ix,region_idx[0],region_idx[1]]>rmspec)
                meantab[ix] = np.nanmean(datain[ix,region_idx[0],region_idx[1]][idxrmspec])
            else:
                meantab[ix] = np.nanmean(datain[ix, region_idx[0], region_idx[1]])

    for ix in range(len(veltab)):
        sumtab[ix] = np.nansum(datain[ix,:,:])

    f, axarr = plt.subplots(1, 2, gridspec_kw = {'width_ratios':[2, 1]})
    f.set_figheight(5)
    f.set_figwidth(18)
    axarr[0].plot(veltab,meantab,drawstyle='steps-mid',color='black',linewidth=0.5)
    im = axarr[1].imshow(tab[:,:],origin='lower',cmap='viridis_r')
    f.colorbar(im, orientation='vertical')

    if r1 != [10,10]:
        axarr[1].contour(r1, origin='lower', color='black', vmin=0, vmax=1,levels=[0,1])
        axarr[1].set_ylim(np.min(np.where(r1[:,:]>0)[0]),np.max(np.where(r1[:,:]>0)[0]))
        axarr[1].set_xlim(np.min(np.where(r1[:,:]>0)[1]),np.max(np.where(r1[:,:]>0)[1]))

#axarr[1].plot([vin,vin],axarr[0].get_ylim(),'--')


    if ylim != [0, 0]:
        axarr[0].set_ylim(ylim)
    if xlim != [0, 0]:
        axarr[0].set_xlim(xlim)

    ax2=axarr[0].twiny()
    ax2.set_xlim(axarr[0].get_xlim())
    #"    print (ax2.get_xlim())\n",
    #"    print (ax2.get_xlim())\n",
    def tick_function(X):
            V=restfreq-(((X)*restfreq/const.c.to('km/s').value))
            V=V/1e9
            return ["%.3f"% z for z in V]
    def kms_to_ghz(v):
        f = restfreq-(((v)*restfreq/const.c.to('km/s').value))
        f = f/1e9
        return f

    def ghz_to_kms(X):
            V=(restfreq-X*1e9)*const.c.to('km/s').value/restfreq
           #V=V*1e9
            return ["%.3f"% z for z in V]

    velspan = ax2.get_xlim()[1]-ax2.get_xlim()[0]
    freq_ticks_loc = np.arange(ax2.get_xlim()[0],ax2.get_xlim()[1],velspan/8.0)
    #tick_start=np.round(float(kms_to_ghz(ax2.get_xlim()[0])),decimals=1)
    #tick_end=np.round(float(kms_to_ghz(ax2.get_xlim()[1])),decimals=1)
    #print (np.round(tick_start,decimals=1),np.round(tick_end,decimals=1))


    ax2.set_xticks(freq_ticks_loc)
    #print (np.arange(tick_end,tick_start,0.1))
    ax2.set_xticklabels(tick_function(freq_ticks_loc))
    #ax2.set_xticklabels(np.arange(tick_start,tick_end,0.1))
    ax2.set_xlabel('Freqeuncy (GHz)',fontsize=14)

    lineghz=np.array([218.98100900, 218.97740600,218.90335550,219.15181800,219.14267450,219.09858090,219.09487100,218.45965290])
    lineloc =np.array([9,5,9,9,5,9,4,9])
    linelabel = ['HCNO 10-9', 'MF','OCS 18-17','CH$_3$NH$_2$','CCS','HCOOH','Glycoladehyde','NH$_2$CHO']

    linepos=ghz_to_kms(lineghz)
    print (linepos)

    #axarr[0].plot([linepos[0],linepos[0]],[axarr[0].get_ylim()[0],axarr[0].get_ylim()[1]],'--',linewidth=0.5)
    for i in range(len(linepos)):
        if float(linepos[i]) > ax2.get_xlim()[0] and float(linepos[i]) < ax2.get_xlim()[1]:
            print (i, '   time around')
            print (linepos[i])
            ax2.text(float(linepos[i])+6.0, lineloc[i], linelabel[i], color='blue', rotation=90,
                bbox={'facecolor': 'white', 'alpha': 0.99, 'edgecolor': 'white'})
            axarr[0].plot([float(linepos[i]), float(linepos[i])], [axarr[0].get_ylim()[0], axarr[0].get_ylim()[1]], '--',
                      color='blue', linewidth=0.5)

    axarr[0].set_xlabel('Velocity [km s$^{-1}$]',fontsize=14)
    axarr[0].set_ylabel('Mean Intensity [K]',fontsize=14)

    if show_patches:
        axarr[0].add_patch(
            patches.Rectangle(
            (vin,axarr[0].get_ylim()[0]),   # (x,y)
            vout-vin,          # width
            axarr[0].get_ylim()[1]-axarr[0].get_ylim()[0],          # height
            alpha=0.2
            )
        )
        axarr[0].add_patch(
            patches.Rectangle(
            (rmsvel[0],axarr[0].get_ylim()[0]),   # (x,y)
            rmsvel[1]-rmsvel[0],          # width
            axarr[0].get_ylim()[1]-axarr[0].get_ylim()[0],          # height
            alpha=0.1,
            color='yellow'
            )
        )
    if rmsreg != [10,10]:
        rms_mom0 = np.nanstd(tab[rmsreg[0]:rmsreg[1],rmsreg[2]:rmsreg[3]])
        print ('RMS of the moment map: ', rms_mom0)

        axarr[1].add_patch(
        patches.Rectangle(
        (rmsreg[0],rmsreg[2]),   # (x,y)
        rmsreg[1]-rmsreg[0],          # width
        rmsreg[3]-rmsreg[2],          # height
        alpha=0.2
        )
    )

    plt.setp(axarr[0].get_yticklabels(), rotation='horizontal', fontsize=14)
    plt.setp(axarr[0].get_xticklabels(), rotation='horizontal', fontsize=14)

    #axarr[0].text(0.9, 0.9, r'' + str(np.nansum(tab)), color='black',
    #        transform=axarr[0].transAxes,
    #        bbox={'facecolor': 'white', 'alpha': 0.99, 'edgecolor': 'white'})

    #axarr[0].text(0.9, 0.8, r'' + str(multi_rmsmom0) +'$\sigma$', color='black',
    #        transform=axarr[0].transAxes,
    #        bbox={'facecolor': 'white', 'alpha': 0.99, 'edgecolor': 'white'})



    plt.savefig('testplots/' + source + '_' + mol + '_moment_map_' + str(vin) + '_' + str(vout) + 'rms'+str(multi_rmsmom0)+'.png',
                format='png')
    plt.show()
    plt.close()


# In[41]:




def custom_moment_c18o(vin,vout,datain,header,rms=0,rmsvel=[0,0],vlsr=8.5,r1=[10,10]):

    size = np.shape(datain[0,:,0])[0]
    tab = np.zeros([size,size])
    nchan = np.shape(datain[:,0,0])[0]
    meantab = np.zeros([nchan,1])
    sumtab = np.zeros([nchan,1])
    rmstab = np.zeros([size,size])


#####
    veltab, chanwidth, nchan, restfreq = SpecaxisHandler(datain,header,vlsr)


    idxvlrs = np.argmin(np.abs(veltab - (0)))
    idxvin = np.argmin(np.abs(veltab - (vin)))
    idxvout = np.argmin(np.abs(veltab - (vout)))
    idxvrmsin = np.argmin(np.abs(veltab - (rmsvel[0])))
    idxvrmsout = np.argmin(np.abs(veltab - (rmsvel[1])))

    idxres1 = np.argmin(np.abs(veltab - (-40.5)))
    idxres2 = np.argmin(np.abs(veltab - (40.5)))

    for m in range(size):
        for n in range(size):

            if np.isnan(datain[0,m,n]):
                continue
            if datain[0,m,n] == 0:
                continue
            tab[m,n] = np.sum(datain[idxvin:idxvout,m,n])*chanwidth
 #           else:
  #              tab[m,n] = np.sum(datain[idxvin:idxvout,m,n])*chanwidth


    region_idx = np.where(r1==1)

#    rmsval = np.std(datain[idxvrmsin:idxvrmsout,region_idx[0],region_idx[1]])
#    print (rmsval)

#    print (peakval,idxpeak)

    for ix in range(len(veltab)):
        if r1 == [10,10]:
            meantab[ix] = np.nanmean(datain[ix,:,:])
        else:
            meantab[ix] = np.nanmean(datain[ix,region_idx[0],region_idx[1]])
    for ix in range(len(veltab)):
        sumtab[ix] = np.nansum(datain[ix,:,:])

    peakval_red = np.max(meantab[idxvlrs:idxvout])
    print (peakval_red)
    idxpeak_red = np.where(meantab==peakval_red)[0][0]
    print (idxpeak_red,veltab[idxpeak_red])

    for idxr in range(len(meantab[idxpeak_red:idxvout])):
        if meantab[idxpeak_red+idxr] < peakval_red*0.1:
            break
        tred = idxpeak_red+idxr

    peakval_blue = np.max(meantab[idxvin:idxvlrs])
    print (peakval_blue)
    idxpeak_blue = np.where(meantab==peakval_blue)[0][0]

    for idxb in range(len(meantab[idxvin:idxpeak_blue])):
        if meantab[idxpeak_blue-idxb] < peakval_blue*0.1:
            break
        tblue = idxpeak_blue-idxb


 #   thr1_idx = np.where(meantab[idxvin:idxvout]>peakval/10.0)
 #   thr1_tab = np.zeros(len(thr1_idx[0]))


 #   print (np.min(veltab[idxvin+thr1_idx[0]]),np.max(veltab[idxvin+thr1_idx[0]]))

    f, axarr = plt.subplots(1, 2, gridspec_kw = {'width_ratios':[2, 1]})
    f.set_figheight(5)
    f.set_figwidth(18)
    axarr[0].plot(veltab,meantab,drawstyle='steps-mid')

    axarr[0].plot(veltab[tblue],0,'*',color='red')
    axarr[0].plot(veltab[tred],0,'*',color='red')

    #axarr[0].plot([idxvin+idxpeak[0],idxvin+idxpeak[0]][-0.1,0.1],'--',color='red')

    axarr[0].plot(veltab,meantab,drawstyle='steps-mid')

    im = axarr[1].imshow(tab[:,:],origin='lower',cmap='jet')
    if r1 != [10,10]:
        axarr[1].contour(r1, origin='lower', color='black', vmin=0, vmax=1,levels=[0,1])
#        axarr[1].set_ylim(np.min(np.where(r1[:,:]>0)[0]),np.max(np.where(r1[:,:]>0)[0]))
#        axarr[1].set_xlim(np.min(np.where(r1[:,:]>0)[1]),np.max(np.where(r1[:,:]>0)[1]))

#axarr[1].plot([vin,vin],axarr[0].get_ylim(),'--')


    axarr[0].set_xlabel('Velocity [km s-1]',fontsize=20)
    axarr[0].set_ylabel('Mean Intensity [K]',fontsize=20)





    axarr[0].add_patch(
        patches.Rectangle(
        (vin,axarr[0].get_ylim()[0]),   # (x,y)
        vout-vin,          # width
        axarr[0].get_ylim()[1]-axarr[0].get_ylim()[0],          # height
        alpha=0.2
        )
    )

    f.colorbar(im, orientation='vertical')
    plt.show()
    plt.close()

    return    np.min(veltab[tblue]), np.max(veltab[tred])


# In[ ]:


def custom_moment_multi(vin,vout,datain,header,rms=0,rmsvel=[0,0],vlsr=8.5,r1=[10,10],rmsreg=[10,10]):

    size = np.shape(datain[0,:,0])[0]
    tab = np.zeros([size,size])
    nchan = np.shape(datain[:,0,0])[0]
    meantab = np.zeros([nchan,1])
    sumtab = np.zeros([nchan,1])

#####
    veltab, chanwidth, nchan, restfreq = SpecaxisHandler(datain,header,vlsr)



    idxvin = np.argmin(np.abs(veltab - (vin)))
    idxvout = np.argmin(np.abs(veltab - (vout)))
    idxvrmsin = np.argmin(np.abs(veltab - (rmsvel[0])))
    idxvrmsout = np.argmin(np.abs(veltab - (rmsvel[1])))

    idxres1 = np.argmin(np.abs(veltab - (-40.5)))
    idxres2 = np.argmin(np.abs(veltab - (40.5)))

    idxrms = []

    for m in range(size):
        for n in range(size):

            if np.isnan(datain[0,m,n]):
                continue
            if datain[0,m,n] == 0:
                continue
            tab[m,n] = np.sum(datain[idxvin:idxvout,m,n])*chanwidth
 #           else:
  #              tab[m,n] = np.sum(datain[idxvin:idxvout,m,n])*chanwidth


    region_idx = np.where(r1==1)


    for ix in range(len(veltab)):
        if r1 == [10,10]:
            meantab[ix] = np.nanmean(datain[ix,:,:])
        else:
            meantab[ix] = np.nanmean(datain[ix,region_idx[0],region_idx[1]])
    for ix in range(len(veltab)):
        sumtab[ix] = np.nansum(datain[ix,:,:])



    f, axarr = plt.subplots(1, 2, gridspec_kw = {'width_ratios':[2, 1]})
    f.set_figheight(5)
    f.set_figwidth(18)
    axarr[0].plot(veltab,meantab,drawstyle='steps-mid')
    im = axarr[1].imshow(tab[:,:],origin='lower',cmap='jet')
    if r1 != [10,10]:
        axarr[1].contour(r1, origin='lower', color='black', vmin=0, vmax=1,levels=[0,1])
#        axarr[1].set_ylim(np.min(np.where(r1[:,:]>0)[0]),np.max(np.where(r1[:,:]>0)[0]))
#        axarr[1].set_xlim(np.min(np.where(r1[:,:]>0)[1]),np.max(np.where(r1[:,:]>0)[1]))

#axarr[1].plot([vin,vin],axarr[0].get_ylim(),'--')



    axarr[0].set_xlabel('Velocity [km s-1]',fontsize=20)
    axarr[0].set_ylabel('Mean Intensity [K]',fontsize=20)



    axarr[0].add_patch(
        patches.Rectangle(
        (vin,axarr[0].get_ylim()[0]),   # (x,y)
        vout-vin,          # width
        axarr[0].get_ylim()[1]-axarr[0].get_ylim()[0],          # height
        alpha=0.2
        )
    )

    axarr[0].add_patch(
        patches.Rectangle(
        (rmsvel[0],axarr[0].get_ylim()[0]),   # (x,y)
        rmsvel[1]-rmsvel[0],          # width
        axarr[0].get_ylim()[1]-axarr[0].get_ylim()[0],          # height
        alpha=0.1,
        color='yellow'
        )
    )

    if rmsreg != [10,10]:
        rms_mom0 = np.nanstd(tab[rmsreg[0]:rmsreg[1],rmsreg[2]:rmsreg[3]])
        print ('RMS of the moment map: ', rms_mom0)

        axarr[1].add_patch(
        patches.Rectangle(
        (rmsreg[0],rmsreg[2]),   # (x,y)
        rmsreg[1]-rmsreg[0],          # width
        armsreg[3]-rmsreg[2],          # height
        alpha=0.2
        )
    )

    f.colorbar(im, orientation='vertical')
    plt.show()
    plt.close()


# In[11]:


def custom_moment_rmsproper_beta(vin,vout,datain,header,regname,rms=0,rmsvel=[0,0],vlsr=8.5,r1=[10,10]):

    size = np.shape(datain[0,:,0])[0]
    tab = np.empty([size,size])
    nchan = np.shape(datain[:,0,0])[0]
    meantab = np.zeros([nchan,1])
    sumtab = np.zeros([nchan,1])

#####
    veltab, chanwidth, nchan, restfreq = SpecaxisHandler(datain,header,vlsr)



    idxvin = np.argmin(np.abs(veltab - (vin)))
    idxvout = np.argmin(np.abs(veltab - (vout)))
    idxvrmsin = np.argmin(np.abs(veltab - (rmsvel[0])))
    idxvrmsout = np.argmin(np.abs(veltab - (rmsvel[1])))

    idxres1 = np.argmin(np.abs(veltab - (-40.5)))
    idxres2 = np.argmin(np.abs(veltab - (40.5)))

    print (rms, idxres1, idxres2)
    print (rms, idxvin, idxvout)

    idxrms = []
    for m in range(size):
        for n in range(size):

            if np.isnan(datain[0,m,n]):
                continue
            if datain[0,m,n] == 0:
                continue
            #applying 3sigma mask
            #applying 3sigma mask:
            if rms>0:
                print ('chuj')
                if regname == 'red': #if red
                    rmslowres = np.std(datain[idxvrmsin:idxres1,m,n])
                    rmshighres = np.std(datain[idxres1:idxvrmsout,m,n])
                    if idxvin < idxres2 and idxvout > idxres2:
                        idxrms1 = np.where(datain[idxvin:idxres2,m,n]>rms*rmshighres)
                        idxrms2 = np.where(datain[idxres2:idxvout,m,n]>rms*rmslowres)
                        idxrms = np.append(idxrms1,idxrms2)
                        print (idxrms)
                        print('japeridole')
                    else:
                        if idxvout < idxres2:
                            idxrms = np.where(datain[idxvin:idxvout,m,n]>rms*rmshighres)
                        if idxvout > idxres2:
                            idxrms = np.where(datain[idxvin:idxvout,m,n]>rms*rmslowres)
                        idxrms = idxrms[0]

                        print('gunwo')
                if regname == 'blue': #if red
                    rmshighres = np.std(datain[idxvrmsin:idxres2,m,n])
                    rmslowres = np.std(datain[idxres2:idxvrmsout,m,n])
                    print (rmshighres, rmslowres)
                    if idxvin < idxres1 and idxvout > idxres1:
                        idxrms1 = np.where(datain[idxvin:idxres1,m,n]>rms*rmslowres)
                        idxrms2 = np.where(datain[idxvin:idxres1,m,n]>rms*rmshighres)
                        idxrms=np.append(idxrms1,idxrms2)
                        print (idxrms)
                        print("imapickle")
                    else:
                        if idxvout < idxres1:
                            idxrms = np.where(datain[idxvin:idxvout,m,n]>rms*rmslowres)
                        if idxvout > idxres1:
                            idxrms = np.where(datain[idxvin:idxvout,m,n]>rms*rmshighres)
                        idxrms = idxrms[0]
                        print("yomama")

                if len(idxrms) > 1:
                    if regname == 'red':
                        maxvel[m,n] = veltab[idxvin+np.max(idxrms)]
                    else:
                        maxvel[m,n] = veltab[idxvin+np.min(idxrms)]
                elif len(idxrms) == 1:
                    if regname == 'red':
                        maxvel[m,n] = veltab[idxvin+idxrms]
                    else:
                        maxvel[m,n] = veltab[idxvin+idxrms]
                else:
                    maxvel[m,n] = 0
             #   idxrms = np.where(datain[idxvin:idxvout,m,n]>rms*rmsval)

    # this part is necessary for observations with different resolutions in one spectrum
    #            for k in range(len(datain[idxvin:idxvout,m,n])): #going through every channel in selected range
    #                if datain[idxvin+k,m,n] > rms*rmsval:
    #                    idxrms.append(k)
                tab[m,n] = np.sum(datain[idxvin:idxvout,m,n][idxrms])*chanwidth
 #           else:
  #              tab[m,n] = np.sum(datain[idxvin:idxvout,m,n])*chanwidth


    for ix in range(len(veltab)):
        if r1== [10,10]:
            meantab[ix] = np.nanmean(datain[ix,:,:])
        else:
            meantab[ix] = np.nanmean(datain[ix,:,:]*r1)
    for ix in range(len(veltab)):
        sumtab[ix] = np.nansum(datain[ix,:,:])

    f, axarr = plt.subplots(1, 2, gridspec_kw = {'width_ratios':[2, 1]})
    f.set_figheight(5)
    f.set_figwidth(18)
    axarr[0].plot(veltab,meantab,drawstyle='steps-mid')
    im = axarr[1].imshow(tab[:,:],origin='lower',cmap='jet')
    if r1[0,0] != 10:
        axarr[1].contour(r1, origin='lower', color='black', vmin=0, vmax=1)
        axarr[1].set_ylim(np.min(np.where(r1[:,:]>0)[0]),np.max(np.where(r1[:,:]>0)[0]))
        axarr[1].set_xlim(np.min(np.where(r1[:,:]>0)[1]),np.max(np.where(r1[:,:]>0)[1]))

#axarr[1].plot([vin,vin],axarr[0].get_ylim(),'--')





    axarr[0].add_patch(
        patches.Rectangle(
        (vin,axarr[0].get_ylim()[0]),   # (x,y)
        vout-vin,          # width
        axarr[0].get_ylim()[1]-axarr[0].get_ylim()[0],          # height
        alpha=0.2
        )
    )

    f.colorbar(im, orientation='vertical')
    plt.show()
    plt.close()


# In[3]:


def SpecaxisHandler(datain,header,vlsr):
    nchan =  header['NAXIS3']
    specaxistype = header['CTYPE3']
    if specaxistype == 'FREQ':
      reffreq = header['CRVAL3']
      restfreq = header['RESTFRQ']
      chanwidth_freq = header['CDELT3']
      refpix = header['CRPIX3']
      #print('Reference frequency: ', reffreq)
      #print('Rest frequency: ', restfreq)
      #print('Channel width: ', float(chanwidth_freq))
      #print('Reference velocity pixel: ', refpix) #from this we know where CRVAL3 refers to
      ## Velocity of the reference channel
      v0 = (restfreq-reffreq)/restfreq*const.c.to('km/s').value # [km/s]
      #print('Reference velocity: ', v0)
      # Velocity resolution
      chanwidth=chanwidth_freq/restfreq*const.c.to('km/s').value # [km/s]
      if float(chanwidth) < 0:
          chan_sign = 'minus'
          chanwidth=chanwidth*(-1.0)
      else:
          chan_sign = 'plus'

      #print (chan_sign)
      #because the channel width was negative we needed to multiply by -1.0
      #print('Velocity resolution: ', chanwidth)
      #this is a bit awry part, but should take into account the fact that header not neccesarily starts from
      #the first channel
      if v0 <= 0:
          v0 = v0-chanwidth*(refpix-1) # HUGE PROBLEM HERE
      else:
          v0 = v0+chanwidth*(refpix-1) # To account for refvelpix, if it's not the first channel!
      #print('Start velocity: ', v0)
    #this part was written specifically for IRS43 - JCMT observations, I am not sure if that will work properly for
    #other data with header "VELO-LSR"
    elif specaxistype == 'VELO-LSR':  # HAS NOT BEEN TESTED YET
      restfreq = header['RESTFREQ']
      startvel = header['CRVAL3']/1000.0
      chanwidth = header['CDELT3']/1000.0
      refvelpix = header['CRPIX3']
      reffreq = header['RESTFREQ']
      v0 = startvel - refvelpix* chanwidth
      if float(chanwidth) < 0:
          chan_sign = 'minus'
          chanwidth=chanwidth*(-1.0)
      else:
          chan_sign = 'plus'

    #print (chan_sign)
    print('---- test:specaxis ---')

    if vlsr > 0:
        delta_freq = np.abs(kms_to_ghz_scalar(vlsr,restfreq)*1e9-restfreq)
    else:
        delta_freq =kms_to_ghz_scalar(vlsr,restfreq)*1e9-restfreq
    print('---- test:specaxis ---', '---- delta_freq',delta_freq)
    print (vlsr)
#it ends by creating a full array of velocities, by simply using the first and last value
    if chan_sign == 'minus':
        velocity = np.arange(v0,v0+chanwidth*(nchan),chanwidth)
        velocity = velocity[0:nchan]-vlsr #shifting all velocities with respect to the vlsr
        frequency =  np.arange(reffreq,reffreq+chanwidth_freq*(nchan),(reffreq+chanwidth_freq*(nchan)-reffreq)/nchan)
        if vlsr < 0:
            frequency = frequency-delta_freq
        else:
            frequency = frequency+delta_freq
    else:
        #print (v0,v0-chanwidth*(nchan),chanwidth)
        velocity = np.arange(v0,v0-chanwidth*(nchan),-chanwidth)
        velocity = velocity[0:nchan]-vlsr #shifting all velocities with respect to the vlsr
        frequency =  np.arange(reffreq,reffreq+chanwidth_freq*(nchan),(reffreq+chanwidth_freq*(nchan)-reffreq)/nchan)
        if vlsr < 0:
            frequency = frequency-delta_freq
        else:
            frequency = frequency+delta_freq

    #print (reffreq, reffreq+chanwidth_freq*(nchan))
    #print (frequency[0],frequency[-1])
   # print (nchan)
    #print (np.shape(velocity),np.shape(frequency))
    return velocity, frequency, chanwidth, nchan, restfreq


# In[7]:


def BeamHandler(beam_table, header, beam_type = 'min'):
    #So beam area in arcsec is as it says, a beam area in arcseconds
#Pixval arcsec is taken from the header the dimension of pixel in arcsec
#altogheter you get a beam area in pixels


    nchan = header['NAXIS3']
    pixval_x = header['CDELT1']
    pixval_arcsec =  (pixval_x)*3600.0
    print ('Pixel value: ', pixval_x, ' deg')
    print ('Pixel size in arcsec: ', pixval_arcsec)


#ok, this is elaborate way to deal with single and multiple beams
#ends up with creating a table with area of the beam. It is the same dimension
#as the velocity table so can be easily compared

    if type(beam_table[0]) is np.ndarray:
       beam_area_arcsec = beam_table[:,0]*beam_table[:,1]*1.1331
       beam_area_pix = beam_area_arcsec/(pixval_arcsec**2)
#   print ('Beam area in pix: ', beam_area_pix)
    else:
       beam_area_arcsec = beam_table[0]*beam_table[1]*1.1331
       beam_area_pix = beam_area_arcsec/(pixval_arcsec**2)
       beam_area_pix = np.full(nchan, beam_area_pix) #broadcast the array to the size of velocity dimension
#   print ('Beam area in pix: ', beam_area_pix)

#key part: we divide every value in the dataset by the beam area in pixel
#but as every channel can have a different beam area we divide going through every channel, for all the pixels.

    beam_area_tab = beam_area_pix[:,None,None]
    if type(beam_table[0]) is np.ndarray:
        if beam_type == 'min':
            beam_a = np.min(beam_table[:,0])
            beam_b = np.min(beam_table[:,1])
            beam_pa = np.min(beam_table[:,2])
        if beam_type == 'max':
            beam_a = np.max(beam_table[:,0])
            beam_b = np.max(beam_table[:,1])
            beam_pa = np.max(beam_table[:,2])
    else:
        beam_a = beam_table[0]
        beam_b = beam_table[1]
        beam_pa = beam_table[2]
    return pixval_x, np.abs(pixval_arcsec), beam_area_tab, beam_a, beam_b, beam_pa


def draw_patch(ax, in_v, out_v, color='r', alpha=0.2):
    ax.add_patch(
        patches.Rectangle(
            (in_v, ax.get_ylim()[0]),  # (x,y)
            out_v - in_v,  # width
            ax.get_ylim()[1] - ax.get_ylim()[0],  # height
            alpha=alpha,
            color=color
        )
    )


def HIFI_resolution(freq):
    # eats freq in hz, throws out beam fwhm
    lam = const.c.value / freq  # meters
    lam_cm = lam * 1e2
    D = 3.5 * 100  # m to cm #herschel mirror
    res_rad = 1.22 * lam_cm / D
    res_arcsec = (res_rad * 180.0 / np.pi) * 3600.0
    return res_arcsec


def cmask(index, radius, array):
        a, b = index
        nx, ny = array.shape
        x, y = np.ogrid[-a:ny - a, -b:nx - b]
        mask = x * x + y * y <= radius * radius
        return mask

def gauss2d(x, y, amp=1, x_cen=0, y_cen=0, x_sigma=0.4, y_sigma=0.4):
        z = amp * np.exp(-((x - x_cen) ** 2 / (2 * x_sigma ** 2) + (y - y_cen) ** 2 / (2 * y_sigma ** 2)))
        return z

def moment_map(source,mol,vin,vout,datain,header,regname,regcolor,rms=1,rmsvel=[0,0],vlsr=8.5,\
               plot=False,r1=[10,10],sigma_cutoff=True,rms_flux=1,rmsmask=True,\
               mom1mask = True, mom1cutoff=3,r1_noise='noise',noise_cutoff=1,noise_mom1cut=False,velreg ='',for_func1=False):
    '''
    :param source:
    :param mol:
    :param vin:
    :param vout:
    :param datain:
    :param header:
    :param regname: region red/blue
    :param rms: set how many x sigma should be the rms in the spectra
    :param rmsvel: array with velocities from which we calcuate the rms
    :param vlsr:
    :param plot:
    :param r1:
    :param sigma_cutoff:
    :param rms_flux: how many sigma for moment0 mask
    :param rmsmask: if True put a mask on mom0 calculation
    :param mom1mask:
    :param mom1cutoff:
    :param r1_noise:
    :param noise_cutoff:
    :param noise_mom1cut:
    :return:
    '''


    print ('|---- moment map script -----|')
    print ('|---- %s --- %s --- %s --- %f;%f----|'%(source,mol,regname,vin,vout))

    # ========= json file with source prop=============================================================================#

    with open('json/sources.json', 'r') as f:
        sources = json.load(f)

    #reading source vlsr
    vlsr = sources[source]['vlsr']


    # ========= definining empty tables ===============================================================================#

    plot_vmax = 0
    size = np.shape(datain[0,:,0])[0]
    mom0 = np.zeros([size,size])
    mom1 = np.zeros([size,size])
    mom0_nomask = np.zeros([size,size])

    maxvel = np.zeros([size,size])
    rmstab = np.zeros([size,size])
    dettab = np.zeros([size,size])



    nchan = np.shape(datain[:,0,0])[0]

    #???
    #meantab = np.zeros([nchan,1])
    #sumtab = np.zeros([nchan,1])

    #sigma_cutoff - using 1sigma cut-off instead of the given vout
    #still need to decide if there should be an outer limit then?

    # ========= spectral axis info ====================================================================================#
    veltab, chanwidth, nchan, restfreq = SpecaxisHandler(datain,header,vlsr)


    #indices of the given velocity limits

    idxvin = np.argmin(np.abs(veltab - (vin)))
    idxvout = np.argmin(np.abs(veltab - (vout)))
    idxvrmsin = np.argmin(np.abs(veltab - (rmsvel[0])))
    idxvrmsout = np.argmin(np.abs(veltab - (rmsvel[1])))

    #print ('Velocity in:', vin)
    #print ('Velocity out:', vout)
    #print ('Index in:', idxvin)
    #print ('Index out:', idxvout)

    if rmsmask:
        print(' |--- Using rmsmask to calculate moment 0 with rms cutoff: %.1f sigma ---|'%(rms_flux))
    if mom1mask:
        print('|--- Calculating mom1 value only for %5.2f sigma cutoff ---|'%(mom1cutoff))

    # ========= values for change in resolution! ======================================================================#








    ####################################################################################################################
    #IMPORTING FLUX FILES
    prefix = '../../Projects/'
    if mol == 'HCN':
        fluximage = prefix + sources[source]['field']+'/'+sources[source]['field']+'_B3_flux.fits' ###### field + mol
    else:
        fluximage = prefix + sources[source]['field']+'/'+sources[source]['field']+'_B6_flux.fits' ###### field + mol
    fluximage = pf.getdata(fluximage)
    if len(np.shape(fluximage)) == 4:
        fluximage = fluximage[0,0,:,:]

    #####################
    # finding the maximum sensitivity pixel
    maxflux = np.where(fluximage == np.nanmax(fluximage))
    print ('---- peak primary beam ',np.nanmax(fluximage))
    print ('---- peak primary beam sensitivity at pixel',maxflux)
    print ('---- size primary beam file',np.shape(fluximage))



    #flux_m = maxflux[0]
    #flux_n = maxflux[1]
    #experimental




    f = plt.figure()
    ax1 = f.add_subplot('111')
    x = np.linspace(0, size, size)
    pixval_x = header['CDELT1']
    pixval_arcsec = np.abs((pixval_x) * 3600.0)

    if mol != 'HCN':
        lam = 0.001300448783239
    else:
        lam = 0.003
    D = 12.0
    x_fwhm = 1.28 * (lam / D * 180.0 / np.pi * 3600.0) / pixval_arcsec
    y_fwhm = 1.28 * (lam / D * 180.0 / np.pi * 3600.0) / pixval_arcsec
    y = np.linspace(0, size, size)
    x, y = np.meshgrid(x, y)
    # print 'gauss value at center', (gauss2d(224,224,x_cen=224,y_cen=224,x_sigma=100.0/2.35482,y_sigma=100.0/2.35482))
    # print 'gauss value at 100,100', (gauss2d(100,100,x_cen=224,y_cen=224,x_sigma=100.0/2.35482,y_sigma=100.0/2.35482))
    im = ax1.imshow( gauss2d(x, y, x_cen=size / 2.0, y_cen=size / 2.0, x_sigma=x_fwhm / 2.35482, y_sigma=y_fwhm / 2.35482),
        cmap='viridis',vmin=0,vmax=1)
    cbar = f.colorbar(im, orientation='vertical')
    plt.savefig(source+'_'+mol+'_gauss.png')
    plt.close()


    flux_m = 800
    flux_n = 600

    if source == 'Emb8N' or source == 'S68N':
        if mol == 'CO':
            smooth = False
            velrms_var1 = [30, 80]
            flux_m = 800
            flux_n = 600
        if mol == 'SiO' :
            smooth = True
            velrms_var1 = [-38, -10]
            velrms_var2 = [-60, -42]
            flux_m = 800
            flux_n = 600
        if mol == 'H2CO':
            smooth = True
            velrms_var1 = [10, 35]
            velrms_var2 = [-38, -27]
            flux_m = 800
            flux_n = 600
        if mol == 'HCN':
            smooth = False
            velrms_var1 = [10,50]
            flux_m = 1350
            flux_n = 1350
    if source == 'SMM1a' or source == 'SMM1b' or source == 'SMM1d':
        if mol == 'CO':
            smooth = False
            velrms_var1 = [-70, -20]
            flux_m = 800
            flux_n = 600
        if mol == 'SiO':
            smooth = True
            velrms_var1 = [-30, -10]
            velrms_var2 = [-60, -42]
            flux_m = 800
            flux_n = 600
        if mol == 'H2CO':
            smooth = True
            velrms_var1 = [-25, -10]
            velrms_var2 = [-38, -27]
            flux_m = 800
            flux_n = 600

        if mol == 'HCN':
            smooth = False
            velrms_var1 = [10,35]
            flux_m = 1200
            flux_n = 1400

    ####plotting primary beam sensitivity


    idxvrms_var1_in = np.argmin(np.abs(veltab - (velrms_var1[0])))
    idxvrms_var1_out = np.argmin(np.abs(veltab - (velrms_var1[1])))


    rms_circle = cmask([flux_m,flux_n], 20, datain[0,:,:])
    idx_rms_circle = np.where(rms_circle)

    #rmscenter_pix_beam = np.nanstd(datain[idxvrms_var1_in:idxvrms_var1_out,idx_rms_circle[0],idx_rms_circle[1]])
    rmscenter_pix_beam = np.nanstd(datain[idxvrms_var1_in:idxvrms_var1_out,idx_rms_circle[0],idx_rms_circle[1]])
    rmscenter_pix_beam = rmscenter_pix_beam*fluximage[flux_m,flux_n]

    if mol == 'HCN':
        if source == 'S68N' or source == 'Emb8N':
            ######## problem!?!??!!?!
            #print ('getting gauss approx instead of the fluxfile!')
            #fluximage = gauss2d(x, y, x_cen=size / 2.0, y_cen=size / 2.0, x_sigma=x_fwhm / 2.35482, y_sigma=y_fwhm / 2.35482)
            rmscenter_pix_beam = rmscenter_pix_beam* gauss2d(flux_m,flux_n, x_cen=size / 2.0, y_cen=size / 2.0, x_sigma=x_fwhm / 2.35482, y_sigma=y_fwhm / 2.35482)


    meantab_rms = np.zeros(np.shape(veltab))
    for i in range(np.shape(veltab)[0]):
        meantab_rms[i] = np.nanmean(datain[i,idx_rms_circle[0],idx_rms_circle[1]])

    if smooth:
        idxvrms_var2_in = np.argmin(np.abs(veltab - (velrms_var2[0])))
        idxvrms_var2_out = np.argmin(np.abs(veltab - (velrms_var2[1])))



    rms_circle = cmask([flux_m, flux_n], 5, datain[0, :, :])
    idx_rms_circle = np.where(rms_circle)


    if smooth:
        rmscenter_pix_beam_lowres = np.nanstd(datain[idxvrms_var2_in:idxvrms_var2_out, idx_rms_circle[0], idx_rms_circle[1]])
        rmscenter_pix_beam_lowres = rmscenter_pix_beam_lowres/fluximage[flux_m,flux_n]
        rmscenter_pix_beam_lowres = rmscenter_pix_beam_lowres


        meantab_rms_lowres = np.zeros(np.shape(veltab))
        for i in range(np.shape(veltab)[0]):
            meantab_rms_lowres[i] = np.nanmean(datain[i, idx_rms_circle[0], idx_rms_circle[1]])

    f = plt.figure()
    ax1 = f.add_subplot('111')
    ax1.text(-20,5,rmscenter_pix_beam)
    ax1.plot(veltab,meantab_rms)
    plt.savefig(source+'_'+mol+'_rms_spectra.png')
    plt.close()

    print ('rms within the circle of radius 20pix.....', rmscenter_pix_beam)
    if smooth:
        print ('rms lowres.....', rmscenter_pix_beam_lowres)





    circle_mask_tab = np.zeros(np.shape(datain[0,:,:]))
    circle_mask_tab[idx_rms_circle] = 1


    f = plt.figure()
    ax1 = f.add_subplot('111')
    im = ax1.imshow(fluximage,origin='lower',vmin=0,vmax=1)
    ax1.plot(flux_n,flux_m,'+',markersize=10)
    cbar = f.colorbar(im, orientation='vertical')
    #ax1.contour(circle_mask_tab, color='black', vmin=0, vmax=1, levels=[0, 1])
    #ax1.set_ylim(flux_n-20,flux_n+20)
    #ax1.set_xlim(flux_m-20,flux_m+20)

    plt.savefig(source+'_'+mol+'_rms_flux.png')
    plt.close()



    ############################################





    if smooth:
       if mol == 'SiO':
                vresred = 40.5
                vresblue = -40.5
       elif mol == 'H2CO':
                vresred = 36.0
                vresblue = -26
       else:
                vresred = 200
                vresblue = -200
    else:
        vresred = 200
        vresblue = -200

    idxvresred = np.argmin(np.abs(veltab - (vresred)))
    idxvresblue = np.argmin(np.abs(veltab - (vresblue)))

    # ========= loop through pixels ===================================================================================#
    #if 'fitsfiles/mom0_' + source + '_' + mol + '_' + str(vin) + '_' + str(vout) + '_rms' + str(
    #        rms_flux) + '.fits' not in glob.glob('fitsfiles/*.fits'):

    for m in range(size):
        for n in range(size):
            if np.isnan(datain[0,m,n]):
                continue    #if nan - skip | checking for first velocity channel
            #if datain[0,m,n] == 0:
            #    continue   #if begins with 0 skip | not sure if necessary?
            if r1[m,n]== 0:
                continue    #if outside region skip | this significantly reduces time

            mom0_nomask[m, n] = np.sum(datain[idxvin:idxvout, m, n]) * chanwidth

            # ========= looking for maxval  =======================================================================#
            if rms > 0: #only if rms threshold set

                #print ('rmsval:', rmscenter_pix_beam)
                #print ('pbcorrection:', fluximage[m,n])

                fluxscaling = fluximage[m,n]
                #rmsval = np.std(datain[idxvrmsin:idxvrmsout, m, n])  # calcuating rms for given spectra
                if mol == 'HCN':
                    if source == 'S68N' or source == 'Emb8N':
                        #becuase I fucked up and got different dimension of flux file
                        #luckily I know the dimension difference 0.000011/0.000027 so we can scale each pixel
                        fluxscaling = gauss2d(flux_m,flux_n, x_cen=size / 2.0, y_cen=size / 2.0, x_sigma=x_fwhm / 2.35482, y_sigma=y_fwhm / 2.35482)


                rmsval = rmscenter_pix_beam/fluxscaling #rms from center and scaled with .flux
                #print ('rmsval after pb correction:', rmsval)
                rmstab[m, n] = rmsval  # creating rmstab


                #old method
                #rms_scaling = np.sqrt(7.21 * 8)  # ratio of beam area
                if smooth:
                    rmsval_lowres = rmscenter_pix_beam_lowres/fluxscaling #rms from center and scaled with .flux


                if regcolor == 'blue':
                    if vin < vresblue:
                        idxrms_a = np.array([[],[]])
                        idxrms_b = np.array([[],[]])
                        idxrms_c = np.array([[],[]])
                        #idxrms_a = np.where(datain[idxvin:idxvresblue, m, n] > rms * rmsval/rms_scaling)
                        idxrms_a = np.where(datain[idxvin:idxvresblue, m, n] > rms * rmsval_lowres)
                        veltab_a = veltab[idxvin:idxvresblue][idxrms_a]
                        datain_a = datain[idxvin:idxvresblue, m, n][idxrms_a]
                        idxrms_b = np.where(datain[idxvresblue:idxvout, m, n] > rms * rmsval)
                        veltab_b = veltab[idxvresblue:idxvout][idxrms_b]
                        datain_b = datain[idxvresblue:idxvout, m, n][idxrms_b]
                        veltab_c = np.append(veltab_a,veltab_b)
                        datain_c = np.append(datain_a,datain_b)
                    else:
                        idxrms_a = np.array([[],[]])
                        idxrms_b = np.array([[],[]])
                        idxrms_c = np.array([[],[]])
                        idxrms_c = np.where(datain[idxvin:idxvout,m,n]>rms*rmsval) #selecting indices above threshold
                        veltab_c = veltab[idxvin:idxvout][idxrms_c]
                        datain_c = datain[idxvin:idxvout,m,n][idxrms_c]

                if regcolor == 'red':
                    if vout > vresred:
                        idxrms_a = np.array([[], []])
                        idxrms_b = np.array([[], []])
                        idxrms_c = np.array([[], []])
                        #idxrms_a = np.where(datain[idxvin:idxvresred, m, n] > rms * rmsval / rms_scaling)
                        idxrms_a = np.where(datain[idxvin:idxvresred, m, n] > rms * rmsval)
                        veltab_a = veltab[idxvin:idxvresred][idxrms_a]
                        datain_a = datain[idxvin:idxvresred, m, n][idxrms_a]
                        idxrms_b = np.where(datain[idxvresred:idxvout, m, n] > rms * rmsval_lowres)
                        veltab_b = veltab[idxvresred:idxvout][idxrms_b]
                        datain_b = datain[idxvresred:idxvout, m, n][idxrms_b]
                        veltab_c = np.append(veltab_a, veltab_b)
                        datain_c = np.append(datain_a, datain_b)
                    else:
                        idxrms_a = np.array([[], []])
                        idxrms_b = np.array([[], []])
                        idxrms_c = np.array([[], []])
                        idxrms_c = np.where(datain[idxvin:idxvout, m, n] > rms * rmsval)  # selecting indices above
                        veltab_c = veltab[idxvin:idxvout][idxrms_c]
                        datain_c = datain[idxvin:idxvout, m, n][idxrms_c]

                if idxrms_c[0].size > 0: #checking if any channel is above threshold
                    if m == 800 and n == 800: print('c >0')

                    if regcolor == 'red':
                        #so when not splitted into two resolutions just go quickly here
                        #otherwise YOU WANT to enter the idxrms_c.size==0 condition
                        maxvel_index = idxvin + np.max(idxrms_c) #setting index where maxvel was
                        maxvel[m,n] = veltab[maxvel_index]  #value of max velocity index
                    elif regcolor == 'blue':
                        maxvel_index = idxvin+np.min(idxrms_c)
                        maxvel[m,n] = veltab[maxvel_index]
                elif idxrms_a[0].size > 0 and idxrms_b[0].size > 0:  #if no pixel above set maxvel as 0
                    if m == 800 and n == 800: print('both >0')

                    if regcolor == 'blue':
                        if np.min(veltab[idxvin+idxrms_a]) < np.min(veltab[idxvresblue+idxrms_b]):
                            maxvel_index = idxvin+np.min(idxrms_a)
                            maxvel[m,n] = veltab[maxvel_index]
                        else:
                            maxvel_index = idxvresblue+np.min(idxrms_b)
                            maxvel[m,n] = veltab[maxvel_index]
                    if regcolor == 'red':
                        if m == 800 and n == 800: print('a: ', np.max(veltab[idxvin+idxrms_a]),'b: ', np.max(veltab[idxvresred+idxrms_b]))

                        if np.max(veltab[idxvin+idxrms_a]) > np.max(veltab[idxvresred+idxrms_b]):
                            if m == 800 and n == 800: print('a > b')
                            maxvel_index = idxvin + np.max(idxrms_a)  # setting index where maxvel was
                            maxvel[m, n] = veltab[maxvel_index]  # value of max velocity index
                        else:
                            if m == 800 and n == 800: print('b > a')
                            maxvel_index = idxvresred + np.max(idxrms_b)  # setting index where maxvel was
                            maxvel[m, n] = veltab[maxvel_index]  # value of max velocity index


                elif idxrms_a[0].size > 0 and idxrms_b[0].size <= 0:  # if no pixel above set maxvel as 0
                    if m == 800 and n == 800: print('a >0')

                    if regcolor == 'blue':
                        maxvel_index = idxvin + np.min(idxrms_a)
                        maxvel[m, n] = veltab[maxvel_index]
                    if regcolor == 'red':
                        maxvel_index = idxvin + np.max(idxrms_a)  # setting index where maxvel was
                        maxvel[m, n] = veltab[maxvel_index]  # value of max velocity index

                elif idxrms_b[0].size > 0 and idxrms_a[0].size <= 0:  # if no pixel above set maxvel as 0
                    if m == 800 and n == 800: print('b >0')

                    if regcolor == 'blue':
                        maxvel_index = idxvresblue + np.min(idxrms_b)
                        maxvel[m, n] = veltab[maxvel_index]
                    if regcolor == 'red':
                        maxvel_index = idxvresred + np.max(idxrms_b)  # setting index where maxvel was
                        maxvel[m, n] = veltab[maxvel_index]  # value of max velocity index
                else:
                    maxvel[m,n] = 0
                    maxvel_index = -99


                if sigma_cutoff:
                    if regcolor == 'red':
                        idxvout_new = maxvel_index
                        idxvin_new = idxvin
                    if regcolor == 'blue':
                        idxvin_new = maxvel_index
                        idxvout_new = idxvout
                else:
                        idxvout_new = idxvout
                        idxvin_new = idxvin

                plotvmax_check = False
                #plotting to check if there are any issues with vmax and rms


#                if source =='Emb8N':
#                   if regname == 'red':
#                        if m == 950 and n ==700:
#                            plotvmax_check = True
#                    if regname == 'blue':
#                        if m == 840 and n ==450:
#                            plotvmax_check = True


                #if plot_vmax < 1:
                if plotvmax_check:

                    print (rmsval)
                    print ('---- vresred ---', vresred)




                    f_vmax = plt.figure(1)
                    ax_vmax = f_vmax.add_subplot('111')
                    ax_vmax.plot(veltab,datain[:,m,n], drawstyle='steps-mid', color='black', linewidth=0.7)
                    draw_patch(ax_vmax,vin,vout)
                    draw_patch(ax_vmax,rmsvel[0],rmsvel[1],color='yellow')
                    ax_vmax.plot([ax_vmax.get_xlim()[0],ax_vmax.get_xlim()[1]],[rms*rmsval,rms*rmsval],'--',color='red', linewidth=0.5)
                    if smooth:
                        ax_vmax.plot([ax_vmax.get_xlim()[0],ax_vmax.get_xlim()[1]],[rms*rmsval_lowres,rms*rmsval_lowres],'--',color='red', linewidth=0.5)

                    ax_vmax.text(ax_vmax.get_xlim()[0]-10.0,rms*rmsval,'%.2f'%(rms*rmsval))
                    ax_vmax.plot([veltab[idxvin_new],veltab[idxvin_new]],[ax_vmax.get_ylim()[0],ax_vmax.get_ylim()[1]],'--',color='red', linewidth=0.5)
                    ax_vmax.plot([veltab[idxvout_new],veltab[idxvout_new]],[ax_vmax.get_ylim()[0],ax_vmax.get_ylim()[1]],'--',color='red', linewidth=0.5)

                    ax_vmax.plot(veltab_c,datain_c,'x',markersize=5)
                    ax_vmax.set_title(str(m)+';'+str(n))
                    plt.savefig('plots/'+source+'_vmax_'+'_'+regname+''+mol+'_'+str(vin)+'_'+str(vout)+'.png')
                    #plt.show()
                    plt.close()
                    plot_vmax += 1 #just plot one per region

            # ========= calculating moment map=======================================================================#
            ########################## THAT's WHAT SHE SAID! #########################################################

            if rmsmask: # if mask applied

                if regcolor == 'blue':
                    if vin < vresblue:
                        #idxrms_a = np.where(datain[idxvin_new:idxvresblue, m, n] > rms * rmsval/rms_scaling)
                        idxrms_a = np.where(datain[idxvin_new:idxvresblue, m, n] > rms_flux * rmsval_lowres)
                        mom0[m, n] = np.sum(datain[idxvin_new:idxvresblue, m, n][idxrms_a]) * chanwidth
                        idxrms_b = np.where(datain[idxvresblue:idxvout_new, m, n] > rms_flux * rmsval)
                        mom0[m, n] = mom0[m,n] + np.sum(datain[idxvresblue:idxvout_new, m, n][idxrms_b]) * chanwidth
                    else:
                        idxrms_c = np.where(datain[idxvin_new:idxvout_new,m,n] > rms_flux * rmsval)
                        mom0[m,n] = np.sum(datain[idxvin_new:idxvout_new,m,n][idxrms_c])*chanwidth

                if regcolor == 'red':
                    if m == 800 and n == 800: print('velin_new ', veltab[idxvin_new])
                    if m == 800 and n == 800: print('velout_new ', veltab[idxvout_new])

                    if vout > vresred:
                        #idxrms_a = np.where(datain[idxvin_new:idxvresred, m, n] > rms * rmsval/rms_scaling)
                        idxrms_a = np.where(datain[idxvin_new:idxvresred, m, n] > rms_flux * rmsval)
                        mom0[m, n] = np.sum(datain[idxvin_new:idxvresred, m, n][idxrms_a]) * chanwidth
                        if m == 800 and n == 800: print ('idx_a ',mom0[m,n])
                        idxrms_b = np.where(datain[idxvresred:idxvout_new, m, n] > rms_flux * rmsval_lowres)
                        mom0[m, n] = mom0[m,n] + np.sum(datain[idxvresred:idxvout_new, m, n][idxrms_b]) * chanwidth
                        if m == 800 and n == 800: print ('idx_b ',mom0[m,n])
                    else:
                        idxrms_c = np.where(datain[idxvin_new:idxvout_new,m,n] > rms_flux * rmsval)
                        mom0[m,n] = np.sum(datain[idxvin_new:idxvout_new,m,n][idxrms_c])*chanwidth
                        if m == 800 and n == 800: print('outta here', mom0[m, n])

                #idxrms_flux = np.where(datain[idxvin_new:idxvout_new,m,n]>rms_flux*rmsval)
                #mom0[m,n] = np.sum(datain[idxvin_new:idxvout_new,m,n][idxrms_flux])*chanwidth

                if mom0[m,n]>0:
                    dettab[m,n]=1


            if mom1mask:
                mom0rms = rmsval*chanwidth*10*mom1cutoff
            else:
                mom0rms = 0.0


##### calculate moment map without rms mask for mom1 subtraction

            for k in range(len(datain[idxvin:idxvout, m, n])):
                mom1[m, n] = datain[idxvin + k, m, n] * veltab[idxvin + k] * chanwidth + mom1[m, n]
            mom1[m, n] = mom1[m, n] / mom0_nomask[m, n]
    pf.writeto('fitsfiles/mom0_' + source + '_' + mol + '_' + str(vin) + '_' + str(vout) + '_rms' + str(
            rms_flux) + '.fits', mom0, clobber=True)
    mom0 = pf.getdata(
          'fitsfiles/mom0_' + source + '_' + mol + '_' + str(vin) + '_' + str(vout) + '_rms' + str(rms_flux) + '.fits')

      #####







    noise_idx = np.where(r1_noise==1)
    momrms = np.std(mom0[noise_idx])
    mom0masktab = np.zeros([size,size])
    mommaskidx = np.where(mom0>momrms*noise_cutoff)
    mom0masktab[mommaskidx]=1

#    if noise_mom1cut:
#        mom1 = mom1*mom0masktab



    region_idx = np.where(r1 == 1)
    if regcolor == 'red':
        vmax_max = (np.max((maxvel * r1)[region_idx]))
    else:
        vmax_max = (np.min((maxvel * r1)[region_idx]))

    if regcolor == 'red':
        avlim = np.where((maxvel * r1)[region_idx] > 0)
    if regcolor == 'blue':
        avlim = np.where((maxvel * r1)[region_idx] < 0)

    print('regular mean velocity', np.nanmean((maxvel * r1)[region_idx]))
    print("mean velocity with masked min values", np.nanmean((maxvel * r1)[region_idx][avlim]))

    #befor central rms:
    #rms_final = np.nanmean((rmstab * r1)[region_idx])

    rms_final = rmscenter_pix_beam
    if smooth:
        rms_final_lowres = rmscenter_pix_beam_lowres


    if mol == 'SiO':
        if regcolor == 'red':
            if vout > 40.5:
                #rms_final = rms_final/rms_scaling
                rms_final = rms_final_lowres
        elif regcolor == 'blue':
            if vin < -40.5:
                #rms_final = rms_final/rms_scaling
                rms_final = rms_final_lowres
    if mol == 'H2CO':
        if regcolor == 'blue':
            if vin < -26:
                #rms_final = rms_final / rms_scaling
                rms_final = rms_final_lowres

    mask_for_abundance=True

    mom0_mean_SiOmask = 0
    mom0_mean_H2COmask = 0
    mom0_mean_HCNmask = 0
    mom0_sum_SiOmask = 0
    mom0_sum_H2COmask = 0
    mom0_sum_HCNmask = 0

    if mask_for_abundance:
        abu_mask = pf.getdata('fitsfiles/ntot_' + source + '_' + 'SiO' + '_' + regname + '_' + velreg+'.fits')
        abu_idx = np.where(abu_mask > 0)
        mom0_sum_SiOmask = np.nansum((mom0*r1)[abu_idx])
        mom0_mean_SiOmask = np.nanmean(((mom0*r1)[abu_idx]))

        abu_mask = pf.getdata('fitsfiles/ntot_' + source + '_' + 'H2CO' + '_' + regname + '_' + velreg+'.fits')
        abu_idx = np.where(abu_mask > 0)
        mom0_sum_H2COmask = np.nansum((mom0*r1)[abu_idx])
        mom0_mean_H2COmask = np.nanmean(((mom0*r1)[abu_idx]))

        w = WCS(prefix + sources[source]['field'] + '/cont_B3/' + sources[source]['field'] + '_' + 'cont' + '.fits')
        w = w.dropaxis(3)  # Remove the Stokes axis
        w = w.dropaxis(2)  # and spectral axis
        w_co = WCS(prefix + sources[source]['field'] + '/' + sources[source]['field'] + '_' + 'CO' + '.fits')
        w_co = w_co.dropaxis(3)  # Remove the Stokes axis
        w_co = w_co.dropaxis(2)  # and spectral axis
        abu_mask = pf.getdata('fitsfiles/ntot_' + source + '_' + 'HCN' + '_' + regname + '_' + velreg+'.fits')
        abu_idx = np.where(abu_mask > 0)
        list_x = np.empty(0)
        list_y = np.empty(0)
        mom0hcn = np.empty(0)

        for m in range(np.shape(abu_mask[0,:])[0]):
            for n in range(np.shape(abu_mask[0,:])[0]):
               if abu_mask[m,n] > 0 :
                   ra,dec =  w.all_pix2world(m, n, 0)
                   co_pix_x, co_pix_y = w_co.all_world2pix(ra, dec, 0)
                   if (int(co_pix_x) in list_x) and (int(co_pix_y) in list_y):
                        continue
                   list_x = np.append(list_x,int(co_pix_x))
                   list_y = np.append(list_y,int(co_pix_y))
                   mom0hcn = np.append(mom0hcn,mom0[int(co_pix_x),int(co_pix_y)])

        mom0_mean_HCNmask = np.nanmean((mom0hcn))



    mom0_sum = np.nansum((mom0*r1)[region_idx])
    mom0_mean =  np.nanmean((mom0*r1)[region_idx])

    print ('mom0 sum', mom0_sum)

    dettab_idx  = np.where(dettab==1)
    mom0_mean_oversigma = np.nanmean(((mom0*r1)[dettab_idx]))
    if np.isnan(mom0_mean_oversigma): mom0_mean_oversigma = 0.0
    if np.isnan(mom0_mean): mom0_mean = 0.0
    if np.isnan(mom0_sum): mom0_sum = 0.0

    print ('mom0 sum', mom0_mean_oversigma)



    maxvel_mean = np.nanmean((maxvel*r1)[region_idx][avlim])
    vmax_mean = np.nanmean((rmstab * r1)[region_idx])

    print ('|--- RMS min: %.2f ---|'%(np.nanmin(rmstab)))
    print ('|--- RMS aver: %.2f ---|'%(np.nanmean(rmstab)))
    print ('|--- RMS  aver region: %.2f ---|'%(np.nanmean((rmstab * r1)[region_idx])))
    print ('|--- RMS  min region: %.2f ---|'%(np.nanmin((rmstab * r1)[region_idx])))

    print ('|--- mom0_mean: %.2f ---|'%(mom0_mean))
    print ('|--- mom0_mean_oversigma: %.2f ---|'%(mom0_mean_oversigma))
    print ('|--- mom0_sum: %.2f ---|'%(mom0_sum))



    mom0_mean_final = mom0_mean_oversigma
#### mean of only 3sigma pixels should be taken to the mass_main or M7 scripts


################################### BEARS, BEETS, BATTLESTAR GALLACTICA! ##############################################
    # here we chceck if the measured mean value is above 3sigma

    if mom0_mean_oversigma < 3 * rms_final:
        flag_mom0 = 'uplim'
        mom0_sum = 3 * rms_final
        mom0_mean_final = 3 * rms_final
    else:
        flag_mom0 = 'det'

    if vin == vout:
        flag_mom0 = 'uplim'
        mom0_sum = 3 * rms_final
        mom0_mean_final = 3 * rms_final


    #Forced uplim for f***g annoying random pixels in some fields:
    if source == 'SMM1b' and velreg == 'EHV' and mol == 'H2CO':
        flag_mom0 = 'uplim'
        mom0_sum = 3 * rms_final
        mom0_mean_final = 3 * rms_final
    if source == 'SMM1a' and velreg == 'fast' and mol == 'SiO' and regname == 'blue':
        flag_mom0 = 'uplim'
        mom0_sum = 3 * rms_final
        mom0_mean_final = 3 * rms_final
    if source == 'S68N' and velreg == 'fast' and mol == 'H2CO' and regname == 'blue':
        flag_mom0 = 'uplim'
        mom0_sum = 3 * rms_final
        mom0_mean_final = 3 * rms_final
    if source == 'SMM1a' and velreg == 'fast' and mol == 'H2CO' and regname == 'blue':
        flag_mom0 = 'uplim'
        mom0_sum = 3 * rms_final
        mom0_mean_final = 3 * rms_final





    #print ('|--- flag_mom0: %s ---|'%(flag_mom0))

    if plot:

        f, axarr = plt.subplots(1, 4, gridspec_kw = {'width_ratios':[1, 1, 1, 1]})
        f.set_figheight(10)
        f.set_figwidth(20)
        im0 = axarr[0].imshow(mom0[:,:],origin='lower',cmap='inferno')
        #axarr[0].contour(mom0masktab, origin='lower', color='black', vmin=0, vmax=1,levels=[0,1])
        #if mask_for_abundance:
        #    axarr[0].contour(mom0[:,:]*abu_mask, origin='lower', color='white', vmin=0, vmax=1,levels=[0,1])

        im1 = axarr[1].imshow(mom1[:,:],origin='lower',cmap='inferno',vmin=vin,vmax=vout)
        im2 = axarr[2].imshow(maxvel[:,:],origin='lower',cmap='inferno',vmin=vin,vmax=vout)
        im3 = axarr[3].imshow(dettab[:,:],origin='lower',cmap='inferno')
        #axarr[3].plot(m,n,'+',markersize=20)

        f.suptitle('Mom 0 mean final:  %.5e | Mom 0 mean_measured: %.5e | Rms final:%.5e | Flag mom0: %s'%(mom0_mean_final,mom0_mean_oversigma,rms_final,flag_mom0),transform=axarr[0].transData)
        #axarr[0].text(0.1,1700, 'Rms final:%.5e'%(rms_final),transform=axarr[0].transData)
        #axarr[0].text(0.1,1800, 'Flag mom0: %s'%(flag_mom0),transform=axarr[0].transData)


        f.colorbar(im0,orientation='horizontal', ax=axarr[0])
        f.colorbar(im1,orientation='horizontal' ,ax=axarr[1])
        f.colorbar(im2,orientation='horizontal', ax=axarr[2])
        f.colorbar(im3,orientation='horizontal', ax=axarr[3])

        axarr[0].set_title('mom0')
        axarr[1].set_title('mom1')
        axarr[2].set_title('maxvel')
        axarr[3].set_title('rms')

        for i in axarr:
            i.set_ylim(np.min(np.where(r1[:,:]>0)[0]),np.max(np.where(r1[:,:]>0)[0]))
            i.set_xlim(np.min(np.where(r1[:,:]>0)[1]),np.max(np.where(r1[:,:]>0)[1]))


        plt.savefig('plots/'+source+'_'+mol+'moment_map_'+regname+'_'+str(vin)+'_'+str(vout)+'_v2.png',format='png')
        #plt.show()
        plt.close()

        print (mom0_mean_SiOmask, mom0_mean_H2COmask, mom0_mean_HCNmask)


    if for_func1  == False:
        return  mom0_sum,\
            rms_final, \
            flag_mom0, \
            mom0_mean_final, \
            mom0_mean_SiOmask, \
            mom0_mean_H2COmask, \
            mom0_mean_HCNmask, \
            rms_final, \
            flag_mom0,\
            maxvel_mean,\
            vmax_max,\
            vmax_mean,\
            rmstab.tolist(),\
            (mom0*r1).tolist(),\
            (mom1*r1).tolist(),\
            (maxvel*r1).tolist(), \
            (dettab).tolist()
    else:
        return (mom0*r1),\
               (mom1*r1)




################# THIS IS THE END OF MOMENT MAP FUNCTION ###############################################################
################# NOTHING TO SEE HERE ##################################################################################


















def Qpart_interpolate(temp, mol):
    Mol_part = {
        'CO': {'part_table': [2.5595, 2.2584, 2.0369, 1.9123, 1.7370, 1.4386, 1.1429, 0.8526, 0.5733, 0.3389, 0.1478],
               'temp_table': [1000, 500, 300, 225, 150, 75, 37.5, 18.75, 9.375, 5, 2.725]},
        'H2CO': {'part_table': [3.7930, 3.4598, 3.2725, 3.0086, 2.5584, 2.1094, 1.6501, 1.1399, 0.6516, 0.3046],
                 'temp_table': [500, 300, 225, 150, 75, 37.5, 18.75, 9.375, 5, 2.725]},
        'HCN': {'part_table': [3.5198, 2.9688, 2.6566, 2.5122, 2.3286, 2.0286, 1.7317, 1.4389, 1.1545, 0.9109, 0.7016],
                'temp_table': [1000, 500, 300, 225, 150, 75, 37.5, 18.75, 9.375, 5, 2.725]},
        'SiO': {'part_table': [3.0656, 2.6949, 2.4613, 2.3354, 2.1594, 1.8593, 1.5602, 1.2632, 0.9703, 0.7115, 0.4736],
                'temp_table': [1000, 500, 300, 225, 150, 75, 37.5, 18.75, 9.375, 5, 2.725]},
        'test': {'part_table': [5, 4, 3, 2, 1],
                 'temp_table': [5, 4, 3, 2, 1]}
    }


    f = interp1d(Mol_part[mol]['temp_table'], Mol_part[mol]['part_table'])
    #xnew = np.linspace(np.min(Mol_part[mol]['temp_table']),
    #                   np.max(Mol_part[mol]['temp_table']), num=50, endpoint=True)

    #plt.plot(Mol_part[mol]['temp_table'], Mol_part[mol]['part_table'], 'o',
    #         xnew, f(xnew), '-',
    #         temp, f(temp), '*')
    print('---- For T = %.1f the log10(Q) = %.4f ---' % (temp, f(temp)))
    return f(temp)

# In[9]:


def M1(mol,trans,T,Rlobe,mom0,mom1,maxvel,dist,pixval_arcsec,plot=False,r1=[10,10],incl_corr=False,vmax_fixed=False,vmax_fix=0.0):
    mu_mass = 2.8# (Kauffmann+2008)
    dist = dist
    h_mass = 1.6737236*1e-27 #hydrogen mass in kg
    Rlobe = Rlobe*1.496e+8 #km                                 #####OUTFLOW SPECIFIC
    atomic_kg = 1.6605*1e-27

    molecules = {
    'CO' : {'atomic_mass': 12.0107+15.999,
          'mass': (12.0107+15.999)*atomic_kg,
          'h2_ratio': 1.2e4,                 # h2 to co abundance ratio (Frerking+1982)
          'Qpart_75':10**1.4386,
          'Qpart_150':10**1.7370,
          'transitions':{
                        '2-1':{
                            'Eu': 16.5962,
                            'Eu_J': 2.291352329373e-22,
                            'gu': 5,
                            'nu': 230.538, #GHz
                            'Aul': 6.910e-07 #s-1
                        },
                        '3-2':{
                            'Eu': 33.1917,
                            'Eu_J': 4.582608013332e-22,
                            'gu': 7,
                            'nu': 345.79598990, #GHz
                            'Aul': 2.497e-06 #s-1
                            }
                        }
            }
    }

    Nu_gu = 1937*molecules['CO']['transitions'][trans]['nu']**2/(molecules['CO']['transitions'][trans]['Aul']            *molecules['CO']['transitions'][trans]['gu'])*(mom0)
    print ('Nu ok')
    print ('Nu_gu tp Ntot conversion:', (10**Qpart_interpolate(T,mol))*            np.exp(molecules['CO']['transitions'][trans]['Eu_J']/(T*const.k_B.value)))

    Ntot = Nu_gu*(10**Qpart_interpolate(T,mol))*            np.exp(molecules['CO']['transitions'][trans]['Eu_J']/(T*const.k_B.value))
    print ('Ntot ok')
    print ('Ntot to M conversion',molecules['CO']['h2_ratio']*mu_mass*h_mass*(pixval_arcsec*dist*pixval_arcsec*dist*1.496e+13*1.496e+13)/const.M_sun.value )
    M = Ntot*molecules['CO']['h2_ratio']*mu_mass*h_mass*(pixval_arcsec*dist*pixval_arcsec*dist*1.496e+13*1.496e+13)/const.M_sun.value

    print('Ok, so far...')
    if vmax_fixed:
        M_mom = M*np.abs(vmax_fix)
        Fout = (M*vmax_fix*vmax_fix)/Rlobe*525600*60 #to yr
    else:
        M_mom = M*np.abs(maxvel)
        Fout = (M*maxvel*maxvel)/Rlobe*525600*60 #to yr
    if incl_corr == True:
        #for M1 we use c1 correction, see Table 3 in vdMarel2013
        #for now hard coded value for 10degree.
        Fout = Fout * 0.28
    if r1 == [10,10]:
        r1 = np.ones(np.shape(M))





    if plot:
        f, axarr = plt.subplots(1, 4, gridspec_kw = {'width_ratios':[1,1,1,1]})
        f.set_figheight(9)
        f.set_figwidth(18)
        im0 = axarr[0].imshow(M_mom[:,:]*r1,origin='lower',cmap='jet') #plotting central part only
        im1 = axarr[1].imshow(M[:,:]*r1,origin='lower',cmap='jet') #plotting central part only
        im2 = axarr[2].imshow(Nu_gu[:,:]*r1,origin='lower',cmap='jet') #plotting central part only
        im3 = axarr[3].imshow(Ntot[:,:]*r1,origin='lower',cmap='jet') #plotting central part only

        axarr[0].set_title('M_mom')
        axarr[1].set_title('M')
        axarr[2].set_title('Nu_gu')
        axarr[3].set_title('Ntot')



        for i in axarr:
            i.set_ylim(np.min(np.where(r1[:,:]>0)[0]),np.max(np.where(r1[:,:]>0)[0]))
            i.set_xlim(np.min(np.where(r1[:,:]>0)[1]),np.max(np.where(r1[:,:]>0)[1]))



        f.colorbar(im0, orientation='horizontal',ax=axarr[0])
        f.colorbar(im1, orientation='horizontal',ax=axarr[1])
        f.colorbar(im2, orientation='horizontal',ax=axarr[2])
        f.colorbar(im3, orientation='horizontal',ax=axarr[3])

        plt.show()
#plt.savefig('test.eps')
        plt.close()

    print ('Outflow mass (wolololo): ', np.nansum(M*r1), 'Msol')
    print ('Outflow momentum (M1): ', np.nansum(M_mom*r1), 'Msol km s-1 ')
    print ('Outflow force (M1): ', np.nansum(Fout*r1), 'Msol km s-1 yr-1')
    #return Nu_gu.tolist(), Ntot.tolist(), M.tolist(), M_mom.tolist(), Fout.tolist(),\
    return np.nansum(M*r1), np.nansum(M_mom*r1), np.nansum(Fout*r1)

#must be an easy way to sum all over regions

#print (mom0slow[750,910],mom1slow[750,910],maxvelslow[750,910],Foutslow[750,910])


# In[4]:


def M7(source,region,velregime,mol,trans,T,Rlobe,values,mom0,mom1,maxvel,dist,pixval_arcsec,r1=[10,10],plot=False,incl_corr=False,vmax_fixed=False,vmax_fix=0.0):
    mu_mass = 2.8# (Kauffmann+2008)
    dist = dist
    h_mass = 1.6737236*1e-27 #hydrogen mass in kg
    Rlobe = Rlobe*1.496e+8 #km                                 #####OUTFLOW SPECIFIC
    atomic_kg = 1.6605*1e-27

    molecules = {
    'CO' : {'atomic_mass': 12.0107+15.999,
          'mass': (12.0107+15.999)*atomic_kg,
          'h2_ratio': 1.2e4,                 # h2 to co abundance ratio (Frerking+1982)
          'Qpart_75':10**1.4386,
          'Qpart_150':10**1.7370,
          'Qpart_300': 10 ** 2.0369,
          'Qpart_500': 10 ** 2.2584,
          'transitions':{
                        '2-1':{
                            'Eu': 16.5962,
                            'Eu_J': 2.291352329373e-22,
                            'gu': 5,
                            'nu': 230.538, #GHz
                            'Aul': 6.910e-07 #s-1
                        },
                        '3-2':{
                            'Eu': 33.1917,
                            'Eu_J': 4.582608013332e-22,
                            'gu': 7,
                            'nu': 345.79598990, #GHz
                            'Aul': 2.497e-06 #s-1
                            }
                        }
            }
    }


    #value without scaling with moment one!
    columns = 1937*molecules['CO']['transitions'][trans]['nu']**2/(molecules['CO']['transitions'][trans]['Aul']\
            *molecules['CO']['transitions'][trans]['gu'])*(mom0)
    #this value have the momentum per pixel
    Nu_gu = 1937*molecules['CO']['transitions'][trans]['nu']**2/(molecules['CO']['transitions'][trans]['Aul']\
                                                                 *molecules['CO']['transitions'][trans]['gu'])*(np.abs(mom1)*mom0)
    Ntot = Nu_gu*(10**Qpart_interpolate(T,mol))*\
           np.exp(molecules['CO']['transitions'][trans]['Eu_J']/(T*const.k_B.value))
    M_mom = Ntot*molecules['CO']['h2_ratio']*mu_mass*h_mass*\
            (pixval_arcsec*dist*pixval_arcsec*dist*1.496e+13*1.496e+13)/const.M_sun.value
    Out_rate = M_mom/Rlobe*525600*60 #M_sol/year

    # rms part
    columns_rms = 1937 * molecules[mol]['transitions'][trans]['nu'] ** 2 / (molecules[mol]['transitions'][trans]['Aul'] \
                                                                       * molecules[mol]['transitions'][trans]['gu']) * \
                values[source][region][velregime][mol]['mom0sum']['rms']

    Nu_gu_rms = columns_rms

    Ntot_rms = Nu_gu_rms * (10**Qpart_interpolate(T,mol)) * np.exp(molecules[mol]['transitions'][trans]['Eu_J'] \
                                                                      / (T * const.k_B.value))

    Ntot_flag = values[source][region][velregime][mol]['mom0mean']['flag']

    Nmol_rms = Ntot_rms * (pixval_arcsec * dist * pixval_arcsec * dist * 1.496e+13 * 1.496e+13)

    M_mom_rms = Ntot_rms * molecules[mol]['mass'] * (
    pixval_arcsec * dist * pixval_arcsec * dist * 1.496e+13 * 1.496e+13) / const.M_sun.value

    Out_rate_rms = M_mom_rms/Rlobe*525600*60 #M_sol/year


    if vmax_fixed:
        Fout = (M_mom*np.abs(vmax_fix))/Rlobe*525600*60 #to yr
        Fout_rms = (M_mom_rms*np.abs(1))/Rlobe*525600*60 #to yr
        Lkin = (M_mom*np.abs(vmax_fix)*np.abs(vmax_fix))/(2*Rlobe) #are units correct?
        Lkin_rms = (M_mom*np.abs(vmax_fix)*np.abs(1))/(2*Rlobe) #are units correct?

    else:
        Fout = (M_mom*np.abs(maxvel))/Rlobe*525600*60 #to yr
        Fout_rms = (M_mom*np.abs(1))/Rlobe*525600*60 #to yr
        Lkin = (M_mom*np.abs(maxvel)*np.abs(maxvel))/(2*Rlobe) #are units correct?
        Lkin_rms = (M_mom*np.abs(1)*np.abs(1))/(2*Rlobe) #are units correct?


    if incl_corr == True:
        #for M7 we use c3 correction, see Table 3 in vdMarel2013
        #for now hard coded value for 10degree.
        Fout = Fout * 1.2

    if r1 == [10,10]:
        r1 = np.ones(np.shape(M_mom))

    print ('Column density of CO (average): ', np.mean(columns),  'cm-2')
    print ('Column density of CO (min): ', np.min(columns),  'cm-2')
    print ('Column density of CO (max): ', np.max(columns),  'cm-2')

    print ('Outflow momentum (M7): ', np.nansum(M_mom*r1), 'Msol km s-1')
    print ('Outflow force (M7): ', np.nansum(Fout*r1), 'Msol km s-1 yr-1')
    print ('Kinetic luminosity (M7): ', np.nansum(Lkin), 'Lsol')


    if plot:
        f, axarr = plt.subplots(1, 3, gridspec_kw = {'width_ratios':[1,1,1]})
        f.set_figheight(9)
        f.set_figwidth(27)
        im0 = axarr[0].imshow(Fout[:,:]*r1,origin='lower',cmap='jet') #plotting central part only
        im1 = axarr[1].imshow(Nu_gu[:,:]*r1,origin='lower',cmap='jet') #plotting central part only
        im2 = axarr[2].imshow(columns[:,:]*r1,origin='lower',cmap='jet') #plotting central part only

        axarr[0].set_title('Fout')
        axarr[1].set_title('Nu_gu')
        axarr[2].set_title('Column density [cm-2]')



        for i in axarr:
            i.set_ylim(np.min(np.where(r1[:,:]>0)[0]),np.max(np.where(r1[:,:]>0)[0]))
            i.set_xlim(np.min(np.where(r1[:,:]>0)[1]),np.max(np.where(r1[:,:]>0)[1]))


        f.colorbar(im0, orientation='horizontal',ax=axarr[0])
        f.colorbar(im1, orientation='horizontal',ax=axarr[1])
        f.colorbar(im2, orientation='horizontal',ax=axarr[2])


        #plt.show()
        plt.savefig('plots/' + source + '_' + mol + 'M7_' + region + velregime + '.png',
         format = 'png')
        plt.close()



    M_mom_sum = np.nansum(M_mom*r1)
    if  M_mom_sum == 0:
        M_mom_sum = 3*M_mom_rms
    Fout_sum  = np.nansum(Fout * r1)
    if  Fout_sum == 0:
        Fout_sum = 3*Fout_rms
    Out_rate_sum = np.nansum(Out_rate * r1)
    if Out_rate_sum == 0:
        Out_rate_sum = 3*Out_rate_rms



#    return Nu_gu.tolist(), Ntot.tolist(), M_mom.tolist(), Fout.tolist(),\
    return M_mom_sum, \
           M_mom_rms,\
           Ntot_flag, \
           Fout_sum, \
           Fout_rms,\
           Ntot_flag, \
           np.nansum(Out_rate*r1), \
           Out_rate_rms,\
           Ntot_flag \

        #must be an easy way to sum all over regions

#print (mom0slow[750,910],mom1slow[750,910],maxvelslow[750,910],Foutslow[750,910])


# # mass_main
#
# Provides tables with Ntot of molecule and values of total number of molecules

# In[16]:


def mass_main(source,mol,tr,T,region,velregime,values,mom0,dettab,r1,pixval_arcsec,plot=False,positive_mask=False):
    mu_mass = 2.8# (Kauffmann+2008)
    h_mass = 1.6737236*1e-27 #hydrogen mass in kg
    dist = 430 #distance to Serpens in parsecs (Ortiz-Leon2016)
    R_lobe = 17*dist*1.496e+8 #km                                 #####OUTFLOW SPECIFIC
    atomic_kg = 1.6605*1e-27



    molecules = {
    'CO' : {'atomic_mass': 12.0107+15.999,
          'mass': (12.0107+15.999)*atomic_kg,
          'h2_ratio': 1.2e4,                 # h2 to co abundance ratio (Frerking+1982)
          'Qpart_75':10**1.4386,
          'transitions':{
                        '2-1':{
                            'Eu': 16.5962,
                            'Eu_J': 2.291352329373e-22,
                            'gu': 5,
                            'nu': 230.538, #GHz
                            'Aul': 6.910e-07 #s-1
                        },
                        '3-2':{
                            'Eu': 33.1917,
                            'Eu_J': 4.582608013332e-22,
                            'gu': 7,
                            'nu': 345.79598990, #GHz
                            'Aul': 2.497e-06 #s-1
                            }
                        }
          },
    'SiO' : {'atomic_mass': 28.0855+15.999,
          'mass': (28.0855+15.999)*atomic_kg,
          'h2_ratio': 3e12,                 # h2 to sio abundance ratio (Zruyis 1989, Martin-Pintado 1992)
          'Qpart_75':10**1.8593, #######/??????????
          'Qpart_18.75':10**1.2632,
          'Qpart_9.375':10**0.9703,
          'transitions':{
                        '5-4':{
                            'Eu': 31.2588,   #K
                            'Eu_J': 4.315742410516e-22, #J
                            'gu': 11,
                            'nu': 217.10498, #GHz
                            'Aul': 5.1965e-04 #s-1
                        }
           }
    },
    'H2CO' : {'atomic_mass': 1.00794+1.00794+12.0107+15.999,
          'mass': (1.00794+1.00794+12.0107+15.999)*atomic_kg,
          'h2_ratio': 1e8,                 #  Sherwood 1980
          'Qpart_9.375':10**1.1399,
          'Qpart_18.75':10**1.2632,
          'transitions':{
                        '3-2':{
                            'Eu': 21.0,   #K #Not sure?!?!?!?
                            'Eu_J': 2.899362439404e-22, #J
                            'gu': 7, ## not sure?!?!?!?!?
                            'nu': 218.222192, #GHz
                            'Aul': 2.818e-04  #s-1
                        }
          }
    },
    'HCN' : {'atomic_mass': 1.00794+12.0107+14.00674,
          'mass': ( 1.00794+12.0107+14.00674)*atomic_kg,
          'h2_ratio': 1e6,                 #  ??????
          'Qpart_9.375':10**1.1545,
          'Qpart_18.75':10**1.2632,
          'transitions':{
                        '1-0':{
                            'Eu': 4.25,   #K #Not sure?!?!?!?
                            'Eu_J': 5.867758250000001e-23, #J
                            'gu': 3.0, ## not sure?!?!?!?!?
                            'nu': 88.63184700, #GHz
                            'Aul': 2.407e-05  #s-1
                        }
          }
    }
            }


    print (mol,tr,T,pixval_arcsec,plot)
    print (molecules[mol]['transitions'][tr]['nu'],
           molecules[mol]['transitions'][tr]['Aul'],
           molecules[mol]['transitions'][tr]['gu'])
    if np.shape(mom0) == 0:
        return -99,  -99,  -99,


    Nu_gu = 1937*molecules[mol]['transitions'][tr]['nu']**2/(molecules[mol]['transitions'][tr]['Aul']\
            *molecules[mol]['transitions'][tr]['gu'])*mom0
    Ntot = Nu_gu*(10**Qpart_interpolate(T,mol))*np.exp(molecules[mol]['transitions'][tr]['Eu_J']/(T*const.k_B.value))
    N_mol = Ntot*(pixval_arcsec*dist*pixval_arcsec*dist*1.496e+13*1.496e+13)
    M_mol = Ntot*molecules[mol]['mass']*(pixval_arcsec*dist*pixval_arcsec*dist*1.496e+13*1.496e+13)/const.M_sun.value
   # M_mom = M_mol*np.abs(maxvel)


 # rms part
    Nu_gu_rms = 1937*molecules[mol]['transitions'][tr]['nu']**2/(molecules[mol]['transitions'][tr]['Aul']\
            *molecules[mol]['transitions'][tr]['gu']) * values[source][region][velregime][mol]['mom0sum']['rms']

    Ntot_rms = Nu_gu_rms*(10**Qpart_interpolate(T,mol))*np.exp(molecules[mol]['transitions'][tr]['Eu_J']\
                /(T*const.k_B.value))
    Ntot_flag = values[source][region][velregime][mol]['mom0mean']['flag']

    Nmol_rms = Ntot_rms*(pixval_arcsec*dist*pixval_arcsec*dist*1.496e+13*1.496e+13)

    M_mol_rms = Ntot_rms*molecules[mol]['mass']*(pixval_arcsec*dist*pixval_arcsec*dist*1.496e+13*1.496e+13)/const.M_sun.value





    if positive_mask:
        idx = np.where(Ntot<0)
        Ntot[idx]=0
        idx = np.where(N_mol<0)
        N_mol[idx]=0
        idx = np.where(M_mol<0)
        M_mol[idx]=0

    region_idx = np.where(r1==1)

    print ('----- %s ---- %s ---- %s ---- %s -----'%(source, mol, region, velregime))


    dettab_idx  = np.where(dettab==1)


    np.nanmean((Ntot*r1)[region_idx])
    print ( mol + ' flag: ', Ntot_flag)
    print ( mol + ' average density: ', np.nanmean((Ntot*r1)[region_idx]))
    print ( mol + ' average density_dettab: ', np.nanmean((Ntot*r1)[dettab_idx]))
    print ( mol + ' rms: ', Nu_gu_rms)
    print ( mol + ' peak density: ', np.nanmax((Ntot*r1)[region_idx]))
    print ( mol + ' rms: ', Nu_gu_rms)
    print ( mol + ' number_density: ', np.nansum(Ntot*r1) )
    print ( mol + ' rms: ', Ntot_rms)
    print ( mol + ' total number: ', np.nansum(N_mol*r1) )
    print ( mol + ' rms: ', Nmol_rms)
    print ( mol + ' mass: ', np.nansum(M_mol*r1), 'Msol')
    print ( mol + ' rms: ', M_mol_rms)

   # print ( mol + 'momentum (M1): ', np.nansum(M_mom*r1), 'Msol km s-1 ')


    if plot:
        f, axarr = plt.subplots(1, 2, gridspec_kw = {'width_ratios':[1,1]})
        f.set_figheight(5)
        f.set_figwidth(15)
        im0 = axarr[0].imshow(N_mol[:,:]*r1,origin='lower',cmap='jet') #plotting central part only
        im1 = axarr[1].imshow(Ntot[:,:]*r1,origin='lower',cmap='jet') #plotting central part only

        axarr[0].set_title('N_mol')
        axarr[1].set_title('N_tot')

        for i in axarr:
            i.set_ylim(np.min(np.where(r1[:,:]>0)[0]),np.max(np.where(r1[:,:]>0)[0]))
            i.set_xlim(np.min(np.where(r1[:,:]>0)[1]),np.max(np.where(r1[:,:]>0)[1]))

        axarr[0].set_title('N_mol')
        axarr[1].set_title('Ntot')




        f.colorbar(im0, orientation='horizontal',ax=axarr[0])
        f.colorbar(im1, orientation='horizontal',ax=axarr[1])

        plt.savefig('plots/'+source+'_'+mol+'mass_main_'+region+'_'+velregime+'.png',format='png')
        plt.close()

        #dettab_idx -> count only the pixels with the detection!

        column_density = np.nanmean((Ntot*r1)[dettab_idx])
        column_density_rms = Ntot_rms
        column_density_flag = Ntot_flag
        if column_density_flag == 'uplim':
            column_density = Ntot_rms*3


    pf.writeto('fitsfiles/ntot_' + source + '_' + mol + '_' + str(region) + '_' + str(velregime) + '.fits', Ntot,clobber=True)

    return  column_density, \
            column_density_rms, \
            column_density_flag, \
            np.nansum(Ntot * r1), \
            Ntot_rms, \
            Ntot_flag, \
            np.nanmax((Ntot*r1)[region_idx]), \
            Ntot_rms, \
            Ntot_flag, \
            np.nansum(N_mol*r1), \
            Nmol_rms, \
            Ntot_flag, \
            np.nansum(M_mol*r1),\
            M_mol_rms,\
            Ntot_flag \



def mass_main_new(source,mol,tr,T,region,velregime,values,pixval_arcsec):
    mu_mass = 2.8# (Kauffmann+2008)
    h_mass = 1.6737236*1e-27 #hydrogen mass in kg
    dist = 430 #distance to Serpens in parsecs (Ortiz-Leon2016)
    R_lobe = 17*dist*1.496e+8 #km                                 #####OUTFLOW SPECIFIC
    atomic_kg = 1.6605*1e-27

    molecules = {
    'CO' : {'atomic_mass': 12.0107+15.999,
          'mass': (12.0107+15.999)*atomic_kg,
          'h2_ratio': 1.2e4,                 # h2 to co abundance ratio (Frerking+1982)
          'Qpart_75':10**1.4386,
          'transitions':{
                        '2-1':{
                            'Eu': 16.5962,
                            'Eu_J': 2.291352329373e-22,
                            'gu': 5,
                            'nu': 230.538, #GHz
                            'Aul': 6.910e-07 #s-1
                        },
                        '3-2':{
                            'Eu': 33.1917,
                            'Eu_J': 4.582608013332e-22,
                            'gu': 7,
                            'nu': 345.79598990, #GHz
                            'Aul': 2.497e-06 #s-1
                            }
                        }
          },
    'SiO' : {'atomic_mass': 28.0855+15.999,
          'mass': (28.0855+15.999)*atomic_kg,
          'h2_ratio': 3e12,                 # h2 to sio abundance ratio (Zruyis 1989, Martin-Pintado 1992)
          'Qpart_75':10**1.8593, #######/??????????
          'Qpart_18.75':10**1.2632,
          'Qpart_9.375':10**0.9703,
          'transitions':{
                        '5-4':{
                            'Eu': 31.2588,   #K
                            'Eu_J': 4.315742410516e-22, #J
                            'gu': 11,
                            'nu': 217.10498, #GHz
                            'Aul': 5.1965e-04 #s-1
                        }
           }
    },
    'H2CO' : {'atomic_mass': 1.00794+1.00794+12.0107+15.999,
          'mass': (1.00794+1.00794+12.0107+15.999)*atomic_kg,
          'h2_ratio': 1e8,                 #  Sherwood 1980
          'Qpart_9.375':10**1.1399,
          'Qpart_18.75':10**1.2632,
          'transitions':{
                        '3-2':{
                            'Eu': 21.0,   #K #Not sure?!?!?!?
                            'Eu_J': 2.899362439404e-22, #J
                            'gu': 7, ## not sure?!?!?!?!?
                            'nu': 218.222192, #GHz
                            'Aul': 2.818e-04  #s-1
                        }
          }
    },
    'HCN' : {'atomic_mass': 1.00794+12.0107+14.00674,
          'mass': ( 1.00794+12.0107+14.00674)*atomic_kg,
          'h2_ratio': 1e6,                 #  ??????
          'Qpart_9.375':10**1.1545,
          'Qpart_18.75':10**1.2632,
          'transitions':{
                        '1-0':{
                            'Eu': 4.25,   #K #Not sure?!?!?!?
                            'Eu_J': 5.867758250000001e-23, #J
                            'gu': 3.0, ## not sure?!?!?!?!?
                            'nu': 88.63184700, #GHz
                            'Aul': 2.407e-05  #s-1
                        }
          }
    }
            }

    Nu_gu = 1937*molecules[mol]['transitions'][tr]['nu']**2/(molecules[mol]['transitions'][tr]['Aul']\
            *molecules[mol]['transitions'][tr]['gu'])*values[source][region][velregime][mol]['mom0mean']['value']
    Ntot = Nu_gu*(10**Qpart_interpolate(T,mol))*np.exp(molecules[mol]['transitions'][tr]['Eu_J']/(T*const.k_B.value))
    N_mol = Ntot*(pixval_arcsec*dist*pixval_arcsec*dist*1.496e+13*1.496e+13)
    M_mol = Ntot*molecules[mol]['mass']*(pixval_arcsec*dist*pixval_arcsec*dist*1.496e+13*1.496e+13)/const.M_sun.value

    Nu_gu_rms = 1937*molecules[mol]['transitions'][tr]['nu']**2/(molecules[mol]['transitions'][tr]['Aul']\
            *molecules[mol]['transitions'][tr]['gu']) * values[source][region][velregime][mol]['mom0mean']['rms']
    Ntot_rms = Nu_gu_rms*(10**Qpart_interpolate(T,mol))*np.exp(molecules[mol]['transitions'][tr]['Eu_J']\
                /(T*const.k_B.value))
    Ntot_flag = values[source][region][velregime][mol]['mom0mean']['flag']
    Nmol_rms = Ntot_rms*(pixval_arcsec*dist*pixval_arcsec*dist*1.496e+13*1.496e+13)
    M_mol_rms = Ntot_rms*molecules[mol]['mass']*(pixval_arcsec*dist*pixval_arcsec*dist*1.496e+13*1.496e+13)/const.M_sun.value

    print ('----- %s ---- %s ---- %s ---- %s -----'%(source, mol, region, velregime))


   # print ( mol + 'momentum (M1): ', np.nansum(M_mom*r1), 'Msol km s-1 ')

    column_density = Ntot
    column_density_rms = Ntot_rms
    column_density_flag = Ntot_flag
    if column_density_flag == 'uplim':
            column_density = Ntot_rms*3

    return  column_density, \
            column_density_rms, \
            column_density_flag





def mass_main_masking(source,mol,mol_mask,tr,T,region,velregime,values,pixval_arcsec):
    mu_mass = 2.8# (Kauffmann+2008)
    h_mass = 1.6737236*1e-27 #hydrogen mass in kg
    dist = 430 #distance to Serpens in parsecs (Ortiz-Leon2016)
    R_lobe = 17*dist*1.496e+8 #km                                 #####OUTFLOW SPECIFIC
    atomic_kg = 1.6605*1e-27

    molecules = {
    'CO' : {'atomic_mass': 12.0107+15.999,
          'mass': (12.0107+15.999)*atomic_kg,
          'h2_ratio': 1.2e4,                 # h2 to co abundance ratio (Frerking+1982)
          'Qpart_75':10**1.4386,
          'transitions':{
                        '2-1':{
                            'Eu': 16.5962,
                            'Eu_J': 2.291352329373e-22,
                            'gu': 5,
                            'nu': 230.538, #GHz
                            'Aul': 6.910e-07 #s-1
                        },
                        '3-2':{
                            'Eu': 33.1917,
                            'Eu_J': 4.582608013332e-22,
                            'gu': 7,
                            'nu': 345.79598990, #GHz
                            'Aul': 2.497e-06 #s-1
                            }
                        }
          },
    'SiO' : {'atomic_mass': 28.0855+15.999,
          'mass': (28.0855+15.999)*atomic_kg,
          'h2_ratio': 3e12,                 # h2 to sio abundance ratio (Zruyis 1989, Martin-Pintado 1992)
          'Qpart_75':10**1.8593, #######/??????????
          'Qpart_18.75':10**1.2632,
          'Qpart_9.375':10**0.9703,
          'transitions':{
                        '5-4':{
                            'Eu': 31.2588,   #K
                            'Eu_J': 4.315742410516e-22, #J
                            'gu': 11,
                            'nu': 217.10498, #GHz
                            'Aul': 5.1965e-04 #s-1
                        }
           }
    },
    'H2CO' : {'atomic_mass': 1.00794+1.00794+12.0107+15.999,
          'mass': (1.00794+1.00794+12.0107+15.999)*atomic_kg,
          'h2_ratio': 1e8,                 #  Sherwood 1980
          'Qpart_9.375':10**1.1399,
          'Qpart_18.75':10**1.2632,
          'transitions':{
                        '3-2':{
                            'Eu': 21.0,   #K #Not sure?!?!?!?
                            'Eu_J': 2.899362439404e-22, #J
                            'gu': 7, ## not sure?!?!?!?!?
                            'nu': 218.222192, #GHz
                            'Aul': 2.818e-04  #s-1
                        }
          }
    },
    'HCN' : {'atomic_mass': 1.00794+12.0107+14.00674,
          'mass': ( 1.00794+12.0107+14.00674)*atomic_kg,
          'h2_ratio': 1e6,                 #  ??????
          'Qpart_9.375':10**1.1545,
          'Qpart_18.75':10**1.2632,
          'transitions':{
                        '1-0':{
                            'Eu': 4.25,   #K #Not sure?!?!?!?
                            'Eu_J': 5.867758250000001e-23, #J
                            'gu': 3.0, ## not sure?!?!?!?!?
                            'nu': 88.63184700, #GHz
                            'Aul': 2.407e-05  #s-1
                        }
          }
    }
            }

    Nu_gu = 1937*molecules[mol]['transitions'][tr]['nu']**2/(molecules[mol]['transitions'][tr]['Aul']\
            *molecules[mol]['transitions'][tr]['gu'])*values[source][region][velregime]['CO_'+mol_mask]['mom0mean']['value']
    Ntot = Nu_gu*(10**Qpart_interpolate(T,mol))*np.exp(molecules[mol]['transitions'][tr]['Eu_J']/(T*const.k_B.value))
    N_mol = Ntot*(pixval_arcsec*dist*pixval_arcsec*dist*1.496e+13*1.496e+13)
    M_mol = Ntot*molecules[mol]['mass']*(pixval_arcsec*dist*pixval_arcsec*dist*1.496e+13*1.496e+13)/const.M_sun.value

    Nu_gu_rms = 1937*molecules[mol]['transitions'][tr]['nu']**2/(molecules[mol]['transitions'][tr]['Aul']\
            *molecules[mol]['transitions'][tr]['gu']) * values[source][region][velregime][mol]['mom0mean']['rms']
    Ntot_rms = Nu_gu_rms*(10**Qpart_interpolate(T,mol))*np.exp(molecules[mol]['transitions'][tr]['Eu_J']\
                /(T*const.k_B.value))
    Ntot_flag = values[source][region][velregime][mol]['mom0mean']['flag']
    Nmol_rms = Ntot_rms*(pixval_arcsec*dist*pixval_arcsec*dist*1.496e+13*1.496e+13)
    M_mol_rms = Ntot_rms*molecules[mol]['mass']*(pixval_arcsec*dist*pixval_arcsec*dist*1.496e+13*1.496e+13)/const.M_sun.value

    print ('----- %s ---- %s ---- %s ---- %s -----'%(source, mol, region, velregime))


   # print ( mol + 'momentum (M1): ', np.nansum(M_mom*r1), 'Msol km s-1 ')

    column_density = Ntot
    column_density_rms = Ntot_rms
    column_density_flag = Ntot_flag
    if column_density_flag == 'uplim':
            column_density = Ntot_rms*3

    return  column_density, \
            column_density_rms, \
            column_density_flag





def moment_plots(mom0,mom1,maxvel,r1=[10,10],mom1lim=[0,10]):

    f, axarr = plt.subplots(1, 3, gridspec_kw = {'width_ratios':[1,1,1]})
    f.set_figheight(5)
    f.set_figwidth(18)
    if r1 == [10,10]:
        im0 = axarr[0].imshow(mom0[:,:],origin='lower',cmap='jet') #plotting central part only
        im1 = axarr[1].imshow(mom1[:,:],origin='lower',cmap='jet',vmin=mom1lim[0],vmax=mom1lim[1]) #plotting central part only
        im2 = axarr[2].imshow(maxvel[:,:],origin='lower',cmap='jet',vmin=mom1lim[0],vmax=mom1lim[1]) #plotting central part only
    else:
        im0 = axarr[0].imshow(mom0[:,:]*r1,origin='lower',cmap='jet') #plotting central part only
        im1 = axarr[1].imshow(mom1[:,:]*r1,origin='lower',cmap='jet',vmin=mom1lim[0],vmax=mom1lim[1]) #plotting central part only
        im2 = axarr[2].imshow(maxvel[:,:]*r1,origin='lower',cmap='jet',vmin=mom1lim[0],vmax=mom1lim[1]) #plotting central part only
        for i in axarr:
            i.set_ylim(np.min(np.where(r1[:,:]>0)[0]),np.max(np.where(r1[:,:]>0)[0]))
            i.set_xlim(np.min(np.where(r1[:,:]>0)[1]),np.max(np.where(r1[:,:]>0)[1]))



    axarr[0].set_title('Moment 0')
    axarr[1].set_title('Moment 1')
    axarr[2].set_title('v_max')




    f.colorbar(im0, orientation='vertical',ax=axarr[0])
    f.colorbar(im1, orientation='vertical',ax=axarr[1])
    f.colorbar(im2, orientation='vertical',ax=axarr[2])

    plt.show()
    plt.close()

def custom_pv(source,pvdiagram,header_pv,velocity,chanwidth,border=[0,0],offset=0,outname= 'pvdiagram'):

    with open('json/sources.json', 'r') as f:
        sources = json.load(f)

    if border == [0,0]:
        border[0] = sources[source]['vellim'][0]
        border[1] = sources[source]['vellim'][1]
        print (border)



    vlsr = sources[source]['vlsr']
    ehvblue = sources[source]['ehvlim'][0]
    ehvred = sources[source]['ehvlim'][1]
    ihvblue = sources[source]['ihvlim'][0]
    ihvred = sources[source]['ihvlim'][1]


    f, axarr = plt.subplots(1, 2, gridspec_kw = {'width_ratios':[20, 1]})
    f.set_figheight(5)
    f.set_figwidth(18)
    transpv = np.matrix.transpose(pvdiagram)
    im = axarr[0].imshow(transpv,origin='lower',cmap='jet', aspect='auto') #plotting central part only
    levels = [0.012, 0.05,0.1,0.2,0.5]
    axarr[0].contour(transpv,levels,linewidth=0.1) #plotting central part only
    f.colorbar(im, orientation='vertical')

#ok this is some fun stuff to make the labels look the way we want them:
#velrange_tab = np.fromstring(velrange[np.where(lnam==paths[j])[0]],sep=',')
    velrange_tab = border
    print (velrange_tab[0],velrange_tab[1])

#getting the pixel value of the central velocity
#sourcevelpix = np.interp(sourcevel,velocity,np.arange(0,nrchan))
    sourcevelchan = np.argmin(np.abs(velocity - (0)))

    minvelpix =  sourcevelchan-abs(velrange_tab[0])/chanwidth
    maxvelpix =  sourcevelchan+velrange_tab[1]/chanwidth
    axarr[0].set_ylim(minvelpix,maxvelpix)
    ysteps = np.array([minvelpix,sourcevelchan-(abs(velrange_tab[0])/2.0)/chanwidth,sourcevelchan,sourcevelchan+(abs(velrange_tab[1])/2.0)/chanwidth,maxvelpix])
    axarr[0].set_yticks(ysteps)
    ylabels_mod = (ysteps-sourcevelchan)*chanwidth
    ylabels_mod = np.round(ylabels_mod,1)
    axarr[0].set_yticklabels(ylabels_mod,fontsize=20)

    px0 = header_pv['CRVAL2']
    pixval = header_pv['CDELT2']
    nrpix = header_pv['NAXIS2']
    xsteps = np.array([0.0,0.25,0.5,0.75,1.0])
    xsteps = xsteps*nrpix
    axarr[0].set_xticks(xsteps)

    xlabels_mod = xsteps*pixval-abs(px0)
    xlabels_mod = np.round(xlabels_mod,1)
    axarr[0].set_xticklabels(xlabels_mod,fontsize=20)



    axarr[0].set_xlabel('Angular offset ["]',fontsize=20)
    axarr[0].set_ylabel('Velocity [km/s]',fontsize=20)


    axarr[0].set_xlim([offset,nrpix-offset])
    axarr[0].plot(axarr[0].get_xlim(),[sourcevelchan+(ehvblue)/chanwidth,sourcevelchan+(ehvblue)/chanwidth],'--',color='black')
    axarr[0].plot(axarr[0].get_xlim(),[sourcevelchan+(ehvred)/chanwidth,sourcevelchan+(ehvred)/chanwidth],'--',color='black')
    axarr[0].plot(axarr[0].get_xlim(),[sourcevelchan+(ihvblue)/chanwidth,sourcevelchan+(ihvblue)/chanwidth],'--',color='black')
    axarr[0].plot(axarr[0].get_xlim(),[sourcevelchan+(ihvred)/chanwidth,sourcevelchan+(ihvred)/chanwidth],'--',color='black')

    plt.savefig(outname + '_' + source + '.png', format='png')
    #plt.show()
    plt.close()


# # Produce array of distances and pick max value

# In[59]:


def rdist_arr(xc,yc,size,r1,pixval_arcsec,dist):
    tab = np.zeros([size,size])
    for m in range(size):
        for n in range(size):
            tab[m,n] = ((np.float(m-xc))**2.+(np.float(n-yc))**2.)**0.5
    print (pixval_arcsec,dist)
    tab = tab*pixval_arcsec*dist
    print (np.shape(tab),np.shape(r1))
    tab = tab*r1
    print (np.max(tab))
    rmax = np.max(tab)
    return rmax, tab
 #           else:
  #              tab[m,n] = np.sum(datain[idxvin:idxvout,m,n])*chanwidth



# # Measure distance to specific pixel

# * xc,yc - reference pixel coords (in pix)
# * xpix, ypix - position to which distance is measured

# In[54]:


def rdist(xc,yc,xpix,ypix,size,r1,pixval_arcsec,dist):
    d = ((np.float(xpix-xc))**2.+(np.float(ypix-yc))**2.)**0.5
    d_au = d*pixval_arcsec*dist
    return d_au
 #           else:
  #              tab[m,n] = np.sum(datain[idxvin:idxvout,m,n])*chanwidth



# # Dataset in (Jy/beam) out in K, including beam handler

# In[17]:


def dataJytoK(source,sources,mol,restfreq,dataset,header,customfile=False,fixed_beam=False,customname='',prefix=''):
    print ('convert JytoK started')

    if header['BUNIT'] == 'K':
        print ('Unit is',header['BUNIT'],'you may continue')
        convert_JytoK = False
    elif header['BUNIT'] == 'Jy/beam':
        print ('Unit is',header['BUNIT'],'you should convert!')
        convert_JytoK = True
    else:
        print ('Unknown unit', header['BUNIT'], 'stop right here')
        convert_JytoK = False

    if convert_JytoK:
        #beam_table =  np.loadtxt(prefix + sources[source]['field']+'/'+sources[source]['field']+'_'+mol+'.beam')  ###### field + mol
        #if mol == 'H2CO':
        #    beam_table = np.loadtxt(prefix + sources[source]['field']+'/' +'Emb8N_h2co_nocontsub.pbcor.fits'+'.beam')
        #else:

        if customfile == False:
            print ('customfile false')
            beam_table =  np.loadtxt(prefix + sources[source]['field']+'/'+sources[source]['field']+'_'+mol+'.beam')  ###### field + mol
        else:
            print ('beam file',customname + '.beam')
            beam_table =  np.loadtxt(customname + '.beam')  ###### field + mol

        pixval_x, pixval_arcsec, beam_area_tab,beam_a,beam_b,beam_pa = BeamHandler(beam_table, header)
        dataset_perpix = dataset/beam_area_tab
        omega = (pixval_x*np.pi/180.0)**2*u.sr
        dataset_K = dataset_perpix[:,:,:]*u.Jy.to(u.K, equivalencies=u.brightness_temperature(omega,restfreq*u.Hz))

    else:
        pixval_x = header['CDELT1']
        pixval_arcsec =  np.abs((pixval_x)*3600.0)
        #print ('Size of the pixel:', pixval_arcsec, 'arcsec')

    print ('convert JytoK ended')
    return dataset_K, pixval_arcsec





# #  Handling most of the fits and JSON stuff

# In[1]:


def FitsHandler1(source,mol,customfile=False,customname='',prefix='../../Projects/',convert_JytoK=True,vlsr=8.5):


    #print ('Read json sources file')
    with open('database/sources.json', 'r') as f:
        sources = json.load(f)
    if customfile:
        #print('Customfile selected', prefix + customname)

        image = customname 
        dataset, header = pf.getdata(image, header=True)

    else:
        #print('Customfile not selected', prefix + sources[source]['field']+'/'+sources[source]['field']+'_'+mol+'.fits')

        vlsr= sources[source]['vlsr']
        image = prefix + sources[source]['field']+'/'+sources[source]['field']+'_'+mol+'.fits'###### field + mol
        dataset, header = pf.getdata(image,header=True)

    pixval_x = header['CDELT1']
    pixval_arcsec = np.abs((pixval_x) * 3600.0)
    #print('Size of the pixel:', pixval_arcsec, 'arcsec')



    print (np.shape(dataset))
    w = WCS(image)
    if (len(np.shape(dataset))) == 4:
        dataset=dataset[0,:,:,:]
        w = w.dropaxis(3)  # Remove the Stokes axis
        w = w.dropaxis(2)
    #elif (len(np.shape(dataset))) == 3:


#    w = w.dropaxis(2)  # and spectral axis




    size = np.shape(dataset[0,:,0])[0]



    velocity, freqtab, chanwidth, nchan, restfreq = SpecaxisHandler(dataset,header,vlsr)



    #print ('before conversion Jy to K')
    if convert_JytoK:
        if customfile == False:
            dataset_K, pixval_arcsec = dataJytoK(source,sources,mol,restfreq,dataset,header,prefix=prefix)
        else:
            dataset_K, pixval_arcsec = dataJytoK(source,sources,mol,restfreq,dataset,header,prefix=prefix,
                                                 customfile=True, customname=customname)

        final_dataset = dataset_K
    else:
        final_dataset = dataset

    return final_dataset, header, size, velocity, chanwidth, nchan, restfreq, pixval_arcsec, w





# In[30]:


def SimplePlot(source,region,mol='CO',mol2_region='core',mol2='C18O',norm=False,xlim=[-15,15]):


    with open('sources.json', 'r') as f:
        sources = json.load(f)

    prefix = '../../Projects/'


    #r1 = pf.getdata(prefix + sources[source]['field']+'/'+source+'_'+region+'.fits')
    r1_C18O_red = pf.getdata(prefix + sources[source]['field']+'/'+sources[source]['field']+'_'+'C18O_red'+'_B6.fits')
    r1_C18O_blue = pf.getdata(prefix + sources[source]['field']+'/'+sources[source]['field']+'_'+'C18O_blue'+'_B6.fits')
#    r1_core = pf.getdata(prefix + sources[source]['field']+'/'+source+'_'+'core'+'.fits')
    r1_red = pf.getdata(prefix + sources[source]['field']+'/'+source+'_'+'red'+'_B6.fits')
    r1_blue = pf.getdata(prefix + sources[source]['field']+'/'+source+'_'+'blue'+'_B6.fits')
#    print (sources[source])

#    if region == 'blue':
#        vin = sources[source]['vellim'][0]
#        vout = sources[source]['velin'][0]
#    if region == 'red':
#        vin = sources[source]['velin'][1]
#        vout = sources[source]['vellim'][1]
    vlsr = sources[source]['vlsr']
    #rmsvel = sources[source]['rms'+region]

#    print (vin,vout,vlsr)

#    size = np.shape(datain[0,:,0])[0]
#    nchan = np.shape(datain[:,0,0])[0]
#    sumtab = np.zeros([nchan,1])
#    veltab, chanwidth, nchan, restfreq = SpecaxisHandler(datain,header,vlsr)

    #idxvin = np.argmin(np.abs(veltab - (vin)))
    #idxvout = np.argmin(np.abs(veltab - (vout)))
    #idxvrmsin = np.argmin(np.abs(veltab - (rmsvel[0])))
    #idxvrmsout = np.argmin(np.abs(veltab - (rmsvel[1])))


#    print ('Iteration starts now!')

    datain, header, size, veltab, chanwidth, nchan, restfreq, pixval_arcsec=FitsHandler1(source,mol)
    region_dict, momtab = CombMomentMeanReg(source,size,nchan,datain,veltab,chanwidth)
    meantab = region_dict[region]['mean']


    idxres1 = np.argmin(np.abs(veltab - (-40.5)))
    idxres2 = np.argmin(np.abs(veltab - (40.5)))

    idxrms = []

    #####

    datain_mol2, header_mol2, size_mol2, veltab_mol2, chanwidth_mol2, nchan_mol2, restfreq_mol2, pixval_arcsec_mol2=FitsHandler1(source,mol2)
    size_mol2 = np.shape(datain_mol2[0,:,0])[0]
    nchan_mol2 = np.shape(datain_mol2[:,0,0])[0]

#    r1_C18O = pf.getdata(prefix + sources[source]['field']+'/'+sources[source]['field']+'_'+'C18O_'+C18O_region+'.fits')
#    meantab_C18O, momtab_C18O = MomentMeanReg(size,nchan_C18O,datain,veltab_C18O,chanwidth_C18O,r1=r1_C18O)
    mol2_dict, momtab_mol2 = CombMomentMeanReg(source,size_mol2,nchan_mol2,datain_mol2,veltab_mol2,chanwidth_mol2,mol=mol2)
    meantab_mol2 = mol2_dict[mol2_region]['mean']
    #function to calculate the moment and mean map of the region


### Spectra !!!



    f, axarr = plt.subplots(1, 3, gridspec_kw = {'width_ratios':[1, 1 , 1]})
    f.set_figheight(5)
    f.set_figwidth(15)

#    if norm:
##        axarr[0].set_ylabel('Fraction of peak',fontsize=20)
 #       axarr[0].plot(veltab,meantab/np.max(meantab),drawstyle='steps-mid')
 #       axarr[0].plot(veltab_mol2,meantab_mol2/np.max(meantab_mol2),drawstyle='steps-mid')

 #   else:
    axarr[0].set_ylabel('Mean Intensity [K]',fontsize=20)
    axarr[0].plot(veltab,meantab,drawstyle='steps-mid')
    axarr[0].plot(veltab_mol2,meantab_mol2,drawstyle='steps-mid')
    axarr[0].set_xlim(xlim)



    axarr[0].set_xlabel('Velocity ($\\varv - \\v_{\\rmLSR}$ [km s-1]',fontsize=20)

    im1 = axarr[1].plot(veltab,meantab/np.max(meantab),drawstyle='steps-mid')
    im1 = axarr[1].plot(veltab_mol2,meantab_mol2/np.max(meantab_mol2),drawstyle='steps-mid')
    axarr[1].set_ylabel('Fraction of peak',fontsize=20)
    axarr[1].set_xlim(xlim)

#    im1 = axarr[1].imshow(veltab,meantab/meantab_C18O)

    print (chanwidth_mol2, chanwidth)
    print (veltab_mol2[-1])

    if (np.abs(chanwidth_mol2 -chanwidth)<0.01):
        ratio_idxin = np.argmin(np.abs(veltab - (veltab_mol2[0])))
        ratio_idxout = np.argmin(np.abs(veltab - (veltab_mol2[-1])))
        ratio_tab = meantab_mol2/meantab[ratio_idxin:ratio_idxout+1]
        axarr[2].plot(veltab_mol2,ratio_tab)
        axarr[2].set_xlim(xlim)
        axarr[2].set_ylim([-0.5,2.0])
    print ('Done')


#    axarr[0].add_patch(
#        patches.Rectangle(
#        (sources[source]['ehvlim'][0],axarr[0].get_ylim()[0]),   # (x,y)
#        sources[source]['ihvlim'][0]-sources[source]['ehvlim'][0],          # width
#        axarr[0].get_ylim()[1]-axarr[0].get_ylim()[0],          # height
#        alpha=0.2,
#        color='r'
#        )
#    )

    axarr[0].plot([sources[source]['ehvlim'][0],sources[source]['ehvlim'][0]],axarr[0].get_ylim(),'--',color='r')
    axarr[0].plot([sources[source]['ihvlim'][0],sources[source]['ihvlim'][0]],axarr[0].get_ylim(),'--',color='g')
    axarr[0].plot([sources[source]['velin'][0],sources[source]['velin'][0]],axarr[0].get_ylim(),'--',color='b')
    axarr[0].plot([sources[source]['ehvlim'][1],sources[source]['ehvlim'][1]],axarr[0].get_ylim(),'--',color='r')
    axarr[0].plot([sources[source]['ihvlim'][1],sources[source]['ihvlim'][1]],axarr[0].get_ylim(),'--',color='g')
    axarr[0].plot([sources[source]['velin'][1],sources[source]['velin'][1]],axarr[0].get_ylim(),'--',color='b')

    axarr[1].plot([sources[source]['ehvlim'][0],sources[source]['ehvlim'][0]],axarr[1].get_ylim(),'--',color='r')
    axarr[1].plot([sources[source]['ihvlim'][0],sources[source]['ihvlim'][0]],axarr[1].get_ylim(),'--',color='g')
    axarr[1].plot([sources[source]['velin'][0],sources[source]['velin'][0]],axarr[1].get_ylim(),'--',color='b')
    axarr[1].plot([sources[source]['ehvlim'][1],sources[source]['ehvlim'][1]],axarr[1].get_ylim(),'--',color='r')
    axarr[1].plot([sources[source]['ihvlim'][1],sources[source]['ihvlim'][1]],axarr[1].get_ylim(),'--',color='g')
    axarr[1].plot([sources[source]['velin'][1],sources[source]['velin'][1]],axarr[1].get_ylim(),'--',color='b')

    axarr[2].plot([sources[source]['ehvlim'][0],sources[source]['ehvlim'][0]],axarr[2].get_ylim(),'--',color='r')
    axarr[2].plot([sources[source]['ihvlim'][0],sources[source]['ihvlim'][0]],axarr[2].get_ylim(),'--',color='g')
    axarr[2].plot([sources[source]['velin'][0],sources[source]['velin'][0]],axarr[2].get_ylim(),'--',color='b')
    axarr[2].plot([sources[source]['ehvlim'][1],sources[source]['ehvlim'][1]],axarr[2].get_ylim(),'--',color='r')
    axarr[2].plot([sources[source]['ihvlim'][1],sources[source]['ihvlim'][1]],axarr[2].get_ylim(),'--',color='g')
    axarr[2].plot([sources[source]['velin'][1],sources[source]['velin'][1]],axarr[2].get_ylim(),'--',color='b')



    outname = ('CO_'+mol+'_spectra')
    plt.savefig(outname+'.eps',format='eps')
    plt.show()
    plt.close()
#### Maps
    f2, axarr = plt.subplots(1, 3, gridspec_kw = {'width_ratios':[1, 1 , 1]})
    f2.set_figheight(5)
    f2.set_figwidth(15)
    axarr[0].imshow(momtab,origin='lower')
    axarr[0].contour(r1_red, origin='lower', color='black', vmin=0, vmax=1,levels=[0,1])
    axarr[0].contour(r1_blue, origin='lower', color='black', vmin=0, vmax=1,levels=[0,1])
    axarr[1].imshow(momtab_mol2,origin='lower')
    axarr[1].contour(r1_C18O_red, origin='lower', color='black', vmin=0, vmax=1,levels=[0,1])
    axarr[1].contour(r1_C18O_blue, origin='lower', color='black', vmin=0, vmax=1,levels=[0,1])


    print ('Done')
    outname = ('CO_'+mol2+'_spectra')
    plt.savefig(outname+'.eps',format='eps')
    plt.show()
    plt.close()










# In[9]:


def SimplePlotDouble(source,region,mol='CO',mol2_region='core',mol2='C18O',norm=False,xlim=[-15,15]):


    with open('sources.json', 'r') as f:
        sources = json.load(f)

    prefix = '../../Projects/'


    #r1 = pf.getdata(prefix + sources[source]['field']+'/'+source+'_'+region+'.fits')
    r1_C18O_red = pf.getdata(prefix + sources[source]['field']+'/'+sources[source]['field']+'_'+'C18O_red'+'_B6.fits')
    r1_C18O_blue = pf.getdata(prefix + sources[source]['field']+'/'+sources[source]['field']+'_'+'C18O_blue'+'_B6.fits')
#    r1_core = pf.getdata(prefix + sources[source]['field']+'/'+source+'_'+'core'+'.fits')
    r1_red = pf.getdata(prefix + sources[source]['field']+'/'+source+'_'+'red'+'_B6.fits')
    r1_blue = pf.getdata(prefix + sources[source]['field']+'/'+source+'_'+'blue'+'_B6.fits')
#    print (sources[source])

#    if region == 'blue':
#        vin = sources[source]['vellim'][0]
#        vout = sources[source]['velin'][0]
#    if region == 'red':
#        vin = sources[source]['velin'][1]
#        vout = sources[source]['vellim'][1]
    vlsr = sources[source]['vlsr']
    #rmsvel = sources[source]['rms'+region]

#    print (vin,vout,vlsr)

#    size = np.shape(datain[0,:,0])[0]
#    nchan = np.shape(datain[:,0,0])[0]
#    sumtab = np.zeros([nchan,1])
#    veltab, chanwidth, nchan, restfreq = SpecaxisHandler(datain,header,vlsr)

    #idxvin = np.argmin(np.abs(veltab - (vin)))
    #idxvout = np.argmin(np.abs(veltab - (vout)))
    #idxvrmsin = np.argmin(np.abs(veltab - (rmsvel[0])))
    #idxvrmsout = np.argmin(np.abs(veltab - (rmsvel[1])))


#    print ('Iteration starts now!')

    datain, header, size, veltab, chanwidth, nchan, restfreq, pixval_arcsec=FitsHandler1(source,mol)
    region_dict= CombMeanReg(source,size,nchan,datain,veltab,chanwidth)
    meantab = region_dict[region]['mean']


    idxres1 = np.argmin(np.abs(veltab - (-40.5)))
    idxres2 = np.argmin(np.abs(veltab - (40.5)))

    idxrms = []

    #####

    datain_mol2, header_mol2, size_mol2, veltab_mol2, chanwidth_mol2, nchan_mol2, restfreq_mol2, pixval_arcsec_mol2=FitsHandler1(source,mol2)
    size_mol2 = np.shape(datain_mol2[0,:,0])[0]
    nchan_mol2 = np.shape(datain_mol2[:,0,0])[0]

#    r1_C18O = pf.getdata(prefix + sources[source]['field']+'/'+sources[source]['field']+'_'+'C18O_'+C18O_region+'.fits')
#    meantab_C18O, momtab_C18O = MomentMeanReg(size,nchan_C18O,datain,veltab_C18O,chanwidth_C18O,r1=r1_C18O)
    mol2_dict = CombMeanReg(source,size_mol2,nchan_mol2,datain_mol2,veltab_mol2,chanwidth_mol2,mol=mol2)
    meantab_mol2 = mol2_dict[mol2_region]['mean']
    #function to calculate the moment and mean map of the region


### Spectra !!!



    f, axarr = plt.subplots(1, 2, gridspec_kw = {'width_ratios':[1, 1]})
    f.set_figheight(5)
    f.set_figwidth(15)



#    if norm:
##        axarr[0].set_ylabel('Fraction of peak',fontsize=20)
 #       axarr[0].plot(veltab,meantab/np.max(meantab),drawstyle='steps-mid')
 #       axarr[0].plot(veltab_mol2,meantab_mol2/np.max(meantab_mol2),drawstyle='steps-mid')

 #   else:
    axarr[0].set_ylabel('Mean Intensity [K]',fontsize=20)
    axarr[0].plot(veltab,meantab,drawstyle='steps-mid')
    axarr[0].plot(veltab_mol2,meantab_mol2,drawstyle='steps-mid')

    axarr[0].set_ylim(-0.1,np.max(meantab)+0.1*np.max(meantab))

    axarr[0].set_xlim(xlim)



    axarr[0].set_xlabel('Velocity [km s-1]',fontsize=20)

    im1 = axarr[1].plot(veltab,meantab/np.max(meantab),drawstyle='steps-mid')
    im1 = axarr[1].plot(veltab_mol2,meantab_mol2/np.max(meantab_mol2),drawstyle='steps-mid')
    axarr[1].set_ylabel('Fraction of peak',fontsize=20)
    axarr[1].set_xlim(xlim)

#    im1 = axarr[1].imshow(veltab,meantab/meantab_C18O)

    print (chanwidth_mol2, chanwidth)
    print (veltab_mol2[-1])



#    axarr[0].add_patch(
#        patches.Rectangle(
#        (sources[source]['ehvlim'][0],axarr[0].get_ylim()[0]),   # (x,y)
#        sources[source]['ihvlim'][0]-sources[source]['ehvlim'][0],          # width
#        axarr[0].get_ylim()[1]-axarr[0].get_ylim()[0],          # height
#        alpha=0.2,
#        color='r'
#        )
#    )

    axarr[0].plot([sources[source]['ehvlim'][0],sources[source]['ehvlim'][0]],axarr[0].get_ylim(),'--',color='r')
    axarr[0].plot([sources[source]['ihvlim'][0],sources[source]['ihvlim'][0]],axarr[0].get_ylim(),'--',color='g')
    axarr[0].plot([sources[source]['velin'][0],sources[source]['velin'][0]],axarr[0].get_ylim(),'--',color='b')
    axarr[0].plot([sources[source]['ehvlim'][1],sources[source]['ehvlim'][1]],axarr[0].get_ylim(),'--',color='r')
    axarr[0].plot([sources[source]['ihvlim'][1],sources[source]['ihvlim'][1]],axarr[0].get_ylim(),'--',color='g')
    axarr[0].plot([sources[source]['velin'][1],sources[source]['velin'][1]],axarr[0].get_ylim(),'--',color='b')

    axarr[1].plot([sources[source]['ehvlim'][0],sources[source]['ehvlim'][0]],axarr[1].get_ylim(),'--',color='r')
    axarr[1].plot([sources[source]['ihvlim'][0],sources[source]['ihvlim'][0]],axarr[1].get_ylim(),'--',color='g')
    axarr[1].plot([sources[source]['velin'][0],sources[source]['velin'][0]],axarr[1].get_ylim(),'--',color='b')
    axarr[1].plot([sources[source]['ehvlim'][1],sources[source]['ehvlim'][1]],axarr[1].get_ylim(),'--',color='r')
    axarr[1].plot([sources[source]['ihvlim'][1],sources[source]['ihvlim'][1]],axarr[1].get_ylim(),'--',color='g')
    axarr[1].plot([sources[source]['velin'][1],sources[source]['velin'][1]],axarr[1].get_ylim(),'--',color='b')

    outname = ('CO_'+mol+'_spectra')
    plt.savefig(outname+'.eps',format='eps')
    plt.show()
    plt.close()











# In[5]:


def MomentMeanReg(size,nchan,datain,veltab,chanwidth,r1):
    tab = np.zeros([size,size])
    meantab = np.zeros([nchan,1])
    sumtab = np.zeros([nchan,1])

#    print (size,nchan,chanwidth)

    idxvin = np.argmin(np.abs(veltab - (-5)))
    idxvout = np.argmin(np.abs(veltab - (5)))

    for m in range(size):
        for n in range(size):
            if np.isnan(datain[0,m,n]):
                continue
            if datain[0,m,n] == 0:
                continue
            tab[m,n] = np.sum(datain[idxvin:idxvout,m,n])*chanwidth

    region_idx = np.where(r1==1)


    for ix in range(len(veltab)):
        if r1 == [10,10]:
            meantab[ix] = np.nanmean(datain[ix,:,:])
        else:
            meantab[ix] = np.nanmean(datain[ix,region_idx[0],region_idx[1]])
    for ix in range(len(veltab)):
        sumtab[ix] = np.nansum(datain[ix,:,:])


    return meantab, tab



# In[6]:


def CombMomentMeanReg(source,size,nchan,datain,veltab,chanwidth,mol='CO'):

    regions = {'red':{},'blue':{},'r+b':{}}

    with open('sources.json', 'r') as f:
        sources = json.load(f)



    idxvin = np.argmin(np.abs(veltab - (-10)))
    idxvout = np.argmin(np.abs(veltab - (10)))

    tab = np.zeros([size,size])
    for m in range(size):
        for n in range(size):
                if np.isnan(datain[0,m,n]):
                    continue
                if datain[0,m,n] == 0:
                    continue
                tab[m,n] = np.sum(datain[idxvin:idxvout,m,n])*chanwidth


    for region in regions:
        if region != 'red' and region != 'blue':
#            print('break free!')
            continue
#        print (region)


        prefix = '../../Projects/'


        if mol == 'C18O':
            r1 = pf.getdata(prefix + sources[source]['field']+'/'+sources[source]['field']+'_'+'C18O_'+region+'_B6.fits')
        elif mol == 'HCN':
            r1 = pf.getdata(prefix + sources[source]['field']+'/'+source+'_'+region+'_B3.fits')
        else:
            r1 = pf.getdata(prefix + sources[source]['field']+'/'+source+'_'+region+'_B6.fits')

#        r1_core = pf.getdata(prefix + sources[source]['field']+'/'+source+'_'+'core'+'.fits')

#        print (sources[source])

#        if region == 'blue':
#            vin = sources[source]['vellim'][0]
#            vout = sources[source]['velin'][0]
#        if region == 'red':
#            vin = sources[source]['velin'][1]
#            vout = sources[source]['vellim'][1]
        vlsr = sources[source]['vlsr']
        rmsvel = sources[source]['rms'+region]



        meantab = np.zeros([nchan,1])
        sumtab = np.zeros([nchan,1])

#        print (size,nchan,chanwidth)


 #           else:
  #              tab[m,n] = np.sum(datain[idxvin:idxvout,m,n])*chanwidth


        region_idx = np.where(r1==1)
        vlsr_idx = np.argmin(np.abs(veltab - (0)))
#        print (vlsr_idx)

        for ix in range(len(veltab)):
            if r1 == [10,10]:
                meantab[ix] = np.nanmean(datain[ix,:,:])
            else:
                meantab[ix] = np.nanmean(datain[ix,region_idx[0],region_idx[1]])
        for ix in range(len(veltab)):
            sumtab[ix] = np.nansum(datain[ix,:,:])

        regions[region]['mean']=meantab
#        print ('cool here!')


#    print ('and now just sum it up!')

    regions['r+b']['mean']=np.append(regions['blue']['mean'][0:vlsr_idx],regions['red']['mean'][vlsr_idx:nchan])
#    print ('Table successfully returned')
    return regions, tab




# In[1]:


def CombMeanReg(source,size,nchan,datain,veltab,chanwidth,mol='CO'):

    regions = {'red':{},'blue':{},'r+b':{}}

    with open('sources.json', 'r') as f:
        sources = json.load(f)


    for region in regions:
        if region == 'r+b':
#            print('break free!')
            continue
#        print (region)


        prefix = '../../Projects/'


        if mol == 'C18O':
            r1 = pf.getdata(prefix + sources[source]['field']+'/'+sources[source]['field']+'_'+'C18O_'+region+'_B6.fits')
        elif mol == 'HCN':
            r1 = pf.getdata(prefix + sources[source]['field']+'/'+source+'_'+region+'_B3.fits')
        else:
            r1 = pf.getdata(prefix + sources[source]['field']+'/'+source+'_'+region+'_B6.fits')

#        r1_core = pf.getdata(prefix + sources[source]['field']+'/'+source+'_'+'core'+'.fits')

#        print (sources[source])

#        if region == 'blue':
#            vin = sources[source]['vellim'][0]
#            vout = sources[source]['velin'][0]
#        if region == 'red':
#            vin = sources[source]['velin'][1]
#            vout = sources[source]['vellim'][1]
        vlsr = sources[source]['vlsr']
        rmsvel = sources[source]['rms'+region]



        meantab = np.zeros([nchan,1])
        sumtab = np.zeros([nchan,1])

#        print (size,nchan,chanwidth)


 #           else:
  #              tab[m,n] = np.sum(datain[idxvin:idxvout,m,n])*chanwidth


        region_idx = np.where(r1==1)
        vlsr_idx = np.argmin(np.abs(veltab - (0)))
#        print (vlsr_idx)

        for ix in range(len(veltab)):
            if r1 == [10,10]:
                meantab[ix] = np.nanmean(datain[ix,:,:])
            else:
                meantab[ix] = np.nanmean(datain[ix,region_idx[0],region_idx[1]])
        for ix in range(len(veltab)):
            sumtab[ix] = np.nansum(datain[ix,:,:])

        regions[region]['mean']=meantab
#        print ('cool here!')


#    print ('and now just sum it up!')

    regions['r+b']['mean']=np.append(regions['blue']['mean'][0:vlsr_idx],regions['red']['mean'][vlsr_idx:nchan])
#    print ('Table successfully returned')
    return regions




def func1_with_moment_func(ax, figtype,source,mol,prefix, vin, vout, regcolor,
                params = None,
                autocontours_check=False,
                customcontours_check=False,
                customcontours='',
                plotcolour = False,
                sourcemask=True,
                sourcemask_color = '',
                plotcont=True,
                plotcontour=True,
                maptype='plotreg',
                custommap = '',
                lowres_out=True,
                ticklabels=False,
                ticklabels_ra=True,
                ticklabels_dec=True,
                rmsprint = False,
                rmsmask = True,
                plotmom1 = False,
                rms_flux = 3,
                calculate_moment = True,
                velregime = None,
                showregion=False,
                yrmspos = 0.5,
                cmap_vmax = 400.0):

    if sourcemask_color == '':
        sourcemask_color = str(regcolor)



    func1_start_time = time.time()

    #print("--- %s minutes --- " % ((time.time() - func1_start_time) / 60.0))


    with open('json/sources.json', 'r') as f:
       sources = json.load(f)
    with open('json/figure1_params.json', 'r') as f:
       params = json.load(f)


    if mol == 'HCN':
        band = 'Band3'
        r1_red = pf.getdata(prefix + sources[source]['field'] + '/' + source + '_' + 'red' + '_B3.fits')
        r1_blue = pf.getdata(prefix + sources[source]['field'] + '/' + source + '_' + 'blue' + '_B3.fits')
        r1_noise = pf.getdata(prefix + sources[source]['field'] + '/' + sources[source]['field'] + '_' + 'noise' + '_B3.fits')
        contimage = prefix + sources[source]['field'] + '/cont_B3/' + sources[source]['field'] + '_' + 'cont' + '.fits'  ###### field + mol
        contimage_beam = prefix + sources[source]['field'] + '/cont_B3/' + sources[source]['field'] + '_' + 'cont' + '.fits.beam'  ###### field + mol

    else:
        band = 'Band6'
        r1_red = pf.getdata(prefix + sources[source]['field'] + '/' + source + '_' + 'red' + '_B6.fits')
        r1_blue = pf.getdata(prefix + sources[source]['field'] + '/' + source + '_' + 'blue' + '_B6.fits')
        r1_noise = pf.getdata(prefix + sources[source]['field'] + '/' + sources[source]['field'] + '_' + 'noise' + '_B6.fits')
        contimage = prefix + sources[source]['field'] + '/cont/' + sources[source]['field'] + '_' + 'cont' + '.fits'  ###### field + mol
        contimage_beam =  prefix + sources[source]['field'] + '/cont/' + sources[source]['field'] + '_' + 'cont' + '.beam'  ###### field + mol

    image = prefix + sources[source]['field'] + '/' + sources[source]['field'] + '_' + mol + '.fits'  ###### field + mol
    w = WCS(contimage)
    #w = w.dropaxis(3)  # Remove the Stokes axis
    w = w.dropaxis(2)  # and spectral axis

    datain, header = pf.getdata(image, header=True)
    print(np.shape(datain))
    datain = datain[0, :, :, :]

    size = np.shape(datain[0,0,:])

    data_continuum, header_continuum = pf.getdata(contimage,header=True)
    data_continuum = data_continuum[0, 0, :, :]

    dataset_K, header, size, velocity, chanwidth, nchan, restfreq, pixval_arcsec,w = FitsHandler1(source, mol)

    datain = dataset_K
    veltab = velocity




    xc_world = sources[source]['coords']['ra']
    yc_world = sources[source]['coords']['dec']


    if maptype == 'custom':
        xi_world, xo_world, yi_world, yo_world = custommap
    else:
        xi_world = sources[source][maptype]['xi']
        xo_world = sources[source][maptype]['xo']
        yi_world = sources[source][maptype]['yi']
        yo_world = sources[source][maptype]['yo']





#    f = plt.figure(figsize=(6,6))

    cmap = mpl.cm.Greys
    cmap.set_under('white')
    cmap.set_over('black')
    cmap.set_bad('white')

    #print (source, band)
    #print (params[source])
    cont_vmin = params[source]['cont'][band]['vmin']
    cont_vmax = params[source]['cont'][band]['vmax']
    cont_stretch = params[source]['cont'][band]['stretch']

    #print(cont_vmin, cont_vmax, cont_stretch)


    if plotcont:
        norm = ImageNormalize (data_continuum, vmin = cont_vmin, vmax= cont_vmax, stretch=LogStretch(cont_stretch))

        beam_table_cont = \
            np.loadtxt(contimage_beam)
        pixval_x_cont, pixval_arcsec_cont, beam_cont_area_tab, beam_cont_a, beam_cont_b, beam_cont_pa\
            = BeamHandler(beam_table_cont, header_continuum)

        ax.imshow(data_continuum, norm = norm, vmin = cont_vmin, vmax= cont_vmax, cmap=cmap)
        rms_cont = np.nanstd(data_continuum)


    momtab = np.zeros([size, size])
    mom1tab = np.zeros([size, size])
    nocut_mom1tab = np.zeros([size, size])
    nocut_momtab = np.zeros([size, size])

    xi_pix, yi_pix = w.all_world2pix(xi_world, yi_world, 0)  # Find pixel correspondig to ra,dec
    xo_pix, yo_pix = w.all_world2pix(xo_world, yo_world, 0)  # Find pixel correspondig to ra,dec

    # if mol != 'HCN' :
    #    ax.set_xlim(xi_pix, xo_pix)
    #    ax.set_ylim(yi_pix, yo_pix)
    ax.set_xlim(xi_pix, xo_pix)
    ax.set_ylim(yi_pix, yo_pix)

    if lowres_out:
        if mol == 'SiO':
            if vout>39.5: vout=39.5
            if vin<-39.5: vin=-39.5
        if mol == 'H2CO':
                if vin < -26: vin = -26

    idxvin = np.argmin(np.abs(veltab - (vin)))
    idxvout = np.argmin(np.abs(veltab - (vout)))

    print ('vin, vout:', vin, vout)


    vrms_in = sources[source][mol]['vrms_'+sourcemask_color][0]
    vrms_out = sources[source][mol]['vrms_'+sourcemask_color][1]
    print ('vrms_in', vrms_in)
    print ('vrms_out', vrms_out)


    idxvrmsin = np.argmin(np.abs(veltab - (vrms_in)))
    idxvrmsout = np.argmin(np.abs(veltab - (vrms_out)))

    noise_idx = np.where(r1_noise == 1)


    if showregion:
        ax.contour(r1_red, levels=[1],
                   colors='red', linewidths=1, transform=ax.get_transform(w))
        ax.contour(r1_blue, levels=[1],
               colors='blue', linewidths=1, transform=ax.get_transform(w))

    if calculate_moment:

        if figtype == 'fig2':
            print('Calculating moment map without threshold for %s %s' % (velregime, mol))
        if figtype == 'fig1':
            print('Calculating moment map without threshold for %s' % (mol))
        print("--- %s minutes --- " % ((time.time() - func1_start_time) / 60.0))
        for m in range(size):
            for n in range(size):
                if np.isnan(datain[0, m, n]):
                    continue
                if datain[0, m, n] == 0:
                    continue
                nocut_momtab[m, n] = np.sum(datain[idxvin:idxvout, m, n]) * chanwidth
                for k in range(len(datain[idxvin:idxvout, m, n])):
                    nocut_mom1tab[m, n] = datain[idxvin + k, m, n] * veltab[idxvin + k] * chanwidth + nocut_mom1tab[m, n]
                nocut_mom1tab[m, n] = nocut_mom1tab[m, n] / nocut_momtab[m, n]
        print('Moment map maximum without threshold for %s' % (mol))
        print(("--- %.2f K km s-1 --- "%(np.nanmax(nocut_momtab))))

    # print ('Done')
        if rmsmask:
            if sourcemask_color == 'red':
                r1 = r1_red
            if sourcemask_color == 'blue':
                r1 = r1_blue
            momtab,mom1tab=moment_map(source, mol, vin, vout, datain, header,
                       sourcemask_color, rms=5, plot=True,
                       r1=r1, sigma_cutoff=True, rmsmask=True,
                       rms_flux=5,
                       mom1mask=True, mom1cutoff=3, velreg=sourcemask_color,for_func1=True)
            print('Using moment map calculated without threshold')
        else:
            momtab = nocut_momtab
            mom1tab = nocut_mom1tab

    # print (mom0[m,n],rmsval,chanwidth






        #print("--- %s minutes --- " % ((time.time() - func1_start_time) / 60.0))

        if mol in params[source]:
            if 'rms' in params[source][mol][sourcemask_color]:
                momrms = params[source][mol][sourcemask_color]['rms']
            else:
                momrms = np.std(momtab[noise_idx])
        else:
            momrms = np.std(momtab[noise_idx])


# if momrms == 0: continue

    nocut_rms = np.nanstd(nocut_momtab[noise_idx])

    momrms = nocut_rms

    print ('-----momrms------: ',momrms)



    if plotmom1:
        reg_mom1 = np.where(momtab>3*momrms)
        reg_mom1_tab = np.zeros([size,size])
        reg_mom1_tab[reg_mom1] = 1

        mom1tab = mom1tab * reg_mom1_tab

        cmap = mpl.cm.seismic
        cmap.set_under('white', alpha=0)
        cmap.set_over('white')
        cmap.set_bad('white')
        im = ax.imshow(mom1tab, cmap=cmap, vmin=vin,vmax=vout)

        plt.colorbar(im, orientation='vertical')


    if plotcontour:

        print ('maxval before region', np.nanmax(momtab))


        if sourcemask:
            if sourcemask_color == 'red':
                momtab = momtab * r1_red
            elif sourcemask_color == 'blue':
                momtab = momtab * r1_blue
        maxval = np.nanmax(momtab)
        nocut_maxval = np.nanmax(nocut_momtab)

        print ('maxval', maxval)
        print ('maxval without threshold', nocut_maxval)



        #modifier for too high rms on the blueshifted Emb8N
        #if source == 'Emb8N':
        #    if sourcemask_color == 'blue':
        #        if mol != 'SiO':
        #            momrms = momrms/2.0


    #autocontours_check = True
    #autocontours = np.array([4, 5, 9, 15, 18, 30, 40, 50, 60, 80, 100])
        #autocontours = np.linspace(momrms*3,maxval,10)/momrms
        if 'levels_'+regcolor in sources[source][mol]:
            autocontours = np.array(sources[source][mol]['levels_'+regcolor])*momrms
        else:
            autocontours = np.array([3,6,9,15,20,40,60,80,100])*momrms
        print (autocontours)
    #print(np.linspace(momrms,maxval,10)/momrms)

        if autocontours_check:
            contours = autocontours
        elif customcontours_check:
            contours = customcontours
        else:
            if mol in params[source]:
                contours =  np.array(params[source][mol][regcolor]['contours'])
            else:
                contours = autocontours

    # axarr[plotidx].contour(momtab,levels=[5,8,10,20,80,120,150],colors=regcolor)
        if contours[0]<maxval:



            if regcolor == 'red':
                if figtype == 'fig1':
                    xrmspos = 0.05
                    if source == 'SMM1d': xrmspos = 0.75
                    yrmspos = 0.55
                if figtype == 'fig2':
                    xrmspos = 0.05
                    yrmspos = 0.55

                if rmsprint:
                    ax.text(xrmspos, yrmspos, r'' + str(round(momrms, 2)), color='#DF1420',
                                transform=ax.transAxes,
                                bbox={'facecolor': 'white', 'alpha': 0.99, 'edgecolor': 'white'})
                ax.contour(momtab, levels=contours,transform=ax.get_transform(w),
                               colors='black',linewidths=1.0)
                ax.contour(momtab, levels=contours,transform=ax.get_transform(w),
                               colors='#DF1420',linewidths=0.8)


            #fontsize=16
            elif regcolor == 'blue':

                if figtype == 'fig1' or figtype == 'fig1_custom':
                    xrmspos = 0.05
                    if source == 'SMM1d': xrmspos = 0.75
                    yrmspos = 0.65
                if figtype == 'fig2':
                    xrmspos = 0.05
                    yrmspos = 0.35


                if rmsprint:
                    ax.text(xrmspos, yrmspos, r'' + str(round(momrms, 2)), color='#2846EE',
                                transform=ax.transAxes,
                                bbox={'facecolor': 'white', 'alpha': 0.99, 'edgecolor': 'white'})
                ax.contour(momtab, levels=contours,transform=ax.get_transform(w),
                               colors='black',linewidths=1.0)
                ax.contour(momtab, levels=contours,transform=ax.get_transform(w),
                               colors='#2846EE',linewidths=0.8)

            elif regcolor == 'black':
                if rmsprint:
                    ax.text(0.05, 0.65, r'' + str(round(momrms, 2)), color='#DF1420',
                                transform=ax.transAxes,fontsize=16,
                                bbox={'facecolor': 'white', 'alpha': 0.99, 'edgecolor': 'white'})
                ax.contour(momtab, levels=contours,transform=ax.get_transform(w),
                               colors='black',linewidths=1.0)

            else:
                if rmsprint:
                    ax.text(0.05, yrmspos, r'' + str(round(momrms, 2)), color=regcolor,
                            transform=ax.transAxes, fontsize=16,
                            bbox={'facecolor': 'white', 'alpha': 0.99, 'edgecolor': 'white'})
                ax.contour(momtab, levels=contours,transform=ax.get_transform(w),
                               colors='black',linewidths=1.2)
                ax.contour(momtab, levels=contours, transform=ax.get_transform(w),
                               colors=regcolor,linewidths=1.0)

        if mol == 'H2CO':
            mollabel = 'H$_2$CO'
        else:
            mollabel = mol

        if figtype == 'fig1':
            ax.text(0.09, 0.76, mollabel, color='black',
                            transform=ax.transAxes,fontsize=22,
                            bbox={'facecolor': 'white', 'alpha': 0.99, 'edgecolor': 'white'})
        if figtype == 'fig2':
            ax.text(0.07, 0.76, velregime, color='black',
                            transform=ax.transAxes,fontsize=22,
                            bbox={'facecolor': 'white', 'alpha': 0.99, 'edgecolor': 'white'})


        ############ if you want line beam

        if (figtype == 'fig1' and mol == 'H2CO') or (figtype == 'fig2' and velregime == 'EHV') or (figtype == 'fig1_custom'):
            beam_table_line= \
                np.loadtxt(prefix + sources[source]['field'] + '/' + sources[source]['field'] + '_' + mol + '.beam')
            pixval_x_line, pixval_arcsec_line, beam_line_area_tab, beam_line_a, beam_line_b, beam_line_pa \
                = BeamHandler(beam_table_line, header)


            beam2 = patches.Ellipse((xi_pix + beam_shift_x, yi_pix + beam_shift_y), width=beam_line_a / pixval_arcsec_line,
                                height=beam_line_b /pixval_arcsec_line, angle=beam_line_pa, facecolor='red',
                                edgecolor='black')

            print(ax.get_xlim()[0] + 10, ax.get_ylim()[0] + 10)
            ax.add_patch(beam2)

            print ('beam size', mol ,beam_line_a, beam_line_b )

        if source == 'Emb8N':
            sourcelabel = 'Ser-emb 8 (N)'
        if source == 'S68N':
            sourcelabel = 'S68N'
        if source == 'SMM1a':
            sourcelabel = 'SMM1-a'
        if source == 'SMM1b':
            sourcelabel = 'SMM1-b'
        if source == 'SMM1d':
            sourcelabel = 'SMM1-d'

    #if figtype == 'fig1':
#
 #       if source == 'SMM1a':
  #          plt.subplots_adjust(top=0.98, bottom=0.06, left=0.07, right=0.95, hspace=0.0,
   #                         wspace=0.0)
    #    if source == 'Emb8N':
     #       plt.subplots_adjust(top=0.78, bottom=0.15, left=0.10, right=0.99, hspace=0.0,
      #                     wspace=0.0)
       # if source == 'S68N':
        #        plt.subplots_adjust(top=0.98, bottom=0.15, left=0.12, right=0.95, hspace=0.0,
        #                            wspace=0.0)
        #else:
        #    plt.subplots_adjust(top=0.98, bottom=0.06, left=0.07, right=0.95, hspace=0.0,
        #                    wspace=0.0)

    if plotcolour:


        if sourcemask:
            if sourcemask_color == 'red':
                momtab = momtab * r1_red
            elif sourcemask_color == 'blue':
                momtab = momtab * r1_blue

        # axarr[plotidx].contour(momtab,levels=[5,8,10,20,80,120,150],colors=regcolor)
        print('plotting colours, whatsup?')
        if regcolor == 'red':
            cmap = plt.cm.Reds
            cmap.set_under(color='white', alpha=0.1)
            imnorm = ImageNormalize(momtab,vmin=3*momrms, vmax=30*momrms, stretch=LogStretch)

            #ax.imshow(momtab, cmap=cmap, vmin=3*momrms, vmax=30*momrms)
            ax.imshow(momtab, cmap=cmap, vmin=3*momrms, vmax=np.max(momtab))

        if regcolor == 'blue':
            cmap = plt.cm.Blues
            imnorm = ImageNormalize(momtab,vmin=3*momrms, vmax=30*momrms, stretch=LogStretch)
            cmap.set_under(color='white', alpha=0.1)

            ax.imshow(momtab, cmap=cmap, vmin=3*momrms, vmax=np.max(momtab))
        if regcolor == 'special':
            cmap = plt.cm.inferno
            cmap.set_under(color='black', alpha=0.3)

            ax.imshow(momtab, cmap=cmap)
 #           ax.contourf(momtab, levels=np.array([3,6,9,12,15,20,30,50])*momrms)

    if figtype == 'fig1':

        if mol == 'CO':
            if source == 'S68N':
                labelx=0.85
                labely=1.04
            elif source == 'Emb8N':
                labelx = 0.85
                labely = 1.04
            elif source ==  'SMM1a':
                labelx = 0.85
                labely = 1.04
            elif source == 'SMM1b':
                labelx = 0.75
                labely = 1.04
            else:
                labelx = 0.85
                labely = 1.04
            ax.text(labelx,labely,sourcelabel,
            transform=ax.transAxes,
            color='black',
            rotation=0,
            fontsize=32,
            bbox={'facecolor': 'white', 'alpha': 0.99, 'edgecolor': 'white'})

    if figtype == 'fig2':
            if velregime == 'fast':
                if source == 'Emb8N':
                    labelx = 0.25
                    labely = 1.04
                else:
                    labelx = 0.4
                    labely = 1.04
                #ax.text(labelx, labely, sourcelabel, transform=ax.transAxes, color='black', rotation=0,
                #        fontsize=32,
                #        bbox={'facecolor': 'white', 'alpha': 0.99, 'edgecolor': 'white'})

    # if regcolor == 'red':
    ##        axarr[plotidx].contourf(momtab,levels= momrms*np.array([4,5,9,15,18,30,40,50,60,80,100]),cmap='Reds')
    #    else:
    #        axarr[plotidx].contourf(momtab,levels= momrms*np.array([4,5,9,15,18,30,40,50,60,80,100]),cmap='Blues')

    #    axarr[plotidx].set_xlim([xcen-xoff,xcen+xoff])
    #    axarr[plotidx].set_ylim([ycen-yoff,ycen+yoff])



    #if source == 'Emb8N':
    #    if mol == 'H2CO':
    #        ax.set_title('H$_2$CO')
    #    else:
    #        ax.set_title(mol)

    #print ('size ratio:',(np.abs((xo_pix-xi_pix)/(yo_pix-yi_pix))))
    #print ('x size:', np.abs(xo_pix-xi_pix))
    #print ('y size:', np.abs(yo_pix-yi_pix))


    size_ratio = (np.abs((xo_pix-xi_pix)/(yo_pix-yi_pix)))
    x_size = np.abs(xo_pix-xi_pix)
    y_size = np.abs(yo_pix-yi_pix)


    coord_style = 'offset'
    zoom_in_pixels = zoom/pixval_arcsec_line


    if coord_style == 'RaDec':
        lon = ax.coords['ra']
        lat = ax.coords['dec']
        lon.display_minor_ticks(True)
        lat.display_minor_ticks(True)
        #if source == 'SMM1d':
        lon.set_axislabel('R.A.')
        lat.set_axislabel('Decl.', minpad=-1)
        lon.set_ticklabel_visible(True)
        lon.set_major_formatter('hh:mm:ss.s')
    if coord_style == 'offset':
        print ('offset')
        ax.set_xticks(x_size/2.0-zoom_in_pixels)



    ax.set_xticklabels(ax.get_xticklabels(),fontsize=4)
    ax.coords.frame.set_linewidth(1)


    #if ticklabels==False:
    #        lon.set_ticklabel_visible(False)
    #        lat.set_ticklabel_visible(False)
    #        lon.set_axislabel('')
    #        lat.set_axislabel('')

    if ticklabels_ra==False:
            lon.set_ticklabel_visible(False)
            lon.set_axislabel('')
    if ticklabels_dec==False:
            lat.set_ticklabel_visible(False)
            lat.set_axislabel("")


    lon.set_ticks(exclude_overlapping=True, size=6)
    lat.set_ticks(exclude_overlapping=True, size=6)

    if (figtype == 'fig1' and mol == 'H2CO') or (figtype == 'fig2' and velregime == 'EHV'):
            scalebar = AnchoredSizeBar(ax.transData,
                               200, str(int(pixval_arcsec*229.887*435))+' AU', 4,
                               pad=0.5,
                               color='black',
                               frameon=False,
                               size_vertical=1)
    #                           fontproperties=fontprops)
            ax.add_artist(scalebar)

    elif (figtype == 'fig1_custom'):
            scalebar = AnchoredSizeBar(ax.transData,
                               200, str(int(pixval_arcsec*229.887*435/1.0))+' AU', 4,
                               pad=0.5,
                               color='black',
                               frameon=False,
                               size_vertical=1)
            ax.add_artist(scalebar)


    #mpl.rcParams['axes.linewidth'] = 5.0

    #rcParams["figure.figsize"] = [x_size/100.0*4.0,y_size/100.0]



    return f,ax,x_size/100.0,y_size/100.0





