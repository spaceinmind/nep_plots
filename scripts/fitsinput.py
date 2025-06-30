def main():
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.patches as mpatches
    from matplotlib.patches import Ellipse
    from matplotlib.legend_handler import HandlerPatch
    from astropy.io import fits
    from astropy.wcs import WCS
    import numpy as np

    class HandlerEllipse(HandlerPatch):
        def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
            center = 0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent
            p = mpatches.Ellipse(xy=center, width=width + xdescent, height=height + ydescent)
            self.update_prop(p, orig_handle, legend)
            p.set_transform(trans)
            return [p]

    n_font = 14
    n_font_legend = 7

    # Central position (J2000)
    ra_c, dec_c = 267.6999817, 66.5685883

    # Source magnitudes
    mags = {
        'HSC i-band': '19.0',
        'HSC z-band': '18.9',
        'SNUCAM I-band': '18.7',
        'MegaCam i-band': '17.8',
        'AKARI N2': '19.6',
        'AKARI N3': '19.8',
        'AKARI N4': '20.5'
    }

    # FITS file info: (filename, hdu_index, width_in_pixels, scale_factor, vmax, label)
    fits_info = [
        ('/fred/oz002/users/sho/nep/calexp-HSC-I-18114-3,6.fits', 1, 220, 1000, 3000.0, 'HSC i-band'),
        ('/fred/oz002/users/sho/nep/calexp-HSC-Z-18114-3,6.fits', 1, 220, 1000, 3000.0, 'HSC z-band'),
        ('/fred/oz002/users/sho/nep/westI.fits', 0, 140, 10, 100.0, 'SNUCAM I-band'),
        ('/fred/oz002/users/sho/nep/np5iswarp_st2.fits', 0, 200, 10.0, 2000, 'MegaCam i-band'),
        ('/fred/oz002/users/sho/nep/NEPW_N2_mosaic_W.fits', 0, 25, 500.0, 1200.0, 'AKARI N2'),
        ('/fred/oz002/users/sho/nep/NEPW_N3_mosaic_W.fits', 0, 25, 500.0, 1400.0, 'AKARI N3'),
        ('/fred/oz002/users/sho/nep/NEPW_N4_mosaic_W.fits', 0, 25, 500.0, 1600.0, 'AKARI N4'),
    ]

    # Load catalog sources (RA/DEC in degrees)
    def load_catalog(path):
        data = fits.open(path)[1].data
        return data['coord_ra'] / (2 * np.pi) * 360, data['coord_dec'] / (2 * np.pi) * 360

    E_ra_i, E_dec_i = load_catalog('/fred/oz002/users/sho/nep/forced_src-HSC-I-18114-3,6.fits')
    E_ra_z, E_dec_z = load_catalog('/fred/oz002/users/sho/nep/forced_src-HSC-Z-18114-3,6.fits')

    def trim_catalog(ra, dec, ra_c, dec_c, width=0.5 / 60):
        mask = (ra > ra_c - width) & (ra < ra_c + width) & (dec > dec_c - width) & (dec < dec_c + width)
        return ra[mask], dec[mask]

    E_ra_trim_i, E_dec_trim_i = trim_catalog(E_ra_i, E_dec_i, ra_c, dec_c)
    E_ra_trim_z, E_dec_trim_z = trim_catalog(E_ra_z, E_dec_z, ra_c, dec_c)

    fig = plt.figure(figsize=(50, 50))
    grid = plt.GridSpec(9, 9, hspace=0.0, wspace=0.0)

    for idx, (fpath, hdu_idx, width, scale, vmax, label) in enumerate(fits_info):
        hdulist = fits.open(fpath)
        hdu = hdulist[hdu_idx]
        header = hdu.header
        wcs = WCS(header)
        px, py = wcs.wcs_world2pix(ra_c, dec_c, 0)
        xmin, xmax = px - width / 2, px + width / 2
        ymin, ymax = py - width / 2, py + width / 2

        row = (idx // 3) * 3
        col = (idx % 3) * 3
        ax = fig.add_subplot(grid[row:row + 3, col:col + 3], projection=wcs)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.text(xmin + width * 0.05, ymin + width * 0.85, label, fontsize=n_font, fontweight='bold')
        ax.text(xmin + width * 0.05, ymin + width * 0.80, mags[label], fontsize=n_font, fontweight='bold')

        ellipse = Ellipse((ra_c, dec_c), width=2.52 / 3600 / np.cos(np.deg2rad(dec_c)),
                          height=2.52 / 3600, edgecolor='red', facecolor='none', linewidth=3,
                          transform=ax.get_transform('fk5'))
        ax.add_patch(ellipse)

        catalog_ra, catalog_dec = (E_ra_trim_i, E_dec_trim_i) if 'i-band' in label else (E_ra_trim_z, E_dec_trim_z)
        for ra, dec in zip(catalog_ra, catalog_dec):
            obj = Ellipse((ra, dec), width=2.52 / 3600 / np.cos(np.deg2rad(dec)), height=2.52 / 3600,
                          edgecolor='green', facecolor='none', linewidth=1.5, transform=ax.get_transform('fk5'))
            ax.add_patch(obj)

        ax.imshow(hdu.data * scale, cmap=cm.Blues, vmin=0., vmax=vmax)

        if idx in [3, 4, 5, 6]:
            ax.set_xlabel('RA (J2000.0)', fontsize=n_font)
            ax.tick_params(axis='x', which='both', bottom=True, labelbottom=True, labelsize=n_font_legend)
        else:
            ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False, labelsize=n_font_legend)

        if idx in [0, 3, 6]:
            ax.set_ylabel('Dec (J2000.0)', fontsize=n_font)
            ax.tick_params(axis='y', which='both', left=True, labelleft=True, labelsize=n_font_legend)
        else:
            ax.tick_params(axis='y', which='both', left=False, labelleft=False, labelsize=n_font_legend)

    plt.show()

if __name__ == "__main__":
    main()