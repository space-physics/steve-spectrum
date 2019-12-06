#!/usr/bin/env python
"""
python FirstSpectrum.py ~/data/2019GL083272

"""
import typing
import typing.io
import argparse
from pathlib import Path
import numpy as np
import xarray
import scipy.signal
from datetime import datetime
from matplotlib.pyplot import figure, show

import stevespectrum as splots

band_head = {'N2p1N01': (424.5, 427.9),  # (0, 1)
             'N2p1N0112': (421.6, 428.3),  # (0, 1) and (1, 2)
             'continuum': (450.0, 530.0)}  # [nm]

feature = ["picket fence", "STEVE"]


def load_spectrum(path: Path) -> xarray.DataArray:
    """ load TRex spectrum text file from
    https://doi.org/10.5281/zenodo.3552801

    data is irregularly formatted, defying usual Numpy reads. Just do it the manual way.
    """

    path = Path(path).expanduser()
    if path.is_file():
        flist = [path]
        path = path.parent
        time: typing.List[typing.Union[str, datetime]] = ["unknown"]
    else:
        flist = [
            path / "TREx_spectrograph_20180410_063045.txt",
            path / "TREx_spectrograph_20180410_064015.txt",
        ]
        time = [datetime(2018, 4, 10, 6, 30, 45), datetime(2018, 4, 10, 6, 40, 15)]
    if not path.is_dir():
        raise NotADirectoryError(
            f"{path} not found; please download and extract the data from https://doi.org/10.5281/zenodo.3552801"
        )

    wavelengths = read_flattxt(path / "wavelength_per_bin.txt", 3)

    arr = None
    for file in flist:
        if arr is None:
            arr = np.array(read_flattxt(file, 4)).reshape((-1, len(wavelengths)))
        else:
            arr = np.stack((arr, np.array(read_flattxt(file, 4)).reshape((-1, len(wavelengths)))))

    if arr.ndim == 2:
        arr = arr[None, :, :]

    # all zeros outside this elevation index range, by inspection
    igood = slice(27, 229)

    # the paper did not specify the type or parameters of the spectrum smoothing used.
    # here we use a typical Savitsky-Golay filter with arbitrary parameters
    # observe with Figure 1 that  windows_length=5, polyorder=3 has a reasonably
    # good match to the article spectrum figures w.r.t. trends and values.
    arr = scipy.signal.savgol_filter(arr, window_length=5, polyorder=3, axis=2)

    # can't have negative intensity, assuming detector bias
    arr[arr < 0] = 0

    dat = xarray.DataArray(
        data=arr,
        name="luminosity (R)",
        dims=("time", "elevation", "wavelength"),
        coords={"time": time, "elevation": range(arr.shape[1]), "wavelength": wavelengths},
    )

    dat = dat[:, igood, :]

    return dat


def read_flattxt(path: Path, skip_header: int) -> typing.List[float]:
    """ read float data from irregularly shaped text file """
    dat: typing.List[float] = []
    with path.open("r") as f:
        skip_rows(f, skip_header)
        for line in f:
            dat += list(map(float, line.split()))
    return dat


def skip_rows(f: typing.io.TextIO, rows: int):
    """ skip rows of file f """
    for _ in range(rows):
        f.readline()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("path", help="path to paper data directory")
    p.add_argument("-p", "--plots", help="plot names to plot", nargs="+")
    P = p.parse_args()

    i_el = [
        {"quiet": 100, "equatorward": 116, "feature": 119, "poleward": 122},
        {"quiet": 90, "equatorward": 99, "feature": 103, "poleward": 107},
    ]

    N = len(i_el)

    i_wl: typing.List[str] = ["427.8", "557.7", "630.0", "450..530"]

    dat = load_spectrum(P.path)

    if dat.shape[0] > N:
        raise ValueError(f"expecting no more than 2 files in {P.path}")
    # %% figure setup
    if not P.plots or "keo" in P.plots:
        fg = figure(120)
        fg.clf()
        splots.plot_keogram(dat, i_el, fg.gca())

    if not P.plots or "el" in P.plots:
        ax220, ax221, ax222 = splots.setup_elevation_plots(dat, feature, band_head['N2p1N01'], N)

    if not P.plots or "ratio" in P.plots:
        fg23 = figure(23)
        fg23.clf()
        ax23 = fg23.subplots(N, 1, sharex=True)
        ax23[-1].set_xlabel("elevation bin (unitless)")
        ax23[-1].set_ylabel("luminosity (Rayleighs)")
        fg23.tight_layout()

    # %% figure loop
    for i, d in enumerate(dat):  # each time/event
        # %% paper plot
        if not P.plots or "paper" in P.plots:
            fg1 = figure(1 + i, figsize=(12, 10))
            fg1.clf()
            axs = fg1.subplots(N, 1, sharex=True)
            fg1.suptitle(feature[i] + ": " + str(dat.time.values)[:-10])
            splots.plot_paper(d, i_el[i], i_wl, axs)

        # %% zoomed paper plot
        if not P.plots or "zoom" in P.plots:
            fg = figure(10 + i, figsize=(18, 16))
            fg.clf()
            axs = fg.subplots(N, 3)
            fg.suptitle(feature[i] + ": " + str(d.time.values)[:-10])
            splots.plot_zoom(d, i_el[i], i_wl, axs)

        # %% lines vs elevation plot
        if not P.plots or "el" in P.plots:
            splots.elevation_plots(d, feature[i], band_head, i_el[i], i_wl, ax220[i], ax221[i], ax222[i])
            ax220[0].legend()
            ax221[0].legend()
            ax222[0].legend()

        if not P.plots or "ratio" in P.plots:
            splots.plot_ratio_elevation(d, band_head['N2p1N01'], i_el[i]["feature"], ax=ax23[i])
            ax23[i].set_title(feature[i] + ": " + str(d.time.values)[:-10])
            ax23[0].legend()

    show()
