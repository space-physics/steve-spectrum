#!/usr/bin/env python
"""
python FirstSpectrum.py ~/data/2019GL083272

Four main sources of discrepancy from paper figures:

1. picking the same elevation bins
2. flipping poleward / equatorward elevation direction
3. spectral smoothing
4. bin width
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

IndexElevation = typing.Sequence[typing.Dict[str, int]]
color = {"equatorward": "red", "feature": "blue", "poleward": "green"}
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
        name="spectrograh",
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


def plot_speclines(
    dat: xarray.DataArray,
    i_el: IndexElevation,
    wl_minmax: typing.Tuple[float, float] = None,
    axs=None,
    j: int = 0,
):
    """
    elevation angles chosen by inspection of keograms
    in Figures 1 and 2
    """
    marker = None if wl_minmax is None else "."
    if axs is None:
        fg = figure(1)
        fg.clf()
        axs = fg.subplots(dat.shape[0], 1, sharex=True)
    else:
        fg = None

    for i in range(dat.shape[0]):
        ax = axs[i]
        for k, v in i_el[i].items():
            ax.plot(
                dat.wavelength, dat[i].loc[v, :].values, label=k, color=color[k], marker=marker
            )
        for w in (427.8, 557.7, 630.0):
            ax.axvline(w, color="black", linestyle="--", alpha=0.5)
        ax.set_title(f"{feature[i]}: " + str(dat.time[i].values)[:-10])
        if j == 0:
            ax.set_ylabel("Luminosity (Rayleighs)")
        ax.grid(True)
        ax.set_ylim(0, None)
        if wl_minmax is None:
            ax.set_xlim(dat.wavelength[0], dat.wavelength[-1])
        else:
            ax.set_xlim(wl_minmax)
    ax.legend()
    if fg is not None:
        ax.set_xlabel("wavelength (nm)")
        fg.suptitle("Original paper Figures 1 and 2")


def plot_bgsubtracted_spectrum(
    dat: xarray.DataArray,
    i_el: IndexElevation,
    wl_minmax: typing.Tuple[float, float] = None,
    axs=None,
    j: int = 0,
):
    """
    elevation angles chosen by inspection of keograms
    in Figures 1 and 2

    Here, we additionally apply background subtraction
    """
    marker = None if wl_minmax is None else "."
    if axs is None:
        fg = figure(1)
        fg.clf()
        axs = fg.subplots(dat.shape[0], 1, sharex=True)
    else:
        fg = None

    for i in range(dat.shape[0]):
        ax = axs[i]

        bg_steve = dat[i].loc[i_el[i]["feature"], :]
        bg_equatorward = dat[i].loc[i_el[i]["equatorward"], :]
        bg_poleward = dat[i].loc[i_el[i]["poleward"], :]
        ax.plot(
            dat.wavelength,
            bg_steve - bg_equatorward,
            label="bg subtract equatorward",
            color=color["equatorward"],
            marker=marker,
        )
        ax.plot(
            dat.wavelength,
            bg_steve - bg_poleward,
            label="bg subtract poleward",
            color=color["poleward"],
            marker=marker,
        )
        for w in (427.8, 557.7, 630.0):
            ax.axvline(w, color="black", linestyle="--", alpha=0.5)
        ax.set_title(f"{feature[i]}: " + str(dat.time[i].values)[:-10])
        if j == 0:
            ax.set_ylabel("Luminosity (Rayleighs)")
        ax.grid(True)
        ax.set_ylim(0, None)
        if wl_minmax is None:
            ax.set_xlim(dat.wavelength[0], dat.wavelength[-1])
        else:
            ax.set_xlim(wl_minmax)

    ax.legend()
    ax.set_xlabel("wavelength (nm)")
    if fg is not None:
        fg.suptitle("background subtracted intensity")


def plot_keogram(dat: xarray.DataArray, i_el: IndexElevation):
    """
    the paper authors provide only two time steps of data.
    We present them together to help show we have used the same elevation angles
    as the paper Figures.

    A keogram typically plots time on one axis and space on the other axis,
    with intensity as the value plotted.
    In this case, we sum the spectrum and are left with time and elevation angle
    as free variables.
    """

    j_el = slice(90, 125)  # arbitrary
    fg = figure(2)
    fg.clf()
    ax = fg.gca()
    keogram = dat.loc[:, j_el, :].sum("wavelength")
    keogram.name = "wavelength-summed luminosity"
    keogram.T.plot(ax=ax)
    for j, i in enumerate(i_el):
        ax.scatter([dat.time[j].values] * 3, i.values(), s=100, c=color.values())

    ax.set_title("keogram")
    ax.set_xlabel("time (UTC)")
    ax.set_ylabel("elevation bin (unitless)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("path", help="path to paper data directory")
    p.add_argument("-paper", help="show paper figure plots", action="store_true")
    P = p.parse_args()

    i_el = [
        {"equatorward": 116, "feature": 119, "poleward": 122},
        {"equatorward": 99, "feature": 103, "poleward": 107},
    ]

    dat = load_spectrum(P.path)

    if P.paper:
        plot_speclines(dat, i_el)
        plot_keogram(dat, i_el)
        plot_bgsubtracted_spectrum(dat, i_el)

    Nt = dat.shape[0]
    fg = figure(10, figsize=(18, 16))
    fg.clf()
    axs = fg.subplots(Nt * 2, 4)
    for j, slim in enumerate([None, (425.0, 430.0), (555.0, 560.0), (625.0, 635.0)]):
        plot_speclines(dat, i_el, slim, axs[:Nt, j], j)
        plot_bgsubtracted_spectrum(dat, i_el, slim, axs[Nt:, j], j)

    fg.tight_layout(pad=1.5, h_pad=1.8)
    show()
