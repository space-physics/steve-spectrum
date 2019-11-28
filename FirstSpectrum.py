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

IndexElevation = typing.Dict[str, int]
color = {"quiet": "black", "equatorward": "red", "feature": "blue", "poleward": "green"}
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


def get_marker(dat: xarray.DataArray) -> str:
    return None if dat.shape[1] > 100 else "."


def plot_speclines(dat: xarray.DataArray, i_el: IndexElevation, ax, j: int = 0):
    """
    elevation angles chosen by inspection of keograms
    in Figures 1 and 2
    """
    marker = get_marker(dat)

    for k, v in i_el.items():
        ax.plot(dat.wavelength, dat.loc[v, :].values, label=k, color=color[k], marker=marker)

    if j == 0:
        ax.set_ylabel("Luminosity (Rayleighs)")
        for w in (427.8, 557.7, 630.0):
            ax.axvline(w, color="black", linestyle="--", alpha=0.5)

    ax.grid(True)
    ax.set_ylim(0, None)
    ax.set_xlim(dat.wavelength[0], dat.wavelength[-1])
    ax.legend()


def plot_bgsubtracted_spectrum(dat: xarray.DataArray, i_el: IndexElevation, ax, j: int = 0):
    """
    elevation angles chosen by inspection of keograms
    in Figures 1 and 2

    Here, we additionally apply background subtraction
    """
    if not dat.ndim == 2:
        raise ValueError("data should be elevation_bin x wavelength")

    marker = get_marker(dat)

    bg_steve = dat.loc[i_el["feature"], :]
    bg_equatorward = dat.loc[i_el["equatorward"], :]
    bg_poleward = dat.loc[i_el["poleward"], :]
    ax.plot(
        dat.wavelength,
        bg_steve - bg_equatorward,
        label="feature $-$ equatorward",
        color=color["equatorward"],
        marker=marker,
    )
    ax.plot(
        dat.wavelength,
        bg_steve - bg_poleward,
        label="feature $-$ poleward",
        color=color["poleward"],
        marker=marker,
    )

    if j == 0:
        ax.set_ylabel("Luminosity (Rayleighs)")
        for w in (427.8, 557.7, 630.0):
            ax.axvline(w, color="black", linestyle="--", alpha=0.5)

    ax.grid(True)
    ax.set_ylim(0, None)
    ax.set_xlim(dat.wavelength[0], dat.wavelength[-1])
    ax.legend()


def plot_keogram(dat: xarray.DataArray, i_el: typing.Sequence[typing.Dict[str, int]]):
    """
    the paper authors provide only two time steps of data.
    We present them together to help show we have used the same elevation angles
    as the paper Figures.

    A keogram typically plots time on one axis and space on the other axis,
    with intensity as the value plotted.
    In this case, we sum the spectrum and are left with time and elevation angle
    as free variables.
    """

    j_el = slice(70, 125)  # arbitrary
    fg = figure(20)
    fg.clf()
    ax = fg.gca()
    keogram = dat.loc[:, j_el, :].sum("wavelength")
    keogram.name = "wavelength-summed luminosity"
    keogram.T.plot(ax=ax)
    for j, i in enumerate(i_el):
        ax.scatter([dat.time[j].values] * 4, i.values(), s=100, c=color.values())

    ax.set_title("keogram")
    ax.set_xlabel("time (UTC)")
    ax.set_ylabel("elevation bin (unitless)")
    # label each pixel column with time
    ax.set_xticks(ax.get_xticks()[::2])  # only two data points
    time = dat.time.values.astype('datetime64[us]').astype(datetime)
    ax.set_xticklabels([t[11:] for t in time.astype(str)])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("path", help="path to paper data directory")
    P = p.parse_args()

    i_el = [
        {"quiet": 100, "equatorward": 116, "feature": 119, "poleward": 122},
        {"quiet": 90, "equatorward": 99, "feature": 103, "poleward": 107},
    ]

    dat = load_spectrum(P.path)

    plot_keogram(dat, i_el)

    for i, d in enumerate(dat):  # each time/event
        fg1 = figure(1 + i, figsize=(12, 10))
        fg1.clf()
        axs1 = fg1.subplots(2, 1, sharex=True)
        fg1.suptitle(feature[i] + ": " + str(d.time.values)[:-10])
        plot_speclines(d, i_el[i], ax=axs1[0])
        plot_bgsubtracted_spectrum(d, i_el[i], ax=axs1[1])
        axs1[1].set_xlabel("wavelength (nm)")

        fg = figure(10 + i, figsize=(18, 16))
        fg.clf()
        axs = fg.subplots(2, 3)
        fg.suptitle(feature[i] + ": " + str(d.time.values)[:-10])
        for j, slim in enumerate([(420.0, 435.0), (550.0, 565.0), (620.0, 640.0)]):
            k = (d.wavelength >= slim[0]) & (d.wavelength < slim[1])
            plot_speclines(d[:, k], i_el[i], axs[0, j], j)
            plot_bgsubtracted_spectrum(d[:, k], i_el[i], axs[1, j], j)
            axs[1, j].set_xlabel("wavelength (nm)")
        # fg.tight_layout(pad=1.5, h_pad=1.8)
    show()
