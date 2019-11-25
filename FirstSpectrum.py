#!/usr/bin/env python
"""
python FirstSpectrumFigure1.py ~/data/2019GL083272

Four main sources of discrepancy from paper figures:

1. picking the same elevation bins
2. flipping poleward / equatorward elevation direction
3. spectral smoothing
4. bin width
"""
import argparse
from pathlib import Path
import numpy as np
import xarray
import typing
from datetime import datetime
from matplotlib.pyplot import figure, show


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
        flist = [path / "TREx_spectrograph_20180410_063045.txt", path / "TREx_spectrograph_20180410_064015.txt"]
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

    dat = xarray.DataArray(
        data=arr,
        dims=("time", "elevation", "wavelength"),
        coords={"time": time, "elevation": np.linspace(0, 180, arr.shape[1]), "wavelength": wavelengths},
    )

    return dat


def read_flattxt(path: Path, skip_header: int) -> typing.List[float]:
    """ read float data from irregularly shaped text file """
    dat: typing.List[float] = []
    with path.open("r") as f:
        skip_rows(f, skip_header)
        for line in f:
            dat += list(map(float, line.split()))
    return dat


def skip_rows(f, rows: int):
    """ skip rows of file f """
    for _ in range(rows):
        f.readline()


def plot_speclines(dat: xarray.DataArray):

    # by inspection of paper
    elevation_deg = [
        {"poleward": 108.0, "equatorward": 100.0, "steve": 104.0},
        {"poleward": 123.0, "equatorward": 117.0, "steve": 120.0},
    ]

    for i in range(dat.shape[0]):
        ax = figure().gca()
        for k, v in elevation_deg[i].items():
            j = abs(dat.elevation - v).argmin().item()
            ax.plot(dat.wavelength, dat[i, j, :].values, label=k)
        for w in (427.8, 557.7, 630.0):
            ax.axvline(w, color="black", linestyle="--", alpha=0.5)
        ax.legend()
        ax.set_title(str(dat.time[i].values)[:-10])
        ax.set_ylabel("Luminosity (Rayleighs)")
        ax.set_xlabel("wavelength (nm)")
        ax.grid(True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("path", help="path to paper data directory")
    P = p.parse_args()

    dat = load_spectrum(P.path)

    plot_speclines(dat)

    show()
