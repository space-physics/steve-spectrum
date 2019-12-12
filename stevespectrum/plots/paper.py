import xarray
import typing

from .wavelength import plot_speclines_wavelength, plot_bgsubtracted_spectrum


def plot_paper(
    dat: xarray.DataArray, i_el: typing.Dict[str, int], i_wl: typing.Sequence[str], axs
):
    plot_speclines_wavelength(dat, i_el, i_wl, ax=axs[0])
    plot_bgsubtracted_spectrum(dat, i_el, ax=axs[1])
    axs[-1].set_xlabel("wavelength (nm)")


def plot_zoom(
    dat: xarray.DataArray, i_el: typing.Dict[str, int], i_wl: typing.Sequence[str], axs
):
    for j, slim in enumerate([(420.0, 435.0), (550.0, 565.0), (620.0, 640.0)]):
        k = (dat.wavelength >= slim[0]) & (dat.wavelength < slim[1])

        plot_speclines_wavelength(dat[:, k], i_el, i_wl, axs[0, j], j)
        plot_bgsubtracted_spectrum(dat[:, k], i_el, axs[1, j], j)

        axs[-1, j].set_xlabel("wavelength (nm)")
    # fg.tight_layout(pad=1.5, h_pad=1.8)
