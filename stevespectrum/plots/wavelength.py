import xarray
import typing

from .base import get_marker, color

IndexElevation = typing.Dict[str, int]


def plot_bgsubtracted_spectrum(dat: xarray.DataArray, i_el: IndexElevation, ax, j: int = 0):
    """
    elevation bins chosen by inspection of keograms
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


def plot_speclines_wavelength(
    dat: xarray.DataArray, i_el: IndexElevation, i_wl: typing.Sequence[str], ax, j: int = 0
):
    """
    plots luminosity vs wavelength for chosen elevation bins

    elevation bins chosen by inspection of keograms
    in Figures 1 and 2
    """
    marker = get_marker(dat)

    for k, v in i_el.items():
        ax.plot(dat.wavelength, dat.loc[v, :].values, label=k, color=color[k], marker=marker)

    if j == 0:
        ax.set_ylabel("Luminosity (Rayleighs)")
        for w in i_wl:
            ax.axvline(w, color="black", linestyle="--", alpha=0.5)

    ax.grid(True)
    ax.set_ylim(0, None)
    ax.set_xlim(dat.wavelength[0], dat.wavelength[-1])
    ax.legend()
