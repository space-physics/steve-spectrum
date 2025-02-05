import xarray
import typing
from datetime import datetime
from matplotlib.pyplot import figure
from matplotlib.colors import LogNorm

from .base import color, sum_bandhead, color_lines

keo_el = slice(70, 125)  # roughly match inset of Figure 1 and 2
# keo_el = slice(None)


def plot_keogram(dat: xarray.DataArray, i_el: typing.Sequence[typing.Dict[str, int]], ax):
    """
    the paper authors provide only two time steps of data.
    We present them together to help show we have used the same elevation angles
    as the paper Figures.

    A keogram typically plots time on one axis and space on the other axis,
    with intensity as the value plotted.
    In this case, we sum the spectrum and are left with time and elevation angle
    as free variables.
    """

    keogram = dat.loc[:, keo_el, :].sum("wavelength")
    keogram.name = "wavelength-summed luminosity"
    keogram.T.plot(ax=ax)
    for j, i in enumerate(i_el):
        ax.scatter([dat.time[j].values] * 4, i.values(), s=100, c=color.values())

    ax.set_title("keogram")
    ax.set_xlabel("time (UTC)")
    ax.set_ylabel("elevation bin (unitless)")
    # label each pixel column with time
    ax.set_xticks(ax.get_xticks()[::2])  # only two data points
    time = dat.time.values.astype("datetime64[us]").astype(datetime)
    ax.set_xticklabels([t[11:] for t in time.astype(str)])


def setup_elevation_plots(
    dat: xarray.DataArray,
    feature: typing.Sequence[str],
    head_limits: typing.Mapping[str, typing.Sequence[float]],
    N: int,
) -> tuple:
    """ create blank figures """

    fg = figure(figsize=(12, 10))
    ax20 = fg.subplots(N, 1, sharex=True)
    ax20[-1].set_xlabel("elevation bin (unitless)")
    ax20[-1].set_ylabel("luminosity (Rayleighs)")
    fg.suptitle("No background subtraction")

    fg = figure(figsize=(12, 10))
    ax21 = fg.subplots(N, 1, sharex=True)
    ax21[-1].set_xlabel("elevation bin (unitless)")
    ax21[-1].set_ylabel("relative luminosity (Rayleighs)")
    fg.suptitle("Equatorward background subtraction")

    fg = figure(figsize=(12, 10))
    ax22 = fg.subplots(N, 1, sharex=True)
    ax22[-1].set_xlabel("elevation bin (unitless)")
    ax22[-1].set_ylabel("relative luminosity (Rayleighs)")
    fg.suptitle("Poleward background subtraction")

    for k in head_limits:
        fg = figure()
        ax = fg.subplots(N, 1, sharex=True)
        plot_spectrum_elevation(dat, feature, head_limits[k], ax, k)

    return ax20, ax21, ax22


def elevation_plots(
    dat: xarray.DataArray,
    feature: str,
    head_limits: typing.Mapping[str, typing.Sequence[float]],
    i_el: typing.Dict[str, int],
    i_wl: typing.List[str],
    ax20,
    ax21,
    ax22,
):
    # %% raw values
    plot_speclines_elevation(dat, head_limits, i_el["feature"], i_wl, ax=ax20)
    ax20.set_title(feature + ": " + str(dat.time.values)[:-10])
    # %% background subtraction
    # note that "background" can be brighter than "feature", which makes negative intensities
    bgsub = dat - dat.loc[i_el["equatorward"], :]
    plot_speclines_elevation(bgsub, head_limits, i_el["feature"], i_wl, ax=ax21)
    ax21.set_title(feature + ": " + str(dat.time.values)[:-10])

    bgsub = dat - dat.loc[i_el["poleward"], :]
    plot_speclines_elevation(bgsub, head_limits, i_el["feature"], i_wl, ax=ax22)
    ax22.set_title(feature + ": " + str(dat.time.values)[:-10])


def plot_spectrum_elevation(
    dat: xarray.DataArray,
    feature: typing.Sequence[str],
    head_limits: typing.Sequence[float],
    ax,
    k: str,
):
    """
    plot luminosity vs. elevation bin
    """
    # keep only elevation bins of interest
    dat = dat.loc[:, keo_el, :]

    j = [
        abs(dat.wavelength.values - head_limits[0]).argmin(),
        abs(dat.wavelength.values - head_limits[1]).argmin(),
    ]

    for i, (d, a) in enumerate(zip(dat, ax)):
        d = d[:, j[0]: j[1] + 1]

        norm = LogNorm() if k == "all" else None
        LOGMIN = 1.0  # arbitrary
        d.plot(ax=a, norm=norm, vmin=max(d.min(), LOGMIN))
        a.set_title(feature[i] + ": " + str(d.time.values)[:-10])

    ax[1].set_ylabel("elevation bin (unitless)")
    ax[0].set_ylabel("")
    ax[1].set_xlabel("wavelength (nm)")
    ax[0].set_xlabel("")


def plot_speclines_elevation(
    dat: xarray.DataArray,
    head_limits: typing.Mapping[str, typing.Sequence[float]],
    i_el: int,
    i_wl: typing.Sequence[str],
    ax,
):
    """
    plot luminosity vs. elevation bin
    """
    # keep only elevation bins of interest
    dat = dat.loc[keo_el, :]

    for i in i_wl:
        if i == "427.8":
            d = sum_bandhead(dat, head_limits["N2p1N01"])
        elif i == "450..530":
            d = sum_bandhead(dat, head_limits["continuum"]) / 5
            ax.annotate('÷ 5', (dat.elevation[0], d.values[0]),
                        xytext=(dat.elevation[0]-1, d.values[0]+40),
                        fontsize='large',
                        arrowprops={'arrowstyle': '-|>'})
        else:
            d = dat.sel(wavelength=i, method="nearest")

        ax.plot(dat.elevation, d.values, label=i, color=color_lines[i])

    ax.axvline(i_el, color="k", linestyle="--")


def plot_ratio_elevation(
    dat: xarray.DataArray, head_limits: typing.Sequence[float], i_el: int, ax
):
    """
    plot intensity ratios
    """
    # keep only elevation bins of interest
    dat = dat.loc[keo_el, :]

    i4278 = sum_bandhead(dat, head_limits)
    i5577 = dat.sel(wavelength=557.7, method="nearest")
    i6300 = dat.sel(wavelength=630.0, method="nearest")

    ax.plot(dat.elevation, i5577 / i4278, label="557.7 / 427.8")
    ax.plot(dat.elevation, i6300 / i4278, label="630.0 / 427.8")
    ax.plot(dat.elevation, i6300 / i5577, label="630.0 / 557.7")

    ax.axvline(i_el, color="k", linestyle="--")
