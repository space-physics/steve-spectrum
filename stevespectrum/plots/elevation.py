import xarray
import typing
from datetime import datetime
from matplotlib.pyplot import figure

from .base import color, sum_bandhead, color_lines

keo_el = slice(70, 125)  # roughly match inset of Figure 1 and 2


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
    head_limits: typing.Sequence[float],
    N: int,
) -> tuple:
    fg20 = figure(20, figsize=(12, 10))
    fg20.clf()
    ax20 = fg20.subplots(N, 1, sharex=True)
    ax20[-1].set_xlabel("elevation bin (unitless)")
    ax20[-1].set_ylabel("luminosity (Rayleighs)")
    # fg20.tight_layout()
    fg20.suptitle("No background subtraction")

    fg21 = figure(21, figsize=(12, 10))
    fg21.clf()
    ax21 = fg21.subplots(N, 1, sharex=True)
    ax21[-1].set_xlabel("elevation bin (unitless)")
    ax21[-1].set_ylabel("luminosity (Rayleighs)")
    # fg21.tight_layout()
    fg21.suptitle("Equatorward background subtraction")

    fg22 = figure(22)
    fg22.clf()
    ax22 = fg22.subplots(N, 1, sharex=True)
    plot_N21N_elevation(dat, feature, head_limits, ax22)

    return ax20, ax21


def elevation_plots(
    dat: xarray.DataArray,
    feature: str,
    head_limits: typing.Mapping[str, typing.Sequence[float]],
    i_el: typing.Dict[str, int],
    i_wl: typing.Sequence[str],
    ax20,
    ax21,
):
    plot_speclines_elevation(dat, head_limits, i_el["feature"], i_wl, ax=ax20)
    ax20.set_title(feature + ": " + str(dat.time.values)[:-10])

    bgsub = dat.values - dat.loc[i_el["equatorward"], :].values
    bgsub[bgsub < 0] = 0.
    bgsub = xarray.DataArray(bgsub, coords=dat.coords, dims=dat.dims, name=dat.name)
    # have to force min to zero, since some of the "background" was brighter than "feature"
    plot_speclines_elevation(bgsub, head_limits, i_el["feature"], i_wl, ax=ax21)
    ax21.set_title(feature + ": " + str(dat.time.values)[:-10])


def plot_N21N_elevation(
    dat: xarray.DataArray, feature: typing.Sequence[str], head_limits: typing.Sequence[float], ax,
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

        d.plot(ax=a)
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
            d = sum_bandhead(dat, head_limits['N2p1N01'])
        elif i == "450..530":
            d = sum_bandhead(dat, head_limits['continuum'])
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
