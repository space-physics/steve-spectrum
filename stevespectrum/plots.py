import xarray
import typing
from datetime import datetime
from matplotlib.pyplot import figure

IndexElevation = typing.Dict[str, int]

keo_el = slice(70, 125)  # roughly match inset of Figure 1 and 2


# https://matplotlib.org/gallery/color/named_colors.html
color = {"quiet": "black", "equatorward": "red", "feature": "blue", "poleward": "green"}
color_lines = {427.8: "blue", 557.7: "yellowgreen", 630.0: "red", "continuum": "xkcd:mauve"}


def plot_paper(
    dat: xarray.DataArray, i_el: typing.Dict[str, int], i_wl: typing.Sequence[float], axs
):
    plot_speclines_wavelength(dat, i_el, i_wl, ax=axs[0])
    plot_bgsubtracted_spectrum(dat, i_el, ax=axs[1])
    axs[-1].set_xlabel("wavelength (nm)")


def plot_zoom(
    dat: xarray.DataArray, i_el: typing.Dict[str, int], i_wl: typing.Sequence[float], axs
):
    for j, slim in enumerate([(420.0, 435.0), (550.0, 565.0), (620.0, 640.0)]):
        k = (dat.wavelength >= slim[0]) & (dat.wavelength < slim[1])

        plot_speclines_wavelength(dat[:, k], i_el, i_wl, axs[0, j], j)
        plot_bgsubtracted_spectrum(dat[:, k], i_el, axs[1, j], j)

        axs[-1, j].set_xlabel("wavelength (nm)")
    # fg.tight_layout(pad=1.5, h_pad=1.8)


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
    fg20.tight_layout()

    fg21 = figure(21, figsize=(12, 10))
    fg21.clf()
    ax21 = fg21.subplots(N, 1, sharex=True)
    ax21[-1].set_xlabel("elevation bin (unitless)")
    ax21[-1].set_ylabel("luminosity (Rayleighs)")
    fg21.tight_layout()

    fg22 = figure(22)
    fg22.clf()
    ax22 = fg22.subplots(N, 1, sharex=True)
    plot_N21N_elevation(dat, feature, head_limits, ax22)

    return ax20, ax21


def elevation_plots(
    dat: xarray.DataArray,
    feature: str,
    head_limits: typing.Sequence[float],
    i_el: typing.Dict[str, int],
    i_wl: typing.Sequence[float],
    ax20,
    ax21,
):
    plot_speclines_elevation(dat, head_limits, i_el["feature"], i_wl, ax=ax20)
    ax20.set_title(feature + ": " + str(dat.time.values)[:-10])

    plot_speclines_elevation(
        dat - dat.loc[i_el["equatorward"], :], head_limits, i_el["feature"], i_wl, ax=ax21
    )
    ax21.set_title(feature + ": " + str(dat.time.values)[:-10])


def get_marker(dat: xarray.DataArray) -> str:
    return None if dat.shape[1] > 100 else "."


def sum_bandhead(dat: xarray.DataArray, head_limits: typing.Sequence[float]) -> xarray.DataArray:
    j = [
        abs(dat.wavelength.values - head_limits[0]).argmin(),
        abs(dat.wavelength.values - head_limits[1]).argmin(),
    ]

    return dat[:, j[0]: j[1] + 1].sum("wavelength")


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


def plot_speclines_wavelength(
    dat: xarray.DataArray, i_el: IndexElevation, i_wl: typing.Sequence[float], ax, j: int = 0
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


def plot_speclines_elevation(
    dat: xarray.DataArray,
    head_limits: typing.Sequence[float],
    i_el: int,
    i_wl: typing.Sequence[float],
    ax,
):
    """
    plot luminosity vs. elevation bin
    """
    # keep only elevation bins of interest
    dat = dat.loc[keo_el, :]

    for i in i_wl:
        if i == 427.8:
            d = sum_bandhead(dat, head_limits)
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
