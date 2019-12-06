import xarray
import typing

# https://matplotlib.org/gallery/color/named_colors.html
color = {"quiet": "black", "equatorward": "red", "feature": "blue", "poleward": "green"}
color_lines = {427.8: "blue", 557.7: "yellowgreen", 630.0: "red", "continuum": "xkcd:mauve"}


def get_marker(dat: xarray.DataArray) -> str:
    return None if dat.shape[1] > 100 else "."


def sum_bandhead(dat: xarray.DataArray, head_limits: typing.Sequence[float]) -> xarray.DataArray:
    j = [
        abs(dat.wavelength.values - head_limits[0]).argmin(),
        abs(dat.wavelength.values - head_limits[1]).argmin(),
    ]

    return dat[:, j[0]: j[1] + 1].sum("wavelength")
