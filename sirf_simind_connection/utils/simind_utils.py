import os
from numbers import Number


class SimindError(Exception):
    """Base exception for SIMIND-related errors."""


class SimindNotFoundError(SimindError):
    """Raised when SIMIND executable is not found."""


def create_window_file(
    lower_bounds: list,
    upper_bounds: list,
    scatter_orders: list,
    output_filename: str = "input",
    energy_window=None,
    lower_ew=None,
    upper_ew=None,
):
    """
    Creates a window file for simind simulation

    Args:
        lower_bounds (list): lower bounds of energy windows
        upper_bounds (list): upper bounds of energy windows
        scatter_orders (list): scatter orders of energy windows
        output_filename (str, optional): name of output file. Defaults to 'input'.
        ! energy_window (str, optional): energy window type can be dew or dew.
            Defaults to None.
        ! lower_ew (list, optional): lower energy window lower and upper bounds.
            Defaults to None.
        ! upper_ew (list, optional): upper energy window lower and upper bounds.
            Defaults to None.
        ! Note that dual and triple energy windows are not yet supported by this
        wrapper. Please define your own energy windows and work out yourself
    """

    # if path suffix is not. win, add it
    if not output_filename.endswith(".win"):
        output_filename += ".win"

    if isinstance(lower_bounds, Number):
        lower_bounds = [lower_bounds]
    if isinstance(upper_bounds, Number):
        upper_bounds = [upper_bounds]
    if isinstance(scatter_orders, Number):
        scatter_orders = [scatter_orders]

    assert len(lower_bounds) == len(upper_bounds) == len(scatter_orders), (
        "lower_bounds, upper_bounds and scatter_orders must have same length"
    )

    # remove previous window file if present
    if os.path.exists(output_filename):
        os.remove(output_filename)

    with open(output_filename, "w") as file:
        for i in range(len(lower_bounds)):
            # for some reason simind doesn't like the last line to have a newline
            # character
            file.write(
                f"{float(lower_bounds[i])},{float(upper_bounds[i])},{int(scatter_orders[i])}\n"
            )
        # simind sometimes doesn't output scatter files unless there's one line
        # with a dedicated scatter order
        # annoyingle this will create one extra total, air, scatter file
        if max(scatter_orders) < 1:
            file.write(f"{float(lower_bounds[i])},{float(upper_bounds[i])},1")
