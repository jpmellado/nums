# save netcdf data for structure coords
# need to pass the coordinates in order corresponding to the array shapes in vars

import numpy as np
import netCDF4 as nc


def save_netcdf(
    times,  # list of times
    coords,  # list of 1d arrays containing the coordinates along each direction
    coord_names,  # names of each coordinate
    vars,  # list of lists, latter being a list of arrays with the fields at different times
    var_names,  # names of each variable
    filename,  # name of the file to use
):
    # creating netcdf
    file_dst = nc.Dataset(filename + ".nc", "w")

    # create dimensions for destiny nc-file
    file_dst.createDimension("t", None)
    for ic, coord in enumerate(coords):
        file_dst.createDimension(coord_names[ic], np.size(coord))

    # create and write independent variables in destiny nc-file using single precision
    t_dst = file_dst.createVariable("t", "f4", ("t",))
    t_dst[:] = times[:]
    for ic, coord in enumerate(coords):
        coord_dst = file_dst.createVariable(coord_names[ic], "f4", (coord_names[ic],))
        coord_dst[:] = coord[:]

    for iv, var in enumerate(vars):
        var_dst = file_dst.createVariable(
            var_names[iv], "f4", ("t",) + tuple(coord_names)
        )
        match np.array(var).ndim:  # could not find a more elegant solution
            case 3:
                var_dst[:, :, :] = np.array(var)
            case 2:
                var_dst[:, :] = np.array(var)
            case 1:
                var_dst[:] = np.array(var)

    file_dst.close()


def read_netcdf(var, filename):
    print("to be done")


# To test this module
def test():
    x = np.linspace(0.0, 10.0, num=10)
    y = np.linspace(0.0, 20.0, num=20)

    u = np.zeros((np.size(y), np.size(x)))

    times = np.linspace(0.0, 1.0, num=10)
    u_checked = []
    for t in times:
        u = u + 1.0
        u_checked.append(np.copy(u))

    save_netcdf(times, [y, x], ["y", "x"], [u_checked], ["u"], "test")


if __name__ == "__main__":
    test()
