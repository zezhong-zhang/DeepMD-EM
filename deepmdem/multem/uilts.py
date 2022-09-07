import numpy as np
from ase import units
import matplotlib.pyplot as plt


def energy2wavelength(energy: float) -> float:
    """
    Calculate relativistic de Broglie wavelength from energy.

    Parameters
    ----------
    energy: float
        Energy [keV].

    Returns
    -------
    float
        Relativistic de Broglie wavelength [Å].
    """
    energy = energy * 1e3
    return (
        units._hplanck
        * units._c
        / np.sqrt(energy * (2 * units._me * units._c**2 / units._e + energy))
        / units._e
        * 1.0e10
    )


def potential_sampling(energy: float, collection_angle: float) -> float:
    """


    Args:
        energy (float): eV
        collection_angle (float): mrad

    Returns:
        pixel size for potential sampling
    """
    return energy2wavelength(energy) / 3 / (collection_angle / 1e3)


def potential_pixel(
    energy: float, collection_angle: float, real_space_length: float
) -> float:
    """
    Args:
        energy (float): eV
        collection_angle (float): mrad
        real_space_length (float): Å

    Returns:
        sampling for potential sampling
    """
    return ceil_to_nearest_even_number(
        real_space_length / potential_sampling(energy, collection_angle)
    )


def probe_sampling(energy: float, convergence_angle: float) -> float:
    """
    Args:
        energy (float): eV
        convergence_angle (float): mrad

    Returns:
        pixel size for scanning probe sampling
    """
    return energy2wavelength(energy) / 4 / (convergence_angle / 1e3)


def probe_pixel(
    energy: float, convergence_angle: float, real_space_length: float
) -> float:
    """
    Args:
        energy (float): eV
        convergence_angle (float): mrad
        real_space_length (float): Å

    Returns:
        sampling for scanning probe sampling
    """
    return int(np.ceil(real_space_length / probe_sampling(energy, convergence_angle)))


def ceil_to_nearest_even_number(num):
    return int(np.ceil(num / 2) * 2)


def ceil_to_nearest_2_power_number(num):
    return int(2 ** np.ceil(np.log2(num)))


def plot_image(image, pixel_size, space="real", title=None, filename=None, tile=None):
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
    import matplotlib.font_manager as fm

    fig, ax = plt.subplots()
    if tile is not None:
        image = np.kron(np.ones(tile), image)
    ax.imshow(image)

    fontprops = fm.FontProperties(size=18)
    if space == "real":
        unit = "1 Å"
    if space == "reciprocal":
        unit = "1/Å"
    scalebar = AnchoredSizeBar(
        ax.transData,
        1 / pixel_size,
        unit,
        "lower left",
        pad=0.1,
        color="white",
        frameon=False,
        size_vertical=1,
        fontproperties=fontprops,
    )
    ax.add_artist(scalebar)
    ax.set_yticks([])
    ax.set_xticks([])
    if title is not None:
        ax.set_title(title)
    if filename is not None:
        plt.savefig(filename)


def filter_image(image, pixel_size, probe_size):
    print(image.shape)
    Y, X = np.mgrid[0 : image.shape[0], 0 : image.shape[1]]
    Y -= image.shape[0] // 2
    X -= image.shape[1] // 2
    Y = Y / (pixel_size[0] * image.shape[0])
    X = X / (pixel_size[1] * image.shape[1])
    R = X**2 + Y**2
    sigma = 1.0 / probe_size
    fft_filter = np.exp(-0.5 * R / sigma**2)
    fft_filter = np.fft.ifftshift(fft_filter)
    fft_data = np.fft.fft2(image)
    fft_data = fft_data * fft_filter
    return np.real(np.fft.ifft2(fft_data))


def fourier_interpolation(image, output_size):
    fft = np.fft.fft2(image)
    fft = np.fft.fftshift(fft)
    fft = padding(fft, output_size[0], output_size[1])
    fft = np.fft.ifftshift(fft)
    out_image = np.real(np.fft.ifft2(fft))
    return out_image


def padding(array, xx, yy):
    """
    :param array: numpy array
    :param xx: desired height
    :param yy: desirex width
    :return: padded array
    """

    h = array.shape[0]
    w = array.shape[1]

    a = (xx - h) // 2
    aa = xx - a - h

    b = (yy - w) // 2
    bb = yy - b - w

    return np.pad(array, pad_width=((a, aa), (b, bb)), mode="constant")


def average_phonon(data_list):
    return np.abs(sum(data_list)) / len(data_list)


def get_displacement(coords, format="rmsd"):
    assert format in ["rmsd", "u2", "Debye-Waller"]
    avg_coords = np.mean(coords, axis=0)
    u2 = np.sum((coords - avg_coords) ** 2 / 3, axis=(0, 2)) / len(coords)
    if format == "rmsd":
        return np.sqrt(u2)
    elif format == "u2":
        return u2
    elif format == "Debye-Waller":
        return u2 * 8 * np.pi**2


def get_mean_position(coords):
    return np.mean(coords, axis=0)


def plot_displacement(coords, format="rmsd", filename=None, bulk_displacement=None):
    assert format in ["rmsd", "u2", "Debye-Waller"]
    displace = get_displacement(coords, format=format)
    avg_coords = get_mean_position(coords)

    plt.plot(avg_coords[:, 2], displace, "*", label="MD")
    if bulk_displacement is not None:
        plt.plot(
            avg_coords[:, 2],
            bulk_displacement * np.ones(len(displace)),
            "-",
            label="bulk",
        )
    plt.xlabel(r"z ($\AA$)")
    if format == "rmsd":
        plt.ylabel(r"RMSD ($\AA$)")
    else:
        plt.ylabel(f"{format} ($\AA^2$)")
    plt.legend()
    if filename is not None:
        plt.savefig(filename)
