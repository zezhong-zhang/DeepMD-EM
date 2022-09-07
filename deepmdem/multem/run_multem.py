from deepmdem.multem.crystal.Al001_crystal import Al001_crystal
import multem
import dpdata
import numpy as np
import time
from deepmdem.multem.dp_to_multem import dp_to_multem
from deepmdem.multem.uilts import (
    potential_pixel,
    probe_pixel,
    energy2wavelength,
    fourier_interpolation,
    filter_image,
)


def run_MULTEM(
    simulation_type: str,
    atoms_conf: str,
    type_map: dict,
    energy: float,
    convergence_angle: float,
    max_scattering_vector: float = None,
    collection_angle: list = None,
    scanning_area: list = None,
    output_size: list = None,
    rmsd: float = 0.0,
    md_phonon: bool = True,
    nphonon: int = 10,
    bwl: bool = True,
):
    st = time.time()
    print("GPU available: %s" % multem.is_gpu_available())

    input_multislice = multem.Input()
    system_conf = multem.SystemConfiguration()

    system_conf.precision = "float"
    system_conf.cpu_ncores = 12
    system_conf.cpu_nthread = 1
    system_conf.device = "device"
    system_conf.gpu_device = 0

    # Set simulation experiment
    input_multislice.simulation_type = simulation_type

    # Electron-Specimen interaction model
    input_multislice.interaction_model = "Multislice"
    input_multislice.potential_type = "Lobato_0_12"

    # Potential slicing
    input_multislice.potential_slicing = "dz_Proj"
    input_multislice.spec_dz = 2.0
    # input_multislice.potential_slicing = "Planes"

    # Electron-Phonon interaction model
    if md_phonon:
        input_multislice.pn_model = "Still_Atom"
    else:
        input_multislice.pn_model = "Frozen_Phonon"
        input_multislice.pn_coh_contrib = 0
        input_multislice.pn_single_conf = False
        input_multislice.pn_nconf = nphonon
        input_multislice.pn_dim = 110
        input_multislice.pn_seed = 300183

    dp_frame = dpdata.System(atoms_conf, fmt="lammps/lmp")
    (
        input_multislice.spec_atoms,
        input_multislice.spec_lx,
        input_multislice.spec_ly,
        input_multislice.spec_lz,
        a,
        b,
        c,
    ) = dp_to_multem(dp_frame=dp_frame, type_map=type_map, rms3d=rmsd)
    # atoms_1 = np.array(input_multislice.asdict()['spec_atoms'])

    # (
    #     input_multislice.spec_atoms,
    #     input_multislice.spec_lx,
    #     input_multislice.spec_ly,
    #     input_multislice.spec_lz,
    #     a,
    #     b,
    #     c,
    #     input_multislice.spec_dz,
    # ) = Al001_crystal(8, 8, 25, 2, rmsd)
    # atoms_2 = np.array(input_multislice.asdict()['spec_atoms'])

    # import matplotlib.pyplot as plt
    # ax = plt.axes(projection='3d')
    # ax.scatter3D(atoms_1[:, 1], atoms_1[:, 2], atoms_1[:, 3], cmap='Greens');
    # plt.savefig('atoms_1.jpg')

    # ax = plt.axes(projection='3d')
    # ax.scatter3D(atoms_2[:, 1], atoms_2[:, 2], atoms_2[:, 3], cmap='Greens');
    # plt.savefig('atoms_2.jpg')

    # Specimen thickness
    input_multislice.thick_type = "Through_Thick"
    input_multislice.thick = [x * input_multislice.spec_dz for x in range(0, 100)]

    # Microscope parameters
    input_multislice.E_0 = energy
    input_multislice.theta = 0.0
    input_multislice.phi = 0.0

    # Illumination model
    input_multislice.illumination_model = "Coherent"
    input_multislice.temporal_spatial_incoh = "Temporal_Spatial"

    # Set the incident wave
    input_multislice.iw_type = "Auto"
    # input_multislice.iw_psi = read_psi_0_multem(input_multislice.nx, input_multislice.ny)
    input_multislice.iw_x = [input_multislice.spec_lx / 2]  # input_multislice.spec_lx/2
    input_multislice.iw_y = [input_multislice.spec_ly / 2]  # input_multislice.spec_ly/2

    # Condenser lens
    input_multislice.cond_lens_m = 0
    input_multislice.cond_lens_c_10 = 0
    input_multislice.cond_lens_c_30 = 0
    input_multislice.cond_lens_c_50 = 0.00
    input_multislice.cond_lens_c_12 = 0.0
    input_multislice.cond_lens_phi_12 = 0.0
    input_multislice.cond_lens_c_23 = 0.0
    input_multislice.cond_lens_phi_23 = 0.0
    input_multislice.cond_lens_inner_aper_ang = 0.0
    input_multislice.cond_lens_outer_aper_ang = convergence_angle

    # defocus spread function
    ti_sigma = multem.iehwgd_to_sigma(32)
    input_multislice.cond_lens_ti_a = 1.0
    input_multislice.cond_lens_ti_sigma = ti_sigma
    input_multislice.cond_lens_ti_beta = 0.0
    input_multislice.cond_lens_ti_npts = 5

    # Source spread function
    si_sigma = multem.hwhm_to_sigma(0.45)
    input_multislice.cond_lens_si_a = 1.0
    input_multislice.cond_lens_si_sigma = si_sigma
    input_multislice.cond_lens_si_beta = 0.0
    input_multislice.cond_lens_si_rad_npts = 8
    input_multislice.cond_lens_si_azm_npts = 412

    # Zero defocus reference
    input_multislice.cond_lens_zero_defocus_type = "First"
    input_multislice.cond_lens_zero_defocus_plane = 0

    # Set the probe
    if simulation_type == "STEM":
        input_multislice.scanning_type = "Area"
        input_multislice.scanning_periodic = True
        input_multislice.scanning_square_pxs = True
        input_multislice.scanning_x0 = scanning_area[0] * input_multislice.spec_lx
        input_multislice.scanning_y0 = scanning_area[1] * input_multislice.spec_ly
        input_multislice.scanning_xe = scanning_area[2] * input_multislice.spec_lx
        input_multislice.scanning_ye = scanning_area[3] * input_multislice.spec_ly
        max_scanning_length = np.max(
            [
                input_multislice.scanning_xe - input_multislice.scanning_x0,
                input_multislice.scanning_ye - input_multislice.scanning_y0,
            ]
        )
        input_multislice.scanning_ns = probe_pixel(
            energy=energy,
            convergence_angle=convergence_angle,
            real_space_length=max_scanning_length,
        )
        input_multislice.detector.type = "Circular"
        input_multislice.detector.cir = collection_angle

    # x-y sampling
    if collection_angle is not None:
        max_collection_angle = max(collection_angle, key=lambda item: item[1])[1]
    else:
        max_collection_angle = max_scattering_vector * energy2wavelength(energy) * 1e3
    input_multislice.nx = potential_pixel(
        energy=energy,
        collection_angle=max_collection_angle,
        real_space_length=input_multislice.spec_lx,
    )
    input_multislice.ny = potential_pixel(
        energy=energy,
        collection_angle=max_collection_angle,
        real_space_length=input_multislice.spec_ly,
    )
    input_multislice.bwl = bwl

    # Do the simulation
    output_multislice = multem.simulate(system_conf, input_multislice)
    print("Time: %.2f" % (time.time() - st))

    if simulation_type == "STEM":
        # probe_size = 0.7  # A
        # pixel_size = (0.1, 0.1)
        data = []
        for i in range(len(output_multislice.data)):
            d = []
            for j in range(len(input_multislice.detector.cir)):
                image = np.array(output_multislice.data[i].image_tot[j])
                if output_size is not None:
                    image = fourier_interpolation(image, output_size)
                # image = filter_image(image, pixel_size, probe_size)
                d.append(image)
            data.append(d)
        pixel_size = max_scanning_length / input_multislice.scanning_ns

    elif simulation_type == "CBED":
        data = []
        for i in range(len(output_multislice.data)):
            m2psi_tot = output_multislice.data[i].m2psi_tot
            if output_size is not None:
                m2psi_tot = fourier_interpolation(m2psi_tot, output_size)
            data.append(np.array(m2psi_tot))
        max_collection_angle_in_A = (
            max_collection_angle / energy2wavelength(energy) / 1e3
        )
        pixel_size = max(
            max_collection_angle_in_A * 2 / input_multislice.nx,
            max_collection_angle_in_A * 2 / input_multislice.ny,
        )
        data = np.array(data)
    fielname = atoms_conf.split(".")[0]
    np.save(fielname + "multem_out.npy", data)
    return data, pixel_size
