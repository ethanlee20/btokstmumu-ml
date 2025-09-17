
import numpy
import pandas


def square_matrix_transform(matrix_dataframe, vector_dataframe):

    """
    Multiply a dataframe of vectors 
    by a dataframe of square matrices.
    Return a dataframe.

    Only works for square matrices.
    """

    if not (
        numpy.sqrt(matrix_dataframe.shape[1]) 
        == vector_dataframe.shape[1]
    ):
        raise ValueError("Matrix must be square.")

    vector_length = vector_dataframe.shape[1]

    transformed_vector_dataframe = pandas.DataFrame(
        data=numpy.zeros(shape=vector_dataframe.shape),
        index=vector_dataframe.index,
        columns=vector_dataframe.columns,
        dtype="float64",
    )

    for i in range(vector_length):
        for j in range(vector_length):
            transformed_vector_dataframe.iloc[:, i] += (
                matrix_dataframe.iloc[:, vector_length * i + j]
                * vector_dataframe.iloc[:, j]
            )

    return transformed_vector_dataframe


def dot_product(vector_dataframe_1, vector_dataframe_2):

    """
    Compute the dot products of two vector dataframes.
    """

    if not (vector_dataframe_1.shape[1] == vector_dataframe_2.shape[1]):
        raise ValueError("Vector dimensions do not match.")
    
    vector_length = vector_dataframe_1.shape[1]

    result_series = pandas.Series(
        data=numpy.zeros(len(vector_dataframe_1)),
        index=vector_dataframe_1.index,
        dtype="float64",
    )

    for dimension in range(vector_length):
        result_series += (
            vector_dataframe_1.iloc[:, dimension] 
            * vector_dataframe_2.iloc[:, dimension]
        )

    return result_series


def vector_magnitude(vector_dataframe):
    
    """
    Compute the magnitude of each vector in a vector dataframe.
    Return a series.
    """

    result_series = numpy.sqrt(dot_product(vector_dataframe, vector_dataframe))
    
    return result_series


def cosine_angle(vector_dataframe_1, vector_dataframe_2):
    
    """
    Find the cosine of the angle between vectors in vector dataframes.
    Return a series.
    """

    result_series = (
        dot_product(vector_dataframe_1, vector_dataframe_2) 
        / (
            vector_magnitude(vector_dataframe_1)
            * vector_magnitude(vector_dataframe_2)
        )
    )

    return result_series


def cross_product_3d(three_vector_dataframe_1, three_vector_dataframe_2):

    """
    Find the cross product of 3-dimensional vectors 
    from two vector dataframes.
    Return a vector dataframe.
    """

    assert (
        three_vector_dataframe_1.shape[1] 
        == three_vector_dataframe_2.shape[1] 
        == 3
    )
    assert (
        three_vector_dataframe_1.shape[0] 
        == three_vector_dataframe_2.shape[0]
    )
    assert (
        three_vector_dataframe_1.index.equals(
            three_vector_dataframe_2.index
        )
    )

    three_vector_dataframe_1 = three_vector_dataframe_1.copy()
    three_vector_dataframe_2 = three_vector_dataframe_2.copy()

    three_vector_dataframe_1.columns = ["x", "y", "z"]
    three_vector_dataframe_2.columns = ["x", "y", "z"]

    cross_product_dataframe = pandas.DataFrame(
        data=numpy.zeros(
            shape=three_vector_dataframe_1.shape
        ),
        index=three_vector_dataframe_1.index,
        columns=three_vector_dataframe_1.columns,
        dtype="float64"
    )

    cross_product_dataframe["x"] = (
        three_vector_dataframe_1["y"] * three_vector_dataframe_2["z"]
        - three_vector_dataframe_1["z"] * three_vector_dataframe_2["y"]
    )
    cross_product_dataframe["y"] = (
        three_vector_dataframe_1["z"] * three_vector_dataframe_2["x"]
        - three_vector_dataframe_1["x"] * three_vector_dataframe_2["z"]
    )
    cross_product_dataframe["z"] = (
        three_vector_dataframe_1["x"] * three_vector_dataframe_2["y"]
        - three_vector_dataframe_1["y"] * three_vector_dataframe_2["x"]
    )

    return cross_product_dataframe


def unit_normal(three_vector_dataframe_1, three_vector_dataframe_2):
    
    """
    For planes specified by two three-vector dataframes,
    calculate the unit normal vectors.
    Return a vector dataframe.
    """

    normal_vector_dataframe = cross_product_3d(
        three_vector_dataframe_1, 
        three_vector_dataframe_2
    )
    
    unit_normal_vector_dataframe = normal_vector_dataframe.divide(
        vector_magnitude(normal_vector_dataframe), 
        axis="index"
    )

    return unit_normal_vector_dataframe



def convert_to_four_momentum_dataframe(dataframe_with_four_columns):
    
    """
    Create a four-momentum dataframe.

    Create a dataframe where each row 
    represents a four-momentum.
    The columns are well labeled.
    """

    four_momentum_dataframe = dataframe_with_four_columns.copy()
    four_momentum_dataframe.columns = ["E", "px", "py", "pz"]
    return four_momentum_dataframe


def convert_to_three_momentum_dataframe(dataframe_with_three_columns):

    """
    Create a three-momentum dataframe.

    Create a dataframe where each row 
    represents a three-momentum.
    The columns are well labeled.
    """

    three_momentum_dataframe = dataframe_with_three_columns.copy()
    three_momentum_dataframe.columns = ["px", "py", "pz"]
    return three_momentum_dataframe


def convert_to_three_velocity_dataframe(dataframe_with_three_columns):

    """
    Create a three-velocity dataframe.

    Create a dataframe where each row 
    represents a three-velocity.
    The columns are well labeled.
    """
    
    three_velocity_dataframe = dataframe_with_three_columns.copy()
    three_velocity_dataframe.columns = ["vx", "vy", "vz"]
    return three_velocity_dataframe


def calculate_invariant_mass_squared_of_two_particles(
    particle_one_four_momentum_dataframe, 
    particle_two_four_momentum_dataframe
):
    
    """
    Compute the squares of the invariant masses for 
    two particle systems.
    """

    particle_one_four_momentum_dataframe = convert_to_four_momentum_dataframe(
        particle_one_four_momentum_dataframe
    )
    particle_two_four_momentum_dataframe = convert_to_four_momentum_dataframe(
        particle_two_four_momentum_dataframe
    )

    sum_of_four_momenta_dataframe = (
        particle_one_four_momentum_dataframe 
        + particle_two_four_momentum_dataframe
    )

    sum_of_three_momenta_dataframe = convert_to_three_momentum_dataframe(
        sum_of_four_momenta_dataframe[["px", "py", "pz"]]
    )

    sum_of_three_momenta_magnitude_squared_dataframe = (
        vector_magnitude(sum_of_three_momenta_dataframe) ** 2
    )

    invariant_mass_squared_dataframe = (
        sum_of_four_momenta_dataframe["E"]**2 
        - sum_of_three_momenta_magnitude_squared_dataframe
    )
    
    return invariant_mass_squared_dataframe


def three_velocity_from_four_momentum_dataframe(four_momentum_dataframe):

    """
    Compute a three-velocity dataframe 
    from a four-momentum dataframe.
    """

    four_momentum_dataframe = convert_to_four_momentum_dataframe(
        four_momentum_dataframe
    )
    
    three_momentum_dataframe = convert_to_three_momentum_dataframe(
        four_momentum_dataframe[["px", "py", "pz"]]
    )

    three_velocity_dataframe = convert_to_three_velocity_dataframe(
        three_momentum_dataframe
        .multiply(1 / four_momentum_dataframe["E"], axis=0)
    )

    return three_velocity_dataframe


def calculate_lorentz_factor_series(three_velocity_dataframe):

    """
    Compute a series of Lorentz factors.
    """

    three_velocity_dataframe = convert_to_three_velocity_dataframe(
        three_velocity_dataframe
    )

    three_velocity_magnitude_series = vector_magnitude(three_velocity_dataframe)

    lorentz_factor_series = 1 / numpy.sqrt(1 - three_velocity_magnitude_series**2)

    return lorentz_factor_series


def compute_lorentz_boost_matrix_dataframe(three_velocity_dataframe):

    """
    Compute a dataframe of Lorentz boost matrices.
    """

    three_velocity_dataframe = convert_to_three_velocity_dataframe(
        three_velocity_dataframe
    )
    three_velocity_magnitude_series = vector_magnitude(three_velocity_dataframe)
    lorentz_factor_series = calculate_lorentz_factor_series(three_velocity_dataframe)

    boost_matrix_dataframe = pandas.DataFrame(
        data=numpy.zeros(shape=(three_velocity_dataframe.shape[0], 16)),
        index=three_velocity_dataframe.index,
        columns=[
            "b00",
            "b01",
            "b02",
            "b03",
            "b10",
            "b11",
            "b12",
            "b13",
            "b20",
            "b21",
            "b22",
            "b23",
            "b30",
            "b31",
            "b32",
            "b33",
        ],
    )

    boost_matrix_dataframe["b00"] = lorentz_factor_series
    boost_matrix_dataframe["b01"] = (
        -lorentz_factor_series 
        * three_velocity_dataframe["vx"]
    )
    boost_matrix_dataframe["b02"] = (
        -lorentz_factor_series 
        * three_velocity_dataframe["vy"]
    )
    boost_matrix_dataframe["b03"] = (
        -lorentz_factor_series 
        * three_velocity_dataframe["vz"]
    )
    boost_matrix_dataframe["b10"] = (
        -lorentz_factor_series 
        * three_velocity_dataframe["vx"]
    )
    boost_matrix_dataframe["b11"] = (
        1
        + (lorentz_factor_series - 1)
        * three_velocity_dataframe["vx"] ** 2
        / three_velocity_magnitude_series**2
    )
    boost_matrix_dataframe["b12"] = (
        (lorentz_factor_series - 1)
        * three_velocity_dataframe["vx"]
        * three_velocity_dataframe["vy"]
        / three_velocity_magnitude_series**2
    )
    boost_matrix_dataframe["b13"] = (
        (lorentz_factor_series - 1)
        * three_velocity_dataframe["vx"]
        * three_velocity_dataframe["vz"]
        / three_velocity_magnitude_series**2
    )
    boost_matrix_dataframe["b20"] = (
        -lorentz_factor_series 
        * three_velocity_dataframe["vy"]
    )
    boost_matrix_dataframe["b21"] = (
        (lorentz_factor_series - 1)
        * three_velocity_dataframe["vy"]
        * three_velocity_dataframe["vx"]
        / three_velocity_magnitude_series**2
    )
    boost_matrix_dataframe["b22"] = (
        1
        + (lorentz_factor_series - 1)
        * three_velocity_dataframe["vy"] ** 2
        / three_velocity_magnitude_series**2
    )
    boost_matrix_dataframe["b23"] = (
        (lorentz_factor_series - 1)
        * three_velocity_dataframe["vy"]
        * three_velocity_dataframe["vz"]
        / three_velocity_magnitude_series**2
    )
    boost_matrix_dataframe["b30"] = (
        -lorentz_factor_series * 
        three_velocity_dataframe["vz"]
    )
    boost_matrix_dataframe["b31"] = (
        (lorentz_factor_series - 1)
        * three_velocity_dataframe["vz"]
        * three_velocity_dataframe["vx"]
        / three_velocity_magnitude_series**2
    )
    boost_matrix_dataframe["b32"] = (
        (lorentz_factor_series - 1)
        * three_velocity_dataframe["vz"]
        * three_velocity_dataframe["vy"]
        / three_velocity_magnitude_series**2
    )
    boost_matrix_dataframe["b33"] = (
        1
        + (lorentz_factor_series - 1)
        * three_velocity_dataframe["vz"] ** 2
        / three_velocity_magnitude_series**2
    )

    return boost_matrix_dataframe


def boost(reference_four_momentum_dataframe, four_vector_dataframe): # four vector?

    """
    Lorentz boost a dataframe of four-vectors 
    to a reference four momentum dataframe.
    """

    reference_three_velocity_dataframe = (
        three_velocity_from_four_momentum_dataframe(
            reference_four_momentum_dataframe
        )
    )

    boost_matrix_dataframe = compute_lorentz_boost_matrix_dataframe(
        reference_three_velocity_dataframe
    )

    transformed_four_vector_dataframe = square_matrix_transform(
        boost_matrix_dataframe, four_vector_dataframe
    )

    return transformed_four_vector_dataframe


def calculate_cosine_theta_ell(
    positive_lepton_four_momentum_dataframe, 
    negative_lepton_four_momentum_dataframe, 
    B_meson_four_momentum_dataframe
):
    
    """
    Find the cosine of the lepton helicity angle 
    for B -> K* ell+ ell-. Return a pandas series.
    """

    positive_lepton_four_momentum_dataframe = convert_to_four_momentum_dataframe(
        positive_lepton_four_momentum_dataframe
    )
    negative_lepton_four_momentum_dataframe = convert_to_four_momentum_dataframe(
        negative_lepton_four_momentum_dataframe
    )
    B_meson_four_momentum_dataframe = convert_to_four_momentum_dataframe(
        B_meson_four_momentum_dataframe
    )

    dilepton_four_momentum_dataframe = (
        positive_lepton_four_momentum_dataframe + 
        negative_lepton_four_momentum_dataframe
    )

    positive_lepton_four_momentum_in_dilepton_frame_dataframe = boost(
        reference_four_momentum_dataframe=dilepton_four_momentum_dataframe, 
        four_vector_dataframe=positive_lepton_four_momentum_dataframe
    )
    positive_lepton_three_momentum_in_dilepton_frame_dataframe = convert_to_three_momentum_dataframe(
        positive_lepton_four_momentum_in_dilepton_frame_dataframe[["px", "py", "pz"]]
    )

    dilepton_four_momentum_in_B_frame_dataframe = boost(
        reference_four_momentum_dataframe=B_meson_four_momentum_dataframe, 
        four_vector_dataframe=dilepton_four_momentum_dataframe
    )
    dilepton_three_momentum_in_B_frame_dataframe = convert_to_three_momentum_dataframe(
        dilepton_four_momentum_in_B_frame_dataframe[["px", "py", "pz"]]
    )

    cosine_theta_ell_series = cosine_angle(
        vector_dataframe_1=dilepton_three_momentum_in_B_frame_dataframe, 
        vector_dataframe_2=positive_lepton_three_momentum_in_dilepton_frame_dataframe
    )

    return cosine_theta_ell_series


def calculate_cosine_theta_K(
    K_four_momentum_dataframe, 
    K_star_four_momentum_dataframe, 
    B_meson_four_momentum_dataframe
):
    
    """
    Find the cosine of the K* helicity 
    angle for B -> K* ell+ ell-.
    """

    K_four_momentum_dataframe = convert_to_four_momentum_dataframe(
        K_four_momentum_dataframe
    )
    K_star_four_momentum_dataframe = convert_to_four_momentum_dataframe(
        K_star_four_momentum_dataframe
    )
    B_meson_four_momentum_dataframe = convert_to_four_momentum_dataframe(
        B_meson_four_momentum_dataframe
    )

    K_four_momentum_in_K_star_frame_dataframe = boost(
        reference_four_momentum_dataframe=K_star_four_momentum_dataframe, 
        four_vector_dataframe=K_four_momentum_dataframe
    )
    K_three_momentum_in_K_star_frame_dataframe = convert_to_three_momentum_dataframe(
        K_four_momentum_in_K_star_frame_dataframe[["px", "py", "pz"]]
    )

    K_star_four_momentum_in_B_frame_dataframe = boost(
        reference_four_momentum_dataframe=B_meson_four_momentum_dataframe, 
        four_vector_dataframe=K_star_four_momentum_dataframe
    )
    K_star_three_momentum_in_B_frame_dataframe = convert_to_three_momentum_dataframe(
        K_star_four_momentum_in_B_frame_dataframe[["px", "py", "pz"]]
    )

    cosine_theta_K_series = cosine_angle(
        vector_dataframe_1=K_star_three_momentum_in_B_frame_dataframe, 
        vector_dataframe_2=K_three_momentum_in_K_star_frame_dataframe
    )

    return cosine_theta_K_series


def calculate_unit_normal_vector_to_K_star_K_plane(
    B_meson_four_momentum_dataframe, 
    K_star_four_momentum_dataframe, 
    K_four_momentum_dataframe
):
    
    """
    Find the unit normal to the plane made 
    by the direction vectors of the K* and K 
    in B -> K* ell+ ell-.
    """

    B_meson_four_momentum_dataframe = convert_to_four_momentum_dataframe(
        B_meson_four_momentum_dataframe
    )
    K_star_four_momentum_dataframe = convert_to_four_momentum_dataframe(
        K_star_four_momentum_dataframe
    )
    K_four_momentum_dataframe = convert_to_four_momentum_dataframe(
        K_four_momentum_dataframe
    )

    K_four_momentum_in_K_star_frame_dataframe = boost(
        reference_four_momentum_dataframe=K_star_four_momentum_dataframe, 
        four_vector_dataframe=K_four_momentum_dataframe
    )
    K_three_momentum_in_K_star_frame_dataframe = convert_to_three_momentum_dataframe(
        K_four_momentum_in_K_star_frame_dataframe[["px", "py", "pz"]]
    )

    K_star_four_momentum_in_B_frame_dataframe = boost(
        reference_four_momentum_dataframe=B_meson_four_momentum_dataframe, 
        four_vector_dataframe=K_star_four_momentum_dataframe
    )
    K_star_three_momentum_in_B_frame_dataframe = convert_to_three_momentum_dataframe(
        K_star_four_momentum_in_B_frame_dataframe[["px", "py", "pz"]]
    )

    unit_normal_vector_to_K_star_K_plane_dataframe = unit_normal(
        three_vector_dataframe_1=K_three_momentum_in_K_star_frame_dataframe, 
        three_vector_dataframe_2=K_star_three_momentum_in_B_frame_dataframe
    )

    return unit_normal_vector_to_K_star_K_plane_dataframe


def calculate_unit_normal_vector_to_dilepton_positive_lepton_plane(
    B_meson_four_momentum_dataframe, 
    positive_lepton_four_momentum_dataframe, 
    negative_lepton_four_momentum_dataframe
):
    
    """
    Find the unit normal to the plane made by
    the direction vectors of the dilepton system and
    the positively charged lepton in B -> K* ell+ ell-.
    """

    B_meson_four_momentum_dataframe = convert_to_four_momentum_dataframe(
        B_meson_four_momentum_dataframe
    )
    positive_lepton_four_momentum_dataframe = convert_to_four_momentum_dataframe(
        positive_lepton_four_momentum_dataframe
    )
    negative_lepton_four_momentum_dataframe = convert_to_four_momentum_dataframe(
        negative_lepton_four_momentum_dataframe
    )

    dilepton_four_momentum_dataframe = (
        positive_lepton_four_momentum_dataframe 
        + negative_lepton_four_momentum_dataframe
    )

    positive_lepton_four_momentum_in_dilepton_frame_dataframe = boost(
        reference_four_momentum_dataframe=dilepton_four_momentum_dataframe, 
        four_vector_dataframe=positive_lepton_four_momentum_dataframe
    )
    positive_lepton_three_momentum_in_dilepton_frame_dataframe = convert_to_three_momentum_dataframe(
        positive_lepton_four_momentum_in_dilepton_frame_dataframe[["px", "py", "pz"]]
    )

    dilepton_four_momentum_in_B_frame_dataframe = boost(
        reference_four_momentum_dataframe=B_meson_four_momentum_dataframe, 
        four_vector_dataframe=dilepton_four_momentum_dataframe
    )
    dilepton_three_momentum_in_B_frame_dataframe = convert_to_three_momentum_dataframe(
        dilepton_four_momentum_in_B_frame_dataframe[["px", "py", "pz"]]
    )

    result = unit_normal(
        three_vector_dataframe_1=positive_lepton_three_momentum_in_dilepton_frame_dataframe, 
        three_vector_dataframe_2=dilepton_three_momentum_in_B_frame_dataframe
    )

    return result


def calculate_cosine_chi(
    B_meson_four_momentum_dataframe,
    K_four_momentum_dataframe,
    K_star_four_momentum_dataframe,
    positive_lepton_four_momentum_dataframe,
    negative_lepton_four_momentum_dataframe
):
    
    """
    Find the cosine of the decay angle chi 
    in B -> K* ell+ ell-.

    Chi is the angle between the K* K decay plane 
    and the dilepton ell+ decay plane.
    """

    unit_normal_vector_to_K_star_K_plane_dataframe = (
        calculate_unit_normal_vector_to_K_star_K_plane(
            B_meson_four_momentum_dataframe=B_meson_four_momentum_dataframe, 
            K_star_four_momentum_dataframe=K_star_four_momentum_dataframe, 
            K_four_momentum_dataframe=K_four_momentum_dataframe
        )
    )
    
    unit_normal_vector_to_dilepton_positive_lepton_plane_dataframe = (
        calculate_unit_normal_vector_to_dilepton_positive_lepton_plane(
            B_meson_four_momentum_dataframe=B_meson_four_momentum_dataframe, 
            positive_lepton_four_momentum_dataframe=positive_lepton_four_momentum_dataframe, 
            negative_lepton_four_momentum_dataframe=negative_lepton_four_momentum_dataframe
        )
    )

    cosine_chi_series = dot_product(
        vector_dataframe_1=unit_normal_vector_to_K_star_K_plane_dataframe,
        vector_dataframe_2=unit_normal_vector_to_dilepton_positive_lepton_plane_dataframe,
    )

    return cosine_chi_series


def find_chi(
    B_meson_four_momentum_dataframe,
    K_four_momentum_dataframe,
    K_star_four_momentum_dataframe,
    positive_lepton_four_momentum_dataframe,
    negative_lepton_four_momentum_dataframe,
):
    
    """
    Find the decay angle chi in B -> K* ell+ ell-.

    Chi is the angle between the K* K decay plane 
    and the dilepton ell+ decay plane.
    It ranges from 0 to 2*pi.
    """

    def calculate_sign_of_chi(
        B_meson_four_momentum_dataframe,
        K_star_four_momentum_dataframe,
        K_four_momentum_dataframe,
        positive_lepton_four_momentum_dataframe,
        negative_lepton_four_momentum_dataframe
    ):

        unit_normal_vector_to_K_star_K_plane_dataframe = (
            calculate_unit_normal_vector_to_K_star_K_plane(
                B_meson_four_momentum_dataframe=B_meson_four_momentum_dataframe, 
                K_star_four_momentum_dataframe=K_star_four_momentum_dataframe, 
                K_four_momentum_dataframe=K_four_momentum_dataframe
            )
        )

        unit_normal_vector_to_dilepton_positive_lepton_plane_dataframe = (
            calculate_unit_normal_vector_to_dilepton_positive_lepton_plane(
                B_meson_four_momentum_dataframe=B_meson_four_momentum_dataframe, 
                positive_lepton_four_momentum_dataframe=positive_lepton_four_momentum_dataframe, 
                negative_lepton_four_momentum_dataframe=negative_lepton_four_momentum_dataframe
            )
        )

        normal_vector_cross_product_dataframe = cross_product_3d(
            three_vector_dataframe_1=unit_normal_vector_to_dilepton_positive_lepton_plane_dataframe,
            three_vector_dataframe_2=unit_normal_vector_to_K_star_K_plane_dataframe
        )

        K_star_four_momentum_dataframe = convert_to_four_momentum_dataframe(
            K_star_four_momentum_dataframe
        )
        K_star_four_momentum_in_B_frame_dataframe = boost(
            reference_four_momentum_dataframe=B_meson_four_momentum_dataframe, 
            four_vector_dataframe=K_star_four_momentum_dataframe
        )
        K_star_three_momentum_in_B_frame_dataframe = convert_to_three_momentum_dataframe(
            K_star_four_momentum_in_B_frame_dataframe[["px", "py", "pz"]]
        )

        dot_product_of_cross_product_and_K_star_three_momentum_series = dot_product(
            vector_dataframe_1=normal_vector_cross_product_dataframe, 
            vector_dataframe_2=K_star_three_momentum_in_B_frame_dataframe
        )

        sign = numpy.sign(dot_product_of_cross_product_and_K_star_three_momentum_series) 

        return sign
    
    def convert_to_positive_angles(chi): 

        return chi.where(chi > 0, chi + 2 * numpy.pi)

    cosine_chi_series = calculate_cosine_chi(
        B_meson_four_momentum_dataframe=B_meson_four_momentum_dataframe,
        K_four_momentum_dataframe=K_four_momentum_dataframe,
        K_star_four_momentum_dataframe=K_star_four_momentum_dataframe,
        positive_lepton_four_momentum_dataframe=positive_lepton_four_momentum_dataframe,
        negative_lepton_four_momentum_dataframe=negative_lepton_four_momentum_dataframe
    )

    sign_of_chi = calculate_sign_of_chi(
        B_meson_four_momentum_dataframe=B_meson_four_momentum_dataframe,
        K_star_four_momentum_dataframe=K_star_four_momentum_dataframe,
        K_four_momentum_dataframe=K_four_momentum_dataframe,
        positive_lepton_four_momentum_dataframe=positive_lepton_four_momentum_dataframe,
        negative_lepton_four_momentum_dataframe=negative_lepton_four_momentum_dataframe
    )

    chi_series = sign_of_chi * numpy.arccos(cosine_chi_series)

    chi_series = convert_to_positive_angles(chi_series)

    return chi_series


def calculate_difference_between_invariant_masses_of_K_pi_system_and_K_star(
    K_four_momentum_dataframe, 
    pi_four_momentum_dataframe
):

    """
    Calcualate the difference between the 
    invariant mass of the K pi system
    and the K*'s invariant mass (PDG value).
    """

    invariant_mass_of_K_star = 0.892

    invariant_mass_of_K_pi_system_dataframe = numpy.sqrt(
        calculate_invariant_mass_squared_of_two_particles(
            particle_one_four_momentum_dataframe=K_four_momentum_dataframe, 
            particle_two_four_momentum_dataframe=pi_four_momentum_dataframe
        )
    )

    difference_series = invariant_mass_of_K_pi_system_dataframe - invariant_mass_of_K_star
    
    return difference_series


def calculate_B_to_K_star_mu_mu_variables(dataframe):

    """
    Calculate detecor and generator level variables of B -> K* mu+ mu- decays.

    Variables: 
    q^2, cosine theta mu, cosine theta K, cosine chi, chi, and the
    difference between K pi invariant mass and K* PDG invariant mass
    """

    B_meson_measured_four_momentum_dataframe = convert_to_four_momentum_dataframe(
        dataframe[["E", "px", "py", "pz"]]
    )
    B_meson_generated_four_momentum_dataframe = convert_to_four_momentum_dataframe(
        dataframe[["mcE", "mcPX", "mcPY", "mcPZ"]]
    )
    positive_muon_measured_four_momentum_dataframe = convert_to_four_momentum_dataframe(
        dataframe[["mu_p_E", "mu_p_px", "mu_p_py", "mu_p_pz"]]
    )
    positive_muon_generated_four_momentum_dataframe = convert_to_four_momentum_dataframe(
        dataframe[["mu_p_mcE", "mu_p_mcPX", "mu_p_mcPY", "mu_p_mcPZ"]]
    )
    negative_muon_measured_four_momentum_dataframe = convert_to_four_momentum_dataframe(
        dataframe[["mu_m_E", "mu_m_px", "mu_m_py", "mu_m_pz"]]
    )
    negative_muon_generated_four_momentum_dataframe = convert_to_four_momentum_dataframe(
        dataframe[["mu_m_mcE", "mu_m_mcPX", "mu_m_mcPY", "mu_m_mcPZ"]]
    )
    K_measured_four_momentum_dataframe = convert_to_four_momentum_dataframe(
        dataframe[["K_p_E", "K_p_px", "K_p_py", "K_p_pz"]]
    )
    K_generated_four_momentum_dataframe = convert_to_four_momentum_dataframe(
        dataframe[["K_p_mcE", "K_p_mcPX", "K_p_mcPY", "K_p_mcPZ"]]
    )
    pi_measured_four_momentum_dataframe = convert_to_four_momentum_dataframe(
        dataframe[["pi_m_E", "pi_m_px", "pi_m_py", "pi_m_pz"]]
    )
    pi_generated_four_momentum_dataframe = convert_to_four_momentum_dataframe(
        dataframe[["pi_m_mcE", "pi_m_mcPX", "pi_m_mcPY", "pi_m_mcPZ"]]
    )
    K_star_measured_four_momentum_dataframe = convert_to_four_momentum_dataframe(
        dataframe[["KST0_E", "KST0_px", "KST0_py", "KST0_pz"]]
    )
    K_star_generated_four_momentum_dataframe = convert_to_four_momentum_dataframe(
        dataframe[["KST0_mcE", "KST0_mcPX", "KST0_mcPY", "KST0_mcPZ"]]
    )

    dataframe = dataframe.copy()

    dataframe[Names_of_Variables().q_squared] = calculate_invariant_mass_squared_of_two_particles(
        particle_one_four_momentum_dataframe=positive_muon_measured_four_momentum_dataframe, 
        particle_two_four_momentum_dataframe=negative_muon_measured_four_momentum_dataframe
    )
    dataframe[f"{Names_of_Variables().q_squared}_mc"] = calculate_invariant_mass_squared_of_two_particles(
        particle_one_four_momentum_dataframe=positive_muon_generated_four_momentum_dataframe, 
        particle_two_four_momentum_dataframe=negative_muon_generated_four_momentum_dataframe
    )
    dataframe[Names_of_Variables().cos_theta_mu] = calculate_cosine_theta_ell(
        positive_lepton_four_momentum_dataframe=positive_muon_measured_four_momentum_dataframe, 
        negative_lepton_four_momentum_dataframe=negative_muon_measured_four_momentum_dataframe, 
        B_meson_four_momentum_dataframe=B_meson_measured_four_momentum_dataframe
    )
    dataframe[f"{Names_of_Variables().cos_theta_mu}_mc"] = calculate_cosine_theta_ell(
        positive_lepton_four_momentum_dataframe=positive_muon_generated_four_momentum_dataframe, 
        negative_lepton_four_momentum_dataframe=negative_muon_generated_four_momentum_dataframe, 
        B_meson_four_momentum_dataframe=B_meson_generated_four_momentum_dataframe
    )
    dataframe[Names_of_Variables().cos_k] = calculate_cosine_theta_K(
        K_four_momentum_dataframe=K_measured_four_momentum_dataframe, 
        K_star_four_momentum_dataframe=K_star_measured_four_momentum_dataframe, 
        B_meson_four_momentum_dataframe=B_meson_measured_four_momentum_dataframe
    )
    dataframe[f"{Names_of_Variables().cos_k}_mc"] = calculate_cosine_theta_K(
        K_four_momentum_dataframe=K_generated_four_momentum_dataframe, 
        K_star_four_momentum_dataframe=K_star_generated_four_momentum_dataframe, 
        B_meson_four_momentum_dataframe=B_meson_generated_four_momentum_dataframe
    )
    dataframe["coschi"] = calculate_cosine_chi(
        B_meson_four_momentum_dataframe=B_meson_measured_four_momentum_dataframe,
        K_four_momentum_dataframe=K_measured_four_momentum_dataframe,
        K_star_four_momentum_dataframe=K_star_measured_four_momentum_dataframe,
        positive_lepton_four_momentum_dataframe=positive_muon_measured_four_momentum_dataframe,
        negative_lepton_four_momentum_dataframe=negative_muon_measured_four_momentum_dataframe,
    )
    dataframe["coschi_mc"] = calculate_cosine_chi(
        B_meson_four_momentum_dataframe=B_meson_generated_four_momentum_dataframe,
        K_four_momentum_dataframe=K_generated_four_momentum_dataframe,
        K_star_four_momentum_dataframe=K_star_generated_four_momentum_dataframe,
        positive_lepton_four_momentum_dataframe=positive_muon_generated_four_momentum_dataframe,
        negative_lepton_four_momentum_dataframe=negative_muon_generated_four_momentum_dataframe,
    )
    dataframe[Names_of_Variables().chi] = find_chi(
        B_meson_four_momentum_dataframe=B_meson_measured_four_momentum_dataframe,
        K_four_momentum_dataframe=K_measured_four_momentum_dataframe,
        K_star_four_momentum_dataframe=K_star_measured_four_momentum_dataframe,
        positive_lepton_four_momentum_dataframe=positive_muon_measured_four_momentum_dataframe,
        negative_lepton_four_momentum_dataframe=negative_muon_measured_four_momentum_dataframe,
    )
    dataframe[f"{Names_of_Variables().chi}_mc"] = find_chi(
        B_meson_four_momentum_dataframe=B_meson_generated_four_momentum_dataframe,
        K_four_momentum_dataframe=K_generated_four_momentum_dataframe,
        K_star_four_momentum_dataframe=K_star_generated_four_momentum_dataframe,
        positive_lepton_four_momentum_dataframe=positive_muon_generated_four_momentum_dataframe,
        negative_lepton_four_momentum_dataframe=negative_muon_generated_four_momentum_dataframe,
    )
    dataframe["invM_K_pi_shifted"] = calculate_difference_between_invariant_masses_of_K_pi_system_and_K_star(
        K_four_momentum_dataframe=K_measured_four_momentum_dataframe,
        pi_four_momentum_dataframe=pi_measured_four_momentum_dataframe
    )
    dataframe["invM_K_pi_shifted_mc"] = calculate_difference_between_invariant_masses_of_K_pi_system_and_K_star(
        K_four_momentum_dataframe=K_generated_four_momentum_dataframe,
        pi_four_momentum_dataframe=pi_generated_four_momentum_dataframe
    )

    return dataframe


def calculate_resolution(detector_level_dataframe, name_of_variable):

    """
    Calculate the resolution.

    The resolution of a variable is defined as the
    reconstructed value minus the MC truth value.

    If the variable is chi, periodicity is accounted for.
    """

    def apply_periodicity(resolution):

        resolution = resolution.where(resolution < numpy.pi, resolution - 2 * numpy.pi)
        resolution = resolution.where(resolution > -numpy.pi, resolution + 2 * numpy.pi)
        return resolution
    
    measured = detector_level_dataframe[name_of_variable]
    generated = detector_level_dataframe[name_of_variable+'_mc']
    resolution = measured - generated

    if name_of_variable == Names_of_Variables().chi:

        resolution = apply_periodicity(resolution)

    return resolution


def calculate_efficiency(generator_level_series, detector_level_series, num_bins, bounds):
    
    """
    Calculate the efficiency per bin.

    The efficiency of bin i is defined as the number of
    detector entries in i divided by the number of generator
    entries in i.

    The error for bin i is calculated as the squareroot of the
    number of detector entries in i divided by the number of
    generator entries in i.
    """

    def make_bins(bounds, num_bins):

        assert len(bounds) == 2
        bin_edges, bin_width = numpy.linspace(start=bounds[0], stop=bounds[1], num=num_bins+1, retstep=True)
        bin_middles = numpy.linspace(start=bounds[0]+bin_width/2, stop=bounds[1]-bin_width/2, num=num_bins)
        return bin_edges, bin_middles

    bin_edges, bin_middles = make_bins(bounds, num_bins)

    generator_level_histogram, _ = numpy.histogram(generator_level_series, bins=bin_edges)
    detector_level_histogram, _ = numpy.histogram(detector_level_series, bins=bin_edges)

    print("num events generator: ", generator_level_histogram.sum())
    print("num events detector: ", detector_level_histogram.sum())

    efficiencies = detector_level_histogram / generator_level_histogram
    errors = numpy.sqrt(detector_level_histogram) / generator_level_histogram

    return bin_middles, efficiencies, errors


def bin_in_var(
    dataframe,
    name_of_binning_variable, 
    start, 
    stop, 
    num_bins,
    return_bin_edges=False,
):
    
    bin_edges = numpy.linspace(start=start, stop=stop, num=num_bins+1)
    bins = pandas.cut(dataframe[name_of_binning_variable], bin_edges, include_lowest=True) # the interval each event falls into
    groupby_binned = dataframe.groupby(bins, observed=False)

    if return_bin_edges:
        return groupby_binned, bin_edges
    return groupby_binned


def calc_bin_middles(start, stop, num_bins):

    bin_edges, step = numpy.linspace(
        start=start,
        stop=stop,
        num=num_bins+1,
        retstep=True,
    )
    bin_middles = bin_edges[:-1] + step/2 
    return bin_middles


def calc_afb_of_q_sq(dataframe, num_bins, start, stop):

    """
    Calcuate Afb as a function of q squared.
    Afb is the forward-backward asymmetry.
    """

    def calc_num_forward(df):

        return df["costheta_mu"][(df["costheta_mu"] > 0) & (df["costheta_mu"] < 1)].count()
    
    def calc_num_backward(df):

        return df["costheta_mu"][(df["costheta_mu"] > -1) & (df["costheta_mu"] < 0)].count()

    def calc_afb(df):

        f = calc_num_forward(df)
        b = calc_num_backward(df)
        return (f - b) / (f + b)

    def calc_afb_err(df):

        f = calc_num_forward(df)
        b = calc_num_backward(df)
        f_stdev = numpy.sqrt(f)
        b_stdev = numpy.sqrt(b)
        return 2*f*b / (f+b)**2 * numpy.sqrt((f_stdev/f)**2 + (b_stdev/b)**2) # this is stdev?

    groupby_binned = bin_in_var(
        dataframe, 
        "q_squared", 
        start,
        stop,
        num_bins
    )
    
    afbs = groupby_binned.apply(calc_afb)
    errs = groupby_binned.apply(calc_afb_err)
    bin_middles = calc_bin_middles(start, stop, num_bins)

    return bin_middles, afbs, errs


def calc_afb_of_q_sq_over_dc9(dataframe, num_bins, start, stop):

    dc9_values = []
    bin_mids_over_dc9 = []
    afbs_over_dc9 = []
    afb_errs_over_dc9 = []

    for dc9, df in (dataframe.groupby("dc9")):

        dc9_values.append(dc9)
        bin_mids, afbs, afb_errs = calc_afb_of_q_sq(df, num_bins, start, stop)
        bin_mids_over_dc9.append(bin_mids)
        afbs_over_dc9.append(afbs)
        afb_errs_over_dc9.append(afb_errs)

    return dc9_values, bin_mids_over_dc9, afbs_over_dc9, afb_errs_over_dc9


def calc_s5_of_q_sq(dataframe, num_bins, start, stop):
    
    def calc_num_forward(df):

        cos_theta_k = df["costheta_K"]
        chi = df["chi"]
        
        return df[
            (((cos_theta_k > 0) & (cos_theta_k < 1)) & ((chi > 0) & (chi < pi/2)))
            | (((cos_theta_k > 0) & (cos_theta_k < 1)) & ((chi > 3*pi/2) & (chi < 2*pi)))
            | (((cos_theta_k > -1) & (cos_theta_k < 0)) & ((chi > pi/2) & (chi < 3*pi/2)))
        ].count().min()

    def calc_num_backward(df):

        cos_theta_k = df["costheta_K"]
        chi = df["chi"]
        
        return df[
            (((cos_theta_k > 0) & (cos_theta_k < 1)) & ((chi > pi/2) & (chi < 3*pi/2)))
            | (((cos_theta_k > -1) & (cos_theta_k < 0)) & ((chi > 0) & (chi < pi/2)))
            | (((cos_theta_k > -1) & (cos_theta_k < 0)) & ((chi > 3*pi/2) & (chi < 2*pi)))
        ].count().min()

    def calc_s5(df):

        f = calc_num_forward(df)
        b = calc_num_backward(df)

        try: 

            s5 = 4/3 * (f - b) / (f + b)

        except ZeroDivisionError:

            print("S5 calculation: division by 0, returning nan")
            s5 = numpy.nan

        return s5

    def calc_s5_err(df):

        """
        Calculate the error of S_5.

        The error is calculated by assuming the forward and backward
        regions have uncorrelated Poisson errors and propagating
        the errors.
        """

        f = calc_num_forward(df)
        b = calc_num_backward(df)

        f_stdev = numpy.sqrt(f)
        b_stdev = numpy.sqrt(b)

        try: 

            err =  4/3 * 2*f*b / (f+b)**2 * numpy.sqrt((f_stdev/f)**2 + (b_stdev/b)**2) # this is stdev?

        except ZeroDivisionError:

            print("S5 error calculation: division by 0, returning nan")
            err = nan
        
        return err

    groupby_binned = bin_in_var(
        dataframe, 
        "q_squared", 
        start,
        stop,
        num_bins,     
    )
    
    s5s = groupby_binned.apply(calc_s5)
    errs = groupby_binned.apply(calc_s5_err)
    bin_middles = calc_bin_middles(start, stop, num_bins)

    return bin_middles, s5s, errs


def calc_s5_of_q_sq_over_dc9(dataframe, num_bins, start, stop):

    dc9_values = []
    bin_mids_over_dc9 = []
    s5s_over_dc9 = []
    s5_errs_over_dc9 = []

    for dc9, df in (dataframe.groupby("dc9")):

        dc9_values.append(dc9)
        bin_middles, s5s, s5_errs = calc_s5_of_q_sq(df, num_bins, start, stop)
        bin_mids_over_dc9.append(bin_middles)
        s5s_over_dc9.append(s5s)
        s5_errs_over_dc9.append(s5_errs)

    return dc9_values, bin_mids_over_dc9, s5s_over_dc9, s5_errs_over_dc9