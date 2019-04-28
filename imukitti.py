#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Python Translation of the IMU Kitti example
"""
__author__   = "Ludwig Auer <ludwig.auer@gmail.com>"

import time

import numpy as np
import pandas as pd
import gtsam as gs

import matplotlib.pyplot as plt
from utils import Vector, Matrix

# iSAM2 estimation parameters
SOLVER_TYPE         = 'CHOLESKY'
FIRST_GPS_POSE      = 1
GPS_SKIP            = 10
SIGMA_INIT_GPS      = 1.0 / 0.07
SIGMA_INIT_POSE     = 1.0
SIGMA_INIT_BIAS_ACC = 0.1
SIGMA_INIT_BIAS_GYR = 5e-5
RELIN_SKIP          = 10
W_CORIOLIS          = Vector([0., 0., 0.])
ZERO_VECTOR         = Vector([0., 0., 0.])
G                   = Vector([0., 0., -9.81])

# Convenience functions
def read_imu_data_from_csv(fn, sep=' '):
    """Reads IMU or GPS data from a csv (or tsv) file

    Args:
        fn (str): Filename of .csv file with IMU measurements.
        sep (str): The column separator used in the file.

    Returns:
        pandas.DataFrame: Pandas DataFrame of IMU measurements.~
    """
    return pd.read_csv(fn, sep=sep)


def read_gps_data_from_csv(fn, sep=','):
    """Reads IMU or GPS data from a csv (or tsv) file. Adds a `Position` column.

    Args:
        fn (str): Filename of input file with GPS measurements.
        sep (str): The column separator used in the file.

    Returns:
        pandas.DataFrame: Pandas DataFrame of GPS measurements.~
    """
    df = pd.read_csv(fn, sep=sep)
    df["Position"] = df.apply(lambda row: gs.Point3(row.X, row.Y, row.Z), axis=1)
    return df

# Convenience functions
def read_imu_metadata_from_csv(fn, sep=' '):
    """Reads IMU metadata from a csv (or tsv) file

    Args:
        fn (str): Filename of .csv file with IMU metadata.
        sep (str): The column separator used in the file.

    Returns:
        pandas.DataFrame: Pandas DataFrame of IMU measurements.~
    """
    return pd.read_csv(fn, sep=sep).ix[0]


def get_initial_pose_noise_model(sigma_init_pose):
    """Sets up noise model for initial pose.

    Args:
        sigma_init_pose (float):

    Returns:
        gtsam.noiseModel_Isotropic.Precisions
    """
    return gs.noiseModel_Isotropic.Precisions(
        Vector(
            [0.] * 3 + \
            [sigma_init_pose] * 3
        )
    )

def get_imu_initial_bias_noise_model(sigma_init_bias_acc, sigma_init_bias_gyr):
    """Sets up noise model for initial IMU bias factor.

    Args:
        sigma_init_bias_acc (float):
        sigma_init_bias_gyr (float):

    Returns:
        gtsam.noiseModel_Isotropic.Sigmas
    """
    return gs.noiseModel_Isotropic.Sigmas(
        Vector(
            [sigma_init_bias_acc] * 3 + \
            [sigma_init_bias_gyr] * 3
        )
    )

def get_imu_bias_between_factor_noise_model(sigma_bias_acc, sigma_bias_gyr):
    """Sets up noise model for the IMU between factor.

    Args:
        sigma_bias_acc (float):
        sigma_bias_gyr (float):

    Returns:
        Vector
    """
    return Vector(
        [sigma_bias_acc] * 3 + \
        [sigma_bias_gyr] * 3
    )

def get_imu_preintegration_params(sigma_acc, sigma_gyr, sigma_int):
    """Sets up a (process) noise model for the IMU preintegration.

    Args:
        sigma_acc (float):
        sigma_gyr (float):
        sigma_int (float):

    Returns:
        gtsam.PreintegrationParams
    """
    imu_params = gs.PreintegrationParams(G) # Uses the G-vector
    imu_params.setAccelerometerCovariance(
        sigma_acc ** 2 * np.identity(3, np.float)
    )
    imu_params.setGyroscopeCovariance(
        sigma_gyr ** 2 * np.identity(3, np.float)
    )
    imu_params.setIntegrationCovariance(
        sigma_int ** 2 * np.identity(3, np.float)
    )
    return imu_params

def get_gps_pos_noise_model(sigma_init_gps):
    """Sets up noise model for initial GPS measurement

    Args:
        sigma_init_gps (float)

    Returns:
        gtsam.noiseModel_Diagonal.Precisions
    """
    return gs.noiseModel_Diagonal.Precisions(
        Vector(
            [0.0] * 3 + \
            [sigma_init_gps] * 3
        )
    )

def get_gps_vel_noise_model():
    """Sets up a noise model for the GPS velocity measurement factor.
    """
    return gs.noiseModel_Isotropic.Sigma(3., 1000.)


if __name__ == "__main__":
    """Main iSAM2 estimation loop.

    Steps:

       (1) we read the measurements
       (2) we create the corresponding factors in the graph
       (3) we solve the graph to obtain the optimal estimate
           of the robot trajectory
    """

    # For results
    velocities = []

    # Read GPS and IMU data
    imu_metadata = read_imu_metadata_from_csv('./data/KittiEquivBiasedImu_metadata.txt')
    imu_data = read_imu_data_from_csv('./data/KittiEquivBiasedImu.txt')
    gps_data = read_gps_data_from_csv('./data/KittiGps_converted.txt')

    # Setup starting pose
    starting_pose = Vector(
        [
            imu_metadata.BodyPtx,
            imu_metadata.BodyPty,
            imu_metadata.BodyPtz,
            imu_metadata.BodyPrx,
            imu_metadata.BodyPry,
            imu_metadata.BodyPrz
        ]
    )

    imu_in_body = gs.Pose3.Expmap(starting_pose)

    if not imu_in_body.equals(gs.Pose3(), 1e-5):
        raise ValueError(
            "Currently only imu_in_body = identity is supported" \
            "(i.e. IMU and body frame are the same)."
        )

    current_pose_global = gs.Pose3(gs.Rot3(), gps_data.ix[FIRST_GPS_POSE].Position)
    current_bias = gs.imuBias_ConstantBias(ZERO_VECTOR, ZERO_VECTOR)
    current_velocity_global = Vector([0., 0., 0.])

    # Setup noise models
    sigma_init_v    = get_gps_vel_noise_model()
    noise_model_gps = get_gps_pos_noise_model(
        SIGMA_INIT_GPS
    )
    sigma_init_b    = get_imu_initial_bias_noise_model(
        SIGMA_INIT_BIAS_ACC,
        SIGMA_INIT_BIAS_GYR
    )
    sigma_init_x    = get_initial_pose_noise_model(
        SIGMA_INIT_POSE
    )
    sigma_between_b = get_imu_bias_between_factor_noise_model(
        imu_metadata.AccelerometerBiasSigma,
        imu_metadata.GyroscopeBiasSigma
    )

    # Setup IMU preintegration parameters
    imu_params = get_imu_preintegration_params(
        imu_metadata.AccelerometerSigma,
        imu_metadata.GyroscopeSigma,
        imu_metadata.IntegrationSigma
    )

    # Setup ISAM2 solver
    isam_params = gs.ISAM2Params()
    isam_params.setFactorization(SOLVER_TYPE)
    isam_params.setRelinearizeSkip(RELIN_SKIP)
    isam = gs.ISAM2(isam_params)

    new_factors = gs.NonlinearFactorGraph()
    new_values  = gs.Values()
    imu_times   = Vector(imu_data.Time)

    # Allows faster indexing in IMU preintegration
    imu_accel   = imu_data[['accelX', 'accelY', 'accelZ']].as_matrix()
    imu_omega   = imu_data[['omegaX', 'omegaY', 'omegaZ']].as_matrix()
    imu_delts   = imu_data[['dt']].as_matrix()

    # measure time
    start = time.time()

    # Main loop: inference is performed at each time step
    for measurement_index in np.arange(FIRST_GPS_POSE, len(gps_data)):

        # At each non=IMU measurement we initialize a new node in the graph
        current_pose_key = gs.symbol(ord('x'), measurement_index)
        current_vel_key = gs.symbol(ord('v'), measurement_index)
        current_bias_key = gs.symbol(ord('b'), measurement_index)
        t = gps_data.ix[measurement_index].Time

        # Some verbose
        if measurement_index == FIRST_GPS_POSE:
            new_values.insert(current_pose_key, current_pose_global)
            new_values.insert(current_vel_key, current_velocity_global)
            new_values.insert(current_bias_key, current_bias)
            new_factors.add(gs.PriorFactorPose3(
                current_pose_key, current_pose_global, sigma_init_x))
            new_factors.add(gs.PriorFactorVector(
                current_vel_key, current_velocity_global, sigma_init_v))
            new_factors.add(gs.PriorFactorConstantBias(
                current_bias_key,
                current_bias,
                sigma_init_b))
        else:
            t_previous = gps_data.ix[measurement_index-1].Time

            # Summarize IMU data between the previous GPS measurement and now
            imu_indices = np.where((imu_times >= t_previous) & (imu_times <= t))
            current_summarized_measurement = gs.PreintegratedImuMeasurements(
                imu_params, current_bias)

            for imu_index in imu_indices[0]:
                current_summarized_measurement.integrateMeasurement(
                    Vector(imu_accel[imu_index]),
                    Vector(imu_omega[imu_index]),
                    imu_delts[imu_index]
                )

            # Create IMU Factor
            new_factors.add(
                gs.ImuFactor(
                    current_pose_key-1,
                    current_vel_key-1,
                    current_pose_key,
                    current_vel_key,
                    current_bias_key,
                    current_summarized_measurement
                )
            )

            # Bias evolution as given in the IMU metadata
            new_factors.add(
                gs.BetweenFactorConstantBias(
                    current_bias_key-1,
                    current_bias_key,
                    gs.imuBias_ConstantBias(ZERO_VECTOR, ZERO_VECTOR),
                    gs.noiseModel_Diagonal.Sigmas(
                        np.sqrt(len(imu_indices[0])) * sigma_between_b
                    )
                )
            )

            # Create GPS factor
            gps_pose = gs.Pose3(
                current_pose_global.rotation(),
                gps_data.ix[measurement_index].Position
            )

            if np.mod(measurement_index + 1, GPS_SKIP) == 0:
                new_factors.add(gs.PriorFactorPose3(
                    current_pose_key,
                    gps_pose,
                    noise_model_gps
                )
            )

            # Add initial values
            new_values.insert(current_pose_key, gps_pose)
            new_values.insert(current_vel_key, current_velocity_global)
            new_values.insert(current_bias_key, current_bias)

            # Update Solve: We accumulate 2*GPS_SKIP GPS measurements before updating
            # the solver for the first time so that the heading becomes observable
            if measurement_index > FIRST_GPS_POSE + 2 * GPS_SKIP:
                velocities.append(np.linalg.norm(current_velocity_global))
                isam.update(new_factors, new_values)
                new_factors = gs.NonlinearFactorGraph()
                new_values = gs.Values()
                current_estimate = isam.calculateEstimate()
                current_pose_global = current_estimate.atPose3(
                    current_pose_key
                )
                current_velocity_global = current_estimate.atVector(
                    current_vel_key
                )
                current_bias = current_estimate.atimuBias_ConstantBias(
                    current_bias_key
                )

    # Plot estimated velocity and compare to GTSAM 3.2.1 and MATLAB
    ref_vel_gtsam321ml = pd.read_csv('./data/ref_vel_gtsam321ml.txt',names=['vx','vy','vz'])
    ref_vel_gtsam4ml = pd.read_csv('./data/ref_vel_gtsam4ml.txt',names=['vx','vy','vz'])
    plt.figure()
    plt.title("Norm of estimated velocity")
    plt.plot(np.sqrt(np.square(ref_vel_gtsam321ml).sum(axis=1)))
    plt.plot(np.sqrt(np.square(ref_vel_gtsam4ml).sum(axis=1)))
    plt.plot(velocities)
    plt.xlabel("Timestep #")
    plt.ylabel("Velocity [m/s]")
    plt.legend(['GTSAM 3.2.1 (Matlab)','GTSAM 4.0 (Matlab)','GTSAM 4.0 (Python)'])
    plt.show()
