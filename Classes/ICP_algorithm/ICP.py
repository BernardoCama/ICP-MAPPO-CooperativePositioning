import sys
import os
import numpy as np
from scipy.linalg import block_diag
import functools

# Directories
cwd = os.path.split(os.path.abspath(__file__))[0]
DB_DIR =  os.path.join(os.path.split(os.path.split(cwd)[0])[0], 'DB')
CLASSES_DIR = os.path.join(cwd, 'Classes')
EXPERIMENTS_DIR = os.path.join(cwd, 'Exp')
PARAMETERS_DIR = os.path.join(cwd, 'Parameters')
sys.path.append(os.path.dirname(CLASSES_DIR))
sys.path.append(os.path.dirname(EXPERIMENTS_DIR))
sys.path.append(os.path.dirname(PARAMETERS_DIR))
sys.path.append(os.path.dirname(cwd))

def copy_numpy_args(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        new_args = [arg.copy() if isinstance(arg, np.ndarray) else arg for arg in args]
        new_kwargs = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in kwargs.items()}
        return func(*new_args, **new_kwargs)
    return wrapper

@copy_numpy_args
def gps_kf(params_dict, vehicle_state_gt, gps_meas, cv_measure_gps):
    vehicle_estimate_kf = np.zeros((4*params_dict.num_agents, params_dict.H))*np.nan
    vehicle_estimate_kf_cov = np.zeros((4*params_dict.num_agents, 4*params_dict.num_agents,
                                        params_dict.H))*np.nan

    for time in range(params_dict.H):
        veh_state = np.reshape(vehicle_state_gt[:, time], (params_dict.num_agents, 4))
        [vehicle_state_prior, vehicle_state_prior_cov] = generate_vehicle_prior(params_dict, veh_state)

        Ht = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])

        if time == 0:
            for i in range(params_dict.num_agents):
                cv_gps_diag = cv_measure_gps[:, i * 2:(i + 1) * 2, time]
                gain = np.matmul(vehicle_state_prior_cov[:, i*4:(i+1)*4], np.matmul(np.transpose(Ht), np.linalg.inv(np.matmul(Ht, np.matmul(vehicle_state_prior_cov[:, i*4:(i+1)*4], np.transpose(Ht))) + cv_gps_diag)))
                vehicle_estimate_kf_cov[i*4:(i + 1)*4, i*4:(i + 1)*4, time] = vehicle_state_prior_cov[:, i*4:(i+1)*4] - np.matmul(gain, np.matmul(Ht, vehicle_state_prior_cov[:, i*4:(i+1)*4]))
                vehicle_estimate_kf[i*4:(i + 1)*4, time] = vehicle_state_prior[i, :] + np.matmul(gain, np.transpose(gps_meas[i, :, time]) - np.matmul(Ht, vehicle_state_prior[i,:]))
                
        else:
            for i in range(params_dict.num_agents):
                if not np.isnan(vehicle_estimate_kf[i*4, time-1]):
                    cv = np.array([[ params_dict.motion_model_A_std_vel**2, 0], [0,  params_dict.motion_model_A_std_vel**2]])

                    # Prediction
                    cv_acc = np.array([[0.5, 0, 0, 0], [0, 0.5, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
                    vehicle_gps_prediction_cov = np.matmul( params_dict.motionMatrixA, np.matmul(vehicle_estimate_kf_cov[i*4:(i+1)*4, i*4:(i+1)*4, time-1], np.transpose( params_dict.motionMatrixA))) + np.matmul(params_dict.motionMatrixB, np.matmul(cv, np.transpose(params_dict.motionMatrixB)))
                    vehicle_gps_prediction = np.matmul( params_dict.motionMatrixA, vehicle_estimate_kf[i*4:(i+1)*4, time-1])

                    # Update
                    cv_gps_diag = cv_measure_gps[:, i*2:(i+1)*2, time]
                    gain = np.matmul(vehicle_gps_prediction_cov, np.matmul(np.transpose(Ht), np.linalg.inv(np.matmul(Ht, np.matmul(vehicle_gps_prediction_cov, np.transpose(Ht))) + cv_gps_diag)))
                    vehicle_estimate_kf_cov[i*4:(i+1)*4, i*4:(i+1)*4, time] = vehicle_gps_prediction_cov - np.matmul(gain, np.matmul(Ht, vehicle_gps_prediction_cov))
                    vehicle_estimate_kf[i*4:(i+1)*4, time] = vehicle_gps_prediction + np.matmul(gain, np.transpose(gps_meas[i, :, time]) - np.matmul(Ht, vehicle_gps_prediction))

    return vehicle_estimate_kf, vehicle_estimate_kf_cov

@copy_numpy_args
def generate_vehicle_prior(params_dict, veh_state):
    vehicle_state_prior = np.zeros((params_dict.num_agents, 4))
    vehicle_state_prior_cov = np.zeros((4, params_dict.num_agents*4))

    for i in range(params_dict.num_agents):
        vehicle_state_prior[i, :] = veh_state[i, :] + np.array([10, 10, 1, 1])
        vehicle_state_prior_cov[:, i*4:(i+1)*4] = np.vstack((np.hstack(( params_dict.prior_A_std_pos*np.eye(2), np.zeros((2, 2)))),
                                                            np.hstack((np.zeros((2, 2)),  params_dict.prior_A_std_vel*np.eye(2)))))
    return vehicle_state_prior, vehicle_state_prior_cov

@copy_numpy_args
def generate_gps_meas(params_dict, vehicle_state_gt):
    cv_measure_gps = np.zeros((2, 2*params_dict.num_agents, params_dict.H))
    gps_meas = np.zeros((params_dict.num_agents, 2, params_dict.H))

    for time in range(params_dict.H):
        veh_state = np.reshape(vehicle_state_gt[:, time], (params_dict.num_agents, 4))
        for veh in range(params_dict.num_agents):
            cv_measure_gps[:, veh*2:(veh+1)*2, time] = np.array([[params_dict.meas_GNSS_std_pos**2, 0],
                                                                [0, params_dict.meas_GNSS_std_pos**2]])
            r_mat = np.linalg.cholesky(cv_measure_gps[:, veh*2:(veh+1)*2, time])
            curr_meas = np.transpose(np.transpose(veh_state[veh, 0:2])) + np.transpose(np.matmul(np.transpose(r_mat), np.random.randn(2, 1)))
            gps_meas[veh, :, time] = curr_meas
    return cv_measure_gps, gps_meas

@copy_numpy_args
def generate_v2f_meas(params_dict, fea_vect, fea_vect_boxes, vehicle_state_gt, conn_features):
    v2fcov_fv = np.zeros((2 * params_dict.num_agents, 2 * params_dict.num_features, params_dict.H))
    z_fv = np.zeros((2 * params_dict.num_agents, params_dict.num_features, params_dict.H))
    z_fv_boxes = np.zeros((params_dict.num_features, 3, 8, params_dict.H, params_dict.num_agents))
    connectivity_matrix_gtall = np.zeros((params_dict.num_agents, params_dict.num_features, params_dict.H))

    for time in range(params_dict.H):
        for veh in range(params_dict.num_agents):
            mu_mea_z_boxes = np.zeros((params_dict.num_features, 3, 8))

            fea = np.reshape(fea_vect[:, time, veh], (params_dict.num_features, 4))
            fea_boxes = fea_vect_boxes[:, :, :, time, veh]
            veh_state = np.reshape(vehicle_state_gt[:, time], (params_dict.num_agents, 4))
            mu_mea_z = np.tile(fea[:, 0:2], (1, params_dict.num_agents)) - np.tile(np.transpose(np.reshape(veh_state[:, 0:2], (-1, 1))), (params_dict.num_features, 1))

            mu_mea_z_boxes[:, 0, :] = fea_boxes[:, 0, :] - veh_state[veh, 0]
            mu_mea_z_boxes[:, 1, :] = fea_boxes[:, 1, :] - veh_state[veh, 1]
            mu_mea_z_boxes[:, 2, :] = fea_boxes[:, 2, :]

            connectivity_matrix_gtall[veh, :, time] = conn_features[veh, :, time]

            for f in range(params_dict.num_features):
                v2fcov_fv[veh*2:(veh+1)*2, f*2:(f+1)*2, time] =  params_dict.meas_A2F_std_dist**2*np.eye(2)
                if connectivity_matrix_gtall[veh, f, time]:
                    z_fv[veh*2:(veh+1)*2, f, time] = mu_mea_z[f, veh*2:(veh+1)*2] + np.random.randn(1, 2) * params_dict.meas_model_A2F_std_dist
                    z_fv_boxes[f, :, :, time, veh] = mu_mea_z_boxes[f, :, :] + np.random.randn(3, 8) * params_dict.meas_model_A2F_std_dist
    return z_fv, z_fv_boxes, v2fcov_fv, connectivity_matrix_gtall

@copy_numpy_args
def icp_with_data_association(params_dict, fea_vect,  connectivity_matrix_gt_all, z_fv_all, z_fv_all_boxes, v2f_cov_fv_all, gps_meas,
                              cv_measure_gps, vehicle_estimate_kf, vehicle_estimate_kf_cov):
    estimate_icpda_cov = np.zeros((4*params_dict.num_agents + 2*params_dict.num_features,
                           4*params_dict.num_agents + 2*params_dict.num_features, params_dict.H))*np.nan
    estimateICPDA = np.zeros((4*params_dict.num_agents + 2*params_dict.num_features, params_dict.H))*np.nan
    numberOfDetection = np.zeros((params_dict.num_features, params_dict.H))
    vehicleNumberOfMeasGT = np.zeros((params_dict.num_agents, params_dict.H))
    incorrectAssociation = np.zeros((params_dict.H, ))
    number_of_total_v2f = np.zeros((params_dict.H, 1))

    previous_z_fv_boxes_vehicles = np.reshape(fea_vect[:, params_dict.start_time-1], (params_dict.num_features, 4))[:,:2]

    for time in range(params_dict.start_time-1, params_dict.H):

        # print(f"Timestep {time+1}/{params_dict.H}")

        fea = np.reshape(fea_vect[:, time], (params_dict.num_features, 4))
        z_fv = z_fv_all[:, :, time]
        z_fv_boxes = z_fv_all_boxes[:, :, :, time, :]

        v2f_cov_fv = v2f_cov_fv_all[:,:, time]
        connectivity_matrix_gt = connectivity_matrix_gt_all[:,:, time]

        [featureStatePrior, featureStatePriorCov] = generate_target_prior(params_dict, fea)

        numberOfDetection[0:params_dict.num_features, time] = np.sum(connectivity_matrix_gt, 0)
        vehicleNumberOfMeasGT[:,time] = np.sum(connectivity_matrix_gt,1)
        number_of_total_v2f[time] = np.sum(connectivity_matrix_gt)

        # Centralized cooperative localization
        targetPrediction = np.zeros((params_dict.num_features, 2))
        targetPredictionCov = np.zeros((2, 2*params_dict.num_features))

        # Targets prediction
        if number_of_total_v2f[time] == 0 or time == params_dict.start_time-1:
            estimateICPDA[:, time] = np.hstack((vehicle_estimate_kf[:, time], np.reshape(featureStatePrior, (2*params_dict.num_features, ))))
            for i in range(params_dict.num_agents):
                estimate_icpda_cov[i*4:(i+1)*4, i*4:(i+1)*4, time] = vehicle_estimate_kf_cov[i*4:(i+1)*4, i*4:(i+1)*4, time]
            for f in range(params_dict.num_features):
                estimate_icpda_cov[f*2+4*params_dict.num_agents:(f+1)*2+4*params_dict.num_agents, f*2+4*params_dict.num_agents:(f+1)*2+4*params_dict.num_agents,time] = featureStatePriorCov[:, f*2:(f+1)*2]
        elif number_of_total_v2f[time] != 0:
            for f in range(params_dict.num_features):
                if numberOfDetection[f, time] >= 1 and numberOfDetection[f, time - 1] >= 1:
                    targetPrediction[f,:] = np.transpose(estimateICPDA[f*2+4*params_dict.num_agents:(f+1)*2+4*params_dict.num_agents, time-1])
                    targetPredictionCov[:, f*2:(f+1)*2] = estimate_icpda_cov[f*2+4*params_dict.num_agents:(f+1)*2+4*params_dict.num_agents, f*2+4*params_dict.num_agents:(f+1)*2+4*params_dict.num_agents, time-1]
                else:
                    targetPredictionCov[:, f*2:(f+1)*2] = featureStatePriorCov[:, f*2:(f+1)*2]
                    targetPrediction[f, :] = featureStatePrior[f, :]

            # Vehicles prediction
            vehicle_prediction = np.zeros((4*params_dict.num_agents, ))
            vehicle_prediction_cov = np.zeros((4, 4*params_dict.num_agents))

            for veh in range(params_dict.num_agents):
                if not np.isnan(all(estimateICPDA[veh*4:(veh+1)*4,time-1])):
                    cv = np.array([[ params_dict.motion_model_A_std_vel**2, 0],
                                  [0,  params_dict.motion_model_A_std_vel**2]])
                    vehicle_prediction_cov[:, veh*4:(veh+1)*4] = np.matmul( params_dict.motionMatrixA, np.matmul(estimate_icpda_cov[veh*4:(veh+1)*4, veh*4:(veh+1)*4, time-1], np.transpose( params_dict.motionMatrixA))) + np.matmul(params_dict.motionMatrixB, np.matmul(cv, np.transpose(params_dict.motionMatrixB)))
                    vehicle_prediction[veh*4:(veh+1)*4] = np.matmul( params_dict.motionMatrixA, estimateICPDA[veh*4:(veh+1)*4, time-1])
                else:
                    vehicle_prediction[veh*4:(veh+1)*4] = vehicle_estimate_kf[veh*4:(veh+1)*4, time]
                    vehicle_prediction_cov[:, veh*4:(veh+1)*4] = vehicle_estimate_kf_cov[veh*4:(veh+1)*4, veh*4:(veh+1)*4, time]

            # Reorder boxes for data association
            z_fv_boxes_vehicles = list()
            id_target = []
            curr_veh_pos_list = []
            for curr_veh in range(params_dict.num_agents):
                sensed_targets = list()
                for f in range(params_dict.num_features):
                    if connectivity_matrix_gt[curr_veh, f]:
                        sensed_targets.append(f)
                curr_veh_pos = np.reshape(np.tile(np.hstack((vehicle_prediction[curr_veh*4:curr_veh*4+2], 0)), (8, )), (8, 3))
                curr_veh_pos_list.append(curr_veh_pos)
                z_fv_boxes_vehicles.append(z_fv_boxes[sensed_targets, :, :, curr_veh] + np.transpose(curr_veh_pos))
                id_target.append(sensed_targets)

            # Perfect DA
            connectivityMatrixICPDA = connectivity_matrix_gt # NUM_VEH X NUM_FEATURES
            misura_FV = z_fv                                 # 2*NUM_VEH X NUM_FEATURES

            # ICP estimate
            [estimate_icpda_cov[:, :, time], estimateICPDA[:, time]] = cooperative_localization(params_dict,
                                                                                               connectivityMatrixICPDA,
                                                                                               gps_meas[:, :, time],
                                                                                               misura_FV, np.sum(vehicleNumberOfMeasGT[:,time]), v2f_cov_fv, cv_measure_gps[:, :, time], targetPredictionCov, vehicle_prediction_cov, targetPrediction, vehicle_prediction)
            # sio.savemat(f'estimateICPDA_time{time}.mat',estimateICPDA)
    return estimateICPDA, estimate_icpda_cov, numberOfDetection, incorrectAssociation

@copy_numpy_args
def cooperative_localization(params_dict, VFconnect, y_v, z_FV, M_z, C_z_FV, Cv_measure_gps, Cf_b, Cv_b, mu_f_b, mu_v_b):

    Nf = params_dict.num_features
    Nv = params_dict.num_agents

    y_v = np.reshape(y_v, -1)

    y_v_tot = np.zeros((2*Nv,))
    Cv_mea_gps = np.zeros((2, 2*Nv))
    C_v_b = np.zeros((4, 4*Nv))
    for i in range(Nv):
        # y_v_tot[i*4:(i+1)*4] = np.hstack((y_v[i*2:(i+1)*2], np.zeros(2, )))
        y_v_tot[i*2:(i+1)*2] = y_v[i*2:(i+1)*2]
        Cv_mea_gps[:, i*2:(i+1)*2] = Cv_measure_gps[:, i*2:(i+1)*2]
        C_v_b[:, i*4:(i+1)*4] = Cv_b[:, i*4:(i+1)*4]

    Cv_gps_diag = np.zeros((2 * Nv, 2*Nv))
    # to build V2F measurements
    meas_z_vec = np.zeros((2 * int(M_z),))
    R_all = np.zeros((2*int(M_z), 2*int(M_z)))
    # to build H
    M_v = np.zeros((2 * int(M_z), 4 * Nv))
    M_f = np.zeros((2 * int(M_z), 2 * Nf))

    # build covariance
    D = np.zeros((4 * Nv, 4 * Nv))
    E = np.zeros((4 * Nv, 2 * Nf))
    G = np.zeros((2 * Nf, 2 * Nf))

    theta_f_prev_t = np.zeros((2 * Nf,))
    theta_v_prev_t = np.zeros((4 * Nv,))
    theta = np.zeros((4 * Nv + 2 * Nf,))*np.nan
    C_theta = np.zeros((4 * Nv + 2 * Nf, 4 * Nv + 2 * Nf))*np.nan

    P = np.hstack((np.eye(2), np.zeros((2, 2))))

    m = 0
    for i in range(Nv):
        for f in range(Nf):
            if VFconnect[i, f] == 1: # for subset of features
                conn = 1
                # build complete set of V2F measurements
                m += 1
                meas_z_vec[(m - 1)*2: m*2] = z_FV[i*2: (i+1)*2, f] # prendo tutte le mis di tutte le fea di tutti e le ordino una sotto l'altra
                # V2F complete  covariance
                R_all[(m - 1)*2: m*2, (m - 1)*2:m*2] = C_z_FV[i*2:(i+1)*2, f*2:(f+1)*2]
                # build H
                M_v[(m - 1)*2:m*2, i*4:(i+1)*4] = -P
                M_f[(m - 1)*2:m*2, f*2:(f+1)*2] = np.eye(2)

                # matrice cov(only measurements are considered now)
                D[i*4:(i+1)*4, i*4:(i+1)*4] = D[i*4:(i+1)*4, i*4:(i+1)*4] + block_diag(np.linalg.inv(C_z_FV[i*2:(i+1)*2, f*2:(f+1)*2]), np.zeros((2, 2)))
                E[i*4:(i+1)*4, f*2:(f+1)*2] = np.vstack((-np.linalg.inv(C_z_FV[i*2:(i+1)*2, f*2:(f+1)*2]), np.zeros((2, 2))))
                G[f*2:(f+1)*2, f*2:(f+1)*2] = G[f*2:(f+1)*2, f*2:(f+1)*2] + np.linalg.inv(C_z_FV[i*2:(i+1)*2, f*2:(f+1)*2])

            # a priori on feature
            if i == 0:
                # belief on feature at previous time
                G[f*2:(f+1)*2, f*2:(f+1)*2] = G[f*2:(f+1)*2, f*2:(f+1)*2] + np.linalg.inv(Cf_b[:, f*2:(f+1)*2])
                theta_f_prev_t[f*2:(f+1)*2] = np.transpose(mu_f_b[f, :])

        # GPS vehicle likelihood + belief on vehicle at previous time
        D[i*4:(i+1)*4, i*4:(i+1)*4] = D[i*4:(i+1)*4, i*4:(i+1)*4] + np.linalg.inv(Cv_b[:, i*4:(i+1)*4]) + np.matmul(np.transpose(P), np.matmul(np.linalg.inv(Cv_measure_gps[:, i*2:(i+1)*2]), P))  # block_diag(np.linalg.inv(Cv_measure_gps[:, i*2:(i+1)*2]), np.zeros((2, 2)))
        theta_v_prev_t[i*4:(i+1)*4] = np.transpose(mu_v_b[i*4:(i+1)*4])

        # GPS complete covariance
        Cv_gps_diag[i*2:(i+1)*2, i*2:(i+1)*2] = Cv_measure_gps[:, i*2:(i+1)*2]

    # Covariance Estimate
    C_theta_inv = np.vstack((np.hstack((D, E)), np.hstack((np.transpose(E), G))))
    C_theta_centr = np.linalg.inv(C_theta_inv)

    # Mean Estimate
    H = np.vstack((np.hstack((np.kron(np.eye(Nv), P), np.zeros((2*Nv, 2*Nf)))), np.hstack((M_v, M_f))))

    rho = np.hstack((y_v_tot, meas_z_vec)) # GPS meas + V2F meas
    theta_prev = np.hstack((theta_v_prev_t, theta_f_prev_t))
    Q_all = block_diag(Cv_gps_diag, R_all)

    theta_centr = theta_prev + np.matmul(C_theta_centr, np.matmul(np.transpose(H), np.matmul(np.linalg.inv(Q_all), (rho - np.matmul(H, theta_prev)))))
    # sio.savemat('prova_icp.mat', {'theta_centr_p': theta_centr, 'C_theta_centr_p': C_theta_centr})

    return C_theta_centr, theta_centr

@copy_numpy_args
def generate_target_prior(params_dict, fea):
    featureStatePriorCov = np.tile(params_dict.pior_F_std_pos**2*np.eye(2), (1, params_dict.num_features))
    featureStatePrior = fea[:, 0:2]
    return featureStatePrior, featureStatePriorCov


###############################
###### ERROR COMPUTATION ######
###############################

@copy_numpy_args
def compute_errors(params_dict, vehicleStateGT, Fea_vect, vehicleEstimateKF, estimateICPDA, connectivity_matrix_gt):
    err_square_GPS = np.zeros((params_dict.num_agents,params_dict.H))
    err_square_ICPDA = np.zeros((params_dict.num_agents,params_dict.H))
    err_square_ICPDA_fea = np.zeros((params_dict.num_features,params_dict.H))

    for time in range(params_dict.H):
        for i in range(params_dict.num_agents):
            # x and y pos
            err_square_GPS[i, time] = (vehicleEstimateKF[i*4, time] - vehicleStateGT[i*4, time])**2 + (vehicleEstimateKF[i*4+1, time]-vehicleStateGT[i*4+1, time])**2
            err_square_ICPDA[i, time] = (estimateICPDA[i*4, time] - vehicleStateGT[i*4, time])**2 + (estimateICPDA[i*4+1, time]-vehicleStateGT[i*4+1, time])**2

        for f in range(params_dict.num_features):
            # x and y pos
            err_square_ICPDA_fea[f, time] = (estimateICPDA[f*2+4*params_dict.num_agents, time] - Fea_vect[f*4, time])**2 + \
                                            (estimateICPDA[f*2+4*params_dict.num_agents+1, time] - Fea_vect[f*4+1, time])**2
            if err_square_ICPDA_fea[f, time] == 0:
                err_square_ICPDA_fea[f, time] = np.nan
    return err_square_GPS, err_square_ICPDA, err_square_ICPDA_fea


def compute_errors_single(params_dict, vehicleStateGT, Fea_vect, vehicleEstimateKF, estimateICPDA, connectivity_matrix_gt):
    err_square_GPS = np.zeros((params_dict.num_agents,params_dict.H))
    err_square_ICPDA = np.zeros((params_dict.num_agents,params_dict.H))
    err_square_ICPDA_fea = np.zeros((params_dict.num_features,params_dict.H))

    for time in range(params_dict.H):
        for i in range(params_dict.num_agents):
            err_square_GPS[i, time] = (vehicleEstimateKF[i*4, time] - vehicleStateGT[i*4, time])**2 + (vehicleEstimateKF[i*4+1, time]-vehicleStateGT[i*4+1, time])**2
            err_square_ICPDA[i, time] = (estimateICPDA[i][0, time] - vehicleStateGT[i*4, time])**2 + (estimateICPDA[i][1, time]-vehicleStateGT[i*4+1, time])**2

    return err_square_GPS, err_square_ICPDA, err_square_ICPDA_fea























