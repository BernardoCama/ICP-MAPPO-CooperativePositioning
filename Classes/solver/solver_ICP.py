import sys
import os
import numpy as np
import matplotlib.pyplot as plt
plt.set_loglevel("error")

# Directories
cwd = os.path.split(os.path.abspath(__file__))[0]
CLASSES_DIR = os.path.join(os.path.split(os.path.split(cwd)[0])[0], 'Classes')
EXPERIMENTS_DIR = os.path.join(os.path.split(os.path.split(cwd)[0])[0], 'Exp')
sys.path.append(os.path.dirname(CLASSES_DIR))
sys.path.append(os.path.dirname(EXPERIMENTS_DIR))
sys.path.append(os.path.dirname(cwd))
from Classes.ICP_algorithm.ICP import generate_gps_meas, generate_v2f_meas, gps_kf, icp_with_data_association, compute_errors

class Solver_ICP(object):
    
    DEFAULTS = {}   
    def __init__(self, params):

        self.params = params
        if not isinstance(params, dict):
            params_dict = params.DEFAULTS
        self.__dict__.update(Solver_ICP.DEFAULTS, **params_dict)

        self.result_test_path = os.path.join(self.output_results_dir, 'output_ICP_test_results.npy')

    def set_baselines(self, baselines):
    
        self.baselines = baselines

    def test(self, data):

        self.motionMatrixA = np.vstack((np.hstack((np.eye(2), self.H*np.eye(2))),
                                              np.hstack((np.zeros((2, 2)), np.eye(2)))))
        self.motionMatrixB = np.vstack((self.H**2/2*np.eye(2), self.H*np.eye(2)))

        # Generate GPS measurements
        # cv_measure_gps: 2 x 2*num_agents x timesteps 
        # gps_meas: num_agents x 2 x timesteps
        # Covariances, gps meas
        cv_measure_gps, gps_meas = generate_gps_meas(self, data['vehicleStateGT'])

        # Generate V2F measurements
        # z_fv_all: 2*num_agents x num_features x timesteps
        # z_fv_all_boxes: num_features x 3 x 8 x timesteps x num_agents
        # v2f_cov_fv_all: 2*num_agents, 2*num_features, timesteps
        # connectivity_matrix_gt_all: num_agents x num_features x timesteps
        # Pointpillars data
        # z_fv_all, z_fv_all_boxes, v2f_cov_fv_all, connectivity_matrix_gt_all = generate_v2f_meas(self, data['Fea_vect'], data['Fea_vect_boxes'], data['vehicleStateGT'], data['conn_features'])
        # Artificial noise on A2F measurements
        z_fv_all, z_fv_all_boxes, v2f_cov_fv_all, connectivity_matrix_gt_all = generate_v2f_meas(self, data['Fea_vect_oracle'], data['Fea_vect_boxes_oracle'], data['vehicleStateGT'], data['conn_features'])

        # GPS tracking
        # vehicle_estimate_kf:  num_inputs (4) * num_agents x timesteps
        # vehicle_estimate_kf_cov:  num_inputs (4) * num_agents x num_inputs (4) * num_agents x timesteps
        vehicle_estimate_kf, vehicle_estimate_kf_cov = gps_kf(self, data['vehicleStateGT'], gps_meas, cv_measure_gps)
 
        # ICP-DA solution
        estimateICPDA, estimateICPDACov, numberOfDetection, incorrectAssociation = icp_with_data_association(self, data['Fea_vect_true'], connectivity_matrix_gt_all,
                                                                                                             z_fv_all, z_fv_all_boxes, v2f_cov_fv_all, gps_meas, cv_measure_gps,
                                                                                                             vehicle_estimate_kf, vehicle_estimate_kf_cov)
        # Compute errors
        err_square_GPS, err_square_ICPDA, err_square_ICPDA_featurePos = compute_errors(self, data['vehicleStateGT'], data['Fea_vect_true'], vehicle_estimate_kf, estimateICPDA, connectivity_matrix_gt_all)

        results = {'GPS_mean':np.moveaxis(vehicle_estimate_kf.reshape(self.num_agents, 4, -1), 2, 0), # (timesteps, num_agents, 4)

                   'ICP_A_mean': np.moveaxis(estimateICPDA[:self.num_agents*4, :].reshape(self.num_agents, 4, -1), 2, 0),  # (timesteps, num_agents, 4)
                   'GPS_absolute_error_pos': np.moveaxis(np.sqrt(err_square_GPS), 1, 0), # (timesteps, num_agents)
                   'ICP_A_absolute_error_pos': np.moveaxis(np.sqrt(err_square_ICPDA), 1, 0), # (timesteps, num_agents)
                   } 

        return results

    def save_test_result(self, test_results):
        np.save(self.result_test_path, test_results, allow_pickle = True)

    def load_test_result(self):
        return np.load(self.result_test_path, allow_pickle = True).tolist()

