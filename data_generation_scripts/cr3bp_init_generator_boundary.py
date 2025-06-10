import numpy as np
import copy as cp
import json
import pickle
import warnings


class CR3BPInitGenerator:

    def __init__(self, initial_guess_sample_mode, thrust, min_mass_to_sample, max_mass_to_sample, min_manifold_length, max_manifold_length):
        self.initial_guess_sample_mode = initial_guess_sample_mode
        self.thrust = thrust
        self.min_mass_to_sample = min_mass_to_sample
        self.max_mass_to_sample = max_mass_to_sample
        self.min_manifold_length = min_manifold_length
        self.max_manifold_length = max_manifold_length

        self._load_in_common_initial_guess()

    def get_earth_initial_guess(self, seed, number_of_segments,
                                maximum_shooting_time, minimum_shooting_time):
        """

        :param seed:
        :param num:
        :return: a list of initial guess for earth mission, specified by seed and number of initial guess required
        """
        self._load_in_common_initial_guess()
        # first three are times, then have 10 x 3 controls, with 10 x 2 angles and 10 x 1 radius. final one is mass

        if self.initial_guess_sample_mode == "from_pickle":
            file_path = "/home/jg3607/Thesis/AAS_paper/results/boundary/feasible_cr3bp_earth_alpha_0.09_seed_118.pkl"   
            file = open(file_path, 'rb')
            data = pickle.load(file)
            file.close()
            control = data[0]["results.control"]
            halo_energy = data[0]["cost_alpha"]
            return halo_energy, [control] 
        
        if self.initial_guess_sample_mode == "from_diffusion":
            file_path = "/home/jg3607/Thesis/Diffusion_model/denoising-diffusion-pytorch/results/generated_initializations/boundary/unet_128_mults_4_4_8_embed_class_256_512_timesteps_500_batch_size_512_cond_drop_0.1_mask_val_-1.0/cr3bp_diffusion_boundary_w_5.0_training_num_100000_num_10000.pkl"   
            file = open(file_path, 'rb')
            data = pickle.load(file)
            file.close()
            return data
        
        # Ground truth initial guess test #############################################################################
        if self.initial_guess_sample_mode == "gt_example":
            optimal_earth_initial_guess = cp.copy(self.gt_initial_guess)
            halo_energy = 1.0
            return halo_energy, [optimal_earth_initial_guess]

        # uniform ######################################################################################################
        if self.initial_guess_sample_mode == "uniform_sample":

            sample_num = 1

            # Sample alpha in the cost
            random_state = np.random.RandomState(seed=seed)
            halo_energy = float(random_state.uniform(0.008, 0.095, 1))

            # Sample initial guess
            theta = random_state.uniform(0, 2 * np.pi, number_of_segments * sample_num)
            psi = random_state.uniform(0, 2 * np.pi, number_of_segments * sample_num)
            r = random_state.uniform(0, 1, number_of_segments * sample_num)

            t_shooting = random_state.uniform(minimum_shooting_time, maximum_shooting_time, sample_num)
            t_init = random_state.uniform(0, 15.0, sample_num)
            t_final = random_state.uniform(0, 15.0, sample_num)

            mass = random_state.uniform(self.min_mass_to_sample, self.max_mass_to_sample, sample_num)
            manifold_start = random_state.uniform(0, 1, sample_num)
            manifold_length = random_state.uniform(self.min_manifold_length, self.max_manifold_length, sample_num)
            

            earth_initial_guess_list = []
            for i in range(sample_num):
                earth_initial_guess = []

                # append time and control initial guess
                earth_initial_guess.append(t_shooting[i])
                earth_initial_guess.append(t_init[i])
                earth_initial_guess.append(t_final[i])

                for j in range(number_of_segments):
                    earth_initial_guess.append(theta[i * number_of_segments + j])
                    earth_initial_guess.append(psi[i * number_of_segments + j])
                    earth_initial_guess.append(r[i * number_of_segments + j])

                earth_initial_guess.append(mass[i])
                earth_initial_guess.append(manifold_start[i])
                earth_initial_guess.append(manifold_length[i])
               

                earth_initial_guess = np.asarray(earth_initial_guess)
                earth_initial_guess_list.append(earth_initial_guess)
            return halo_energy, earth_initial_guess_list


    def _load_in_common_initial_guess(self):

        self.gt_initial_guess = np.array([
            13.529464350683654,
            10.860351024299009,
            12.818513332015005,
            2.491498288715784,
            2.7769798891165474,
            0.5914073326138615,
            0.7968338639548337,
            3.7463522469751442,
            0.0,
            3.6491180004302666,
            3.5631074322491103,
            0.0,
            1.7173472432805899,
            6.187479935808942,
            0.16337440704645623,
            0.2052590014758619,
            5.98948986050412,
            0.0,
            2.2687244672169133,
            2.312118686357815,
            0.0,
            2.7607131083320335,
            1.533746378486957,
            0.0,
            2.6231618870471154,
            3.3967118807336054,
            0.0,
            2.8368112120467868,
            1.6549413500789023,
            0.0,
            3.2748137045536887,
            0.4905510903707979,
            0.0,
            2.4590742577386737,
            3.826424195266761,
            0.0,
            3.267772322359604,
            2.246801180317018,
            0.6428486616190392,
            4.840353755703852,
            5.146648290706465,
            0.0,
            3.9551674257266027,
            4.8904019755671255,
            0.0,
            4.804511141215935,
            2.825735680969815,
            0.0,
            6.188753603178447,
            3.8632512568669846,
            0.0,
            1.645064651878399,
            3.0679849069817546,
            0.0,
            3.8391941681459687,
            0.1148250570628078,
            0.0,
            4.570548120984852,
            4.447565978678258,
            0.0,
            3.3127057323630655,
            4.787944408295277,
            0.0,
            416.20235786623977,])  # mass
