import pickle
import numpy as np
import os
import pydylan

# Load data from the pickle file
def load_data(filepath):
    with open(filepath, 'rb') as file:
        data = pickle.load(file)
    return data

def save_data(directory, control_list, counter):
    # Convert the list of control vectors to a NumPy array
    control_array = np.array(control_list)

    # Save the array to a pickle file
    output_path = os.path.join(directory, f'training_data_boundary_{counter}.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(control_array, f)

    print(f"All control vectors have been combined and saved to {output_path}")

def restructure_to_3_channels(normalized_control):
    """
    Restructure 66-dimensional trajectory data into 3-channel format.
    
    Args:
        normalized_control: Array of shape (67,) where index 0 is halo energy (conditioning)
                          and indices 1-66 are trajectory parameters
    
    Returns:
        channels_data: Array of shape (3, 22) with restructured data
    """
    # Extract trajectory parameters (indices 1-66, excluding halo energy at index 0)
    trajectory_params = normalized_control[1:]
    
    # Initialize 3-channel structure
    channels_data = np.zeros((3, 22))
    
    # Index 1: Time variables (3 parameters)
    channels_data[0, 0] = trajectory_params[0]  # shooting_time -> channel 1
    channels_data[1, 0] = trajectory_params[1]  # initial_coast -> channel 2  
    channels_data[2, 0] = trajectory_params[2]  # final_coast -> channel 3
    
    # Indices 2-21: Control vectors (60 parameters = 20 segments × 3 components)
    for i in range(20):
        # Each segment has 3 control components
        segment_start = 3 + i * 3
        channels_data[0, i + 1] = trajectory_params[segment_start]     # ux -> channel 1
        channels_data[1, i + 1] = trajectory_params[segment_start + 1] # uy -> channel 2
        channels_data[2, i + 1] = trajectory_params[segment_start + 2] # uz -> channel 3
    
    # Index 22: Final parameters (3 parameters)
    channels_data[0, 21] = trajectory_params[63]  # final_fuel_mass -> channel 1
    channels_data[1, 21] = trajectory_params[64]  # halo_period -> channel 2
    channels_data[2, 21] = trajectory_params[65]  # manifold_length -> channel 3
    
    return channels_data

def get_halo_period(halo_energy):
    earth = pydylan.Body("Earth")
    moon = pydylan.Body("Moon")

    cr3bp = pydylan.eom.CR3BP(primary=earth, secondary=moon)
    libration_point_information = cr3bp.find_equilibrium_point(pydylan.enum.LibrationPoint.L1)
    desired_orbit_energy = libration_point_information[1] + halo_energy

    halo = pydylan.periodic_orbit.Halo(cr3bp, pydylan.enum.LibrationPoint.L1, desired_orbit_energy, 8000.)
    assert halo.solve_for_orbit() == pydylan.enum.OrbitGenerationResult.Success
    
    return halo.orbit_period

def spherical_to_cart(r,alpha,beta):
    rx = r*np.cos(alpha)*np.cos(beta)
    ry = r*np.sin(alpha)*np.cos(beta)
    rz = r*np.sin(beta)
    return np.array([rx,ry,rz])

if __name__ == "__main__":
    
    # Path to results of the indirect method with ACT
    directory = '/scratch/gpfs/jg3607/AAS_paper/results/boundary/'
    num_of_segments = 20
    thrust = 1
    min_shooting_time = 0
    max_shooting_time = 40
    min_coast_time = 0
    max_coast_time = 15
    min_halo_energy = 0.008
    max_halo_energy = 0.095
    min_final_fuel_mass = 408 #700-292 => cut off value at 90%
    max_final_fuel_mass = 470
    min_manifold_length = 5
    max_manifold_length = 11

    control_list = []   
    classifier = 1
    counter = 0
    for filename in os.listdir(directory):
        if filename.endswith('.pkl'):
            filepath = os.path.join(directory, filename)
            #print(filepath)
            data = load_data(filepath)
            control = data[0]["results.control"]

            if control[-3] > min_final_fuel_mass:
                # Normalize classifier
                classifier = data[0]["cost_alpha"]
                classifier_norm = (classifier-min_halo_energy)/(max_halo_energy-min_halo_energy)

                # Normalize time variables
                shooting_time_norm = (control[0] - min_shooting_time)/(max_shooting_time-min_shooting_time)
                initial_coast_norm = (control[1] - min_coast_time)/(max_coast_time-min_coast_time)
                final_coast_norm = (control[2] - min_coast_time)/(max_coast_time-min_coast_time)

                normalized_control = np.array([classifier_norm,shooting_time_norm,initial_coast_norm,final_coast_norm])

                # Transform control vector to cartesian and normalize
                for i in range(num_of_segments):
                    control_cart = pydylan.transformations.spherical_to_cartesian(control[3+3*i:3+3*(i+1)])
                    control_cart_2 = spherical_to_cart(control[5+3*i],control[3+3*i],control[4+3*i])
                    control_cart_norm = (control_cart - thrust*np.array([-1,-1,-1]))/(2*thrust)
                    normalized_control = np.append(normalized_control,control_cart_norm)

                # Normalize final fuel mass
                final_fuel_mass_norm = (control[-3]-min_final_fuel_mass)/(max_final_fuel_mass-min_final_fuel_mass)

                # Normalize halo orbit manifold control variables
                halo_period_norm = control[-2]/get_halo_period(classifier)
                manifold_length_norm = (control[-1]-min_manifold_length)/(max_manifold_length-min_manifold_length)

                control_vec_end = np.array([final_fuel_mass_norm,halo_period_norm,manifold_length_norm])
                normalized_control = np.append(normalized_control,control_vec_end)

                # Restructure to 3-channel format
                channels_data = restructure_to_3_channels(normalized_control)
                
                # Debug: Print shapes and sample data
                if counter < 3:  # Only print first 3 samples to avoid spam
                    print(f"Sample {counter}:")
                    print(f"  Original normalized_control shape: {normalized_control.shape}")
                    print(f"  Channels_data shape: {channels_data.shape}")
                    print(f"  Halo energy (classifier): {classifier_norm}")
                    print(f"  Channel 1 (time + ux + fuel): {channels_data[0, :5]}...")  # First 5 values
                    print(f"  Channel 2 (time + uy + halo): {channels_data[1, :5]}...")
                    print(f"  Channel 3 (time + uz + manifold): {channels_data[2, :5]}...")
                    print("  ---")
                
                # Store both the 3-channel data and the halo energy (classifier)
                # Format: [halo_energy, 3_channel_data]
                control_list.append([classifier_norm, channels_data])
                counter += 1
            
                #save intermediate results
                if counter % 10000 == 0:
                    save_data(directory, control_list, counter)

                if counter == 150000:
                    break

