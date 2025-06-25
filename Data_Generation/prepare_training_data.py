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

                control_list.append(normalized_control)
                counter += 1
            
                #save intermediate results
                if counter % 10000 == 0:
                    save_data(directory, control_list, counter)

                if counter == 150000:
                    break

