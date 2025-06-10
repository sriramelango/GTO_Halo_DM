import sys
from locale import str

sys.path.append('../')
sys.path.append('./')
sys.path.append('./support_scripts')
sys.path.append('../support_scripts')
sys.path.append('./init_generator')

import pydylan
from support_scripts.support import get_GTO_in_CR3BP_units
import numpy as np
import time
import os
import pickle
import argparse
import matplotlib.pyplot as plt

from cr3bp_init_generator_boundary import CR3BPInitGenerator

class OrbitSolverTimeout(Exception):
    pass

class CR3BPEarthMissionWarmstartSimulatorBoundary:

    def __init__(self, seed, quiet_snopt, number_of_segments, maximum_shooting_time,
                 minimum_shooting_time,
                 initial_guess_sample_mode, start_bdry, end_bdry, thrust, solver_mode,
                 min_mass_to_sample, max_mass_to_sample, snopt_time_limit,
                 result_folder, halo_energy):
        self.seed = seed
        self.quiet_snopt = quiet_snopt
        self.number_of_segments = number_of_segments
        self.maximum_shooting_time = maximum_shooting_time
        self.minimum_shooting_time = minimum_shooting_time
        self.initial_guess_sample_mode = initial_guess_sample_mode
        self.start_bdry = start_bdry
        self.end_bdry = end_bdry
        self.thrust = thrust
        self.solver_mode = solver_mode
        self.min_mass_to_sample = min_mass_to_sample
        self.max_mass_to_sample = max_mass_to_sample
        self.snopt_time_limit = snopt_time_limit
        self.halo_energy = None
        self.result_folder = result_folder
        self.min_manifold_length = 5
        self.max_manifold_length = 11
        self.halo_energy = 0.008 + halo_energy * (0.095-0.008) #halo_energy

        self.init_generator = CR3BPInitGenerator(initial_guess_sample_mode=self.initial_guess_sample_mode,
                                                 thrust=self.thrust,
                                                 min_mass_to_sample=self.min_mass_to_sample,
                                                 max_mass_to_sample=self.max_mass_to_sample,
                                                 min_manifold_length = self.min_manifold_length,
                                                 max_manifold_length = self.max_manifold_length)
                                                 

    def run(self):

        halo_energy, earth_initial_guess_list = self.init_generator.get_earth_initial_guess(seed=self.seed,
                                                                               number_of_segments=self.number_of_segments,
                                                                               maximum_shooting_time=self.maximum_shooting_time,
                                                                               minimum_shooting_time=self.minimum_shooting_time)
        #print(f"current alpha is {self.halo_energy}")

        result_data = []
        for earth_initial_guess in earth_initial_guess_list:
            result_data.append(self.simulate(earth_initial_guess=earth_initial_guess))

        # Print statistics of the result
        #self.print_statistics(result_data=result_data, earth_initial_guess_list=earth_initial_guess_list)

        # Save data
        if not os.path.isdir(self.result_folder):
            os.makedirs(self.result_folder, exist_ok=True)

        if result_data[0]["feasibility"] and result_data[0]["snopt_inform"] == 1:
            file_path = f"{self.result_folder}/snopt_feasible_cr3bp_earth_energy_{self.halo_energy:.3f}_seed_{self.seed}.pkl"
            with open(file_path, 'wb') as f:
                pickle.dump(result_data, f)
            #print(f"{file_path} successfully saved!")
        elif result_data[0]["feasibility"]:
            file_path = f"{self.result_folder}/feasible_cr3bp_earth_alpha_{self.halo_energy:.3f}_seed_{self.seed}.pkl"
            with open(file_path, 'wb') as f:
                pickle.dump(result_data, f)
            #print(f"{file_path} successfully saved!")

    def simulate(self, earth_initial_guess):

        pydylan.set_logging_severity(pydylan.enum.error)

        # Set up environment and thruster #############################################################################
        earth = pydylan.Body("Earth")
        moon = pydylan.Body("Moon")

        cr3bp = pydylan.eom.CR3BP(primary=earth, secondary=moon)
        libration_point_information = cr3bp.find_equilibrium_point(pydylan.enum.LibrationPoint.L1)
        desired_orbit_energy = libration_point_information[1] + self.halo_energy

        halo = pydylan.periodic_orbit.Halo(cr3bp, pydylan.enum.LibrationPoint.L1, desired_orbit_energy, 8000.)
        #halo.solve_for_orbit() == pydylan.enum.OrbitGenerationResult.Success
        assert halo.solve_for_orbit() == pydylan.enum.OrbitGenerationResult.Success

        thruster_parameters = pydylan.ThrustParameters(fuel_mass=700., dry_mass=300., Isp=1000., thrust=1.0)

        gto_spiral = pydylan.phases.lowthrust_spiral(cr3bp,
                                                     get_GTO_in_CR3BP_units(),
                                                     thruster_parameters)
        gto_spiral.evaluate(time_of_flight=self.start_bdry)  # original: 6.48423370092

        # Earth mission  ###############################################################################################
        # snopt setting
        snopt_options = pydylan.SNOPT_options_structure()
        snopt_options.derivative_mode = pydylan.enum.derivative_mode_type.analytic
        snopt_options.quiet_SNOPT = self.quiet_snopt
        snopt_options.time_limit = self.snopt_time_limit  # 500.0
        snopt_options.total_iteration_limit = 15000
        #snopt_options.save_all_SNOPT_evaluations = True
        snopt_options.optimality_tolerance = 1E-3  # TODO： set up optimality tolerance
        # Configure the solver mode
        if self.solver_mode == "feasible":
            snopt_options.solver_mode = pydylan.enum.solver_mode_type.feasible  # Configure to use feasible mode.
        else:
            snopt_options.solver_mode = pydylan.enum.solver_mode_type.optimal

        mbh_options = pydylan.MBH_options_structure()

        # earth mission
        # TODO: reset thrust parameters here. Since in the previous section, we use thrust=1.0 to compute gto_spiral

        thruster_parameters = pydylan.ThrustParameters(fuel_mass=700., dry_mass=300., Isp=1000., thrust=self.thrust)
        thruster_parameters.fuel_mass = gto_spiral.get_states()[-1, 6]

        phase_options = pydylan.phase_options_structure()
        phase_options.number_of_segments = self.number_of_segments  # previously, 10

        phase_options.maximum_initial_coast_time = 15.
        phase_options.maximum_final_coast_time = 15.
        phase_options.maximum_shooting_time = self.maximum_shooting_time  # previously, 15
        phase_options.minimum_shooting_time = self.minimum_shooting_time
        # phase_options.control_coordinate_transcription = pydylan.enum.polar
        phase_options.control_coordinate_transcription = pydylan.enum.spherical

        # the start states are the final of gto_spiral
        earth_mission_start = pydylan.FixedBoundaryCondition(gto_spiral.get_final_states())
        # left integrate from the some point of the halo manifold, use the arc as the end boundary condition
        earth_mission_end = pydylan.LibrationOrbitBoundaryCondition(halo, pydylan.enum.PerturbationDirection.StableLeft, np.asarray([halo.orbit_energy, 0 * halo.orbit_period, self.min_manifold_length]), np.asarray([halo.orbit_energy, 1 * halo.orbit_period, self.max_manifold_length]))

        # Major optimization for earth mission #################################################################
        earth_mission = pydylan.Mission(cr3bp, earth_mission_start, earth_mission_end, pydylan.enum.snopt)  # specify the mode of the mission,

        earth_mission.add_phase_options(phase_options)
        earth_mission.set_thruster_parameters(thruster_parameters)
        earth_mission.add_control_initial_guess(earth_initial_guess)
        #earth_initial_guess.append(np.random.ra)  # TODO: in first version of simulator, use add_control_initial_guess
        #print(f"initial guess is {earth_initial_guess}")

        # Main solving function
        start_time = time.time()
        # earth_mission.optimize(snopt_options, mbh_options)
        if self.halo_energy == None:
            print("halo_energy is not sampled!")
            exit()
        earth_mission.optimize(snopt_options, mbh_options) 

        end_time = time.time()
        solving_time = end_time - start_time

        # results
        #print("\n")
        #print("is the solution for this intialization feasible?", earth_mission.is_best_solution_feasible())
        #print("\n")

        # assert earth_mission.is_best_solution_feasible()  # TODO: remove the assert

        #self._output_control_to_screen(earth_mission.get_control_state())
        #print("\n")
        #print("--------------------------------------------------------------------------------------------------")

        results = earth_mission.evaluate_and_return_solution(earth_mission.get_control_state(),pydylan.enum.transcription_type.ForwardBackwardShooting)
        feasibility = earth_mission.is_best_solution_feasible()

        problem_results = earth_mission.get_all_feasible_solutions()
        #print('snppt evaluation shape', problem_results[0].snopt_control_evaluations.shape)

        # Get snopt control evaluations if solution is feasible
        #if earth_mission.is_best_solution_feasible():
        #    print("The solution is feasible, getting snopt control evaluation data")
        #    snopt_control_evaluations = problem_results[0].snopt_control_evaluations
        #else:
        #    print("From this initalization, the solution is infeasible")
        #    snopt_control_evaluations = None

        # TODO: to save space, only save data that are used: result.control, feasibility, snopt_control_inform,
        if earth_mission.is_best_solution_feasible():
            result_data = {"results.control": results.control,
                           "feasibility": feasibility,
                           "snopt_control_evaluations": problem_results[0].snopt_control_evaluations,
                           "snopt_inform": problem_results[0].snopt_inform,
                           "thrust": self.thrust,
                           "solving_time": solving_time,
                           "cost_alpha": self.halo_energy}
    
            #manifold_arc = halo.generate_manifold_arc(results.control[-2],results.control[-1],pydylan.enum.PerturbationDirection.StableLeft)
            #self.plot(gto_spiral.get_states(),manifold_arc.mani_states,results)
        else:
            result_data = {"results.control": None,
                           "feasibility": feasibility,
                           "snopt_control_evaluations": None,
                           "snopt_inform": None,
                           "thrust": None,
                           "solving_time": solving_time,
                           "cost_alpha": self.halo_energy}

        return result_data

    def _output_control_to_screen(self, control):
        print('\nThe control vector:')
        for entry in control:
            print('{},'.format(entry))

    def print_statistics(self, result_data, earth_initial_guess_list):
        # Statistics
        # Feasible num
        print(f"alpha in the cost is {self.halo_energy}")

        total_num = 0
        feasible_num = 0
        for result in result_data:
            if result["feasibility"]:
                feasible_num += 1
            total_num += 1
        print("Feasible solution ratio is", feasible_num / total_num)

        # Average final mass for feasible solution
        feasible_final_mass_sum = 0
        for result in result_data:
            if result["feasibility"]:
                feasible_final_mass_sum += result["results.control"][-1]
        if feasible_num == 0:
            print(f"No feasible solution!")
        else:
            print(
                f"Average final mass for feasible solution is {feasible_final_mass_sum / feasible_num:.2f}")

        # Local optimal num
        total_num = 0
        local_optimal_num = 0
        for result in result_data:
            if result["feasibility"] and result["snopt_inform"] == 1:
                local_optimal_num += 1
            total_num += 1
        print("Local optimal solution ratio is", local_optimal_num / total_num)

        # Average final mass for local optimal solution
        local_optimal_final_mass_sum = 0
        for result in result_data:
            if result["snopt_inform"] == 1:
                local_optimal_final_mass_sum += result["results.control"][-1]
        if local_optimal_num == 0:
            print(f"No local optimal solution!")
        else:
            print(
                f"Average final mass for local optimal solution is {local_optimal_final_mass_sum / local_optimal_num:.2f}")

        # Print snopt inform
        for i in range(len(result_data)):
            result = result_data[i]
            if result["feasibility"]:
                print(f"for the {i} -th solutions, the snopt inform is {result_data[i]['snopt_inform']}, solving time "
                      f"is {result_data[i]['solving_time']:.3f}, initial time var [{earth_initial_guess_list[i][0]:.3f}, {earth_initial_guess_list[i][1]:.3f}, {earth_initial_guess_list[i][2]:.3f}],"
                      f" final time var [{result_data[i]['results.control'][0]:.3f}, {result_data[i]['results.control'][1]:.3f}, {result_data[i]['results.control'][2]:.3f}]")
                
    def plot(self,gto_spiral,halo_manifold_arc,results):
        fig, ax = plt.subplots()
        ax.grid()
        ax.set_xlabel(r'X (DU)', fontsize=12)
        ax.set_ylabel(r'Y (DU)', fontsize=12)
        ax.set_title(r'GTO to EM$\mathcal{L}_1$ Halo Low-Thrust Transfer', fontsize=14)
        ax.plot(halo_manifold_arc[:, 0], halo_manifold_arc[:, 1], color='Grey')
        ax.plot(gto_spiral[:, 0], gto_spiral[:, 1], color='DodgerBlue')
        ax.plot(results.states[:, 0], results.states[:, 1], color='LimeGreen')
        fig.savefig(f"{self.result_folder}/earth_mission_seed_{self.seed}.png", dpi=100)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CR3BP earth mission simulator')
    parser.add_argument('--start_bdry',
                        help='specify the start boundary condition, how many seconds for gto spiral',
                        default='6.48423370092')
    parser.add_argument('--end_bdry',
                        help='specify the end boundary condition, how many seconds to left integrate from the halo manifold',
                        default='8.')
    parser.add_argument('--thrust',
                        help='specify the thrust',
                        default='1.')
    parser.add_argument('--control_segment_num',
                        help='specify the number of time segments',
                        default='20')
    parser.add_argument('--initial_guess_sample_mode',
                        help='specify the sample mode for initialization generator, e.g. uniform_sample, gt_example',
                        default='uniform_sample')
    parser.add_argument('--solver_mode',
                        help='specify the solver mode, feasible or optimal',
                        default='optimal')
    parser.add_argument('--maximum_shooting_time',
                        help='specify the maximum shooting time',
                        default='40.0')
    parser.add_argument('--minimum_shooting_time',
                        help='specify the minimum shooting time',
                        default='0.0')
    parser.add_argument('--min_mass_to_sample',
                        help='The lower bound for mass in the uniform sampling',
                        default=350.0)
    parser.add_argument('--max_mass_to_sample',
                        help='The upper bound for mass in the uniform sampling',
                        default=450.0)
    parser.add_argument('--snopt_time_limit',
                        help='maximum time allowed for snopt',
                        default=500.0)
    parser.add_argument('--seed',
                        help='random seed to sample the conditional parameter and the initial guess',
                        default=0)
    parser.add_argument('--seed_step',
                        help='random seed to sample the conditional parameter and the initial guess',
                        default=10)
    parser.add_argument('--result_folder',
                        help='the directory to save the result',
                        default='/home/jg3607/Thesis/AAS_paper/results/boundary/test_boundary/')
    parser.add_argument('--halo_energy',
                        help='halo energy level',
                        default=0.5)

    args = parser.parse_args()

    start_bdry = float(args.start_bdry)
    end_bdry = float(args.end_bdry)
    thrust = float(args.thrust)
    control_segment_num = int(args.control_segment_num)
    initial_guess_sample_mode = args.initial_guess_sample_mode
    solver_mode = args.solver_mode
    maximum_shooting_time = float(args.maximum_shooting_time)
    minimum_shooting_time = float(args.minimum_shooting_time)
    min_mass_to_sample = float(args.min_mass_to_sample)
    max_mass_to_sample = float(args.max_mass_to_sample)
    snopt_time_limit = float(args.snopt_time_limit)
    seed = int(args.seed)
    seed_step = int(args.seed_step)
    result_folder = args.result_folder
    halo_energy = float(args.halo_energy)

    print(f"condition seed is {seed}")

    print(f"initial guess sample mode is {initial_guess_sample_mode}")
    print(f"start boundary is {start_bdry}")
    print(f"end boundary is {end_bdry}")
    print(f"thrust limit is {thrust}")
    print(f"time segment num is {control_segment_num}")
    print(f"solver mode is {solver_mode}")
    print(f"shooting time bound is [{minimum_shooting_time}, {maximum_shooting_time}]")
    print(f"minimum mass to sample is {min_mass_to_sample}, maximum mass to sample is {max_mass_to_sample}")
    print(f"snopt time limit is {snopt_time_limit}")
    print(f"result folder is {result_folder}")

    for initial_guess_seed in range(seed,seed+seed_step):
        simulator = CR3BPEarthMissionWarmstartSimulatorBoundary(seed=initial_guess_seed,
                                                                quiet_snopt=True,
                                                                number_of_segments=control_segment_num,
                                                                maximum_shooting_time=maximum_shooting_time,
                                                                minimum_shooting_time=minimum_shooting_time,
                                                                initial_guess_sample_mode=initial_guess_sample_mode,
                                                                start_bdry=start_bdry,
                                                                end_bdry=end_bdry,
                                                                thrust=thrust,
                                                                solver_mode=solver_mode,
                                                                min_mass_to_sample=min_mass_to_sample,
                                                                max_mass_to_sample=max_mass_to_sample,
                                                                snopt_time_limit=snopt_time_limit,
                                                                result_folder=result_folder,
                                                                halo_energy = halo_energy)
        simulator.run()
