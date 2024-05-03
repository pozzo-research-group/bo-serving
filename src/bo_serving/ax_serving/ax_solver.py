from ax.modelbridge.dispatch_utils import choose_generation_strategy
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.modelbridge_utils import get_pending_observation_features
from ax.modelbridge.registry import ModelRegistryBase, Models

from ax.utils.testing.core_stubs import get_branin_experiment, get_branin_search_space

from ax.modelbridge.factory import get_GPEI

from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.utils.measurement.synthetic_functions import hartmann6
from ax.utils.notebook.plotting import init_notebook_plotting, render


def clean_results_json(results_json):
    """"
    Clean results json since tuples are lost
    """
    cleaned_json = {}
    #print(results_json)
    for key, value in results_json.items():
        if isinstance(value, list):
            if len(value) == 2:
                cleaned_json[key] = tuple(value)
        else:
            cleaned_json[key] = value

    return cleaned_json



class AxSolver():
   
    def get_ax_object(self, n_params, n_random_trials, n_bo_trials):

        gs = GenerationStrategy(
            steps=[
                GenerationStep(
                    model=Models.SOBOL,
                    num_trials=n_random_trials,  # How many trials should be produced from this generation step
                    min_trials_observed=1,  # How many trials need to be completed to move to next model
                    max_parallelism=1,  # Max parallelism for this step
                    model_kwargs={"seed": 999},  # Any kwargs you want passed into the model
                    model_gen_kwargs={},  # Any kwargs you want passed to `modelbridge.gen`
                ),
                # 2. Bayesian optimization step
                GenerationStep(
                    model = get_GPEI,
                    num_trials=n_bo_trials,  # No limitation on how many trials should be produced from this step
                    max_parallelism=1
                ),
            ]
        )
        ax_client = AxClient(generation_strategy = gs)

        experiment = ax_client.create_experiment(
                name="color_matching",
                parameters=[{"name":f"x{i+1}", "type":"range", "bounds":[0.0, 1.0], "value_type":"float"} for i in range(n_params-1)], 
                objectives={"euclidean": ObjectiveProperties(minimize=True)},
                parameter_constraints=[' + '.join([f'x{i+1}' for i in range(n_params-1)]) + " <= 1.0"]  # Optional.
        )

        return ax_client

    def __init__(self, n_params, n_random_trials, n_bo_trials):

        self.ax_client = self.get_ax_object(n_params, n_random_trials, n_bo_trials)
        self.n_params = n_params


    def ask(self):
        query_point, trial_ind = self.ax_client.get_next_trial()
        self.open_trial = trial_ind
        ax_ratios = list(query_point.values())
        # 3rd volume is fixed by selection of other 2. But then BO isn't learning this parameter explicitly...
        r3 = 1 - sum(ax_ratios)
        ax_ratios.append(r3)
        param_dict = {f'x{i+1}':r for i, r in zip(range(self.n_params), ax_ratios)}
        return param_dict, trial_ind
    
    def update(self, trial_index, raw_data):
        

        self.ax_client.complete_trial(trial_index = trial_index, raw_data = raw_data)

        return
