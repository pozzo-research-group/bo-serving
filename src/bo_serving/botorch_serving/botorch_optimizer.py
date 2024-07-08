import numpy as np
import botorch
import torch

torch.set_default_dtype(torch.double)

from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.models.transforms.outcome import Standardize 
from botorch.optim import optimize_acqf
from botorch.utils.transforms import normalize, unnormalize

from gpytorch.kernels import MaternKernel

import torch

from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP, FixedNoiseGP
from gpytorch.mlls import ExactMarginalLogLikelihood

import json

class ExperimentData():
    """"
    Fancy cache for experimental data 
    """

    def __init__(self, n_dims_x):
        self.x_data = None
        self.y_data = None
        self.n_dims_x = n_dims_x
        self.open_trials = {}
        self.n_complete_trials = 0
        self.extra_data = {}
        


    def complete_trial(self, trial_index, new_y_data, extra_data = None):
        """
        Add results from a new trial 

        new_x_params: torch tensor or np array nx1
        new_y_data: str, float, or int
        """

        assert trial_index in self.open_trials.keys(), 'Trial index is not in set of open trials'
        assert isinstance(new_y_data, float) or isinstance(new_y_data, int) or isinstance(new_y_data, str)

        new_x = self.open_trials[trial_index]

        new_x_params = torch.tensor(list(new_x.values())).reshape(1,-1)



        try:
            new_y_data = float(new_y_data)
        except:
            raise ValueError(f'Could not cast new_y_data {new_y_data} as float')
        new_y_data = torch.tensor(new_y_data).reshape(1,-1)


        if self.x_data is None:
            x_data = new_x_params
        else:
            x_data = torch.cat([self.x_data, new_x_params], dim = 0)

        if self.y_data is None:
            y_data = new_y_data
        else:
            y_data = torch.cat([self.y_data, new_y_data], dim = 0)

        self.x_data = x_data
        self.y_data = y_data


        #print('got extra data: ', extra_data)
        self.extra_data[trial_index] = extra_data
        # close out trial bookkeeping
        self.n_complete_trials += 1
        del self.open_trials[trial_index]

       # print('X shape: ', self.x_data.shape)
        #print('Y shape: ', self.y_data.shape)

    def get_data(self):
        return self.x_data, self.y_data
    
    def register_new_trial(self, parameterization):
    
        assert len(parameterization) == self.n_dims_x, f'Parameterization must have {self.n_dims_x} entries'

        if self.x_data is None:
            trial_index = 0
        elif len(self.open_trials) == 0:
            trial_index = len(self.x_data)
        else:
            trial_index = max(list(self.open_trials.keys())) + 1

        self.open_trials[trial_index] = parameterization

        return trial_index


class BoTorchOptimizer():
    def __init__(self, bounds, n_dims_x, batch_size, n_random_trials, n_bo_trials, uniqueid, task = 'maximize', nu = 5/2):
        self.model_name = "gp"
        self.model = None
        self.acq_func = None
        self.initial_bounds = bounds
        self.tensor_bounds = torch.tensor(self.initial_bounds).transpose(-1, -2)
        self.task = task
        self.batch = batch_size
        self.design_space_dim = len(bounds)
        self.output_dim = 1
        self.n_dims_x = n_dims_x
        self.nu = nu
        self.n_random_trials = n_random_trials
        self.n_bo_trials = n_bo_trials
        self.ExperimentData = ExperimentData(n_dims_x = n_dims_x )
        self.uniqueid = uniqueid

    @staticmethod
    def data_utils(data):

        if isinstance(data, np.ndarray):
            data_ = torch.from_numpy(data)
        else:
            data_ = torch.from_numpy(np.array(data))
        
        return data_
    

    def generate_random_parameterization(self):
        # Generate n_samples random color samples presented as proportions of stock colors volumes
        param_arr = np.random.dirichlet(np.ones(self.n_dims_x), 1)[0]
        print('shuffled array')
        np.random.shuffle(param_arr)
        return param_arr
        #return self.optimizer.ask()
    
    def get_next_trial(self):


        if self.ExperimentData.n_complete_trials < self.n_random_trials:
            new_x = self.generate_random_parameterization()
            print('New parameterization generated with random sampling')

        else:
            new_x = self._ask()
            print('New parameterization generated with BO')
        

        parameterization = {f'x{i}':val for i, val in enumerate(new_x)}

        trial_index = self.ExperimentData.register_new_trial(parameterization)

        return parameterization, trial_index
    
    def update(self, trial_index, raw_data, extra_data = None):
        """
        Externally callable function to update with new data

        trial_index: int
        raw_data: int float or str
        """

        self.ExperimentData.complete_trial(trial_index, raw_data, extra_data = extra_data)

        self.update_surrogate(*self.ExperimentData.get_data())

        if extra_data is not None:
            self.generate_observability_data()


    def update_surrogate(self, x_data, y_data):

        x_data = self.data_utils(x_data)

        if self.task == 'maximize':
            y_data = self.data_utils(y_data)
            best= y_data.max()
        elif self.task == 'minimize':
            y_data = -1*self.data_utils(y_data)
            best = y_data.min()
        else:
            raise ValueError(f'Task must be either maximize or minimize, not {self.task}')

        normalized_x = normalize(x_data, self.tensor_bounds)
        self.initialize_model(normalized_x, y_data)
      
        acquisition = LogExpectedImprovement(self.model, best_f = best)
        self.acq_func = acquisition
        return 
    
    def _ask(self):


        indices = torch.arange(self.tensor_bounds.shape[1])
        coeffs = torch.ones(self.tensor_bounds.shape[1])
        constraints = [(indices, coeffs, 1)]

        normalized_candidates, acqf_values = optimize_acqf(
            self.acq_func, 
            self.tensor_bounds, 
            q=self.batch, 
            num_restarts=5, 
            raw_samples=10, 
            return_best_only=True,
            sequential=False,
            options={"batch_limit": 1, "maxiter": 10, "with_grad":True}, 
            equality_constraints=constraints
            )
        # calculate acquisition values after rounding
        new_x = unnormalize(normalized_candidates.detach(), bounds=self.tensor_bounds) 

        return new_x.numpy().squeeze()

    def initialize_model(self,x_data, y_data ):
        """
        Initialize and fit a new single task GP with Matern kernel
        
        :param x_data: All X data to train model on
        :type x_data: torch tensor or np array
        :param y_data: All associated y data to train model on 
        :type y_data: torch tensor or np array
        """
        print('Initializing model')
        kernel = MaternKernel(nu = self.nu)
        gp_model = SingleTaskGP(self.data_utils(x_data), self.data_utils(y_data), outcome_transform=Standardize(m=1), covar_module=kernel).to(x_data)

        mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)

        fit_gpytorch_mll(mll)
        
        self.mll = mll
        self.model = gp_model

        return
    
    def new_model_for_observability(self):
        """
        This is a nasty hack to get around a pickling issue.

        When I make predictions on x values for observability, something unpickleable gets attached to 
        BoTorchOptimizer object, breaking caching. this handles model fitting here to avoid this. Downside is we fit 
        model twice. Should figure out a real fix


        """
        xdata, ydata = self.ExperimentData.get_data()
        ## Code from update_surrogate()
        x_data = self.data_utils(xdata)

        if self.task == 'maximize':
            y_data = self.data_utils(ydata)
            best= y_data.max()
        elif self.task == 'minimize':
            y_data = -1*self.data_utils(ydata)
            best = y_data.min()
        else:
            raise ValueError(f'Task must be either maximize or minimize, not {self.task}')

        normalized_x = normalize(x_data, self.tensor_bounds)


        ## Code from initialize_model
        kernel = MaternKernel(nu = self.nu)
        gp_model = SingleTaskGP(self.data_utils(normalized_x), self.data_utils(y_data), outcome_transform=Standardize(m=1), covar_module=kernel).to(normalized_x)

        mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)

        fit_gpytorch_mll(mll)
      
        ## end initialize model code
        posterior = gp_model.posterior(self.data_utils(normalized_x))
        mean  = posterior.mean.detach().reshape(-1).tolist()
        if self.task == 'minimize':
            mean = [-1*val for val in mean]

        del kernel
        del gp_model
        del mll
        del posterior

        return mean
        
    
    def generate_observability_data(self):
        """"
        Generate a data file of observability metrics, for example for streamlit dashboard

        Things that should go here:
        - Most recent image
        - All the colors that have been observed along with their loss scores
        - Model predictions for all tested points

        """

        # get observed rgb for all trials
        trials = list(self.ExperimentData.extra_data.keys())

        try:
            observed_rgb = [self.ExperimentData.extra_data[trial_ind]['observed_rgb'] for trial_ind in trials]
        except KeyError:
            observed_rgb = None



        # get most recent image from extra_data
        try:
            image = self.ExperimentData.extra_data[max(trials)]['image']
        except KeyError:
            image = None

        
        try:
            color_swatch = self.ExperimentData.extra_data[max(trials)]['color_swatch']
        except KeyError:
            color_swatch = None
        # get model predictions for parity plot

        x_data = self.ExperimentData.x_data.detach().clone()
        true_y = self.ExperimentData.y_data.detach().clone()
        true_y = true_y.reshape(-1).tolist()

        pred_y = self.new_model_for_observability()

        

        data = {}

        data['observed_rgb'] = observed_rgb
        data['model_prediction'] = {'y_true':true_y, 'y_pred':pred_y}
        data['image'] = image
        data['color_swatch'] = color_swatch

        with open(f'servedata_{self.uniqueid}.json', 'wt') as f:
            f.write(json.dumps(data))

        #with open('mostrecentimage.jpg', 'wb') as f:
        #    f.write(image)
            
        del self.model
        del self.mll
        del x_data
        del true_y
        del data

        return 