""" Module for PINN """
import torch
import torch.optim.lr_scheduler as lrs
import numpy as np
from .problem import AbstractProblem
# from .label_tensor import LabelTensor
from pina import LabelTensor
from .utils import merge_tensors, PinaDataset



torch.pi = torch.acos(torch.zeros(1)).item() * 2  # which is 3.1415927410125732


class PINN(object):

    def __init__(self,
                 problem,
                 model,
                 optimizer=torch.optim.Adam,
                 optimizer_kwargs=None,
                 lr=0.001,
                 lr_scheduler_type=lrs.ConstantLR,
                 lr_scheduler_kwargs={"factor": 1, "total_iters": 0},
                 regularizer=0.00001,
                 batch_size=None,
                 dtype=torch.float32,
                 device='cpu',
                 error_norm='mse'):
        '''
        :param AbstractProblem problem: the formualation of the problem.
        :param torch.nn.Module model: the neural network model to use.
        :param torch.optim.Optimizer optimizer: the neural network optimizer to
            use; default is `torch.optim.Adam`.
        :param dict optimizer_kwargs: Optimizer constructor keyword args.
        :param float lr: the learning rate; default is 0.001.
        :param torch.optim.LRScheduler lr_scheduler_type: Learning
            rate scheduler.
        :param dict lr_scheduler_kwargs: LR scheduler constructor keyword args.
        :param float regularizer: the coefficient for L2 regularizer term.
        :param type dtype: the data type to use for the model. Valid option are
            `torch.float32` and `torch.float64` (`torch.float16` only on GPU);
            default is `torch.float64`.
        :param str device: the device used for training; default 'cpu'
            option include 'cuda' if cuda is available.
        :param (str, int) error_norm: the loss function used as minimizer,
            default mean square error 'mse'. If string options include mean
            error 'me' and mean square error 'mse'. If int, the p-norm is
            calculated where p is specifined by the int input.
        :param int batch_size: batch size for the dataloader; default 5.
        '''

        if dtype == torch.float64:
            raise NotImplementedError('only float for now')

        self.problem = problem

        # self._architecture = architecture if architecture else dict()
        # self._architecture['input_dimension'] = self.problem.domain_bound.shape[0]
        # self._architecture['output_dimension'] = len(self.problem.variables)
        # if hasattr(self.problem, 'params_domain'):
        # self._architecture['input_dimension'] += self.problem.params_domain.shape[0]

        self.error_norm = error_norm

        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError
        self.device = torch.device(device)

        self.dtype = dtype
        self.history_loss = {}

        self.model = model
        self.model.to(dtype=self.dtype, device=self.device)

        self.truth_values = {}
        self.input_pts = {}

        self.trained_epoch = 0

        if not optimizer_kwargs:
            optimizer_kwargs = {}
        optimizer_kwargs['lr'] = lr
        self.optimizer = optimizer(
            self.model.parameters(), weight_decay=regularizer, **optimizer_kwargs)
        self._lr_scheduler = lr_scheduler_type(
            self.optimizer, **lr_scheduler_kwargs)

        self.batch_size = batch_size
        self.data_set = PinaDataset(self)

    @property
    def problem(self):
        """ The problem formulation."""
        return self._problem

    @problem.setter
    def problem(self, problem):
        """
        Set the problem formulation."""
        if not isinstance(problem, AbstractProblem):
            raise TypeError
        self._problem = problem

    def _compute_norm(self, vec):
        """
        Compute the norm of the `vec` one-dimensional tensor based on the
        `self.error_norm` attribute.     

        .. todo: complete

        :param torch.Tensor vec: the tensor
        """
        if isinstance(self.error_norm, int):
            return torch.linalg.vector_norm(vec, ord=self.error_norm,  dtype=self.dytpe)
        elif self.error_norm == 'mse':
            return torch.mean(vec.pow(2))
        elif self.error_norm == 'me':
            return torch.mean(torch.abs(vec))
        else:
            raise RuntimeError

    def save_state(self, filename):
        """
        Save the state of the model.

        :param str filename: the filename to save the state to.
        """
        checkpoint = {
            'epoch': self.trained_epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'optimizer_class': self.optimizer.__class__,
            'history': self.history_loss,
            'input_points_dict': self.input_pts,
        }

        # TODO save also architecture param?
        # if isinstance(self.model, DeepFeedForward):
        #    checkpoint['model_class'] = self.model.__class__
        #    checkpoint['model_structure'] = {
        #    }
        torch.save(checkpoint, filename)

    def load_state(self, filename):
        """
        Load the state of the model.
        
        :param str filename: the filename to load the state from.
        """

        checkpoint = torch.load(filename)
        self.model.load_state_dict(checkpoint['model_state'])

        self.optimizer = checkpoint['optimizer_class'](self.model.parameters())
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])

        self.trained_epoch = checkpoint['epoch']
        self.history_loss = checkpoint['history']

        self.input_pts = checkpoint['input_points_dict']

        return self

    def span_pts(self, *args, **kwargs):
        """
        Generate a set of points to span the `Location` of all the conditions of
        the problem.

        >>> pinn.span_pts(n=10, mode='grid')
        >>> pinn.span_pts(n=10, mode='grid', location=['bound1'])
        >>> pinn.span_pts(n=10, mode='grid', variables=['x'])
        """

        if all(key in kwargs for key in ['n', 'mode']):
            argument = {}
            argument['n'] = kwargs['n']
            argument['mode'] = kwargs['mode']
            argument['variables'] = self.problem.input_variables
            arguments = [argument]
        elif any(key in kwargs for key in ['n', 'mode']) and args:
            raise ValueError("Don't mix args and kwargs")
        elif isinstance(args[0], int) and isinstance(args[1], str):
            argument = {}
            argument['n'] = int(args[0])
            argument['mode'] = args[1]
            argument['variables'] = self.problem.input_variables
            arguments = [argument]
        elif all(isinstance(arg, dict) for arg in args):
            arguments = args
        else:
            raise RuntimeError

        # when 'locations' is not defined, default is 'all'
        locations = kwargs.get('locations', 'all')
        seed = kwargs.get('seed', None)
        if locations == 'all':
            locations = [condition for condition in self.problem.conditions]
        for location in locations:
            condition = self.problem.conditions[location]

            samples = tuple(condition.location.sample(
                            argument['n'],
                            argument['mode'],
                            variables=argument['variables'],
                            seed=seed)
                            for argument in arguments)

            # function "merge_tensors" outputs pts coordinates with the labels
            pts = merge_tensors(samples)
            self.input_pts[location] = pts

    def update_pts(self, N_pts=100, weight_mass=0.5, *args, **kwargs):
        """
        My new function for adaptive sampling
        """

        # when 'locations' is not defined, default is 'D1' and 'vorticity'
        locations = kwargs.get('locations', ['D1', 'vorticity'])
        domain_pts = self.input_pts[locations[0]].shape[0]
        # when 'seed' is not defined, default is 'None', which means there is no random seed
        seed = kwargs.get('seed', None)

        # domain_pts = self.input_pts['D1'].shape[0]

        # number of points newly added w.r.t. mass eqn
        N_mass = int((domain_pts - N_pts) * weight_mass)
        # number of points newly added w.r.t. momentum eqn
        N_momentum = domain_pts - N_mass - N_pts
        # For evaluation of vorticity magnitude, "eval_pts_MarginFactor * N_pts" number of points will be used
        eval_pts_MarginFactor = 3

        for location in locations:
            condition = self.problem.conditions[location]
            if location == 'vorticity':
                # Select top N_pts among N_pts_MarginFactor*N_pts randomly sampled candidates
                # Caution: too large "N_pts_MarginFactor" can make the newly added pts be concentrated on the small region
                pts_eval_vorti = condition.location.sample(eval_pts_MarginFactor*N_pts, 'random', seed=seed)
                pts_eval_vorti.requires_grad_(True)
                predicted = self.model(pts_eval_vorti)
                residuals = condition.function[0](pts_eval_vorti, predicted)
                # Note that condition.data_weight of the 'vorticity' location was set as zero to disregard it during training
                local_loss = 1. * residuals


            elif location == 'D1' or location == 'D2':
                # pts_eval_pde = self.input_pts[location]
                pts_eval_pde = condition.location.sample(eval_pts_MarginFactor*(N_mass+N_momentum), 'random', seed=seed)
                pts_eval_pde.requires_grad_(True)
                predicted = self.model(pts_eval_pde)
                residuals = condition.function[0](pts_eval_pde, predicted)
                local_loss = condition.data_weight * residuals

            if location == 'vorticity':
                new_sorted, new_indices = torch.sort(local_loss, dim=0, descending=True)
            elif location == 'D1':
                old_D1_sorted, old_D1_indices = torch.sort(local_loss, dim=0, descending=True)
            elif location == 'D2':
                old_D2_sorted, old_D2_indices = torch.sort(local_loss, dim=0, descending=True)

        # First, top N_mass pts w.r.t. mass loss are selected from the previous collocation pts
        D1_new_pts = pts_eval_pde[old_D1_indices[:,0]][:N_mass]
        # Then, top N_momentum pts w.r.t. momentum loss are selected from the previous collocation pts, except pts contained in 'D1_new_pts'
        D2_new_indices = torch.from_numpy(np.setdiff1d(old_D2_indices[:,0].numpy(), old_D1_indices[:,0][:N_mass].numpy()))
        D2_new_pts = pts_eval_pde[D2_new_indices][:N_momentum]
        # Lastly, top N_pts pts w.r.t. vorticity are selected from the newly sampled dataset 'pts_eval'
        vorti_new_pts = pts_eval_vorti[new_indices[:,0]][:N_pts]
        merged_pts = torch.concatenate((D1_new_pts,
                                        D2_new_pts,
                                        vorti_new_pts,
                                        ), dim=0).reshape(-1, 2)

        merged_pts = LabelTensor(merged_pts, ['x', 'y'])
        for location in locations:
            self.input_pts[location] = merged_pts

    def train(self, stop=100, frequency_print=2, save_loss=1, trial=None):

        self.model.train()
        epoch = 0
        # Add all condition with `input_points` to dataloader
        for condition in list(set(self.problem.conditions.keys()) - set(self.input_pts.keys())):
            self.input_pts[condition] = self.problem.conditions[condition]

        data_loader = self.data_set.dataloader

        header = []
        for condition_name in self.problem.conditions:
            condition = self.problem.conditions[condition_name]

            if hasattr(condition, 'function'):
                if isinstance(condition.function, list):
                    for function in condition.function:
                        header.append(f'{condition_name}{function.__name__}')

                    continue

            header.append(f'{condition_name}')

        while True:

            losses = []

            for condition_name in self.problem.conditions:
                condition = self.problem.conditions[condition_name]

                for batch in data_loader[condition_name]:

                    single_loss = []

                    if hasattr(condition, 'function'):
                        pts = batch[condition_name]
                        pts = pts.to(dtype=self.dtype, device=self.device)
                        pts.requires_grad_(True)
                        pts.retain_grad()

                        predicted = self.model(pts)
                        for function in condition.function:
                            residuals = function(pts, predicted)
                            local_loss = (
                                condition.data_weight*self._compute_norm(
                                    residuals))
                            single_loss.append(local_loss)
                    elif hasattr(condition, 'output_points'):
                        pts = condition.input_points.to(
                            dtype=self.dtype, device=self.device)
                        predicted = self.model(pts)
                        residuals = predicted - \
                            condition.output_points.to(
                                device=self.device, dtype=self.dtype)  # TODO fix
                        local_loss = (
                            condition.data_weight*self._compute_norm(residuals))
                        single_loss.append(local_loss)

                    self.optimizer.zero_grad()
                    # adaptive sampling하면서 backward 과정에서 retain_graph=True가 필요해짐
                    sum(single_loss).backward(retain_graph=True)
                    self.optimizer.step()

                # Append to list "losses" each loss at each condition
                losses.append(sum(single_loss))

            self._lr_scheduler.step()

            if save_loss and (epoch % save_loss == 0 or epoch == 0):
                self.history_loss[epoch] = [
                    loss.detach().item() for loss in losses]

            if trial:
                import optuna
                trial.report(sum(losses), epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

            if isinstance(stop, int):
                if epoch == stop:
                    print('[epoch {:05d}] {:.6e} '.format(
                        self.trained_epoch, sum(losses).item()), end='')
                    for loss in losses:
                        print('{:.6e} '.format(loss.item()), end='')
                    print()
                    break
            elif isinstance(stop, float):
                if sum(losses) < stop:
                    break

            if epoch % frequency_print == 0 or epoch == 1:
                print('       {:5s}  {:12s} '.format('', 'sum'),  end='')
                for name in header:
                    print('{:12.12s} '.format(name), end='')
                print()

                print('[epoch {:05d}] {:.6e} '.format(
                    self.trained_epoch, sum(losses).item()), end='')
                for loss in losses:
                    print('{:.6e} '.format(loss.item()), end='')
                print()

            self.trained_epoch += 1
            epoch += 1

        self.model.eval()

        return sum(losses).item()

    # def error(self, dtype='l2', res=100):

    #     import numpy as np
    #     if hasattr(self.problem, 'truth_solution') and self.problem.truth_solution is not None:
    #         pts_container = []
    #         for mn, mx in self.problem.domain_bound:
    #             pts_container.append(np.linspace(mn, mx, res))
    #         grids_container = np.meshgrid(*pts_container)
    #         Z_true = self.problem.truth_solution(*grids_container)

    #     elif hasattr(self.problem, 'data_solution') and self.problem.data_solution is not None:
    #         grids_container = self.problem.data_solution['grid']
    #         Z_true = self.problem.data_solution['grid_solution']
    #     try:
    #         unrolled_pts = torch.tensor([t.flatten() for t in grids_container]).T.to(
    #             dtype=self.dtype, device=self.device)
    #         Z_pred = self.model(unrolled_pts)
    #         Z_pred = Z_pred.detach().numpy().reshape(grids_container[0].shape)

    #         if dtype == 'l2':
    #             return np.linalg.norm(Z_pred - Z_true)/np.linalg.norm(Z_true)
    #         else:
    #             # TODO H1
    #             pass
    #     except:
    #         print("")
    #         print("Something went wrong...")
    #         print(
    #             "Not able to compute the error. Please pass a data solution or a true solution")

    def cal_loss(self, N_eval=200):
        locations = ["D1", "D2"]
        losses = []
        for location in locations:
            condition = self.problem.conditions[location]
            pts_eval = condition.location.sample(N_eval, 'grid')
            pts_eval.requires_grad_(True)
            predicted = self.model(pts_eval)
            residuals = condition.function[0](pts_eval, predicted)
            losses.append(torch.mean(condition.data_weight * residuals, dim=0))
        return losses