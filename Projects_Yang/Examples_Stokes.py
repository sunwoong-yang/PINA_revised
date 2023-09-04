import numpy as np
import torch

from pina.problem import SpatialProblem
from pina.operators import nabla, grad, div
from pina import Condition, Span, LabelTensor

import argparse
import sys
import numpy as np
import torch
from torch.nn import ReLU, Tanh, Softplus

from pina import PINN, LabelTensor, Plotter
from pina.model import FeedForward
from pina.adaptive_functions import AdaptiveSin, AdaptiveCos, AdaptiveTanh
# from problems.stokes import Stokes


# ===================================================== #
#                                                       #
#  This script implements the two dimensional           #
#  Stokes problem. The Stokes class is defined          #
#  inheriting from SpatialProblem. We  denote:          #
#           ux --> field variable velocity along x      #
#           uy --> field variable velocity along y      #
#           p --> field variable pressure               #
#           x,y --> spatial variables                   #
#                                                       #
# ===================================================== #

class Stokes(SpatialProblem):

    # assign output/ spatial variables
    output_variables = ['ux', 'uy', 'p']
    spatial_domain = Span({'x': [-2, 2], 'y': [-1, 1]})

    # define the momentum equation
    def momentum(input_, output_):
        nabla_ = torch.hstack((LabelTensor(nabla(output_.extract(['ux']), input_), ['x']),
            LabelTensor(nabla(output_.extract(['uy']), input_), ['y'])))
        return - nabla_ + grad(output_.extract(['p']), input_)

    # define the continuity equation
    def continuity(input_, output_):
        return div(output_.extract(['ux', 'uy']), input_)

    # define the inlet velocity
    def inlet(input_, output_):
        value = 2 * (1 - input_.extract(['y'])**2)
        return output_.extract(['ux']) - value

    # define the outlet pressure
    def outlet(input_, output_):
        value = 0.0
        return output_.extract(['p']) - value

    # define the wall condition
    def wall(input_, output_):
        value = 0.0
        return output_.extract(['ux', 'uy']) - value

    # define vorticity condition
    def vorticity(input_, output_):
        u_y = grad(output_, input_, components=['ux'], d=['y'])
        v_x = grad(output_, input_, components=['uy'], d=['x'])
        return torch.abs(u_y - v_x)

    # problem condition statement
    conditions = {
        'gamma_top': Condition(location=Span({'x': [-2, 2], 'y':  1}), function=wall),
        'gamma_bot': Condition(location=Span({'x': [-2, 2], 'y': -1}), function=wall),
        'gamma_out': Condition(location=Span({'x':  2, 'y': [-1, 1]}), function=outlet),
        'gamma_in':  Condition(location=Span({'x': -2, 'y': [-1, 1]}), function=inlet),
        'D1': Condition(location=Span({'x': [-2, 2], 'y': [-1, 1]}), function=momentum),
        'D2': Condition(location=Span({'x': [-2, 2], 'y': [-1, 1]}), function=continuity),
        'vorticity': Condition(location=Span({'x': [-2, 2], 'y': [-1, 1]}), function=vorticity, data_weight=0.),
    }

stokes_problem = Stokes()
model = FeedForward(
    layers=[10, 10, 10, 10],
    output_variables=stokes_problem.output_variables,
    input_variables=stokes_problem.input_variables,
    func=Softplus,
)

pinn = PINN(
    stokes_problem,
    model,
    lr=0.006,
    error_norm='mse',
    regularizer=1e-8)

# if args.s:

pinn.span_pts(200, 'grid', locations=['gamma_top', 'gamma_bot', 'gamma_in', 'gamma_out'])
pinn.span_pts(2000, 'latin', seed=42, locations=['D1'])
pinn.span_pts(2000, 'latin', seed=42, locations=['D2'])
# Initial sampling is required for 'vorticity'
pinn.span_pts(1, 'random', locations=['vorticity'])

update = False
if update:
    pinn.train(2000, 100)
    pinn.update_pts(100, weight_mass=0.7, seed=40, locations=['D1', 'D2', 'vorticity'])
    pinn.train(500, 100)
    pinn.update_pts(100, weight_mass=0.7, seed=41, locations=['D1', 'D2', 'vorticity'])
    pinn.train(500, 100)
    pinn.update_pts(100, weight_mass=0.7, seed=43, locations=['D1', 'D2', 'vorticity'])
    pinn.train(500, 100)
    pinn.update_pts(100, weight_mass=0.7, seed=44, locations=['D1', 'D2', 'vorticity'])
    pinn.train(500, 100)
    pinn.update_pts(100, weight_mass=0.7, seed=45, locations=['D1', 'D2', 'vorticity'])
    pinn.train(500, 100)
    pinn.update_pts(100, weight_mass=0.7, seed=46, locations=['D1', 'D2', 'vorticity'])
else:
    pinn.train(5000, 100)

plotter = Plotter()
plotter.plot(pinn, components='ux')
plotter.plot(pinn, components='uy')
plotter.plot(pinn, components='p')