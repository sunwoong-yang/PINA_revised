# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 18:52:20 2023

@author: KIM
"""

import numpy as np
import torch
from torch.nn import Softplus

from pina.problem import SpatialProblem
from pina.operators import nabla, grad, div, advection
from pina import Condition, Span, LabelTensor
from pina.model import FeedForward
from pina import Condition, Span, PINN, Plotter
import matplotlib.pyplot as plt
import time
# ========================================================================== #
#                                                                            #
#  This script implements steady two dimensional Lid-driven Cavity problem.  #
#  Steady two dimensional Lid-driven Cavity problem is inheriting from       #
#  SpatialProblem.                                                           #
#  We  denote:                                                               #
#           ux --> field variable velocity along x                           #
#           uy --> field variable velocity along y                           #
#           p --> field variable pressure                                    #
#           x,y --> spatial variables                                        #
#                                                                            #
# ========================================================================== #


class Cavity(SpatialProblem):
    # assign output/ spatial variables
    output_variables = ['ux', 'uy', 'p']
    spatial_domain = Span({'x': [0, 1], 'y': [0, 1]})
    
    #define the momentum equation
    def momentum(input_, output_):
        
        adv_u = advection(output_.extract(['ux','uy']), input_, ['ux','uy'])        
                
        nabla_ = torch.hstack((LabelTensor(nabla(output_.extract(['ux']), input_), ['x']),
            LabelTensor(nabla(output_.extract(['uy']), input_), ['y'])))
        Re = 100.0
        
        grad_p = grad(output_.extract(['p']), input_)
        return adv_u + grad_p - 1/Re * nabla_
        
    
    #define the continuity equation
    def continuity(input_, output_):
        a = ['ux', 'uy']
        return div(output_.extract(a), input_)
    
    # define the wall condition
    def wall(input_, output_):
        value = 0.0
        return output_.extract(['ux', 'uy']) -value
    
    # define the lid-ux condition
    def lid_ux (input_, output_):
        value = 1.0
        return output_.extract(['ux']) -value
    
    # define the lid-uy condition
    def lid_uy (input_, output_):
        value = 0.0
        return output_.extract(['uy']) -value
    
    # problem condition statement
    conditions = {
        'wall_left': Condition(location=Span({'x': 0, 'y': [0, 1]}), function=wall),
        'wall_bottom': Condition(location=Span({'x': [0, 1], 'y': 0}), function=wall),
        'wall_right': Condition(location=Span({'x':  1, 'y': [0, 1]}), function=wall),
        'lid_ux':  Condition(location=Span({'x': [0, 1], 'y': 1}), function=lid_ux),
        'lid_uy':  Condition(location=Span({'x': [0, 1], 'y': 1}), function=lid_uy),
        'D1': Condition(location=Span({'x': [0, 1], 'y': [0, 1]}), function=momentum),
        'D2': Condition(location=Span({'x': [0, 1], 'y': [0, 1]}), function=continuity),
    }
# def generate_samples_and_train(model, problem):
#     pinn = PINN(problem, model, lr=1e-3, regularizer=1e-8)
#     pinn.span_pts(2048, 'latin', seed=42, locations=['D1', 'D2'])
#     pinn.span_pts(512, 'grid', seed=42, locations=['wall_left', 'wall_bottom', 'wall_right', 'lid_ux', 'lid_uy'])
#     pinn.train(1000, 500) # 20000 iter
#     return pinn

time_init = time.time()
problem = Cavity()

model = FeedForward(
    layers=[20]*5,
    func=torch.nn.Tanh,
    output_variables=problem.output_variables,
    input_variables=problem.input_variables
)

pinn = PINN(problem, model, lr=1e-3, regularizer=1e-8)
pinn.span_pts(1000, 'latin', seed=42, locations=['D1', 'D2'])
pinn.span_pts(400, 'grid', seed=42, locations=['wall_left', 'wall_bottom', 'wall_right', 'lid_ux', 'lid_uy'])
pinn.train(20000, 500) # 20000 iter
print(f"Cavity wo vorti test loss: {pinn.cal_loss()}")

plotter = Plotter()
plotter.plot(pinn, components='ux', filename="ux_Vanially")
plotter.plot(pinn, components='uy', filename="uy_Vanially")
plotter.plot(pinn, components='p', filename="p_Vanially")
print(f"Total time wo vorti: {time.time()-time_init} [s]")
