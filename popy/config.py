"""
Global variables and project configuration settings are defined here.

Access these variables by importing this module and using the dot notation.
For example, to access the PROJECT_PATH variable, use the following code:

import config
path = config.PROJECT_PATH
"""

import os
import xarray as xr
import numpy as np
import matplotlib.colors as mcolors

import os

file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(file_path)
grandparent_dir = os.path.dirname(parent_dir)
PROJECT_PATH_LOCAL = grandparent_dir  # add your local path here

### PROCESSING NEUIRAL DATA ###

# sampling rate of spikes and behav during preprocessing
PREPROCESSING_SAMPLING_RATE = 1000

# for time normalization, we use the following epoch lengths
EPOCH_LENS = np.array([1., 1., 0.5, 0.5, 0.5, 4.])


### PLOTTING ###
COLORS = {
    'target_1': '#90c6f2ff', 
    'target_2': '#ffb273ff', 
    'target_3': '#dea8ddff',

    #'LPFC': 'tab:blue',
    'MCC': '#868789fa',
    'dLPFC': '#44b2e0ff',
    'vLPFC': '#fc8e10ff',

    'ka': '#885bb2ff',
    'Monkey KA': '#885bb2ff',
    'po': '#9aaf49ff',
    'Monkey PO': '#9aaf49ff',
    'ka_simulation': 'tab:brown',
    'po_simulation': 'tab:red',
    'yu_DCZ': 'tab:blue',
    'yu_sham': 'tab:orange',

    1: '#238823', 0: '#D2222D',
    1.0: '#238823', 0.0: '#D2222D',
    '1': '#238823', '0': '#D2222D',
    
    'Foraging': '#3268b7ff',
    'Standard RL-s': '#e8465aff',
    'Standard RL': '#e8465aff',
    'Inferential RL-s': '#edaf40ff',
    'Inferential RL': '#edaf40ff',
    'WSLS': "#9e9e9e",
    'Bayesian': "#a86514ff",
    'none': '#cccccc',
    'PROBLEM': '#cccccc'
    }

'''def hex_to_rgb(hex_color):
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hex(rgb_color):
    """Convert RGB tuple to hex color."""
    return '#{:02X}{:02X}{:02X}'.format(*rgb_color)



def interpolate_color(c1, c2, factor):
    return tuple(
        int(c1[i] + (c2[i] - c1[i]) * factor)
        for i in range(3)
    )


def value_gradient(num_steps):
    if num_steps < 2:   
        return ['#D2222D'] if num_steps == 1 else []

    red = hex_to_rgb('#E8642A')
    white = hex_to_rgb("#F2C894")  # Soft off-white
    green = hex_to_rgb('#4A6B9A')

    half = num_steps // 2
    remainder = num_steps - half * 2

    left = [
        rgb_to_hex(interpolate_color(red, white, i / (half if half else 1)))
        for i in range(half)
    ]
    right = [
        rgb_to_hex(interpolate_color(white, green, i / (half + remainder if half + remainder else 1)))
        for i in range(half + remainder + 1)
    ]

    return left + right[1:]'''

def value_gradient(n=256):
    extended_colors = [
        '#227835ff',  # Deeper blue (12.5% position)
        '#3ca854ff',  # Your original dark pastel blue (25% position)
        '#84d1a6ff',  # Pale Cerulean (37.5% position)
        "#bcdfccff",  # Peach-Orange (50% position)
        "#ebd0d2ff",  # Pastel Orange (62.5% position)
        '#ed8e91ff',  # Light Coral (75% position)
        '#bd3a37ff',  # Atomic Tangerine (75% position)
        '#831b1dff'   # Deeper orange-red (87.5% position)
    ][::-1]

    # Create continuous colormap
    cmap = mcolors.LinearSegmentedColormap.from_list("extended_custom", extended_colors, N=n)
    return [cmap(i / (n - 1)) for i in range(n)]


# Optimal parameters for the models
MODEL_PARAMS = {'ka': {'alpha': 0.40963354578309075, 'V0': 0.11827469139505799, 'beta': 10.638021927670694},
                'po': {'alpha': 0.3037981353357479, 'V0': 0.18398040544466904, 'beta': 6.973374756267803},
                'yu_sham': {'alpha': 0.4513693025955279, 'V0': 0.16954116683423484, 'beta': 7.678119299510347},
                'yu_DCZ': {'alpha': 0.4942628307389808, 'V0': 0.1283662338249597, 'beta': 6.46241106842488}}
MODEL_PARAMS['ka_simulation'] = MODEL_PARAMS['ka']
MODEL_PARAMS['po_simulation'] = MODEL_PARAMS['po']
MODEL_PARAMS['yu_sham_simulation'] = MODEL_PARAMS['yu_sham']
MODEL_PARAMS['yu_DCZ_simulation'] = MODEL_PARAMS['yu_DCZ']

MODEL_PARAMS_RL = {'ka': {'alpha': 0.12667346795497197, 'beta': 8.706042488026716},
                   'po': {'alpha': 0.14989319361701048, 'beta': 4.613720171756842},
                   'yu_sham': {'alpha': 0.20112907842232014, 'beta': 5.335652953142322},
                   'yu_DCZ': {'alpha': 0.20897838964272342, 'beta': 4.78188888988298}}

# Cage plotting things
coord_chars = [chr(i) for i in range(97, 97+19)]

def_coords_ka = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                    0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                    0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
                    0., 0., 0.],
                [0., 0., 0., 1., 1., 1., 1., 1., 1., 0., 1., 1., 1., 1., 1., 1.,
                    1., 0., 0.],
                [0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0.,
                    1., 1., 0.],
                [0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                    1., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 0.,
                    0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                    0., 0., 0.],
                [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                    0., 0., 0.],
                [0., 0., 0., 0., 1., 1., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0.,
                    0., 0., 0.],
                [0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0.,
                    0., 0., 0.],
                [0., 0., 0., 0., 1., 0., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0.,
                    0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 0., 0.,
                    0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 1., 1.,
                    0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 1., 1.,
                    0., 0., 0.],
                [0., 0., 1., 1., 1., 1., 0., 1., 0., 0., 0., 0., 0., 0., 1., 1.,
                    0., 0., 0.],
                [0., 0., 0., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                    0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                    0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                    0., 0., 0.]]).T
def_coords_po = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                    0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                    0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 2.,
                    2., 0., 0.],
                [0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 1., 2., 1., 0., 0., 1.,
                    0., 1., 0.],
                [0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 2., 3., 1., 1., 0., 0.,
                    1., 0., 0.],
                [0., 0., 0., 0., 5., 2., 0., 0., 0., 0., 1., 1., 2., 1., 0., 0.,
                    0., 0., 0.],
                [0., 0., 0., 0., 1., 2., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1.,
                    0., 0., 0.],
                [0., 0., 0., 1., 2., 1., 1., 0., 0., 0., 0., 0., 1., 1., 0., 1.,
                    0., 0., 0.],
                [0., 0., 0., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 1., 2., 1.,
                    0., 0., 0.],
                [0., 0., 0., 1., 2., 1., 1., 0., 0., 0., 0., 0., 0., 1., 0., 3.,
                    0., 0., 0.],
                [0., 0., 0., 1., 2., 1., 1., 0., 0., 0., 0., 0., 0., 0., 1., 2.,
                    1., 0., 0.],
                [0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
                    2., 0., 0.],
                [0., 0., 0., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
                    2., 0., 0.],
                [0., 0., 0., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                    1., 2., 0.],
                [0., 0., 0., 1., 2., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
                    0., 1., 0.],
                [0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                    0., 0., 0.],
                [0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                    0., 0., 0.],
                [0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                    0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                    0., 0., 0.]])

GRID_DEFAULT = {'ka': xr.DataArray(def_coords_ka.T, 
                        coords={"x": coord_chars,
                            "y": coord_chars,
                            'loc_x': ('x', np.flip(np.arange(-9, 10, 1))),
                            'loc_y': ('y', np.arange(-9, 10, 1))},
                        dims=("x", "y"))
                ,
                "po": xr.DataArray(def_coords_po.T, 
                        coords={"x": coord_chars,
                            "y": coord_chars,
                            'loc_x': ('x', np.flip(np.arange(-9, 10, 1))),
                            'loc_y': ('y', np.arange(-9, 10, 1))},
                        dims=("x", "y"))
                }
