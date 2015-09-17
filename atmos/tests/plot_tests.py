# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 11:11:20 2015

@author: mcgibbon
"""
from __future__ import division, unicode_literals
import nose
import numpy as np
from atmos import plot
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import image_comparison


@image_comparison(baseline_images=['linear_skewT'],
                  extensions=['png', 'pdf'])
def test_linear_skewT():
    plt.style.use('bmh')
    fig, ax = plt.subplots(1, 1, figsize=(6, 6),
                           subplot_kw={'projection': 'skewT'})
    ax.plot(np.linspace(1e3, 100, 100), np.linspace(0, -50, 100))
    fig.tight_layout()


@image_comparison(baseline_images=['linear_skewT_with_barbs'],
                  extensions=['png', 'pdf'])
def test_linear_skewT_with_barbs():
    plt.style.use('bmh')
    fig, ax = plt.subplots(1, 1, figsize=(6, 6),
                           subplot_kw={'projection': 'skewT'})
    ax.plot(np.linspace(1e3, 100, 100), np.linspace(0, -50, 100))
    ax.plot_barbs(np.linspace(1e3, 100, 10), np.linspace(50, -50, 10),
                  np.linspace(-50, 50, 10), xloc=0.95)
    fig.tight_layout()


if __name__ == '__main__':
    nose.run()
