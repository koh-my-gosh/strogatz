#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from tqdm import tqdm
from scipy import optimize
import matplotlib.pyplot as plt
from typing import Callable, List

class dynamics2D(object):
    def __init__(
        self, F: list[Callable[[int], int]], xlim: tuple, ylim: tuple, 
        n_interval: int = 5, fig_scale: float = 1.0, ax = None
    ):
        assert len(F) == 2
        assert len(xlim) == 2 and len(ylim) == 2
        self.F = F
        self.xlim = xlim
        self.ylim = ylim
        self.xwidth = xlim[1] - xlim[0]
        self.ywidth = ylim[1] - ylim[0]
        self.n_interval = n_interval
        self.fig_scale = fig_scale
        self.init_meshgrid()
        self.get_UV()
        self.init_axes(ax=ax)
        
    def init_axes(self, ax=None):
        if ax is None:
            self.fig, self.ax = plt.subplots(figsize=(self.xwidth*self.fig_scale, self.ywidth*self.fig_scale))
        else:
            self.ax = ax
        self.ax.set_xlim(*self.xlim)
        self.ax.set_ylim(*self.ylim)
        self.ax.grid()
        self.ax.axhline(y=0, color="grey", linewidth=1.8)
        self.ax.axvline(x=0, color="grey", linewidth=1.8)
        
    def init_meshgrid(self):
        _x = np.linspace(self.xlim[0], self.xlim[1], self.xwidth*self.n_interval)
        _y = np.linspace(self.ylim[0], self.ylim[1], self.ywidth*self.n_interval)
        _X, _Y = np.meshgrid(_x, _y)
        self.XY = np.zeros(shape=(*_X.shape, 2))
        self.XY[..., 0] = _X
        self.XY[..., 1] = _Y
    
    def get_UV(self):
        self.UV = np.zeros_like(self.XY)
        self.UV[..., 0] = self.F[0](self.XY)
        self.UV[..., 1] = self.F[1](self.XY)
        
    def add_direction_field(self):
        norms = np.linalg.norm(self.UV, axis=-1, keepdims=True)
        norm_UV = self.UV/norms
        self.ax.quiver(self.XY[..., 0], self.XY[..., 1], norm_UV[..., 0], norm_UV[..., 1], norms, cmap="viridis")
    
    def add_nullclines(self):
        self.ax.contour(self.XY[..., 0], self.XY[..., 1], self.UV[..., 0], levels=[0], cmap="hsv")
        self.ax.contour(self.XY[..., 0], self.XY[..., 1], self.UV[..., 1], levels=[0], cmap="hsv")
        
        extracted_points = []
        for i in range(2):
            _obj = self.ax.contour(self.XY[..., 0], self.XY[..., 1], self.UV[..., i], levels=[0], cmap="hsv")
            _points = []
            for path in _obj.get_paths():
                v = path.vertices
                _points.append(v)
            extracted_points.append(np.array(_points[0]))
        dist = np.linalg.norm(extracted_points[0][:, None, :]-extracted_points[1][None, :, :], axis=-1)
        return dist
    def add_fixed_points(self, fixed_points: list[list], fp_types: list[str], markersize: float = 12):
        fp = np.array(fixed_points)
        fpt = np.array(fp_types)
        assert fp.shape[-1] == 2
        assert fpt.shape[0] == len(fp_types)
        
        # FIXME: fp_types
        self.ax.plot(fp[..., 0], fp[..., 1], marker=".", markersize=markersize, color="tab:red", linewidth=0.0)
        
        saddle_idx = np.where(fpt == "saddle")[0]
        self.ax.plot(fp[saddle_idx, 0], fp[saddle_idx, 1], marker=".", markersize=0.5*markersize, color="white", linewidth=0.0)
        
        unstable_idx = np.where(fpt == "unstable")[0]
        self.ax.plot(fp[unstable_idx, 0], fp[unstable_idx, 1], marker=".", markersize=0.5*markersize, color="white", linewidth=0.0)
        
    def add_flows(self, initials: list[list], max_length: float = 10.0):
        # init_x = np.array(initials)
        self.ax.streamplot(
            self.XY[..., 0], self.XY[..., 1], self.UV[..., 0], self.UV[..., 1], 
            start_points=initials, color='tab:orange', linewidth=1.0, maxlength=max_length
        )