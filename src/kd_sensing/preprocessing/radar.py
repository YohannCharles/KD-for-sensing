from __future__ import annotations

import numpy as np


class RadarKPI:
    def __init__(self):
        self.N_r = 4
        self.S = 15
        self.f_0 = 77
        self.F_s = 5
        self.N_sample = 256
        self.N_chirp = 128
        self.T_PRI = 65
        self.T_active = self.N_sample / self.F_s
        self.BW = self.S * self.T_active
        self.f_c = self.f_0 + 1e-3 * self.BW / 2
        self.range_res = 3 * 1e8 / (2 * self.BW * 1e6)
        self.range_max = self.N_sample * self.range_res
        self.velocity_res = (
            (3 * 1e8 / (self.f_c * 1e9)) / (2 * self.T_PRI * 1e-6 * self.N_chirp) * 3.6
        )
        self.velocity_max = (3 * 1e8 / (self.f_c * 1e9)) / (4 * self.T_PRI * 1e-6) * 3.6
        self.measurement_offset_angle = 4 * np.pi / 180
        self.angle_start = 0 - self.measurement_offset_angle
        self.angle_end = np.pi - self.measurement_offset_angle
        self.num_of_angle = 64

    def to_dict(self) -> dict[str, float]:
        return {
            "active_chirp_duration_us": self.T_active,
            "chirp_bandwidth_mhz": self.BW,
            "center_frequency_ghz": self.f_c,
            "range_resolution_m": self.range_res,
            "maximum_range_m": self.range_max,
            "velocity_resolution_kmh": self.velocity_res,
            "maximum_velocity_kmh": self.velocity_max,
        }


def Radar_Cube(radar_data, fft_tuple, remove_mean: bool = True):
    fft_angle, fft_range, fft_velocity = fft_tuple
    range_dft = np.fft.fft(radar_data, n=fft_range, axis=1)
    if remove_mean:
        range_dft = range_dft - np.mean(range_dft, axis=2, keepdims=True)
    doppler_dft = np.fft.fft(range_dft, n=fft_velocity, axis=2)
    angle_dft = np.fft.fft(doppler_dft, n=fft_angle, axis=0)
    return np.fft.fftshift(angle_dft, axes=(0, 2))


def Range_Doppler(radar_cube, mean: bool = True, log_scale: bool = True):
    result = np.mean(np.abs(radar_cube), axis=0) if mean else np.sum(np.abs(radar_cube), axis=0)
    return np.log2(1 + result) if log_scale else result


def Range_Angle(radar_cube, mean: bool = True, log_scale: bool = True):
    result = np.mean(np.abs(radar_cube), axis=2).T if mean else np.sum(np.abs(radar_cube), axis=2).T
    return np.log2(1 + result) if log_scale else result


def Doppler_Angle(radar_cube, mean: bool = True, log_scale: bool = True):
    result = np.mean(np.abs(radar_cube), axis=1).T if mean else np.sum(np.abs(radar_cube), axis=1).T
    return np.log2(1 + result) if log_scale else result

