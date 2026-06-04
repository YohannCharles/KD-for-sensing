from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from kd_sensing.utils.geometry import (
    beam_indices_to_angles,
    gps_to_local_xy,
    gps_to_rsu_aod,
    load_beam_angle_table,
    wrap_to_pi,
)


def test_wrap_to_pi_local_yaw_and_latlon_fallback(tmp_path: Path):
    wrapped = wrap_to_pi(np.asarray([3.0 * math.pi, -3.0 * math.pi, math.pi + 0.1]))
    assert np.all(wrapped < math.pi)
    assert np.all(wrapped >= -math.pi)
    assert math.isclose(float(wrapped[0]), -math.pi, abs_tol=1e-8)
    assert math.isclose(float(wrapped[1]), -math.pi, abs_tol=1e-8)
    assert math.isclose(float(wrapped[2]), -math.pi + 0.1, abs_tol=1e-8)

    local = gps_to_local_xy(user_x=3.0, user_y=4.0, rsu_x=1.0, rsu_y=1.0)
    assert local.source == "local_xy"
    assert math.isclose(local.x, 2.0)
    assert math.isclose(local.y, 3.0)

    aod = gps_to_rsu_aod(
        {"user_x": 0.0, "user_y": 1.0, "rsu_x": 0.0, "rsu_y": 0.0, "rsu_yaw": math.pi / 2.0},
        {"yaw_unit": "radians", "yaw_zero_axis": "x"},
    )
    assert math.isclose(aod.theta_gps, 0.0, abs_tol=1e-6)
    assert math.isclose(aod.distance_to_rsu, 1.0, abs_tol=1e-6)

    latlon = gps_to_rsu_aod(
        {"user_lat": 40.0001, "user_lon": -73.0, "rsu_lat": 40.0, "rsu_lon": -73.0},
        {"coordinate_frame": "lat_lon", "missing_rsu_yaw": "use_default", "default_rsu_yaw": 0.0},
    )
    assert latlon.coordinate_source == "equirectangular"
    assert latlon.distance_to_rsu > 10.0
    assert math.isclose(latlon.theta_gps, math.pi / 2.0, rel_tol=0.0, abs_tol=1e-3)

    table_path = tmp_path / "angles.npy"
    np.save(table_path, np.linspace(-1.0, 1.0, 8, dtype=np.float32))
    table = load_beam_angle_table({"beam_angle_table": str(table_path)}, num_beams=8)
    assert table.beam_angle_source == str(table_path)
    assert math.isclose(float(beam_indices_to_angles(7, table)), 1.0, abs_tol=1e-6)

    fallback = load_beam_angle_table({}, num_beams=8)
    assert fallback.beam_angle_source == "dft_ula_approximation"
    assert fallback.angles.shape == (8,)
