"""
OpenCV visualization window for FactorySorting grasp camera.

This shows an offscreen camera from the same robosuite simulation while the
main MuJoCo viewer can stay on a global camera such as birdview.
"""

import platform

import numpy as np

from robosuite.environments.factory_sorting.load_factory_sorting_evalization import base_robosuite_env


DEFAULT_GRASP_VISUAL_CAMERA = "robot0_robotview"
DEFAULT_GRASP_VISUAL_HEIGHT = 360
DEFAULT_GRASP_VISUAL_WIDTH = 360
DEFAULT_GRASP_VISUAL_WINDOW_NAME = "FactorySorting grasp view"


class GraspVisualWindow:
    def __init__(
        self,
        camera=DEFAULT_GRASP_VISUAL_CAMERA,
        height=DEFAULT_GRASP_VISUAL_HEIGHT,
        width=DEFAULT_GRASP_VISUAL_WIDTH,
        window_name=DEFAULT_GRASP_VISUAL_WINDOW_NAME,
        enabled=True,
    ):
        self.camera = camera
        self.height = int(height)
        self.width = int(width)
        self.window_name = window_name
        self.enabled = bool(enabled)
        self._has_window = False
        self._warned = False
        self._cv2 = None

        if self.enabled:
            try:
                import cv2

                self._cv2 = cv2
            except ImportError:
                self.enabled = False
                print("grasp_visual_window_warning: OpenCV cv2 is not installed; disabling grasp window.")

    def render(self, env):
        if not self.enabled or self._cv2 is None:
            return

        raw_env = base_robosuite_env(env)
        try:
            frame = raw_env.sim.render(
                camera_name=self.camera,
                height=self.height,
                width=self.width,
            )
        except Exception as exc:
            if not self._warned:
                print(f"grasp_visual_window_warning: failed to render camera '{self.camera}': {exc}")
                self._warned = True
            return

        frame = np.flip(frame[..., ::-1], axis=0)
        self._cv2.imshow(self.window_name, frame)
        if platform.system() != "Darwin" and not self._has_window:
            self._cv2.moveWindow(self.window_name, 20, 20)
        self._cv2.waitKey(1)
        self._has_window = True

    def close(self):
        if self.enabled and self._cv2 is not None and self._has_window:
            self._cv2.destroyWindow(self.window_name)
            self._cv2.waitKey(1)
        self._has_window = False
