from __future__ import annotations

from dataclasses import dataclass

try:
    from djitellopy import Tello
except ImportError:  # pragma: no cover - runtime dependency
    Tello = None


@dataclass(slots=True)
class TelloState:
    connected: bool = False
    flying: bool = False


class TelloController:
    """Minimal wrapper around djitellopy.Tello for flight control."""

    def __init__(self) -> None:
        if Tello is None:
            raise RuntimeError("djitellopy is not installed in the active environment.")
        self._tello = Tello()
        self._state = TelloState()

    def connect(self) -> None:
        self._tello.connect()
        self._state.connected = True

    def enable_video(self) -> None:
        try:
            self._tello.streamoff()
        except Exception:
            pass
        self._tello.streamon()

    def takeoff(self) -> None:
        if not self._state.connected:
            raise RuntimeError("Tello not connected")
        self._tello.takeoff()
        self._state.flying = True

    def land(self) -> None:
        if not self._state.connected:
            return
        try:
            self._tello.land()
        finally:
            self._state.flying = False

    def send_velocity(
        self, left_right: int, forward_backward: int, up_down: int, yaw: int
    ) -> None:
        def clamp(v: int) -> int:
            return max(-100, min(100, int(v)))

        if not self._state.connected:
            return

        self._tello.send_rc_control(
            clamp(left_right), clamp(forward_backward), clamp(up_down), clamp(yaw)
        )

    def end(self) -> None:
        try:
            if self._state.flying:
                self.land()
        finally:
            try:
                self._tello.end()
            except Exception:
                pass
            self._state.connected = False
