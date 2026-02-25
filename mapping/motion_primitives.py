import time
from picarx import Picarx

CM_PER_SEC_FWD = 25.0
RAD_PER_SEC_LEFT = 0.4
RAD_PER_SEC_RIGHT = 0.42

FWD_SPEED = 45
TURN_SPEED_BASE = 28
TURN_SPEED = 22
STEER_TRIM_DEG = 4


class Motion:
    def __init__(self, px: Picarx):
        self.px = px

    def stop(self):
        self.px.stop().  

    def time_for_distance_cm(self, dist_cm: float) -> float:
        return dist_cm / CM_PER_SEC_FWD

    def time_for_left_angle(self, angle_rad: float) -> float:
        return angle_rad / RAD_PER_SEC_LEFT

    def time_for_right_angle(self, angle_rad: float) -> float:
        return angle_rad / RAD_PER_SEC_RIGHT

    def forward_for(self, t_sec: float, speed: int = FWD_SPEED) -> float:
        if STEER_TRIM_DEG:
            self.px.set_dir_servo_angle(STEER_TRIM_DEG)
        self.px.forward(speed)
        time.sleep(t_sec)
        self.px.stop()
        if STEER_TRIM_DEG:
            self.px.set_dir_servo_angle(0)
        return CM_PER_SEC_FWD * t_sec

    def reverse_for(self, t_sec: float, speed: int = FWD_SPEED) -> float:
        if STEER_TRIM_DEG:
            self.px.set_dir_servo_angle(STEER_TRIM_DEG)
        self.px.backward(speed)
        time.sleep(t_sec)
        self.px.stop()
        if STEER_TRIM_DEG:
            self.px.set_dir_servo_angle(0)
        return CM_PER_SEC_FWD * t_sec

    def tank_left_for(self, t_sec: float, speed: int = TURN_SPEED) -> float:
        drive_t = t_sec * (TURN_SPEED_BASE / max(1.0, float(speed)))
        #fixed:scale runtime when speed changes from calibration
        self.px.set_motor_speed(1, speed)
        self.px.set_motor_speed(2, speed)
        time.sleep(drive_t)
        self.px.stop()
        return RAD_PER_SEC_LEFT * t_sec

    def tank_right_for(self, t_sec: float, speed: int = TURN_SPEED) -> float:
        drive_t = t_sec * (TURN_SPEED_BASE / max(1.0, float(speed)))
        #debug:right turn is negative heading by convention
        self.px.set_motor_speed(1, -speed)
        self.px.set_motor_speed(2, -speed)
        time.sleep(drive_t)
        self.px.stop()
        return -RAD_PER_SEC_RIGHT * t_sec

    def tank_left_angle(self, angle_rad: float) -> float:
        return self.tank_left_for(self.time_for_left_angle(angle_rad))

    def tank_right_angle(self, angle_rad: float) -> float:
        return self.tank_right_for(self.time_for_right_angle(angle_rad))
