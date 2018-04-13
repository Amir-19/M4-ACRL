import math
import numpy as np

first_bin = 0
last_bin = 60
FA_scheme = 2
# from niko shared in slacks and servo.py . I just added "ang"
def parse_data(data):
    ang = (data[1] * 256 + data[0] - 0x200) * math.radians(300.0) / 1024.0
    position = (data[0] + data[1] * 256 - 512) * 5 * math.pi / 3069
    speed = (data[2] + data[3] * 256) * 5 * math.pi / 3069
    load = (data[5] & 3) * 256 + data[4] * (1 - 2 * bool(data[5] & 4))
    direction = 0
    if speed > 3:
        direction = -1
    else:
        direction = 1
    voltage = data[6] / 10
    temperature = data[7]

    return [ang, position, speed, load, voltage, temperature,direction]

def is_approx_equal(a,b,degree = 1e-2):
    is_app_eq = (abs(a - b) <= max(1e-4 * max(abs(a), abs(b)), degree))
    return is_app_eq

def read_data(servo):
    read_all = [0x02, 0x24, 0x08]
    data = servo.send_instruction(read_all, servo.servo_id)
    return parse_data(data)

def policy_robot(servo,ang,dir):
    if is_approx_equal(ang,1.5):
        servo.move_angle(-1.5, blocking=False)
        dir = -1
    elif is_approx_equal(ang,-1.5):
        servo.move_angle(1.5, blocking=False)
        dir = 1
    return dir

def feature_vector(state):
    fvector = np.zeros(last_bin+1)
    fvector[state] = 1.0
    return fvector

def Q_feature_vector(state, action):
    fav = np.zeros(len(state) * 2)
    ind = np.where(state == 1)[0]
    fav[action * len(state) + ind] = 1
    return fav

def get_angle_bin(ang,dir,bins):

    if FA_scheme == 1:
        if dir == 1:
            ang_f_bin = ang + 1.5
        elif dir == -1:
            ang_f_bin = 4.5 - ang
    elif FA_scheme == 2:
        ang_f_bin = ang + 1.5

    return np.digitize(ang_f_bin, bins)
def reward_disc(ang):
    goal = 0
    return -1 * np.abs(ang - goal)