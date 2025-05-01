import gym
import numpy as np
import math
import pybullet as p
from pybullet_utils import bullet_client as bc
from simple_driving.resources.car import Car
from simple_driving.resources.plane import Plane
from simple_driving.resources.goal import Goal
import matplotlib.pyplot as plt
import time
import os

RENDER_HEIGHT = 720
RENDER_WIDTH = 960

class SimpleDrivingEnv(gym.Env):
    metadata = {'render.modes': ['human', 'fp_camera', 'tp_camera']}

    def __init__(self, isDiscrete=True, renders=False):
        if (isDiscrete):
            self.action_space = gym.spaces.Discrete(9)
        else:
            self.action_space = gym.spaces.box.Box(
                low=np.array([-1, -.6], dtype=np.float32),
                high=np.array([1, .6], dtype=np.float32))
        self.observation_space = gym.spaces.box.Box(
            low=np.array([-40, -40, -40, -40], dtype=np.float32),
            high=np.array([40, 40, 40, 40], dtype=np.float32))
        self.np_random, _ = gym.utils.seeding.np_random()

        if renders:
          self._p = bc.BulletClient(connection_mode=p.GUI)
        else:
          self._p = bc.BulletClient()

        self.reached_goal = False
        self._timeStep = 0.01
        self._actionRepeat = 50
        self._renders = renders
        self._isDiscrete = isDiscrete
        self.car = None
        self.goal_object = None
        self.goal = None
        self.obstacles = []
        self.obstacle = None
        self.done = False
        self.prev_dist_to_goal = None
        self.rendered_img = None
        self.render_rot_matrix = None
        self.reset()
        self._envStepCounter = 0

    def step(self, action):
        # Feed action to the car and get observation of car's state
        if (self._isDiscrete):
            fwd = [-1, -1, -1, 0, 0, 0, 1, 1, 1]
            steerings = [-0.6, 0, 0.6, -0.6, 0, 0.6, -0.6, 0, 0.6]
            throttle = fwd[action]
            steering_angle = steerings[action]
            action = [throttle, steering_angle]
        self.car.apply_action(action)
        for i in range(self._actionRepeat):
          self._p.stepSimulation()
          if self._renders:
            time.sleep(self._timeStep)

          carpos, carorn = self._p.getBasePositionAndOrientation(self.car.car)
          goalpos, goalorn = self._p.getBasePositionAndOrientation(self.goal_object.goal)
          car_ob = self.getExtendedObservation()

          if self._termination():
            self.done = True
            break
          self._envStepCounter += 1

        # Compute reward as L2 change in distance to goal
        dist_to_goal = math.sqrt(((carpos[0] - goalpos[0]) ** 2 +
                                  (carpos[1] - goalpos[1]) ** 2))
        reward = -dist_to_goal
        self.prev_dist_to_goal = dist_to_goal

        # Add a small reward for moving forward toward the goal
        if hasattr(self, 'prev_pos'):
            prev_dist = math.sqrt(self.prev_pos[0]**2 + self.prev_pos[1]**2)
            curr_dist = math.sqrt(carpos[0]**2 + carpos[1]**2)
            if prev_dist > curr_dist:  # Moving closer to origin/goal
                reward += 0.5
        self.prev_pos = carpos

        # Done by reaching goal
        if dist_to_goal < 1.5 and not self.reached_goal:
            self.done = True
            self.reached_goal = True
            reward += 50.0  # Add bonus reward for reaching the goal

        # Add collision detection with obstacles
        for obstacle in self.obstacles:
            closest_points = self._p.getClosestPoints(self.car.car, obstacle, distance=0.5)
            if len(closest_points) > 0:  # Collision detected
                reward -= 30.0  # Strong penalty for hitting obstacle
                break

        ob = car_ob
        return ob, reward, self.done, False, dict()

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self):
        self._p.resetSimulation()
        self._p.setTimeStep(self._timeStep)
        self._p.setGravity(0, 0, -10)
        # Reload the plane and car
        Plane(self._p)
        self.car = Car(self._p)
        self._envStepCounter = 0
        self.obstacles = []

        # Set the goal to a random target
        x = (self.np_random.uniform(5, 9) if self.np_random.integers(2) else
             self.np_random.uniform(-9, -5))
        y = (self.np_random.uniform(5, 9) if self.np_random.integers(2) else
             self.np_random.uniform(-9, -5))
        self.goal = (x, y)
        self.done = False
        self.reached_goal = False

        # Visual element of the goal
        self.goal_object = Goal(self._p, self.goal)
        
        # Add multiple obstacles to the environment
        num_obstacles = self.np_random.integers(2, 4)  # 2-3 obstacles
        for i in range(num_obstacles):
            # Place obstacles away from goal and start position
            obstacle_x = self.np_random.uniform(-7, 7)
            obstacle_y = self.np_random.uniform(-7, 7)
            
            # Ensure obstacle isn't too close to goal or start
            while (abs(obstacle_x - self.goal[0]) < 3 and abs(obstacle_y - self.goal[1]) < 3) or \
                  (abs(obstacle_x) < 3 and abs(obstacle_y) < 3):
                obstacle_x = self.np_random.uniform(-7, 7)
                obstacle_y = self.np_random.uniform(-7, 7)
                
            obstacle = self._p.loadURDF(
                fileName=os.path.join(os.path.dirname(__file__), "../resources/simplegoal.urdf"),
                basePosition=[obstacle_x, obstacle_y, 0],
                useFixedBase=True)
            
            self.obstacles.append(obstacle)
        
        # Always have at least one obstacle
        if not self.obstacles:
            obstacle = self._p.loadURDF(
                fileName=os.path.join(os.path.dirname(__file__), "../resources/simplegoal.urdf"),
                basePosition=[self.np_random.uniform(-5, 5), self.np_random.uniform(-5, 5), 0],
                useFixedBase=True)
            self.obstacles.append(obstacle)
            
        # Set the closest obstacle for state representation
        self.obstacle = self.obstacles[0]  # Default to first obstacle
        self.prev_pos = self.car.get_observation()  # Store initial position

        # Get observation to return
        carpos = self.car.get_observation()
        self.prev_dist_to_goal = math.sqrt(((carpos[0] - self.goal[0]) ** 2 +
                                           (carpos[1] - self.goal[1]) ** 2))
        
        # Find closest obstacle for observation
        min_dist = float('inf')
        for obs in self.obstacles:
            obs_pos, _ = self._p.getBasePositionAndOrientation(obs)
            dist = math.sqrt(((carpos[0] - obs_pos[0]) ** 2 + 
                             (carpos[1] - obs_pos[1]) ** 2))
            if dist < min_dist:
                min_dist = dist
                self.obstacle = obs
        
        car_ob = self.getExtendedObservation()
        return np.array(car_ob, dtype=np.float32), {}

    def render(self, mode='human'):
        if mode == "fp_camera":
            # Base information
            car_id = self.car.get_ids()
            proj_matrix = self._p.computeProjectionMatrixFOV(fov=80, aspect=1,
                                                       nearVal=0.01, farVal=100)
            pos, ori = [list(l) for l in
                        self._p.getBasePositionAndOrientation(car_id)]
            pos[2] = 0.2

            # Rotate camera direction
            rot_mat = np.array(self._p.getMatrixFromQuaternion(ori)).reshape(3, 3)
            camera_vec = np.matmul(rot_mat, [1, 0, 0])
            up_vec = np.matmul(rot_mat, np.array([0, 0, 1]))
            view_matrix = self._p.computeViewMatrix(pos, pos + camera_vec, up_vec)

            # Display image
            (_, _, px, _, _) = self._p.getCameraImage(width=RENDER_WIDTH,
                                                      height=RENDER_HEIGHT,
                                                      viewMatrix=view_matrix,
                                                      projectionMatrix=proj_matrix,
                                                      renderer=p.ER_BULLET_HARDWARE_OPENGL)
            frame = np.array(px)
            frame = frame[:, :, :3]
            return frame

        elif mode == "tp_camera":
            car_id = self.car.get_ids()
            base_pos, orn = self._p.getBasePositionAndOrientation(car_id)
            view_matrix = self._p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=base_pos,
                                                                    distance=20.0,
                                                                    yaw=40.0,
                                                                    pitch=-35,
                                                                    roll=0,
                                                                    upAxisIndex=2)
            proj_matrix = self._p.computeProjectionMatrixFOV(fov=60,
                                                             aspect=float(RENDER_WIDTH) / RENDER_HEIGHT,
                                                             nearVal=0.1,
                                                             farVal=100.0)
            (_, _, px, _, _) = self._p.getCameraImage(width=RENDER_WIDTH,
                                                      height=RENDER_HEIGHT,
                                                      viewMatrix=view_matrix,
                                                      projectionMatrix=proj_matrix,
                                                      renderer=p.ER_BULLET_HARDWARE_OPENGL)
            frame = np.array(px)
            frame = frame[:, :, :3]
            return frame
        else:
            return np.array([])

    def getExtendedObservation(self):
        carpos, carorn = self._p.getBasePositionAndOrientation(self.car.car)
        goalpos, goalorn = self._p.getBasePositionAndOrientation(self.goal_object.goal)
        invCarPos, invCarOrn = self._p.invertTransform(carpos, carorn)
        goalPosInCar, goalOrnInCar = self._p.multiplyTransforms(invCarPos, invCarOrn, goalpos, goalorn)
        
        # Find closest obstacle for the observation
        min_dist = float('inf')
        closest_obstacle = self.obstacle
        
        for obs in self.obstacles:
            obs_pos, _ = self._p.getBasePositionAndOrientation(obs)
            dist = math.sqrt(((carpos[0] - obs_pos[0]) ** 2 + 
                             (carpos[1] - obs_pos[1]) ** 2))
            if dist < min_dist:
                min_dist = dist
                closest_obstacle = obs
        
        # Use the closest obstacle for state representation
        obstaclepos, obstacleorn = self._p.getBasePositionAndOrientation(closest_obstacle)
        obstaclePosInCar, obstacleOrnInCar = self._p.multiplyTransforms(invCarPos, invCarOrn, obstaclepos, obstacleorn)

        observation = [goalPosInCar[0], goalPosInCar[1], obstaclePosInCar[0], obstaclePosInCar[1]]
        return observation

    def _termination(self):
        return self._envStepCounter > 2000

    def close(self):
        self._p.disconnect()
