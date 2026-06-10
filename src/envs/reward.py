import numpy as np


class ReachTargetReward:
    def __init__(
        self,
        goal_threshold: float = 0.005,
        linearity_coeff: float = 3.0,
        deviation_coeff: float = 1.0,
        orientation_coeff: float = 0.3,
        smooth_coeff: float = 0.1,
        collision_coeff: float = 1.0,
        joint_limit_coeff: float = 0.5,
        success_reward: float = 100.0,
    ):
        self.goal_threshold = goal_threshold
        self.linearity_coeff = linearity_coeff
        self.deviation_coeff = deviation_coeff
        self.orientation_coeff = orientation_coeff
        self.smooth_coeff = smooth_coeff
        self.collision_coeff = collision_coeff
        self.joint_limit_coeff = joint_limit_coeff
        self.success_reward = success_reward
        self.min_linearity_error = np.inf

    def reset(self):
        self.min_linearity_error = np.inf

    def distance_reward(self, dist_to_goal: float) -> float:
        if dist_to_goal < self.goal_threshold:
            return self.success_reward
        elif dist_to_goal < 3 * self.goal_threshold:
            t = (dist_to_goal - self.goal_threshold) / (2 * self.goal_threshold)
            return self.success_reward * (1 - t) + (1.0 / (1.0 + dist_to_goal)) * t
        else:
            return 1.0 / (1.0 + dist_to_goal)

    def linearity_reward_and_penalty(
        self, ee_pos: np.ndarray, start_ee_pos: np.ndarray, goal: np.ndarray
    ) -> tuple:
        start_to_goal = goal - start_ee_pos
        start_to_goal_norm = np.linalg.norm(start_to_goal)

        if start_to_goal_norm < 1e-6:
            return 0.0, 0.0

        start_to_current = ee_pos - start_ee_pos
        projection_ratio = np.dot(start_to_current, start_to_goal) / (start_to_goal_norm ** 2)
        projection_ratio = np.clip(projection_ratio, 0.0, 1.0)
        projected_point = start_ee_pos + projection_ratio * start_to_goal
        linearity_error = np.linalg.norm(ee_pos - projected_point)

        lin_reward = self.linearity_coeff / (1.0 + linearity_error)

        deviation_penalty = 0.0
        if linearity_error < self.min_linearity_error:
            self.min_linearity_error = linearity_error
        else:
            deviation_penalty = self.deviation_coeff * (linearity_error - self.min_linearity_error)

        return lin_reward, deviation_penalty

    def orientation_penalty(self, ee_z_axis: np.ndarray) -> float:
        target_orient = np.array([0.0, 0.0, -1.0])
        orient_norm = np.linalg.norm(ee_z_axis)
        if orient_norm < 1e-8:
            return 0.0
        ee_orient_norm = ee_z_axis / orient_norm
        dot_product = np.dot(ee_orient_norm, target_orient)
        angle_error = np.arccos(np.clip(dot_product, -1.0, 1.0))
        return self.orientation_coeff * angle_error

    def smooth_penalty(self, action: np.ndarray, prev_action: np.ndarray) -> float:
        action_diff = action - prev_action
        return self.smooth_coeff * np.linalg.norm(action_diff)

    def collision_penalty(self, mj_model, mj_data) -> float:
        penalty = 0.0
        for i in range(mj_data.ncon):
            contact = mj_data.contact[i]
            body1 = mj_model.geom_bodyid[contact.geom1]
            body2 = mj_model.geom_bodyid[contact.geom2]
            if body1 != body2 and body1 != 0 and body2 != 0:
                if abs(body1 - body2) > 1:
                    penalty += self.collision_coeff
        return penalty

    def joint_limit_penalty(self, mj_model, joint_angles: np.ndarray) -> float:
        penalty = 0.0
        joint_ranges = mj_model.jnt_range[:7]
        for i in range(7):
            min_angle = joint_ranges[i][0]
            max_angle = joint_ranges[i][1]
            if joint_angles[i] < min_angle:
                penalty += self.joint_limit_coeff * (min_angle - joint_angles[i])
            elif joint_angles[i] > max_angle:
                penalty += self.joint_limit_coeff * (joint_angles[i] - max_angle)
        return penalty

    def compute(
        self,
        ee_pos: np.ndarray,
        ee_z_axis: np.ndarray,
        joint_angles: np.ndarray,
        action: np.ndarray,
        prev_action: np.ndarray,
        start_ee_pos: np.ndarray,
        goal: np.ndarray,
        mj_model,
        mj_data,
    ) -> tuple:
        dist_to_goal = np.linalg.norm(ee_pos - goal)
        dist_reward = self.distance_reward(dist_to_goal)
        lin_reward, dev_penalty = self.linearity_reward_and_penalty(ee_pos, start_ee_pos, goal)
        orient_penalty = self.orientation_penalty(ee_z_axis)
        smooth_pen = self.smooth_penalty(action, prev_action)
        collision_pen = self.collision_penalty(mj_model, mj_data)
        joint_pen = self.joint_limit_penalty(mj_model, joint_angles)

        total_reward = (
            dist_reward
            + lin_reward
            - collision_pen
            - smooth_pen
            - orient_penalty
            - joint_pen
            - dev_penalty
        )

        return total_reward, dist_to_goal
