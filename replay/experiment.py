#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
跟驰实验CARLA仿真脚本 (重构版)
Car-Following Experiment Simulation

支持：
1. 驾驶舱硬件接入（UDP通信，完整协议）
2. 键盘控制（PC测试模式）
3. 跟驰场景数据采集
4. 多视角相机（独立脚本）
5. 多显示器支持
6. 录制回放功能

实验流程：
Phase 1: 被试手动驾驶，采集纵向控制数据
Phase 2: (离线) 训练个性化模型
Phase 3: 被试作为乘客体验自动驾驶

用法:
    # 键盘测试模式
    python car_following_experiment.py --keyboard
    
    # 驾驶舱模式
    python car_following_experiment.py --cabin
    
    # 指定显示器
    python car_following_experiment.py --keyboard --display 0
    
    # 自动驾驶模式（加载已训练模型）
    python car_following_experiment.py --autopilot --model ./models/self_style.pth

按键说明:
    W/↑         : 油门
    S/↓         : 制动
    A/D         : 转向
    Q           : 切换倒档
    Space       : 手刹
    P           : 切换自动驾驶
    M           : 切换手动/自动变速
    ,/.         : 升降档
    
    L           : 切换车灯
    Shift+L     : 远光灯
    Z/X         : 左/右转向灯
    
    F1          : 开始/停止数据采集
    F2          : 切换驾驶模式 (手动/自动)
    F3          : 切换前车行为 (恒速/变速, 仅跟驰实验)
    F4          : 切换到下一个实验并重启 (共4组)
    F5-F10      : 直接切换实验 1-4 并重启
    TAB         : 切换相机视角
    C           : 切换天气
    V           : 切换地图层
    B           : 加载/卸载地图层
    
    R           : 录制图像
    Ctrl+R      : 开始/停止录制仿真
    Ctrl+P      : 回放录制
    
    Backspace   : 重启场景
    ESC         : 退出
"""

from __future__ import print_function

import glob
import os
import sys
import time
import math
import random
import argparse
import struct
import socket
import csv
import json
import weakref
import re
import logging
from datetime import datetime
from collections import deque

# 添加CARLA路径
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
from carla import ColorConverter as cc

try:
    import pygame
    from pygame.locals import KMOD_CTRL, KMOD_SHIFT
    from pygame.locals import K_0, K_9, K_BACKQUOTE, K_BACKSPACE
    from pygame.locals import K_COMMA, K_DOWN, K_ESCAPE, K_F1, K_F2, K_F3
    from pygame.locals import K_F4, K_F5, K_F6, K_F7, K_F8, K_F9, K_F10
    from pygame.locals import K_LEFT, K_PERIOD, K_RIGHT, K_SLASH, K_SPACE
    from pygame.locals import K_TAB, K_UP, K_a, K_b, K_c, K_d, K_f, K_g
    from pygame.locals import K_h, K_i, K_l, K_m, K_n, K_o, K_p, K_q
    from pygame.locals import K_r, K_s, K_t, K_v, K_w, K_x, K_z
    from pygame.locals import K_MINUS, K_EQUALS
except ImportError:
    raise RuntimeError('无法导入pygame，请确保已安装: pip install pygame')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('无法导入numpy，请确保已安装: pip install numpy')


# ==============================================================================
# -- 全局常量 ------------------------------------------------------------------
# ==============================================================================

# 驾驶舱通信配置
CABIN_IP = '192.168.0.20'
CABIN_PORT = 3232

# 默认参数
DEFAULT_LEAD_SPEED = 20.0  # m/s
MIN_SPACING = 5.0  # 最小间距
SAFE_TIME_HEADWAY = 1.5  # 安全时距

# 实验组：跟驰仅保留 irregular + 3组超车速度阶段
FOLLOWING_EXPERIMENT_TYPES = (
    'following_irregular',
)
OVERTAKING_SPEEDS_KMH = (35.0, 50.0, 65.0)
EXPERIMENT_START_X = 120.0
EXPERIMENT_START_Y = -1.75


def kmh_to_ms(kmh):
    return kmh / 3.6


# ==============================================================================
# -- 全局函数 ------------------------------------------------------------------
# ==============================================================================

def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


def get_actor_blueprints(world, filter, generation):
    bps = world.get_blueprint_library().filter(filter)
    if generation.lower() == "all":
        return bps
    if len(bps) == 1:
        return bps
    try:
        int_generation = int(generation)
        if int_generation in [1, 2, 3]:
            bps = [x for x in bps if int(x.get_attribute('generation')) == int_generation]
            return bps
        else:
            print("警告: Actor Generation 无效")
            return []
    except (ValueError, AttributeError):
        print("警告: Actor Generation 无效")
        return []


# ==============================================================================
# -- 前车行为控制器 ------------------------------------------------------------
# ==============================================================================

class LeadVehicleController:
    """前车行为控制器 - 支持固定曲线和随机化"""

    # 速度变化约束
    MIN_SPEED = 12.0       # 最小速度 m/s (43 km/h)
    MAX_SPEED = 28.0       # 最大速度 m/s (100 km/h)
    MAX_ACCELERATION = 2.0 # 最大加速度 m/s²
    MAX_DECELERATION = 3.0 # 最大减速度 m/s²

    SMOOTH_FREQ_HZ = 1.0 / 70.0
    AGGRESSIVE_FREQ_HZ = 1.0 / 28.0
    IRREGULAR_F0_HZ = 1.0 / 90.0
    IRREGULAR_K_HZ_PER_S = 1.0 / 2400.0
    OVERTAKING_RAMP_TIME_S = 16.0
    FOLLOWING_BASE_SPEED_MS = 65.0 / 3.6   # 跟驰曲线中心速度 (km/h 换算为 m/s)
    FOLLOWING_AMPLITUDE_MS = 15.0 / 3.6    # 跟驰速度正弦幅值 (km/h)
    FOLLOWING_STARTUP_RAMP_S = 18.0
    
    def __init__(self, lead_vehicle, base_speed=DEFAULT_LEAD_SPEED,
                 random_mode=False, random_seed=None, follow_road=True,
                 experiment_type='following_irregular'):
        self.vehicle = lead_vehicle
        self.base_speed = base_speed
        self.current_target_speed = base_speed
        self.start_time = None
        self.behavior_mode = 'fixed'  # 'constant', 'fixed', 'random' (默认使用固定变速曲线)
        self.follow_road = follow_road  # True: 沿道路行驶; False: 直线行驶
        self.experiment_type = experiment_type
        
        # 记录初始方向（用于直线行驶模式）
        self.initial_yaw = None
        if lead_vehicle:
            self.initial_yaw = lead_vehicle.get_transform().rotation.yaw
        
        # 随机化配置
        self.random_mode = random_mode
        self.random_seed = random_seed if random_seed else int(time.time())
        self.rng = random.Random(self.random_seed)
        
        # 速度变化历史（用于记录）
        self.speed_history = []
        self.last_speed_change_time = 0
        
        # 随机化速度曲线（运行时生成）
        self.random_speed_profile = []
        
    def start(self):
        """启动前车控制，默认使用固定速度曲线"""
        self.start_time = None  # 延迟到第一次 update 时初始化
        self.sim_time_offset = None  # 仿真时间偏移
        self.speed_history = []
        self.last_speed_change_time = 0
        
        # 超车实验默认恒速；跟驰实验默认固定曲线
        if self.experiment_type == 'overtaking':
            self.behavior_mode = 'constant'
        elif self.random_mode:
            self.behavior_mode = 'random'
            self._generate_random_profile()
        else:
            self.behavior_mode = 'fixed'  # 默认使用固定变速曲线

        print(f"[前车控制] 实验类型: {self.experiment_type}, 模式: {self.behavior_mode}, 基准速度: {self.base_speed * 3.6:.0f} km/h")
            
    def _generate_random_profile(self, duration=600):
        """生成随机速度曲线
        
        Args:
            duration: 总时长（秒）
        """
        self.random_speed_profile = []
        current_time = 0
        current_speed = self.base_speed
        
        print(f"[前车控制] 生成随机速度曲线 (seed={self.random_seed})")
        
        while current_time < duration:
            # 记录当前点
            self.random_speed_profile.append((current_time, current_speed))
            
            # 随机决定下一次变化的时间间隔 (15-45秒)
            interval = self.rng.uniform(15, 45)
            current_time += interval
            
            # 随机决定速度变化
            # 70% 概率变化，30% 概率保持
            if self.rng.random() < 0.7:
                # 计算允许的速度变化范围（基于最大加减速度和时间间隔）
                max_speed_change = min(
                    self.MAX_ACCELERATION * interval,
                    self.MAX_DECELERATION * interval
                )
                
                # 随机选择加速或减速
                speed_change = self.rng.uniform(-max_speed_change, max_speed_change)
                new_speed = current_speed + speed_change
                
                # 限制在合理范围内
                new_speed = max(self.MIN_SPEED, min(self.MAX_SPEED, new_speed))
                
                # 避免过小的变化
                if abs(new_speed - current_speed) < 1.0:
                    new_speed = current_speed
                    
                current_speed = new_speed
            
            # 有30%概率回归基准速度
            if self.rng.random() < 0.3:
                current_speed = self.base_speed
                
        # 最后回到基准速度
        self.random_speed_profile.append((current_time, self.base_speed))
        
        # 打印生成的曲线
        print(f"[前车控制] 生成了 {len(self.random_speed_profile)} 个速度点:")
        for t, s in self.random_speed_profile[:10]:
            print(f"  {t:.0f}s: {s:.1f} m/s ({s*3.6:.0f} km/h)")
        if len(self.random_speed_profile) > 10:
            print(f"  ... 共 {len(self.random_speed_profile)} 个点")
            
    def update(self, sim_time=None):
        """更新前车速度
        
        Args:
            sim_time: CARLA 仿真时间（秒），如果为 None 则使用系统时间
        """
        if self.vehicle is None:
            return
        
        # 使用仿真时间计算 elapsed
        if sim_time is not None:
            if self.sim_time_offset is None:
                self.sim_time_offset = sim_time
            elapsed = sim_time - self.sim_time_offset
        else:
            # 回退到系统时间
            if self.start_time is None:
                self.start_time = time.time()
            elapsed = time.time() - self.start_time
            
        if self.behavior_mode == 'constant':
            if self.experiment_type == 'overtaking':
                target_speed = self._get_overtaking_target_speed(elapsed)
            else:
                target_speed = self.base_speed
        else:
            # 选择速度曲线
            if self.behavior_mode == 'random' and self.random_speed_profile:
                profile = self.random_speed_profile
                profile_elapsed = min(max(elapsed, 0.0), profile[-1][0])
                target_speed = profile[0][1]
                prev_time, prev_speed = profile[0]

                for t, speed in profile:
                    if profile_elapsed >= t:
                        prev_time, prev_speed = t, speed
                        target_speed = speed
                    else:
                        # 线性插值实现平滑过渡
                        if t > prev_time:
                            alpha = (profile_elapsed - prev_time) / (t - prev_time)
                            alpha = min(1.0, max(0.0, alpha))
                            target_speed = prev_speed + alpha * (speed - prev_speed)
                        break
            else:
                target_speed = self._get_following_target_speed(elapsed)
                    
        # 记录速度变化
        if abs(target_speed - self.current_target_speed) > 0.1:
            self.speed_history.append({
                'time': elapsed,
                'target_speed': target_speed,
                'mode': self.behavior_mode
            })
                    
        self.current_target_speed = target_speed
        
        # 使用基于 waypoint 的控制（让前车能够沿道路行驶、拐弯）
        self._apply_waypoint_control(target_speed)
    
    def _apply_waypoint_control(self, target_speed):
        """车辆控制：可选沿道路行驶或直线行驶"""
        try:
            transform = self.vehicle.get_transform()
            velocity = self.vehicle.get_velocity()
            current_speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
            current_yaw = transform.rotation.yaw
            
            if self.follow_road:
                # === 沿道路行驶模式 ===
                world = self.vehicle.get_world()
                carla_map = world.get_map()
                current_wp = carla_map.get_waypoint(transform.location)
                
                if current_wp is None:
                    return
                
                next_wps = current_wp.next(max(5.0, current_speed * 0.5))
                if not next_wps:
                    return
                target_wp = next_wps[0]
                
                target_loc = target_wp.transform.location
                current_loc = transform.location
                
                target_vector = carla.Vector3D(
                    target_loc.x - current_loc.x,
                    target_loc.y - current_loc.y,
                    0
                )
                target_yaw = math.degrees(math.atan2(target_vector.y, target_vector.x))
            else:
                # === 直线行驶模式（忽略道路弯曲）===
                if self.initial_yaw is None:
                    self.initial_yaw = current_yaw
                target_yaw = self.initial_yaw
            
            # 计算角度差
            yaw_diff = target_yaw - current_yaw
            while yaw_diff > 180:
                yaw_diff -= 360
            while yaw_diff < -180:
                yaw_diff += 360
            
            # 转向控制
            steer = max(-1.0, min(1.0, yaw_diff / 30.0))
            
            # 速度控制 (PI 控制器)
            speed_error = target_speed - current_speed
            
            # 积分项（用于消除稳态误差）
            if not hasattr(self, '_speed_integral'):
                self._speed_integral = 0.0
            self._speed_integral += speed_error * 0.05  # dt ≈ 0.05s
            self._speed_integral = max(-10.0, min(10.0, self._speed_integral))  # 限幅
            
            # PI 控制
            Kp = 0.5   # 比例增益
            Ki = 0.08  # 积分增益
            control_signal = Kp * speed_error + Ki * self._speed_integral
            
            # 速度显著低于目标（>1 m/s）时直接全油门，保证能到达 65km/h 等较高目标速度
            if speed_error > 1.0:
                throttle = 1.0
                brake = 0.0
            elif control_signal > 0.1:
                # 需要加速
                throttle = min(1.0, 0.5 + control_signal * 0.3)
                brake = 0.0
            elif speed_error < -1.0:
                # 显著超过目标：主动刹车
                throttle = 0.0
                brake = min(1.0, (-speed_error) * 0.3)
            elif control_signal < -0.1:
                # 需要减速
                throttle = 0.0
                brake = min(1.0, -control_signal * 0.2)
            else:
                # 保持速度 - 根据目标速度计算维持油门
                # 65 km/h ≈ 18 m/s 需要约 0.55 油门；提高上限避免在高目标速度下被裁掉
                maintain_throttle = 0.4 + (target_speed / 25.0) * 0.3
                throttle = min(1.0, maintain_throttle)
                brake = 0.0
            
            control = carla.VehicleControl(
                throttle=throttle,
                steer=steer,
                brake=brake,
                hand_brake=False,
                reverse=False,
                manual_gear_shift=False
            )
            self.vehicle.apply_control(control)
            
        except Exception:
            # 回退：直接设置速度
            forward = self.vehicle.get_transform().get_forward_vector()
            velocity = carla.Vector3D(
                forward.x * target_speed,
                forward.y * target_speed,
                0
            )
            try:
                self.vehicle.set_target_velocity(velocity)
            except Exception:
                pass
            
    def toggle_mode(self):
        """切换行为模式: constant -> fixed -> random -> constant"""
        if self.experiment_type == 'overtaking':
            self.behavior_mode = 'constant'
            return 'constant (超车实验前车恒速)'

        if self.behavior_mode == 'constant':
            self.behavior_mode = 'fixed'
            self.start_time = time.time()
            return 'fixed (固定曲线)'
        elif self.behavior_mode == 'fixed':
            self.behavior_mode = 'random'
            self.start_time = time.time()
            self._generate_random_profile()
            return f'random (随机, seed={self.random_seed})'
        else:
            self.behavior_mode = 'constant'
            return 'constant (恒速)'

    def _get_following_target_speed(self, elapsed):
        t = max(0.0, elapsed)
        base = self.FOLLOWING_BASE_SPEED_MS
        amp = self.FOLLOWING_AMPLITUDE_MS

        # 跟驰实验开始阶段：前车从 0 平滑加速到基准速度
        if t < self.FOLLOWING_STARTUP_RAMP_S:
            x = t / self.FOLLOWING_STARTUP_RAMP_S
            s = x * x * (3.0 - 2.0 * x)  # smoothstep
            return base * s

        t_wave = t - self.FOLLOWING_STARTUP_RAMP_S

        if self.experiment_type == 'following_aggressive':
            # 高频正弦：相对缓变更快的速度起伏
            wave = math.sin(2.0 * math.pi * self.AGGRESSIVE_FREQ_HZ * t_wave)
            target = base + amp * wave
        elif self.experiment_type == 'following_irregular':
            # 频率可变正弦：在慢频与快频之间切换，体现“有快有慢”
            slow_wave = math.sin(2.0 * math.pi * (1.0 / 95.0) * t_wave)
            fast_wave = math.sin(2.0 * math.pi * (1.0 / 26.0) * t_wave + 0.8)
            # blend 在 [0,1] 内慢速摆动；高值时更快、低值时更慢
            blend = 0.5 * (1.0 + math.sin(2.0 * math.pi * (1.0 / 120.0) * t_wave - math.pi / 2.0))
            wave = (1.0 - blend) * slow_wave + blend * fast_wave
            target = base + amp * wave
        else:
            # 低频正弦：平缓加减速
            wave = math.sin(2.0 * math.pi * self.SMOOTH_FREQ_HZ * t_wave)
            target = base + amp * wave

        return max(0.0, min(self.MAX_SPEED, target))

    def _get_overtaking_target_speed(self, elapsed):
        """超车实验前车速度：从 0 平滑过渡到目标速度，避免拐点。"""
        t = max(0.0, elapsed)
        if t >= self.OVERTAKING_RAMP_TIME_S:
            return self.base_speed

        x = t / self.OVERTAKING_RAMP_TIME_S
        # smoothstep: 3x^2 - 2x^3，端点速度与加速度连续
        s = x * x * (3.0 - 2.0 * x)
        return self.base_speed * s
            
    def get_speed_history(self):
        """获取速度变化历史"""
        return self.speed_history
        
    def get_random_seed(self):
        """获取随机种子（用于复现）"""
        return self.random_seed


# ==============================================================================
# -- 数据采集器 ----------------------------------------------------------------
# ==============================================================================

class DataCollector:
    """跟驰数据采集器"""
    
    def __init__(self):
        self.is_collecting = False
        self.data_buffer = []
        self.save_path = None
        self.file_handle = None
        self.csv_writer = None
        self.start_time = None
        self.frame_count = 0
        self.prev_acceleration = 0.0
        self.prev_time = None
        self.buffer_size = 100  # 缓冲区大小
        self.meta_path = None
        self.metadata = None
        self.world_start_time_s = None
        self.world_end_time_s = None
        
        self.columns = [
            'timestamp', 'real_world_time', 'real_world_unix', 'frame',
            'ego_speed', 'ego_acceleration', 'ego_jerk',
            'ego_pos_x', 'ego_pos_y', 'ego_yaw',
            'lead_pos_x', 'lead_pos_y', 'lead_yaw',
            'lead_speed', 'lead_acceleration', 'lead_target_speed',
            'distance_headway', 'time_headway', 'relative_speed', 'ttc',
            'throttle', 'brake', 'steer', 'longitudinal_control',
            'control_mode', 'gear', 'lead_behavior_mode'
        ]
        
    def _write_metadata(self):
        if not self.meta_path or self.metadata is None:
            return
        with open(self.meta_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)

    def start(self, save_path, lead_controller=None, world_start_time_s=None):
        self.save_path = save_path
        self.data_buffer = []
        self.is_collecting = True
        self.start_time = time.time()
        self.frame_count = 0
        self.prev_acceleration = 0.0
        self.prev_time = None
        self.world_start_time_s = float(world_start_time_s) if world_start_time_s is not None else None
        self.world_end_time_s = None
        
        dir_name = os.path.dirname(save_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        
        self.file_handle = open(save_path, 'w', newline='', encoding='utf-8')
        self.csv_writer = csv.DictWriter(self.file_handle, fieldnames=self.columns)
        self.csv_writer.writeheader()
        
        # 保存实验元数据
        self.meta_path = save_path.replace('.csv', '_metadata.json')
        metadata = {
            'start_time': datetime.now().isoformat(),
            'data_file': save_path,
            'columns': self.columns,
            'world_start_time_s': self.world_start_time_s,
        }
        if lead_controller:
            metadata['lead_vehicle'] = {
                'behavior_mode': lead_controller.behavior_mode,
                'base_speed': lead_controller.base_speed,
                'random_seed': lead_controller.random_seed,
                'random_mode': lead_controller.random_mode,
            }
            if lead_controller.random_speed_profile:
                metadata['lead_vehicle']['random_speed_profile'] = [
                    {'time': t, 'speed': s} for t, s in lead_controller.random_speed_profile
                ]
        
        self.metadata = metadata
        self._write_metadata()
        print(f"[数据采集] 元数据 -> {self.meta_path}")
        
        print(f"[数据采集] 开始 -> {save_path}")
        return True
        
    def stop(self, world_end_time_s=None):
        self.is_collecting = False
        
        # 写入剩余缓冲数据
        if self.data_buffer and self.csv_writer:
            for row in self.data_buffer:
                self.csv_writer.writerow(row)
            self.data_buffer.clear()
        
        if self.file_handle:
            self.file_handle.close()
            self.file_handle = None
        
        self.world_end_time_s = float(world_end_time_s) if world_end_time_s is not None else None
        if self.metadata is not None:
            self.metadata['end_time'] = datetime.now().isoformat()
            self.metadata['world_end_time_s'] = self.world_end_time_s
            if self.world_start_time_s is not None and self.world_end_time_s is not None:
                self.metadata['world_duration_s'] = self.world_end_time_s - self.world_start_time_s
            self._write_metadata()
            
        duration = time.time() - self.start_time if self.start_time else 0
        print(f"[数据采集] 停止")
        print(f"  采集时长: {duration:.1f}秒")
        if self.world_start_time_s is not None:
            print(f"  实验开始世界时间: {self.world_start_time_s:.3f}s")
        if self.world_end_time_s is not None:
            print(f"  实验结束世界时间: {self.world_end_time_s:.3f}s")
        if self.world_start_time_s is not None and self.world_end_time_s is not None:
            print(f"  世界时间时长: {self.world_end_time_s - self.world_start_time_s:.3f}s")
        print(f"  数据帧数: {self.frame_count}")
        print(f"  保存路径: {self.save_path}")
        return self.save_path
        
    def collect(self, ego_vehicle, lead_vehicle, control, control_mode='manual', 
                lead_controller=None):
        if not self.is_collecting:
            return None
            
        try:
            wall_time_unix = time.time()
            current_time = wall_time_unix - self.start_time
            real_world_time = datetime.now().astimezone().isoformat(timespec='milliseconds')
            
            # 自车状态
            ego_transform = ego_vehicle.get_transform()
            ego_velocity = ego_vehicle.get_velocity()
            ego_acc = ego_vehicle.get_acceleration()
            
            ego_speed = math.sqrt(ego_velocity.x**2 + ego_velocity.y**2 + ego_velocity.z**2)
            ego_acceleration = math.sqrt(ego_acc.x**2 + ego_acc.y**2 + ego_acc.z**2)
            
            if ego_acc.x * ego_velocity.x + ego_acc.y * ego_velocity.y < 0:
                ego_acceleration = -ego_acceleration
                
            # 计算jerk
            ego_jerk = 0.0
            if self.prev_time is not None:
                dt = current_time - self.prev_time
                if dt > 0.001:
                    ego_jerk = (ego_acceleration - self.prev_acceleration) / dt
            self.prev_acceleration = ego_acceleration
            self.prev_time = current_time
            
            # 前车状态
            lead_transform = lead_vehicle.get_transform()
            lead_velocity = lead_vehicle.get_velocity()
            lead_acc = lead_vehicle.get_acceleration()
            
            lead_speed = math.sqrt(lead_velocity.x**2 + lead_velocity.y**2 + lead_velocity.z**2)
            lead_acceleration = math.sqrt(lead_acc.x**2 + lead_acc.y**2 + lead_acc.z**2)
            if lead_acc.x * lead_velocity.x + lead_acc.y * lead_velocity.y < 0:
                lead_acceleration = -lead_acceleration
                
            # 跟驰特征 - 使用实际车辆尺寸
            ego_length = ego_vehicle.bounding_box.extent.x * 2
            lead_length = lead_vehicle.bounding_box.extent.x * 2
            
            distance = math.sqrt(
                (ego_transform.location.x - lead_transform.location.x)**2 +
                (ego_transform.location.y - lead_transform.location.y)**2
            )
            distance_headway = max(0, distance - (ego_length + lead_length) / 2)
            
            relative_speed = ego_speed - lead_speed
            time_headway = distance_headway / ego_speed if ego_speed > 0.5 else 999.0
            
            if relative_speed > 0.01:
                ttc = distance_headway / relative_speed
            else:
                ttc = 999.0
            ttc = min(ttc, 999.0)
            
            longitudinal_control = control.throttle - control.brake
            
            # 获取前车控制信息
            lead_target_speed = lead_controller.current_target_speed if lead_controller else lead_speed
            lead_behavior_mode = lead_controller.behavior_mode if lead_controller else 'unknown'
            
            data = {
                'timestamp': round(current_time, 4),
                'real_world_time': real_world_time,
                'real_world_unix': round(wall_time_unix, 3),
                'frame': self.frame_count,
                'ego_speed': round(ego_speed, 4),
                'ego_acceleration': round(ego_acceleration, 4),
                'ego_jerk': round(ego_jerk, 4),
                'ego_pos_x': round(ego_transform.location.x, 2),
                'ego_pos_y': round(ego_transform.location.y, 2),
                'ego_yaw': round(ego_transform.rotation.yaw, 2),
                'lead_pos_x': round(lead_transform.location.x, 2),
                'lead_pos_y': round(lead_transform.location.y, 2),
                'lead_yaw': round(lead_transform.rotation.yaw, 2),
                'lead_speed': round(lead_speed, 4),
                'lead_acceleration': round(lead_acceleration, 4),
                'lead_target_speed': round(lead_target_speed, 4),
                'distance_headway': round(distance_headway, 2),
                'time_headway': round(min(time_headway, 999.0), 2),
                'relative_speed': round(relative_speed, 4),
                'ttc': round(ttc, 2),
                'throttle': round(control.throttle, 4),
                'brake': round(control.brake, 4),
                'steer': round(control.steer, 4),
                'longitudinal_control': round(longitudinal_control, 4),
                'control_mode': control_mode,
                'gear': control.gear,
                'lead_behavior_mode': lead_behavior_mode
            }
            
            # 缓冲写入
            self.data_buffer.append(data)
            if len(self.data_buffer) >= self.buffer_size:
                for row in self.data_buffer:
                    self.csv_writer.writerow(row)
                self.data_buffer.clear()
                self.file_handle.flush()
                
            self.frame_count += 1
            return data
            
        except Exception as e:
            print(f"[数据采集] 错误: {e}")
            return None


# ==============================================================================
# -- 世界管理 ------------------------------------------------------------------
# ==============================================================================

class World:
    """世界管理类"""
    
    def __init__(self, carla_world, hud, args):
        self.world = carla_world
        self.sync = args.sync
        self.actor_role_name = args.rolename
        self.hud = hud
        self.args = args
        
        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print(f'RuntimeError: {error}')
            print('服务器无法发送OpenDRIVE文件，请检查地图配置')
            sys.exit(1)
            
        # 车辆
        self.player = None
        self.lead_vehicle = None
        self.lead_controller = None
        
        # 传感器
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.imu_sensor = None
        self.radar_sensor = None
        self.camera_manager = None
        
        # 数据采集
        self.data_collector = DataCollector()

        # 实验管理：可按 scope 拆分为仅跟驰 / 仅超车 / 全部
        self.experiment_scope = getattr(args, 'experiment_scope', 'all')
        self.experiment_plan = []
        if self.experiment_scope in ('all', 'following'):
            for exp_type in FOLLOWING_EXPERIMENT_TYPES:
                self.experiment_plan.append({
                    'type': exp_type,
                })
        if self.experiment_scope in ('all', 'overtaking'):
            for speed_kmh in OVERTAKING_SPEEDS_KMH:
                self.experiment_plan.append({
                    'type': 'overtaking',
                    'speed_kmh': speed_kmh,
                })
        if not self.experiment_plan:
            # 兜底：至少保留一个跟驰实验，避免空计划导致索引错误
            self.experiment_plan = [{'type': 'following_irregular'}]
        self.experiment_index = 0
        # 实验时长：跟驰默认 180s，超车默认 120s
        self.following_experiment_duration_s = float(getattr(args, 'following_experiment_duration_s', 180.0))
        self.overtaking_experiment_duration_s = float(getattr(args, 'overtaking_experiment_duration_s', 120.0))
        self.experiment_start_sim_time = None

        # 每个实验开始前的冷却时间：用于先打开窗口/录制，再开始让前车起步
        # 冷却期内跳过 lead_controller.update，并强制前车速度为 0
        self.experiment_cooldown_s = float(getattr(args, 'experiment_cooldown_s', 10.0))
        self._experiment_cooldown_active = False
        self._experiment_cooldown_start_sim_time = None
        self.cooldown_remaining_s = 0.0
        self.experiment_remaining_s = 0.0
        self.experiment_countdown_active = False

        # 自动采集：冷却结束后开始；切换实验/重启/退出时停止
        self._experiment_record_started = False
        # 超车实验提示流程：开始提示“先进行跟驰/随后变道超车”
        self._overtaking_prompt_pending = False
        
        # 配置
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = args.filter
        self._actor_generation = args.generation
        self._gamma = args.gamma
        
        # 状态
        self.recording_enabled = False
        self.recording_start = 0
        self.constant_velocity_enabled = False
        self.show_vehicle_telemetry = False
        self.doors_are_open = False
        
        # 地图层管理
        self.current_map_layer = 0
        self.map_layer_names = [
            carla.MapLayer.NONE,
            carla.MapLayer.Buildings,
            carla.MapLayer.Decals,
            carla.MapLayer.Foliage,
            carla.MapLayer.Ground,
            carla.MapLayer.ParkedVehicles,
            carla.MapLayer.Particles,
            carla.MapLayer.Props,
            carla.MapLayer.StreetLights,
            carla.MapLayer.Walls,
            carla.MapLayer.All
        ]
        
        self.restart()
        self.world.on_tick(hud.on_world_tick)

    def _use_experiment_mode(self):
        return getattr(self.args, 'enable_experiment_mode', False)

    def _get_current_experiment(self):
        if not self._use_experiment_mode():
            return None
        return self.experiment_plan[self.experiment_index]

    def _get_experiment_label(self):
        exp = self._get_current_experiment()
        if not exp:
            return "默认实验"
        total = len(self.experiment_plan)
        following_idx = sum(
            1 for i in range(self.experiment_index + 1)
            if self.experiment_plan[i]['type'].startswith('following_')
        )
        overtaking_idx = sum(
            1 for i in range(self.experiment_index + 1)
            if self.experiment_plan[i]['type'] == 'overtaking'
        )
        if exp['type'] == 'following_irregular':
            return f"实验{self.experiment_index + 1}/{total} 跟驰-不规则变速"
        return f"实验{self.experiment_index + 1}/{total} 超车{int(overtaking_idx)} {exp['speed_kmh']:.0f}km/h"

    def _get_experiment_sidebar_title(self):
        """侧边栏实验组标题：按当前 experiment_scope 动态编号。"""
        exp = self._get_current_experiment()
        if not exp:
            return ""
        if exp['type'].startswith('following_'):
            idx = sum(
                1 for i in range(self.experiment_index + 1)
                if self.experiment_plan[i]['type'].startswith('following_')
            )
            return f"跟驰实验{idx}"
        idx = sum(
            1 for i in range(self.experiment_index + 1)
            if self.experiment_plan[i]['type'] == 'overtaking'
        )
        return f"超车实验{idx}"

    def _get_effective_lead_speed(self):
        exp = self._get_current_experiment()
        if exp:
            if exp['type'].startswith('following_'):
                return self.args.lead_speed
            return kmh_to_ms(exp['speed_kmh'])
        return self.args.lead_speed

    def _get_effective_experiment_type(self):
        exp = self._get_current_experiment()
        if exp:
            return exp['type']
        return 'following_irregular'

    def _get_effective_experiment_duration_s(self):
        exp = self._get_current_experiment()
        if exp and exp['type'].startswith('following_'):
            return self.following_experiment_duration_s
        return self.overtaking_experiment_duration_s

    def _get_effective_lead_distance(self):
        base_dist = getattr(self.args, 'lead_distance', 75.0)
        if self._get_effective_experiment_type() == 'overtaking':
            return max(base_dist, 75.0)
        return base_dist

    def _get_experiment_start_spawn(self):
        if not self._use_experiment_mode():
            return None
        if not (getattr(self.args, 'straight_road', False) or getattr(self.args, 'opendrive', None)):
            return None
        target_x = getattr(self.args, 'experiment_start_x', EXPERIMENT_START_X)
        target_y = getattr(self.args, 'experiment_start_y', EXPERIMENT_START_Y)
        try:
            wp = self.map.get_waypoint(
                carla.Location(x=target_x, y=target_y, z=2.0),
                project_to_road=True,
                lane_type=carla.LaneType.Driving)
        except Exception:
            wp = None
        if wp is None:
            return None
        t = wp.transform
        t.location.z += 0.3
        return t

    def _apply_spawn_right_offset(self, transform):
        """将生成点向右侧车道移动（优先按车道中心），必要时再做几何偏移。"""
        if transform is None:
            return None

        # 拷贝一份，避免原始 spawn_points 被原地修改
        shifted = carla.Transform(
            carla.Location(
                x=transform.location.x,
                y=transform.location.y,
                z=transform.location.z,
            ),
            carla.Rotation(
                pitch=transform.rotation.pitch,
                yaw=transform.rotation.yaw,
                roll=transform.rotation.roll,
            ),
        )

        right_offset_m = float(getattr(self.args, 'spawn_right_offset', 0.0) or 0.0)
        if abs(right_offset_m) < 1e-6:
            return shifted

        # 优先使用 waypoint 邻接车道移动，确保落在可驾驶车道中心
        try:
            base_wp = self.map.get_waypoint(
                shifted.location,
                project_to_road=True,
                lane_type=carla.LaneType.Driving
            )
        except Exception:
            base_wp = None

        if base_wp is not None:
            move_right = right_offset_m > 0
            remain = abs(right_offset_m)
            current_wp = base_wp
            moved = False

            while remain > 1e-6:
                next_wp = current_wp.get_right_lane() if move_right else current_wp.get_left_lane()
                if next_wp is None or next_wp.lane_type != carla.LaneType.Driving:
                    break
                current_wp = next_wp
                remain = max(0.0, remain - max(float(getattr(current_wp, "lane_width", 0.0) or 0.0), 0.1))
                moved = True

            if moved:
                shifted = current_wp.transform
                shifted.location.z += 0.3
                return shifted

        # 如果没有邻接可驾驶车道，再退回几何偏移（兼容非标准地图）
        right_vec = shifted.get_right_vector()
        shifted.location.x += right_vec.x * right_offset_m
        shifted.location.y += right_vec.y * right_offset_m
        return shifted

    def _cleanup_role_vehicles(self):
        """
        清理场景中残留的 hero / lead_vehicle，避免重启后叠车导致抖动或生成失败。
        """
        role_names = {self.actor_role_name, 'lead_vehicle'}
        try:
            for actor in self.world.get_actors().filter('vehicle.*'):
                try:
                    role_name = actor.attributes.get('role_name', '')
                except Exception:
                    role_name = ''
                if role_name in role_names:
                    actor.destroy()
        except Exception:
            pass

    def switch_to_experiment(self, index):
        if not self._use_experiment_mode():
            self.hud.notification("未启用实验模式", seconds=2.0)
            return
        # 切换实验前强制停止数据采集，保证每个实验独立文件
        if self.data_collector and self.data_collector.is_collecting:
            sim_time = self.hud.simulation_time if hasattr(self.hud, 'simulation_time') else None
            self.data_collector.stop(world_end_time_s=sim_time)
        self.experiment_index = max(0, min(len(self.experiment_plan) - 1, index))
        self.experiment_start_sim_time = None
        print(f"[实验切换] -> {self._get_experiment_label()}")
        self.restart()
        self.hud.notification(self._get_experiment_label(), seconds=3.0)

    def switch_to_next_experiment(self):
        if not self._use_experiment_mode():
            self.hud.notification("未启用实验模式", seconds=2.0)
            return
        # 切换实验前强制停止数据采集，保证每个实验独立文件
        if self.data_collector and self.data_collector.is_collecting:
            sim_time = self.hud.simulation_time if hasattr(self.hud, 'simulation_time') else None
            self.data_collector.stop(world_end_time_s=sim_time)
        self.experiment_index = (self.experiment_index + 1) % len(self.experiment_plan)
        self.experiment_start_sim_time = None
        print(f"[实验切换] -> {self._get_experiment_label()}")
        self.restart()
        self.hud.notification(self._get_experiment_label(), seconds=3.0)
        
    def restart(self):
        self.player_max_speed = 1.589
        self.player_max_speed_fast = 3.713

        # 只有启用 6 组实验切换时，才做“每次实验开始前冷却”
        if self._use_experiment_mode():
            # 重启/重建车辆时，强制停止数据采集（避免跨实验写入同一个文件）
            if self.data_collector and self.data_collector.is_collecting:
                sim_time = self.hud.simulation_time if hasattr(self.hud, 'simulation_time') else None
                self.data_collector.stop(world_end_time_s=sim_time)
            self._experiment_record_started = False
            self._overtaking_prompt_pending = False
            self._experiment_cooldown_active = True
            self._experiment_cooldown_start_sim_time = None
            self.cooldown_remaining_s = self.experiment_cooldown_s
        else:
            self._experiment_cooldown_active = False
            self._experiment_cooldown_start_sim_time = None
            self.cooldown_remaining_s = 0.0
            self._overtaking_prompt_pending = False
        
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 0
        
        # 自车蓝图
        blueprint_list = get_actor_blueprints(self.world, self._actor_filter, self._actor_generation)
        if not blueprint_list:
            raise ValueError("找不到车辆蓝图")
        blueprint = blueprint_list[0]
        blueprint.set_attribute('role_name', self.actor_role_name)
        
        if blueprint.has_attribute('terramechanics'):
            blueprint.set_attribute('terramechanics', 'true')
        if blueprint.has_attribute('color'):
            blueprint.set_attribute('color', '255,255,255')  # 白色自车
        if blueprint.has_attribute('is_invincible'):
            blueprint.set_attribute('is_invincible', 'false')
        if blueprint.has_attribute('speed'):
            self.player_max_speed = float(blueprint.get_attribute('speed').recommended_values[1])
            self.player_max_speed_fast = float(blueprint.get_attribute('speed').recommended_values[2])
            
        # 重启前无条件清理旧对象 + 残留同角色车辆，避免“多前车/自车抖动”
        try:
            self.destroy()
        except Exception:
            pass
        self._cleanup_role_vehicles()
        self.player = None
        self.lead_vehicle = None
        self.lead_controller = None
            
        # 选择生成点
        spawn_points = self.map.get_spawn_points()
        
        # 如果没有预设生成点（OpenDRIVE地图），使用 waypoint 系统创建
        if not spawn_points:
            print('没有预设生成点，使用 waypoint 系统查找...')
            
            # 使用 waypoint 系统找到道路上的有效位置
            waypoints = self.map.generate_waypoints(100.0)  # 每100米一个
            if waypoints:
                # 筛选正向驾驶车道 (lane_id < 0 表示右侧/正向车道)
                driving_wps = [wp for wp in waypoints 
                              if wp.lane_type == carla.LaneType.Driving and wp.lane_id < 0]
                
                if driving_wps:
                    # 按 x 坐标排序，选择起点附近的 waypoint
                    driving_wps.sort(key=lambda wp: wp.transform.location.x)
                    # 选择第2个（跳过起点，给前车留空间）
                    start_wp = driving_wps[min(1, len(driving_wps)-1)]
                    spawn_point = start_wp.transform
                    spawn_point.location.z += 0.3  # 略微抬高
                    spawn_points = [spawn_point]
                    print(f"通过 waypoint 找到生成点: ({spawn_point.location.x:.1f}, {spawn_point.location.y:.1f}, {spawn_point.location.z:.1f})")
                else:
                    # 没有正向车道，尝试任意驾驶车道
                    driving_wps = [wp for wp in waypoints if wp.lane_type == carla.LaneType.Driving]
                    if driving_wps:
                        driving_wps.sort(key=lambda wp: wp.transform.location.x)
                        start_wp = driving_wps[min(1, len(driving_wps)-1)]
                        spawn_point = start_wp.transform
                        spawn_point.location.z += 0.3
                        spawn_points = [spawn_point]
                        print(f"使用任意车道生成点: ({spawn_point.location.x:.1f}, {spawn_point.location.y:.1f})")
            
            # 如果 waypoint 系统也失败，使用硬编码位置
            if not spawn_points:
                print("waypoint 系统失败，使用默认位置...")
                spawn_point = carla.Transform(
                    carla.Location(x=100.0, y=-1.75, z=2.0),  # 第一车道中心，z=2让车落地
                    carla.Rotation(pitch=0, yaw=0, roll=0)
                )
                spawn_points = [spawn_point]
                    
        if not spawn_points:
            print('错误: 无法找到或创建生成点')
            sys.exit(1)
        
        # 使用指定的生成点或默认
        spawn_index = getattr(self.args, 'spawn_point', None)
        if spawn_index is not None:
            if 0 <= spawn_index < len(spawn_points):
                spawn_point = spawn_points[spawn_index]
                print(f"使用指定生成点 [{spawn_index}]: ({spawn_point.location.x:.1f}, {spawn_point.location.y:.1f})")
            else:
                print(f"警告: 生成点索引 {spawn_index} 超出范围 (0-{len(spawn_points)-1})，使用默认")
                spawn_point = spawn_points[0]
        else:
            # Town04 默认使用最长直道生成点
            # 通过 --list-spawns 扫描得出: 生成点 353 有 1250m 直道
            if 'Town04' in self.map.name:
                best_spawn = 353  # 1250m 直道
                if best_spawn < len(spawn_points):
                    spawn_point = spawn_points[best_spawn]
                    print(f"Town04: 使用最长直道生成点 [{best_spawn}] (1250m)")
                elif len(spawn_points) > 0:
                    spawn_point = spawn_points[0]
                    print(f"Town04: 使用备选生成点 [0]")
            else:
                spawn_point = spawn_points[0]
                if hasattr(spawn_point, 'location'):
                    print(f"使用默认生成点: ({spawn_point.location.x:.1f}, {spawn_point.location.y:.1f})")
        
        # 6km 直道实验总是从起点附近重新开始，保证每次重启都跑完整路段
        exp_spawn = self._get_experiment_start_spawn()
        if exp_spawn is not None:
            spawn_point = exp_spawn
            print(f"[实验] {self._get_experiment_label()} | 起点重置: ({spawn_point.location.x:.1f}, {spawn_point.location.y:.1f})")

        def _clone_transform(t):
            return carla.Transform(
                carla.Location(x=t.location.x, y=t.location.y, z=t.location.z),
                carla.Rotation(pitch=t.rotation.pitch, yaw=t.rotation.yaw, roll=t.rotation.roll)
            )

        right_offset_m = float(getattr(self.args, 'spawn_right_offset', 0.0) or 0.0)
        base_spawn_point = _clone_transform(spawn_point)
        spawn_point = self._apply_spawn_right_offset(spawn_point)
        if abs(right_offset_m) > 1e-6:
            print(
                f"[生成偏移] 向右平移 {right_offset_m:.2f}m -> "
                f"({spawn_point.location.x:.1f}, {spawn_point.location.y:.1f})"
            )

        # 生成自车：优先尝试偏移后的右车道点，并在同车道前后少量探测可生成位置
        candidate_spawns = []
        candidate_spawns.append(_clone_transform(spawn_point))
        for dz in [0.3, 0.8]:
            t = _clone_transform(spawn_point)
            t.location.z += dz
            candidate_spawns.append(t)

        if abs(right_offset_m) > 1e-6:
            try:
                wp = self.map.get_waypoint(
                    spawn_point.location,
                    project_to_road=True,
                    lane_type=carla.LaneType.Driving
                )
            except Exception:
                wp = None

            if wp is not None:
                for step in [5.0, 10.0, -5.0, -10.0]:
                    next_wp = None
                    if step > 0:
                        cands = wp.next(step)
                    else:
                        cands = wp.previous(abs(step))
                    if cands:
                        next_wp = cands[0]
                    if next_wp is not None:
                        t = _clone_transform(next_wp.transform)
                        t.location.z += 0.3
                        candidate_spawns.append(t)

        # 去重（按位置近似）
        dedup = []
        seen = set()
        for t in candidate_spawns:
            key = (round(t.location.x, 2), round(t.location.y, 2), round(t.location.z, 2))
            if key in seen:
                continue
            seen.add(key)
            dedup.append(t)
        candidate_spawns = dedup

        self.player = None
        for cand in candidate_spawns:
            self.player = self.world.try_spawn_actor(blueprint, cand)
            if self.player is not None:
                spawn_point = cand
                break

        # 偏移失败时，回退原始生成点，避免直接退出
        if self.player is None and abs(right_offset_m) > 1e-6:
            self.player = self.world.try_spawn_actor(blueprint, base_spawn_point)
            if self.player is not None:
                spawn_point = base_spawn_point
                print("[生成偏移] 右车道生成失败，已回退到原始车道生成。")

        retry_count = 0
        while self.player is None and retry_count < 10:
            raw_spawn = random.choice(spawn_points)
            shifted_spawn = self._apply_spawn_right_offset(raw_spawn)

            # 先试偏移后的点，再试原始点
            self.player = self.world.try_spawn_actor(blueprint, shifted_spawn)
            if self.player is not None:
                spawn_point = shifted_spawn
                break

            self.player = self.world.try_spawn_actor(blueprint, raw_spawn)
            if self.player is not None:
                spawn_point = raw_spawn
                break
            retry_count += 1
            
        if self.player is None:
            print("错误: 无法生成自车")
            return
        
        # 等待车辆在服务器端完全创建
        if self.sync:
            self.world.tick()
            self.world.tick()  # 多等一帧确保稳定
        else:
            self.world.wait_for_tick()
            
        # 验证车辆是否还存在（可能因碰撞被销毁）
        try:
            _ = self.player.get_location()
        except RuntimeError:
            print("错误: 自车生成后被销毁，请检查生成位置")
            self.player = None
            return
            
        self.show_vehicle_telemetry = False
        self.modify_vehicle_physics(self.player)
        print(f"自车已生成: {get_actor_display_name(self.player)}")
        
        # 生成前车（超车实验会自动拉大初始间距）
        lead_dist = self._get_effective_lead_distance()
        self._spawn_lead_vehicle(spawn_point, distance=lead_dist)
        
        # 等待确保前车完全创建
        if self.sync:
            self.world.tick()
            self.world.tick()  # 多等一帧确保物理稳定
        
        # 设置两车初速度
        # 4组实验中前车统一从 0 起步，再按策略加速；cabin 下自车也从 0 起步
        if getattr(self.args, 'input_mode', 'keyboard') == 'cabin':
            initial_speed = 0.0
        else:
            initial_speed = self._get_effective_lead_speed()
        
        # 设置自车初始速度
        forward = spawn_point.get_forward_vector()
        ego_velocity = carla.Vector3D(
            forward.x * initial_speed,
            forward.y * initial_speed,
            0
        )
        self.player.set_target_velocity(ego_velocity)
        
        # 再次设置前车初始速度（确保与自车同步）
        if self.lead_vehicle:
            lead_forward = self.lead_vehicle.get_transform().get_forward_vector()
            lead_initial_speed = 0.0 if self._use_experiment_mode() else self._get_effective_lead_speed()
            lead_velocity = carla.Vector3D(
                lead_forward.x * lead_initial_speed,
                lead_forward.y * lead_initial_speed,
                0
            )
            self.lead_vehicle.set_target_velocity(lead_velocity)
        
        # 等待速度设置生效
        if self.sync:
            self.world.tick()
        
        target_speed = self._get_effective_lead_speed() * 3.6
        print(f"自车初速度: {initial_speed * 3.6:.0f} km/h, 前车初速度: {lead_initial_speed * 3.6 if self.lead_vehicle else 0:.0f} km/h, 前车目标速度: {target_speed:.0f} km/h")
        print(f"初始车距: {lead_dist}m")
        
        # 验证车辆仍然存在后再创建传感器
        if self.player is None:
            print("错误: 无法创建传感器，车辆不存在")
            return
        
        # OpenDRIVE 地图需要额外等待车辆位置同步
        if getattr(self.args, 'opendrive', None) or getattr(self.args, 'straight_road', False):
            print("等待车辆位置同步...")
            for _ in range(10):  # 等待更多 tick
                if self.sync:
                    self.world.tick()
                else:
                    self.world.wait_for_tick()
            # 验证位置不再是 (0,0,0)
            loc = self.player.get_location()
            print(f"车辆实际位置: ({loc.x:.1f}, {loc.y:.1f}, {loc.z:.1f})")
            
        # 设置传感器
        try:
            self.collision_sensor = CollisionSensor(self.player, self.hud)
            self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
            self.gnss_sensor = GnssSensor(self.player)
            self.imu_sensor = IMUSensor(self.player)
            self.camera_manager = CameraManager(self.player, self.hud, self._gamma)
            self.camera_manager.transform_index = cam_pos_index
            self.camera_manager.set_sensor(cam_index, notify=False)
        except RuntimeError as e:
            print(f"错误: 创建传感器失败 - {e}")
            return
        
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)
        
        if self.sync:
            self.world.tick()
        else:
            self.world.wait_for_tick()
            
    def _spawn_lead_vehicle(self, ego_spawn_point, distance=30.0):
        blueprint_library = self.world.get_blueprint_library()
        
        # 前车蓝图 - 增加容错
        lead_blueprints = blueprint_library.filter('vehicle.tesla.model3')
        if not lead_blueprints:
            lead_blueprints = blueprint_library.filter('vehicle.*')
        if not lead_blueprints:
            print("错误: 找不到前车蓝图")
            return
            
        lead_bp = lead_blueprints[0]
        lead_bp.set_attribute('role_name', 'lead_vehicle')
        if lead_bp.has_attribute('color'):
            lead_bp.set_attribute('color', '0,0,255')  # 蓝色前车
        
        # 对于直道地图，直接在前方生成
        if getattr(self.args, 'straight_road', False) or getattr(self.args, 'opendrive', None):
            # 直道地图：直接在 X 轴正方向前方生成
            lead_location = carla.Location(
                x=ego_spawn_point.location.x + distance,
                y=ego_spawn_point.location.y,
                z=ego_spawn_point.location.z + 0.3
            )
            lead_transform = carla.Transform(lead_location, ego_spawn_point.rotation)
            self.lead_vehicle = self.world.try_spawn_actor(lead_bp, lead_transform)
            if self.lead_vehicle:
                print(f"前车生成于直道前方 {distance}m 处")
        else:
            # CARLA 内置地图：使用 waypoint 系统
            ego_waypoint = self.map.get_waypoint(ego_spawn_point.location)
            if ego_waypoint:
                lead_waypoints = ego_waypoint.next(distance)
                if lead_waypoints:
                    lead_wp = lead_waypoints[0]
                    lead_transform = lead_wp.transform
                    lead_transform.location.z += 0.5
                    
                    road_curvature = self._check_road_curvature(ego_waypoint, distance)
                    if road_curvature > 0.01:
                        print(f"警告: 当前道路有弯道 (曲率: {road_curvature:.4f})")
                    
                    self.lead_vehicle = self.world.try_spawn_actor(lead_bp, lead_transform)
                    if self.lead_vehicle:
                        print(f"前车生成于道路 waypoint，距离: {distance}m")
        
        # 如果上述方法失败，使用传统方法
        if self.lead_vehicle is None:
            print("尝试备用方法生成前车...")
            forward = ego_spawn_point.get_forward_vector()
            for offset in [distance, distance-5, distance+5, distance+10, distance+15]:
                lead_location = carla.Location(
                    x=ego_spawn_point.location.x + forward.x * offset,
                    y=ego_spawn_point.location.y + forward.y * offset,
                    z=ego_spawn_point.location.z + 0.5
                )
                lead_transform = carla.Transform(lead_location, ego_spawn_point.rotation)
                self.lead_vehicle = self.world.try_spawn_actor(lead_bp, lead_transform)
                if self.lead_vehicle:
                    print(f"前车生成于偏移 {offset}m 处")
                    break
                    
        if self.lead_vehicle:
            self.modify_vehicle_physics(self.lead_vehicle)
            follow_road = not getattr(self.args, 'straight_drive', False)
            self.lead_controller = LeadVehicleController(
                self.lead_vehicle, 
                base_speed=self._get_effective_lead_speed(),
                random_mode=getattr(self.args, 'lead_random', False),
                random_seed=getattr(self.args, 'lead_seed', None),
                follow_road=follow_road,
                experiment_type=self._get_effective_experiment_type()
            )
            self.lead_controller.start()
            
            # 设置前车初始速度
            lead_init_speed = 0.0 if self._use_experiment_mode() else self._get_effective_lead_speed()
            lead_forward = self.lead_vehicle.get_transform().get_forward_vector()
            lead_initial_velocity = carla.Vector3D(
                lead_forward.x * lead_init_speed,
                lead_forward.y * lead_init_speed,
                0
            )
            self.lead_vehicle.set_target_velocity(lead_initial_velocity)
            
            mode_str = "直线行驶" if not follow_road else "沿道路行驶"
            print(f"前车已生成: {get_actor_display_name(self.lead_vehicle)} ({mode_str})")
            print(f"前车初速度: {lead_init_speed * 3.6:.0f} km/h, 目标速度: {self._get_effective_lead_speed() * 3.6:.0f} km/h")
        else:
            print("错误: 无法生成前车")
            
    def _check_road_curvature(self, start_waypoint, distance):
        """检查道路曲率，返回平均曲率值（越小越直）"""
        try:
            current_wp = start_waypoint
            total_angle_change = 0.0
            step = 5.0  # 每5米检查一次
            prev_yaw = current_wp.transform.rotation.yaw
            
            traveled = 0.0
            while traveled < distance:
                next_wps = current_wp.next(step)
                if not next_wps:
                    break
                current_wp = next_wps[0]
                current_yaw = current_wp.transform.rotation.yaw
                
                # 计算航向角变化
                angle_diff = abs(current_yaw - prev_yaw)
                if angle_diff > 180:
                    angle_diff = 360 - angle_diff
                total_angle_change += angle_diff
                
                prev_yaw = current_yaw
                traveled += step
            
            # 返回单位距离的平均角度变化（度/米）
            return total_angle_change / distance if distance > 0 else 0
        except Exception:
            return 0
            
    def modify_vehicle_physics(self, actor):
        try:
            physics_control = actor.get_physics_control()
            physics_control.use_sweep_wheel_collision = True
            actor.apply_physics_control(physics_control)
        except Exception:
            pass
            
    def next_weather(self, reverse=False):
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification('天气: %s' % preset[1])
        self.player.get_world().set_weather(preset[0])
        
    def next_map_layer(self, reverse=False):
        self.current_map_layer += -1 if reverse else 1
        self.current_map_layer %= len(self.map_layer_names)
        selected = self.map_layer_names[self.current_map_layer]
        self.hud.notification('地图层: %s' % selected)
        
    def load_map_layer(self, unload=False):
        selected = self.map_layer_names[self.current_map_layer]
        if unload:
            self.hud.notification('卸载地图层: %s' % selected)
            self.world.unload_map_layer(selected)
        else:
            self.hud.notification('加载地图层: %s' % selected)
            self.world.load_map_layer(selected)
            
    def toggle_radar(self):
        if self.radar_sensor is None:
            self.radar_sensor = RadarSensor(self.player)
        elif self.radar_sensor.sensor is not None:
            self.radar_sensor.sensor.destroy()
            self.radar_sensor = None
            
    def tick(self, clock):
        # 获取仿真时间并传递给前车控制器
        sim_time = self.hud.simulation_time if hasattr(self.hud, 'simulation_time') else None
        if self.lead_controller:
            if self._experiment_cooldown_active:
                # 冷却期内保持前车速度为 0，并跳过 lead_controller.update（避免提前起步）
                # 冷却期内不采集数据（避免记录到“启动前静止”阶段以外的数据）
                if self.data_collector and self.data_collector.is_collecting:
                    self.data_collector.stop(world_end_time_s=sim_time)
                if sim_time is not None:
                    if self._experiment_cooldown_start_sim_time is None:
                        self._experiment_cooldown_start_sim_time = sim_time
                    elapsed = sim_time - self._experiment_cooldown_start_sim_time
                    remaining = self.experiment_cooldown_s - elapsed
                    self.cooldown_remaining_s = max(0.0, remaining)
                    if remaining <= 0.0:
                        self._experiment_cooldown_active = False
                        self.cooldown_remaining_s = 0.0
                        exp = self._get_current_experiment() if self._use_experiment_mode() else None
                        if exp:
                            if exp.get('type') == 'overtaking':
                                self.hud.center_instruction("先进行跟驰\n随后变道超车", seconds=1.5)
                            else:
                                self.hud.center_instruction("请进行跟驰", seconds=1.5)
                            self._overtaking_prompt_pending = (exp.get('type') == 'overtaking')
                        # 倒计时结束：自动开始数据采集（每个实验独立文件）
                        if not self._experiment_record_started and self.lead_controller:
                            exp_id = self.experiment_index + 1
                            ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]  # 毫秒级唯一
                            save_dir = f'./experiment_data/{ts}_exp{exp_id}'
                            save_path = os.path.join(save_dir, 'driving_data.csv')
                            self.data_collector.start(
                                save_path,
                                self.lead_controller,
                                world_start_time_s=sim_time,
                            )
                            self._experiment_record_started = True
                            self.hud.notification(f'自动开始数据采集 (实验{exp_id})', seconds=2.0)
                    else:
                        if self.lead_vehicle:
                            self.lead_vehicle.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
            if not self._experiment_cooldown_active:
                self.lead_controller.update(sim_time)
                if self._overtaking_prompt_pending and self.lead_vehicle:
                    lead_vel = self.lead_vehicle.get_velocity()
                    lead_speed = math.sqrt(lead_vel.x**2 + lead_vel.y**2 + lead_vel.z**2)
                    target_speed = self._get_effective_lead_speed()
                    # 达到目标速度（允许小幅容差）后结束等待状态，并提示可超车
                    if lead_speed >= max(0.0, target_speed * 0.95):
                        self._overtaking_prompt_pending = False
                        self.hud.center_instruction("可超车", seconds=1.5)

        # 实验模式：冷却结束后开始计时（跟驰默认 180s，超车默认 120s）
        if self._use_experiment_mode() and sim_time is not None:
            if not self._experiment_cooldown_active:
                if self.experiment_start_sim_time is None:
                    self.experiment_start_sim_time = sim_time
                elapsed = sim_time - self.experiment_start_sim_time
                current_duration = self._get_effective_experiment_duration_s()
                self.experiment_remaining_s = max(0.0, current_duration - elapsed)
                self.experiment_countdown_active = True
                if elapsed >= current_duration:
                    self.experiment_countdown_active = False
                    self.experiment_remaining_s = 0.0
                    self.switch_to_next_experiment()
                    return
            else:
                self.experiment_remaining_s = 0.0
                self.experiment_countdown_active = False
        else:
            self.experiment_remaining_s = 0.0
            self.experiment_countdown_active = False
        
        # 直道保护（默认关闭，只在明确指定 --auto-reset 时启用）
        if getattr(self.args, 'auto_reset', False) and self.player:
            self._check_straight_protection()
            
        self.hud.tick(self, clock)
    
    def _check_straight_protection(self):
        """检测前方道路曲率，接近弯道时警告或重置"""
        # 检查是否禁用
        if getattr(self.args, 'no_straight_protect', False):
            return
            
        try:
            player_loc = self.player.get_location()
            player_wp = self.map.get_waypoint(player_loc)
            
            if player_wp is None:
                return
            
            auto_reset = getattr(self.args, 'auto_reset', False)
            
            # 检测前方 100 米的曲率
            lookahead = 100.0
            next_wps = player_wp.next(lookahead)
            
            if not next_wps:
                # 接近道路末端
                if auto_reset:
                    if not hasattr(self, '_end_warning_shown'):
                        self.hud.notification("接近道路末端，3秒后自动重置...", seconds=3.0)
                        self._end_warning_shown = True
                        self._reset_countdown = time.time()
                    elif time.time() - self._reset_countdown > 3.0:
                        self._end_warning_shown = False
                        print("\n[自动重置] 道路末端，重置到起点...")
                        self.restart()
                else:
                    if not hasattr(self, '_end_warning_shown') or not self._end_warning_shown:
                        self.hud.notification("接近道路末端，按 Backspace 重置", seconds=5.0)
                        self._end_warning_shown = True
                return
            
            self._end_warning_shown = False
            
            # 计算曲率（通过航向角变化）
            current_yaw = player_wp.transform.rotation.yaw
            target_yaw = next_wps[0].transform.rotation.yaw
            
            yaw_diff = abs(target_yaw - current_yaw)
            if yaw_diff > 180:
                yaw_diff = 360 - yaw_diff
            
            # 如果曲率超过阈值（15度）
            curve_threshold = 15
            if yaw_diff > curve_threshold:
                if auto_reset:
                    if not hasattr(self, '_curve_reset_pending') or not self._curve_reset_pending:
                        self.hud.notification(f"前方弯道 ({yaw_diff:.0f}°)，3秒后自动重置...", seconds=3.0)
                        self._curve_reset_pending = True
                        self._curve_reset_time = time.time()
                    elif time.time() - self._curve_reset_time > 3.0:
                        self._curve_reset_pending = False
                        print(f"\n[自动重置] 检测到弯道 ({yaw_diff:.0f}°)，重置到起点...")
                        self.restart()
                else:
                    if not hasattr(self, '_curve_warning_shown') or not self._curve_warning_shown:
                        self.hud.notification(f"前方弯道 ({yaw_diff:.0f}°)，按 Backspace 重置", seconds=5.0)
                        self._curve_warning_shown = True
            else:
                self._curve_warning_shown = False
                self._curve_reset_pending = False
                
        except Exception:
            pass
        
    def render(self, display):
        if self.camera_manager:
            self.camera_manager.render(display)
        else:
            # 调试: camera_manager 不存在
            if not hasattr(self, '_render_warn_shown'):
                print("[Render] 警告: camera_manager 为 None!")
                self._render_warn_shown = True
        self.hud.render(display)
        
    def destroy_sensors(self):
        if self.camera_manager and self.camera_manager.sensor:
            self.camera_manager.sensor.destroy()
            self.camera_manager.sensor = None
            self.camera_manager.index = None
            
    def destroy(self):
        if self.radar_sensor is not None:
            self.toggle_radar()
            
        sensors = [
            self.camera_manager.sensor if self.camera_manager else None,
            self.collision_sensor.sensor if self.collision_sensor else None,
            self.lane_invasion_sensor.sensor if self.lane_invasion_sensor else None,
            self.gnss_sensor.sensor if self.gnss_sensor else None,
            self.imu_sensor.sensor if self.imu_sensor else None,
        ]
        for sensor in sensors:
            if sensor is not None:
                sensor.stop()
                sensor.destroy()
                
        if self.player is not None:
            self.player.destroy()
        if self.lead_vehicle is not None:
            self.lead_vehicle.destroy()


# ==============================================================================
# -- 键盘/驾驶舱控制 -----------------------------------------------------------
# ==============================================================================

class VehicleController:
    """车辆控制器 - 支持键盘和驾驶舱（改进版UDP协议）"""
    
    def __init__(self, world, args):
        self.world = world
        self.args = args
        self._control = carla.VehicleControl()
        self._steer_cache = 0.0
        self._lights = carla.VehicleLightState.NONE
        
        # 控制模式
        self.control_mode = 'manual'
        self.input_mode = args.input_mode
        self._autopilot_enabled = False
        
        # 自动驾驶模型
        self.autopilot_model = None
        
        # 驾驶舱UDP通信
        self.cabin_socket = None
        self.cabin_ip = args.cabin_ip
        self.cabin_port = args.cabin_port
        self._cabin_echo_interval = float(getattr(args, 'cabin_echo_interval', 0.0) or 0.0)
        self._last_cabin_echo_time = 0.0
        self._last_cabin_apply_echo_time = 0.0
        self._cabin_stuck_since = None
        self._last_cabin_nudge_time = 0.0
        # 发送给 SCANeR 的阻力反馈输出（上一帧稳定值）
        self._ffb_output = 0.0
        
        if self.input_mode == 'cabin':
            self._setup_cabin_connection()
            
        # 物理控制
        self._physics_control = None
        if world.player:
            try:
                self._physics_control = world.player.get_physics_control()
            except RuntimeError as e:
                print(f"警告: 无法获取物理控制 - {e}")
            
        # 力反馈相关
        self._previous_steer_angle = 0
        self._previous_time = time.time()
        
        # Ackermann控制
        self._ackermann_enabled = False
        self._ackermann_control = carla.VehicleAckermannControl()
        self._ackermann_reverse = 1
        
    def _setup_cabin_connection(self):
        try:
            self.cabin_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.cabin_socket.settimeout(0.1)
            self.cabin_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1000)
            print(f"[驾驶舱] UDP连接已初始化")
            print(f"  发送地址: {self.cabin_ip}:{self.cabin_port}")
        except Exception as e:
            print(f"[驾驶舱] 连接失败: {e}")
            self.cabin_socket = None
            
    def parse_events(self, client, world, clock, sync_mode):
        current_lights = self._lights
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True
                    
                # Backspace: 重启
                elif event.key == K_BACKSPACE:
                    if self._autopilot_enabled:
                        world.player.set_autopilot(False)
                        world.restart()
                        world.player.set_autopilot(True)
                    else:
                        world.restart()
                        
                # F1: 开始/停止数据采集
                elif event.key == K_F1:
                    if world._use_experiment_mode():
                        world.hud.notification("4组实验模式：采集由冷却倒计时自动控制，F1 无效", seconds=2.0)
                    else:
                        if world.data_collector.is_collecting:
                            sim_time = world.hud.simulation_time if hasattr(world.hud, 'simulation_time') else None
                            world.data_collector.stop(world_end_time_s=sim_time)
                            world.hud.notification("数据采集已停止")
                        else:
                            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                            save_path = f'./experiment_data/{timestamp}/driving_data.csv'
                            sim_time = world.hud.simulation_time if hasattr(world.hud, 'simulation_time') else None
                            world.data_collector.start(
                                save_path,
                                world.lead_controller,
                                world_start_time_s=sim_time,
                            )
                            world.hud.notification("数据采集已开始")
                        
                # F2: 切换控制模式
                elif event.key == K_F2:
                    if self.control_mode == 'manual':
                        self.control_mode = 'autopilot'
                        world.hud.notification("切换到自动驾驶模式")
                    else:
                        self.control_mode = 'manual'
                        world.hud.notification("切换到手动驾驶模式")
                        
                # F3: 切换前车行为
                elif event.key == K_F3:
                    if world.lead_controller:
                        mode = world.lead_controller.toggle_mode()
                        world.hud.notification(f"前车行为: {mode}")

                # F4: 切换下一个实验（按当前实验计划）
                elif event.key == K_F4:
                    world.switch_to_next_experiment()

                # F5-F10: 直接切换实验 1-4（若当前计划不足，超出的按键给提示）
                elif event.key in (K_F5, K_F6, K_F7, K_F8, K_F9, K_F10):
                    key_to_index = {
                        K_F5: 0,
                        K_F6: 1,
                        K_F7: 2,
                        K_F8: 3,
                        K_F9: 4,
                        K_F10: 5,
                    }
                    target_idx = key_to_index[event.key]
                    if target_idx < len(world.experiment_plan):
                        world.switch_to_experiment(target_idx)
                    else:
                        world.hud.notification(
                            f"当前模式仅有 {len(world.experiment_plan)} 组实验",
                            seconds=1.8
                        )
                        
                # Tab: 切换相机
                elif event.key == K_TAB:
                    world.camera_manager.toggle_camera()
                    
                # C: 切换天气
                elif event.key == K_c and pygame.key.get_mods() & KMOD_SHIFT:
                    world.next_weather(reverse=True)
                elif event.key == K_c:
                    world.next_weather()
                    
                # V: 切换地图层
                elif event.key == K_v and pygame.key.get_mods() & KMOD_SHIFT:
                    world.next_map_layer(reverse=True)
                elif event.key == K_v:
                    world.next_map_layer()
                    
                # B: 加载/卸载地图层
                elif event.key == K_b and pygame.key.get_mods() & KMOD_SHIFT:
                    world.load_map_layer(unload=True)
                elif event.key == K_b:
                    world.load_map_layer()
                    
                # G: 雷达可视化
                elif event.key == K_g:
                    world.toggle_radar()
                    
                # N: 下一个传感器
                elif event.key == K_n or event.key == K_BACKQUOTE:
                    world.camera_manager.next_sensor()
                    
                # 1-9: 选择传感器
                elif event.key > K_0 and event.key <= K_9:
                    index_ctrl = 9 if pygame.key.get_mods() & KMOD_CTRL else 0
                    world.camera_manager.set_sensor(event.key - 1 - K_0 + index_ctrl)
                    
                # R: 录制
                elif event.key == K_r and not (pygame.key.get_mods() & KMOD_CTRL):
                    world.camera_manager.toggle_recording()
                elif event.key == K_r and (pygame.key.get_mods() & KMOD_CTRL):
                    if world.recording_enabled:
                        client.stop_recorder()
                        world.recording_enabled = False
                        world.hud.notification("录制已停止")
                    else:
                        client.start_recorder("car_following_recording.rec")
                        world.recording_enabled = True
                        world.hud.notification("录制已开始")
                        
                # Ctrl+P: 回放
                elif event.key == K_p and (pygame.key.get_mods() & KMOD_CTRL):
                    client.stop_recorder()
                    world.recording_enabled = False
                    current_index = world.camera_manager.index
                    world.destroy_sensors()
                    self._autopilot_enabled = False
                    world.player.set_autopilot(False)
                    world.hud.notification("回放中...")
                    client.replay_file("car_following_recording.rec", world.recording_start, 0, 0)
                    world.camera_manager.set_sensor(current_index)
                    
                # Ctrl +/-: 调整回放起始时间
                elif event.key == K_MINUS and (pygame.key.get_mods() & KMOD_CTRL):
                    world.recording_start -= 10 if pygame.key.get_mods() & KMOD_SHIFT else 1
                    world.hud.notification(f"回放起始: {world.recording_start}秒")
                elif event.key == K_EQUALS and (pygame.key.get_mods() & KMOD_CTRL):
                    world.recording_start += 10 if pygame.key.get_mods() & KMOD_SHIFT else 1
                    world.hud.notification(f"回放起始: {world.recording_start}秒")
                    
                # O: 开关门
                elif event.key == K_o:
                    try:
                        if world.doors_are_open:
                            world.hud.notification("关门")
                            world.doors_are_open = False
                            world.player.close_door(carla.VehicleDoor.All)
                        else:
                            world.hud.notification("开门")
                            world.doors_are_open = True
                            world.player.open_door(carla.VehicleDoor.All)
                    except Exception:
                        pass
                        
                # T: 遥测
                elif event.key == K_t:
                    if world.show_vehicle_telemetry:
                        world.player.show_debug_telemetry(False)
                        world.show_vehicle_telemetry = False
                        world.hud.notification("遥测已关闭")
                    else:
                        try:
                            world.player.show_debug_telemetry(True)
                            world.show_vehicle_telemetry = True
                            world.hud.notification("遥测已开启")
                        except Exception:
                            pass
                            
                # H: 帮助
                elif event.key == K_h or (event.key == K_SLASH and pygame.key.get_mods() & KMOD_SHIFT):
                    world.hud.help.toggle()
                    
                # 车辆控制按键
                if isinstance(self._control, carla.VehicleControl):
                    # F: Ackermann控制
                    if event.key == K_f:
                        self._ackermann_enabled = not self._ackermann_enabled
                        world.hud.notification("Ackermann控制 %s" % 
                                             ("开启" if self._ackermann_enabled else "关闭"))
                                             
                    # Q: 倒档
                    if event.key == K_q:
                        if not self._ackermann_enabled:
                            self._control.gear = 1 if self._control.reverse else -1
                        else:
                            self._ackermann_reverse *= -1
                            self._ackermann_control = carla.VehicleAckermannControl()
                            
                    # M: 手动/自动变速
                    elif event.key == K_m:
                        self._control.manual_gear_shift = not self._control.manual_gear_shift
                        self._control.gear = world.player.get_control().gear
                        world.hud.notification('%s变速' % 
                                             ('手动' if self._control.manual_gear_shift else '自动'))
                                             
                    # ,/.: 升降档
                    elif self._control.manual_gear_shift and event.key == K_COMMA:
                        self._control.gear = max(-1, self._control.gear - 1)
                    elif self._control.manual_gear_shift and event.key == K_PERIOD:
                        self._control.gear = self._control.gear + 1
                        
                    # P: 自动驾驶
                    elif event.key == K_p and not (pygame.key.get_mods() & KMOD_CTRL):
                        if self.input_mode == 'cabin':
                            # 对齐 DriveSim 的驾驶舱控制思路：驾驶舱接管时不让 CARLA autopilot 抢控制
                            self._autopilot_enabled = False
                            world.player.set_autopilot(False)
                            world.hud.notification('驾驶舱模式下已禁用CARLA自动驾驶')
                        else:
                            if not self._autopilot_enabled and not sync_mode:
                                print("警告: 异步模式下自动驾驶可能不稳定")
                            self._autopilot_enabled = not self._autopilot_enabled
                            world.player.set_autopilot(self._autopilot_enabled)
                            world.hud.notification('CARLA自动驾驶 %s' % ('开启' if self._autopilot_enabled else '关闭'))
                        
                    # L: 车灯控制
                    elif event.key == K_l and pygame.key.get_mods() & KMOD_CTRL:
                        current_lights ^= carla.VehicleLightState.Special1
                    elif event.key == K_l and pygame.key.get_mods() & KMOD_SHIFT:
                        current_lights ^= carla.VehicleLightState.HighBeam
                    elif event.key == K_l:
                        if not self._lights & carla.VehicleLightState.Position:
                            world.hud.notification("位置灯")
                            current_lights |= carla.VehicleLightState.Position
                        else:
                            world.hud.notification("近光灯")
                            current_lights |= carla.VehicleLightState.LowBeam
                        if self._lights & carla.VehicleLightState.LowBeam:
                            world.hud.notification("雾灯")
                            current_lights |= carla.VehicleLightState.Fog
                        if self._lights & carla.VehicleLightState.Fog:
                            world.hud.notification("车灯关闭")
                            current_lights ^= carla.VehicleLightState.Position
                            current_lights ^= carla.VehicleLightState.LowBeam
                            current_lights ^= carla.VehicleLightState.Fog
                            
                    # I: 内饰灯
                    elif event.key == K_i:
                        current_lights ^= carla.VehicleLightState.Interior
                        
                    # Z/X: 转向灯
                    elif event.key == K_z:
                        current_lights ^= carla.VehicleLightState.LeftBlinker
                    elif event.key == K_x:
                        current_lights ^= carla.VehicleLightState.RightBlinker
                        
        # 获取控制输入
        # 驾驶舱模式强制关闭 CARLA autopilot，避免油门/制动被覆盖
        if self.input_mode == 'cabin' and self._autopilot_enabled:
            self._autopilot_enabled = False
            world.player.set_autopilot(False)

        if not self._autopilot_enabled:
            # 驾驶舱优先：不受 F2「跟驰自动驾驶」影响，否则油门被 IDM 覆盖；也不走 Ackermann 分支
            if self.input_mode == 'cabin' and self.cabin_socket:
                self._parse_cabin_input(world)
            elif self.control_mode == 'manual':
                self._parse_keyboard_input(pygame.key.get_pressed(), clock.get_time())
                self._control.reverse = self._control.gear < 0
            else:
                self._compute_autopilot_control(world)
                
            # 车灯自动控制
            if self._control.brake:
                current_lights |= carla.VehicleLightState.Brake
            else:
                current_lights &= ~carla.VehicleLightState.Brake
            if self._control.reverse:
                current_lights |= carla.VehicleLightState.Reverse
            else:
                current_lights &= ~carla.VehicleLightState.Reverse
                
            if current_lights != self._lights:
                self._lights = current_lights
                world.player.set_light_state(carla.VehicleLightState(self._lights))
                
            # 应用控制（驾驶舱协议只填充 VehicleControl；若误开 Ackermann(F)，apply_ackermann 会忽略油门）
            if self.input_mode == 'cabin' or not self._ackermann_enabled:
                world.player.apply_control(self._control)
            else:
                world.player.apply_ackermann_control(self._ackermann_control)
                self._control = world.player.get_control()

            if self.input_mode == 'cabin' and self._cabin_echo_interval > 0:
                now = time.time()
                if now - self._last_cabin_apply_echo_time >= self._cabin_echo_interval:
                    self._last_cabin_apply_echo_time = now
                    v_apply = world.player.get_velocity()
                    speed_apply = 3.6 * math.sqrt(v_apply.x**2 + v_apply.y**2 + v_apply.z**2)
                    c_apply = world.player.get_control()
                    sim_time = world.hud.simulation_time if hasattr(world, 'hud') and hasattr(world.hud, 'simulation_time') else None
                    sim_time_str = f"{sim_time:.2f}s" if sim_time is not None else "N/A"
                    print(
                        "[驾驶舱控制生效] "
                        f"world_time={sim_time_str} "
                        f"车速={speed_apply:.2f} km/h "
                        f"throttle={c_apply.throttle:.3f} brake={c_apply.brake:.3f} "
                        f"hand_brake={int(c_apply.hand_brake)} gear={c_apply.gear} "
                        f"manual={int(c_apply.manual_gear_shift)} reverse={int(c_apply.reverse)}"
                    )

            # 驾驶舱起步助推：控制量正常但长期低速时，给一个轻微前向速度脉冲帮助车辆脱离静止
            if self.input_mode == 'cabin':
                now = time.time()
                v_apply = world.player.get_velocity()
                speed_apply = 3.6 * math.sqrt(v_apply.x**2 + v_apply.y**2 + v_apply.z**2)
                c_apply = world.player.get_control()
                stuck_condition = (
                    c_apply.throttle > 0.45 and
                    c_apply.brake < 0.05 and
                    not c_apply.hand_brake and
                    not c_apply.reverse and
                    speed_apply < 0.4
                )
                if stuck_condition:
                    if self._cabin_stuck_since is None:
                        self._cabin_stuck_since = now
                    elif now - self._cabin_stuck_since > 1.2 and now - self._last_cabin_nudge_time > 0.8:
                        fwd = world.player.get_transform().get_forward_vector()
                        world.player.set_target_velocity(carla.Vector3D(fwd.x * 2.0, fwd.y * 2.0, 0.0))
                        self._last_cabin_nudge_time = now
                        sim_time = world.hud.simulation_time if hasattr(world, 'hud') and hasattr(world.hud, 'simulation_time') else None
                        sim_time_str = f"{sim_time:.2f}s" if sim_time is not None else "N/A"
                        print(f"[驾驶舱起步助推] world_time={sim_time_str} throttle={c_apply.throttle:.3f} speed={speed_apply:.2f}km/h -> nudge 2.0m/s")
                else:
                    self._cabin_stuck_since = None
                
        # 采集数据
        if world.data_collector.is_collecting and world.lead_vehicle:
            world.data_collector.collect(
                world.player, 
                world.lead_vehicle, 
                self._control,
                self.control_mode,
                world.lead_controller
            )
            
        return False
        
    def _parse_keyboard_input(self, keys, milliseconds):
        # 油门
        if keys[K_UP] or keys[K_w]:
            if not self._ackermann_enabled:
                self._control.throttle = min(self._control.throttle + 0.05, 1.0)
            else:
                self._ackermann_control.speed += round(milliseconds * 0.005, 2) * self._ackermann_reverse
        else:
            if not self._ackermann_enabled:
                self._control.throttle = 0.0
                
        # 制动
        if keys[K_DOWN] or keys[K_s]:
            if not self._ackermann_enabled:
                self._control.brake = min(self._control.brake + 0.1, 1.0)
            else:
                self._ackermann_control.speed -= min(abs(self._ackermann_control.speed), 
                                                     round(milliseconds * 0.005, 2)) * self._ackermann_reverse
                self._ackermann_control.speed = max(0, abs(self._ackermann_control.speed)) * self._ackermann_reverse
        else:
            if not self._ackermann_enabled:
                self._control.brake = 0.0
                
        # 转向
        steer_increment = 5e-4 * milliseconds
        if keys[K_LEFT] or keys[K_a]:
            if self._steer_cache > 0:
                self._steer_cache = 0
            else:
                self._steer_cache -= steer_increment
        elif keys[K_RIGHT] or keys[K_d]:
            if self._steer_cache < 0:
                self._steer_cache = 0
            else:
                self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0
            
        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
        
        if not self._ackermann_enabled:
            self._control.steer = round(self._steer_cache, 2)
            self._control.hand_brake = keys[K_SPACE]
        else:
            self._ackermann_control.steer = round(self._steer_cache, 2)
            
    def _parse_cabin_input(self, world):
        """解析驾驶舱输入 (完整UDP协议)"""
        if not self.cabin_socket:
            return
            
        try:
            # 获取车辆状态
            v = world.player.get_velocity()
            c = world.player.get_control()
            speed = 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)
            if not math.isfinite(speed):
                speed = 0.0
            
            # 计算发动机转速（模拟）
            engine_rpm = 0.0
            if self._physics_control and c.gear > 0:
                gear = self._physics_control.forward_gears[min(c.gear, len(self._physics_control.forward_gears)-1)]
                engine_rpm = speed * gear.ratio * 100  # 简化计算
            if not math.isfinite(engine_rpm):
                engine_rpm = 0.0
                
            # 发送上一帧反馈值，避免在同一帧“先发后算”导致的不稳定
            fValue1_send = float(self._ffb_output) if math.isfinite(self._ffb_output) else 0.0
            
            # 发送数据到驾驶舱
            data_send = struct.pack(
                '<Lffffffffffffffff', 
                15, engine_rpm, speed, fValue1_send,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            )
            
            self.cabin_socket.sendto(data_send, (self.cabin_ip, self.cabin_port))
            
            # 接收驾驶舱数据
            message, _ = self.cabin_socket.recvfrom(164)
            data = struct.unpack('<LfffffffffLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL', message)
            
            if self._cabin_echo_interval > 0:
                now = time.time()
                if now - self._last_cabin_echo_time >= self._cabin_echo_interval:
                    self._last_cabin_echo_time = now
                    preview = message[: min(48, len(message))].hex()
                    sim_time = world.hud.simulation_time if hasattr(world, 'hud') and hasattr(world.hud, 'simulation_time') else None
                    sim_time_str = f"{sim_time:.2f}s" if sim_time is not None else "N/A"
                    print(
                        "[驾驶舱回传] "
                        f"world_time={sim_time_str} "
                        f"len={len(message)} "
                        f"hdr={data[0]} "
                        f"tx_speed={speed:.2f}km/h tx_rpm={engine_rpm:.1f} "
                        f"油门={data[1]:.3f} 制动={data[2]:.3f} f3={data[3]:.3f} "
                        f"转向={data[4]:.3f} 手刹={data[5]:.3f} "
                        f"开关量有效={data[10]} "
                        f"D/N/P/R={data[33]}/{data[35]}/{data[34]}/{data[36]} "
                        f"| hex48={preview}"
                    )
            
            lights = 0
            # 对齐 DriveSim1005：驾驶舱输入下固定使用手动挡逻辑
            self._control.manual_gear_shift = True
            
            if data[0] > 0:
                # 处理模拟量: 油门(data[1]), 制动(data[2]), 方向盘(data[4]), 手刹(data[5])
                
                # 油门 - 平滑过渡
                if self._control.throttle <= data[1] + 0.1:
                    self._control.throttle = data[1]
                else:
                    self._control.throttle *= 0.95
                    
                # 制动 - 限制最大值
                if data[2] < 0.9:
                    self._control.brake = data[2]
                else:
                    self._control.brake = 0.9
                    
                # 方向盘
                self._control.steer = -data[4] / 4
                
                # 力反馈计算
                if data[4]:
                    time_difference = 0.05
                    steer_angle_difference = (-data[4] / 3) - self._previous_steer_angle
                    steer_speed = abs(steer_angle_difference / time_difference)
                    DAMPING_COEFFICIENT = 0.2
                    damping = -DAMPING_COEFFICIENT * steer_speed
                    feedback_non_linear = -3 * math.atan(data[4])
                    fValue1 = feedback_non_linear + damping
                    MAX_FEEDBACK = 0.8
                    if abs(fValue1) > MAX_FEEDBACK:
                        fValue1 = MAX_FEEDBACK if fValue1 > 0 else -MAX_FEEDBACK
                    DEADZONE = 0.1
                    if abs(data[4]) < DEADZONE:
                        fValue1 = 0
                    if not math.isfinite(fValue1):
                        fValue1 = 0.0
                    self._ffb_output = float(fValue1)
                    self._previous_steer_angle = -data[4] / 3
                else:
                    self._ffb_output = 0.0
                    
                # 手刹
                if data[5] > 0:
                    self._control.hand_brake = True
                    lights |= carla.VehicleLightState.Brake
                else:
                    self._control.hand_brake = False
                    
            # 处理开关量信号
            if data[10] > 0:
                # 转向灯
                if data[13] > 0:
                    lights |= carla.VehicleLightState.LeftBlinker
                if data[14] > 0:
                    lights |= carla.VehicleLightState.RightBlinker
                    
                # 车灯
                if data[17] > 0:  # 近光灯
                    lights |= carla.VehicleLightState.LowBeam
                    lights |= carla.VehicleLightState.Position
                if data[18] > 0:  # 远光灯
                    lights |= carla.VehicleLightState.HighBeam
                    lights |= carla.VehicleLightState.Position
                    
                # 挡位
                self._control.reverse = False
                if data[33] > 0:  # D档：使用2档（更高极速；低速扭矩不足时可改回1档）
                    self._control.gear = 2
                elif data[35] > 0:  # N档
                    self._control.gear = 0
                elif data[34] > 0:  # P档
                    self._control.gear = 0
                elif data[36] > 0:  # R档
                    self._control.gear = -1
                    self._control.reverse = True
                    
                # 更新车灯
                if lights != self._lights:
                    self._lights = lights
                    world.player.set_light_state(carla.VehicleLightState(lights))
                    
        except socket.timeout:
            pass
        except Exception as e:
            print(f"[驾驶舱] 通信错误: {e}")
            
    def _compute_autopilot_control(self, world):
        """计算自动驾驶控制 (IDM模型)"""
        if not world.lead_vehicle:
            return
            
        ego_vel = world.player.get_velocity()
        lead_vel = world.lead_vehicle.get_velocity()
        
        ego_speed = math.sqrt(ego_vel.x**2 + ego_vel.y**2 + ego_vel.z**2)
        lead_speed = math.sqrt(lead_vel.x**2 + lead_vel.y**2 + lead_vel.z**2)
        
        ego_loc = world.player.get_location()
        lead_loc = world.lead_vehicle.get_location()
        distance = math.sqrt((ego_loc.x - lead_loc.x)**2 + (ego_loc.y - lead_loc.y)**2)
        
        # IDM参数
        desired_thw = SAFE_TIME_HEADWAY
        min_spacing = MIN_SPACING
        max_acc = 2.0
        comfortable_dec = 2.0
        desired_speed = 25.0
        
        # IDM加速度计算
        delta_v = ego_speed - lead_speed
        s_star = min_spacing + ego_speed * desired_thw + ego_speed * delta_v / (2 * math.sqrt(max_acc * comfortable_dec))
        
        if distance > 0:
            acceleration = max_acc * (1 - (ego_speed / desired_speed)**4 - (s_star / distance)**2)
        else:
            acceleration = -comfortable_dec
            
        # 转换为油门/制动
        if acceleration > 0:
            self._control.throttle = min(acceleration / max_acc, 1.0)
            self._control.brake = 0.0
        else:
            self._control.throttle = 0.0
            self._control.brake = min(-acceleration / comfortable_dec, 1.0)
            
        self._control.steer = 0.0
        
    def get_control(self):
        return self._control
        
    @staticmethod
    def _is_quit_shortcut(key):
        return key == K_ESCAPE or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)


# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================

def find_chinese_font():
    """查找系统中支持中文的字体"""
    chinese_font_names = [
        'notosanscjk', 'notosanssc', 'notosanstc', 'notosanshk',
        'wenquanyimicrohei', 'wenquanyizenhei', 'wenquanyi',
        'droidsansfallback', 'droidsans',
        'microsoftyahei', 'yahei', 'simhei', 'simsun',
        'arialuni', 'arial unicode',
        'dejavusans', 'freesans', 'liberation'
    ]
    
    available_fonts = pygame.font.get_fonts()
    
    for font_name in chinese_font_names:
        for available in available_fonts:
            if font_name in available.lower().replace(' ', '').replace('-', ''):
                font_path = pygame.font.match_font(available)
                if font_path:
                    return font_path
    
    font_paths = [
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf',
        '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
        '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
        '/usr/share/fonts/wenquanyi/wqy-microhei/wqy-microhei.ttc',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
    ]
    
    for path in font_paths:
        if os.path.exists(path):
            return path
    
    return None


class HUD:
    """信息显示"""
    
    def __init__(self, width, height):
        self.dim = (width, height)
        self._init_fonts()
        self._notifications = FadingText(
            self._font_chinese_large,
            (width, 40), (0, height - 40)
        )
        self.help = HelpText(self._font_chinese, width, height)
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()
        self._show_ackermann_info = False
        self._ackermann_control = carla.VehicleAckermannControl()
        self._cooldown_remaining_s = 0.0
        self._experiment_remaining_s = 0.0
        self._experiment_countdown_active = False
        self._center_instruction_text = ''
        self._center_instruction_seconds_left = 0.0
        self._center_overlay_color = (255, 255, 255)
        self._center_overlay_shadow_color = (0, 0, 0)
        self.show_lane_invasion_notification = True
        
    def _init_fonts(self):
        chinese_font = find_chinese_font()
        
        if chinese_font:
            print(f"使用中文字体: {os.path.basename(chinese_font)}")
            self._font_mono = pygame.font.Font(chinese_font, 14)
            self._font_chinese = pygame.font.Font(chinese_font, 18)
            self._font_chinese_large = pygame.font.Font(chinese_font, 20)
            self._font_cooldown = pygame.font.Font(chinese_font, 64)
            self._font_countdown_small = pygame.font.Font(chinese_font, 32)
        else:
            print("警告: 未找到中文字体，使用默认字体")
            font_name = 'courier' if os.name == 'nt' else 'mono'
            fonts = [x for x in pygame.font.get_fonts() if font_name in x]
            default_font = 'ubuntumono'
            mono = default_font if default_font in fonts else (fonts[0] if fonts else None)
            mono = pygame.font.match_font(mono) if mono else None
            self._font_mono = pygame.font.Font(mono, 14) if mono else pygame.font.Font(None, 14)
            self._font_chinese = pygame.font.Font(pygame.font.get_default_font(), 18)
            self._font_chinese_large = pygame.font.Font(pygame.font.get_default_font(), 20)
            self._font_cooldown = pygame.font.Font(pygame.font.get_default_font(), 64)
            self._font_countdown_small = pygame.font.Font(pygame.font.get_default_font(), 32)
        
    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame
        self.simulation_time = timestamp.elapsed_seconds
        
    def tick(self, world, clock):
        self._notifications.tick(world, clock)
        delta_seconds = 1e-3 * clock.get_time()
        self._center_instruction_seconds_left = max(
            0.0, self._center_instruction_seconds_left - delta_seconds
        )
        if self._center_instruction_seconds_left <= 0.0:
            self._center_instruction_text = ''
        self._cooldown_remaining_s = float(getattr(world, 'cooldown_remaining_s', 0.0) or 0.0)
        self._experiment_remaining_s = float(getattr(world, 'experiment_remaining_s', 0.0) or 0.0)
        self._experiment_countdown_active = bool(getattr(world, 'experiment_countdown_active', False))
        if not self._show_info:
            return
            
        t = world.player.get_transform()
        v = world.player.get_velocity()
        c = world.player.get_control()
        
        speed = 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)
        
        self._info_text = [
            'Server: %16.0f FPS' % self.server_fps,
            'Client: %16.0f FPS' % clock.get_fps(),
            '',
            '位置: (%5.1f, %5.1f)' % (t.location.x, t.location.y),
            '挡位: %15s' % {-1: 'R', 0: 'N'}.get(c.gear, c.gear),
            '',
            '油门: %14.2f' % c.throttle,
            '制动: %14.2f' % c.brake,
            '方向: %14.2f' % c.steer,
        ]

        # 行车信息（自车速度 / 前车速度 / 车头间距）
        if world.lead_vehicle:
            lead_loc = world.lead_vehicle.get_location()
            lead_vel = world.lead_vehicle.get_velocity()
            lead_speed = 3.6 * math.sqrt(lead_vel.x**2 + lead_vel.y**2 + lead_vel.z**2)

            distance = math.sqrt(
                (t.location.x - lead_loc.x)**2 +
                (t.location.y - lead_loc.y)**2
            )

            self._info_text.extend([
                '',
                '--- 行车信息 ---',
                '自车速度: %9.0f km/h' % speed,
                '前车速度: %9.0f km/h' % lead_speed,
                '车头间距: %9.1f m' % distance,
            ])
        else:
            self._info_text.extend([
                '',
                '--- 行车信息 ---',
                '自车速度: %9.0f km/h' % speed,
            ])
        
        # 数据采集状态
        if world.data_collector.is_collecting:
            self._info_text.extend([
                '',
                '[录制中] %d 帧' % world.data_collector.frame_count
            ])
            
        # 录制状态
        if world.recording_enabled:
            self._info_text.append('[仿真录制中]')
            
    def show_ackermann_info(self, enabled):
        self._show_ackermann_info = enabled
        
    def update_ackermann_control(self, control):
        self._ackermann_control = control
        
    def toggle_info(self):
        self._show_info = not self._show_info
        
    def notification(self, text, seconds=2.0):
        self._notifications.set_text(text, seconds=seconds)

    def center_instruction(self, text, seconds=1.5):
        self._center_instruction_text = str(text)
        self._center_instruction_seconds_left = max(0.0, float(seconds))

    def _render_center_overlay(self, display, text):
        lines = [part for part in str(text).splitlines() if part != ""]
        if not lines:
            return
        rendered = [
            (
                self._font_cooldown.render(line, True, self._center_overlay_color),
                self._font_cooldown.render(line, True, self._center_overlay_shadow_color),
            )
            for line in lines
        ]
        line_gap = 10
        total_h = sum(s.get_height() for s, _ in rendered) + line_gap * max(0, len(rendered) - 1)
        ty = (self.dim[1] - total_h) // 2
        for surface, shadow in rendered:
            tx = (self.dim[0] - surface.get_width()) // 2
            display.blit(shadow, (tx + 3, ty + 3))
            display.blit(surface, (tx, ty))
            ty += surface.get_height() + line_gap

    def _render_top_right_countdown(self, display, text):
        surface = self._font_countdown_small.render(text, True, self._center_overlay_color)
        shadow = self._font_countdown_small.render(text, True, self._center_overlay_shadow_color)
        tx = max(12, self.dim[0] - surface.get_width() - 12)
        ty = 12
        display.blit(shadow, (tx + 2, ty + 2))
        display.blit(surface, (tx, ty))

    @staticmethod
    def _fmt_mm_ss(seconds):
        seconds = max(0.0, float(seconds))
        mm = int(seconds // 60)
        ss = int(seconds % 60)
        return f"{mm:02d}:{ss:02d}"

    def render(self, display):
        if self._show_info:
            info_surface = pygame.Surface((250, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if item:
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)
        # 冷却中：中央显示倒计时；否则在实验进行中右上角显示倒计时
        if self._cooldown_remaining_s > 0.0:
            self._render_center_overlay(display, f"倒计时: {self._fmt_mm_ss(self._cooldown_remaining_s)}")
        else:
            if self._center_instruction_text and self._center_instruction_seconds_left > 0.0:
                self._render_center_overlay(display, self._center_instruction_text)
            if self._experiment_countdown_active and self._experiment_remaining_s > 0.0:
                self._render_top_right_countdown(display, f"倒计时: {self._fmt_mm_ss(self._experiment_remaining_s)}")
        self.help.render(display)


class FadingText:
    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(min(255, int(127.5 * self.seconds_left)))

    def render(self, display):
        display.blit(self.surface, self.pos)


class HelpText:
    """帮助文本显示"""
    
    def __init__(self, font, width, height):
        self.font = font
        self.dim = (680, 680)
        self.pos = ((width - self.dim[0]) / 2, (height - self.dim[1]) / 2)
        self._visible = False
        self._render()
        
    def toggle(self):
        self._visible = not self._visible
        
    def _render(self):
        lines = __doc__.split('\n')
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for i, line in enumerate(lines):
            text = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text, (4, i * 22))
            
    def render(self, display):
        if self._visible:
            display.blit(self.surface, self.pos)


# ==============================================================================
# -- 传感器 --------------------------------------------------------------------
# ==============================================================================

class CollisionSensor:
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)


class LaneInvasionSensor:
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self._parent = parent_actor
        self.hud = hud
        if parent_actor.type_id.startswith("vehicle."):
            world = self._parent.get_world()
            bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
            self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
        if not getattr(self.hud, 'show_lane_invasion_notification', True):
            return
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        self.hud.notification('压线: %s' % ' and '.join(text))


class GnssSensor:
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(bp, carla.Transform(carla.Location(x=1.0, z=2.8)), attach_to=self._parent)
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude


class IMUSensor:
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.accelerometer = (0.0, 0.0, 0.0)
        self.gyroscope = (0.0, 0.0, 0.0)
        self.compass = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.imu')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda sensor_data: IMUSensor._IMU_callback(weak_self, sensor_data))

    @staticmethod
    def _IMU_callback(weak_self, sensor_data):
        self = weak_self()
        if not self:
            return
        limits = (-99.9, 99.9)
        self.accelerometer = (
            max(limits[0], min(limits[1], sensor_data.accelerometer.x)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.y)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.z)))
        self.gyroscope = (
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.x))),
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.y))),
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.z))))
        self.compass = math.degrees(sensor_data.compass)


class RadarSensor:
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        bound_x = 0.5 + self._parent.bounding_box.extent.x
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        bound_z = 0.5 + self._parent.bounding_box.extent.z
        
        self.velocity_range = 7.5
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.radar')
        bp.set_attribute('horizontal_fov', str(35))
        bp.set_attribute('vertical_fov', str(20))
        self.sensor = world.spawn_actor(
            bp,
            carla.Transform(carla.Location(x=bound_x + 0.05, z=bound_z + 0.05), carla.Rotation(pitch=5)),
            attach_to=self._parent)
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda radar_data: RadarSensor._Radar_callback(weak_self, radar_data))

    @staticmethod
    def _Radar_callback(weak_self, radar_data):
        self = weak_self()
        if not self:
            return
        current_rot = radar_data.transform.rotation
        for detect in radar_data:
            azi = math.degrees(detect.azimuth)
            alt = math.degrees(detect.altitude)
            fw_vec = carla.Vector3D(x=detect.depth - 0.25)
            carla.Transform(
                carla.Location(),
                carla.Rotation(pitch=current_rot.pitch + alt, yaw=current_rot.yaw + azi, roll=current_rot.roll)
            ).transform(fw_vec)
            
            norm_velocity = detect.velocity / self.velocity_range
            r = int(max(0.0, min(1.0, 1.0 - norm_velocity)) * 255.0)
            g = int(max(0.0, min(1.0, 1.0 - abs(norm_velocity))) * 255.0)
            b = int(max(0.0, min(1.0, min(0.0, norm_velocity) + 1.0)) * 255.0)
            self._parent.get_world().debug.draw_point(
                radar_data.transform.location + fw_vec,
                size=0.075,
                life_time=0.06,
                persistent_lines=False,
                color=carla.Color(r, g, b))


class CameraManager:
    def __init__(self, parent_actor, hud, gamma_correction):
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        self._camera_recording_path = None
        
        bound_x = 0.5 + self._parent.bounding_box.extent.x
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        bound_z = 0.5 + self._parent.bounding_box.extent.z
        Attachment = carla.AttachmentType

        self._camera_transforms = [
            # 驾驶员视角
            (carla.Transform(carla.Location(x=0.5, z=1.4)), Attachment.Rigid),
            # 第三人称
            (carla.Transform(carla.Location(x=-5.0, z=3.0), carla.Rotation(pitch=-15)), Attachment.SpringArmGhost),
            # 俯视
            (carla.Transform(carla.Location(x=-2.0, z=15.0), carla.Rotation(pitch=-70)), Attachment.SpringArmGhost),
            # 前方远视
            (carla.Transform(carla.Location(x=bound_x + 10, z=3.0), carla.Rotation(pitch=-10, yaw=180)), Attachment.Rigid),
        ]

        self.transform_index = 0
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB', {}],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)', {}],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)', {}],
            ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic)', {}],
            ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)', {}],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette, 'Camera Semantic Segmentation (CityScapes)', {}],
        ]
        
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            bp = bp_library.find(item[0])
            bp.set_attribute('image_size_x', str(hud.dim[0]))
            bp.set_attribute('image_size_y', str(hud.dim[1]))
            if bp.has_attribute('gamma'):
                bp.set_attribute('gamma', str(gamma_correction))
            item.append(bp)
        self.index = None

    def toggle_camera(self):
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.set_sensor(self.index, notify=False, force_respawn=True)

    def next_sensor(self):
        self.set_sensor((self.index + 1) % len(self.sensors))

    def set_sensor(self, index, notify=True, force_respawn=False):
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None else (force_respawn or (self.sensors[index][2] != self.sensors[self.index][2]))
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index][0],
                attach_to=self._parent,
                attachment_type=self._camera_transforms[self.transform_index][1])
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def toggle_recording(self):
        self.recording = not self.recording
        if self.recording:
            self._camera_recording_path = f'_camera_out/{datetime.now().strftime("%Y%m%d_%H%M%S")}'
            os.makedirs(self._camera_recording_path, exist_ok=True)
        self.hud.notification('录制 %s' % ('开启' if self.recording else '关闭'))

    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))
        else:
            # 调试: surface 为 None
            if not hasattr(self, '_surface_warn_shown'):
                print("[CameraManager.render] 警告: surface 为 None，等待图像...")
                self._surface_warn_shown = True

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        image.convert(self.sensors[self.index][1])
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        # 调试: 每100帧打印一次
        if not hasattr(self, '_frame_count'):
            self._frame_count = 0
        self._frame_count += 1
        if self._frame_count == 1:
            print(f"[Camera] 收到第一帧图像 {image.width}x{image.height}")
        if self.recording:
            image.save_to_disk(f'{self._camera_recording_path}/{image.frame:08d}')


# ==============================================================================
# -- 主循环 --------------------------------------------------------------------
# ==============================================================================

def game_loop(args):
    pygame.init()
    pygame.font.init()
    world = None
    original_settings = None
    sim_world = None

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(60.0)

        def _get_world_with_retry(max_tries=6, per_try_timeout=10.0):
            """等待CARLA Simulator就绪（Editor切图/首次启动可能较慢）"""
            last_err = None
            for i in range(1, max_tries + 1):
                try:
                    client.set_timeout(per_try_timeout)
                    return client.get_world()
                except RuntimeError as e:
                    last_err = e
                    msg = str(e)
                    if 'time-out' in msg or 'time out' in msg or 'timeout' in msg:
                        print(f"等待CARLA就绪... ({i}/{max_tries})")
                        time.sleep(1.0)
                        continue
                    raise
            raise last_err

        # 加载地图
        opendrive_file = None
        if args.straight_road:
            # 使用内置直道地图
            opendrive_file = os.path.join(os.path.dirname(__file__), 'maps', 'straight_road.xodr')
            print(f"加载内置直道地图: {opendrive_file}")
        elif args.opendrive:
            opendrive_file = args.opendrive
            print(f"加载自定义OpenDRIVE地图: {opendrive_file}")
        
        if opendrive_file:
            # 加载 OpenDRIVE 地图
            if not os.path.exists(opendrive_file):
                print(f"错误: 找不到地图文件 {opendrive_file}")
                return
            with open(opendrive_file, 'r', encoding='utf-8') as f:
                opendrive_content = f.read()
            
            # OpenDRIVE 参数
            vertex_distance = 2.0  # 顶点间距(米)
            max_road_length = 500.0  # 最大道路网格长度(米)
            wall_height = 1.0  # 边界墙高度
            extra_width = 0.6  # 额外道路宽度
            
            sim_world = client.generate_opendrive_world(
                opendrive_content,
                carla.OpendriveGenerationParameters(
                    vertex_distance=vertex_distance,
                    max_road_length=max_road_length,
                    wall_height=wall_height,
                    additional_width=extra_width,
                    smooth_junctions=True,
                    enable_mesh_visibility=True
                )
            )
            print("=" * 50)
            print("直道地图加载成功！")
            print("  - 长度: 15公里（足够10分钟实验）")
            print("  - 双向四车道，完全笔直")
            print("=" * 50)
        else:
            if args.map:
                print(f"加载地图: {args.map}")
                try:
                    sim_world = client.load_world(args.map)
                except RuntimeError as e:
                    msg = str(e).lower()
                    if 'map not found' in msg or 'not found' in msg:
                        sim_world = _get_world_with_retry()
                        try:
                            current_map_name = sim_world.get_map().name
                        except Exception:
                            current_map_name = '(unknown)'
                        print("警告: client.load_world() 找不到该地图，将使用当前已加载地图继续运行。")
                        print(f"  请求地图: {args.map}")
                        print(f"  当前地图: {current_map_name}")
                    else:
                        raise
            else:
                sim_world = _get_world_with_retry()
                try:
                    current_map_name = sim_world.get_map().name
                except Exception:
                    current_map_name = '(unknown)'
                print(f"使用当前已加载地图: {current_map_name}")
        
        # 列出生成点（如果请求）
        if args.list_spawns:
            carla_map = sim_world.get_map()
            spawn_points = carla_map.get_spawn_points()
            print(f"\n=== {args.map} 生成点扫描 ({len(spawn_points)} 个) ===")
            print("正在测量各生成点前方直道长度...")
            print("-" * 80)
            
            results = []
            for i, sp in enumerate(spawn_points):
                waypoint = carla_map.get_waypoint(sp.location)
                if waypoint is None:
                    continue
                
                # 测量直道长度
                straight_len = 0
                initial_yaw = waypoint.transform.rotation.yaw
                current_wp = waypoint
                
                for _ in range(400):  # 最多检测 2000m
                    next_wps = current_wp.next(5.0)
                    if not next_wps:
                        break
                    next_wp = next_wps[0]
                    yaw_diff = abs(next_wp.transform.rotation.yaw - initial_yaw)
                    if yaw_diff > 180:
                        yaw_diff = 360 - yaw_diff
                    if yaw_diff > 10:  # 曲率阈值 10 度
                        break
                    straight_len += 5.0
                    current_wp = next_wp
                
                results.append({
                    'idx': i, 'len': straight_len,
                    'x': sp.location.x, 'y': sp.location.y,
                    'road': waypoint.road_id, 'lane': waypoint.lane_id
                })
            
            # 按直道长度排序
            results.sort(key=lambda x: x['len'], reverse=True)
            
            print("\n*** 最长直道 TOP 15 ***\n")
            for r in results[:15]:
                print(f"  生成点 [{r['idx']:3d}]: {r['len']:6.0f}m 直道 | "
                      f"位置: ({r['x']:7.1f}, {r['y']:7.1f}) | 道路: {r['road']}")
            
            print("\n" + "-" * 80)
            if results:
                best = results[0]
                print(f"推荐: python car_following_experiment.py --keyboard --spawn-point {best['idx']}")
                print(f"      (直道长度: {best['len']:.0f}m)")
            print("-" * 80)
            return

        # 同步模式
        if args.sync:
            original_settings = sim_world.get_settings()
            settings = sim_world.get_settings()
            if not settings.synchronous_mode:
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = 0.05
            sim_world.apply_settings(settings)
            
            traffic_manager = client.get_trafficmanager()
            traffic_manager.set_synchronous_mode(True)
            print("同步模式已启用")

        # 创建显示 - 支持多显示器
        display_flags = pygame.HWSURFACE | pygame.DOUBLEBUF
        if args.fullscreen:
            display_flags |= pygame.FULLSCREEN
            
        display = pygame.display.set_mode(
            (args.width, args.height),
            display_flags,
            display=args.display)
        display.fill((0, 0, 0))
        pygame.display.flip()
        pygame.display.set_caption("跟驰实验 - 主视角")

        hud = HUD(args.width, args.height)
        hud.show_lane_invasion_notification = bool(getattr(args, 'show_lane_invasion_notification', False))
        world = World(sim_world, hud, args)
        controller = VehicleController(world, args)

        if args.sync:
            sim_world.tick()
        else:
            sim_world.wait_for_tick()

        clock = pygame.time.Clock()
        
        # 显示操作说明
        print("\n" + "=" * 60)
        print("跟驰实验操作说明 (重构版)")
        print("=" * 60)
        print("  W/↑       : 油门          S/↓      : 制动")
        print("  A/D       : 转向          Q        : 切换倒档")
        print("  Space     : 手刹          M        : 手动/自动变速")
        print("  P         : CARLA自动驾驶 F        : Ackermann控制")
        print("  ")
        print("  L         : 切换车灯      Shift+L  : 远光灯")
        print("  Z/X       : 左/右转向灯   I        : 内饰灯")
        print("  ")
        print("  F1        : 开始/停止数据采集")
        print("  F2        : 切换手动/跟驰自动驾驶")
        print("  F3        : 切换前车行为(恒速/变速, 仅跟驰实验)")
        print("  F4        : 切换下一实验并重启")
        print("  F5~F10    : 直接切换实验1~6并重启(不足6组时按键会提示)")
        print("  TAB       : 切换相机视角")
        print("  C         : 切换天气      V        : 切换地图层")
        print("  R         : 录制图像      Ctrl+R   : 录制仿真")
        print("  Ctrl+P    : 回放录制")
        print("  H         : 显示帮助")
        print("  Backspace : 重启场景      ESC      : 退出")
        print("=" * 60)
        print(f"\n输入模式: {args.input_mode}")
        print(f"显示器: {args.display}")
        print(f"压线提示显示: {'开启' if hud.show_lane_invasion_notification else '关闭'}")
        print(f"生成右移偏移(--spawn-right-offset): {args.spawn_right_offset:.2f} m")
        ls_ms = args.lead_speed
        print(f"前车基准速度(--lead-speed): {ls_ms:.2f} m/s ({ls_ms * 3.6:.1f} km/h)")
        if args.enable_experiment_mode:
            if args.experiment_scope == 'following':
                print("实验计划: 仅跟驰(不规则变速) 共1组，每次重启跑完整6km")
            elif args.experiment_scope == 'overtaking':
                print("实验计划: 仅超车(35/50/65 km/h) 共3组，每次重启跑完整6km")
            else:
                print("实验计划: 1组跟驰(不规则变速) + 3组超车(35/50/65 km/h)")
            print(f"实验时长: 跟驰 {args.following_experiment_duration_s:.0f}s, 超车 {args.overtaking_experiment_duration_s:.0f}s")
            print(f"当前实验: {world._get_experiment_label()}")
            eff = world._get_effective_lead_speed()
            print(
                f"当前实验前车目标速度: {eff:.2f} m/s ({eff * 3.6:.1f} km/h) "
                f"(跟驰段用上面基准速度; 超车段为实验设定)"
            )
        if args.input_mode == 'cabin' and float(getattr(args, 'cabin_echo_interval', 0) or 0) > 0:
            ce = float(args.cabin_echo_interval)
            print(f"驾驶舱回传打印: 每 {ce:.2f}s 一行（关闭: --cabin-echo-interval 0）")
        print()
        

        # 主循环
        loop_count = 0
        while True:
            if args.sync:
                sim_world.tick()
            clock.tick_busy_loop(30)
            
            # 调试: 每100帧报告状态
            loop_count += 1
            if loop_count == 1:
                print(f"[MainLoop] 进入主循环")
            if loop_count == 30:
                cam_surface = world.camera_manager.surface if world.camera_manager else None
                print(f"[MainLoop] 30帧后: camera_manager={world.camera_manager is not None}, surface={cam_surface is not None}")
            
            # 检查车辆是否存在
            if world.player is None:
                print("错误: 车辆不存在，尝试重新生成...")
                world.restart()
                if world.player is None:
                    print("无法生成车辆，退出程序")
                    return
            
            if controller.parse_events(client, world, clock, args.sync):
                return
                
            world.tick(clock)
            world.render(display)
            pygame.display.flip()

    finally:
        if original_settings and sim_world:
            sim_world.apply_settings(original_settings)

        if world is not None:
            if world.data_collector.is_collecting:
                sim_time = world.hud.simulation_time if hasattr(world.hud, 'simulation_time') else None
                world.data_collector.stop(world_end_time_s=sim_time)
            world.destroy()

        pygame.quit()


# ==============================================================================
# -- 入口 ----------------------------------------------------------------------
# ==============================================================================

def main():
    argparser = argparse.ArgumentParser(description='跟驰实验CARLA仿真 (重构版)')
    
    # CARLA连接
    argparser.add_argument('--host', default='127.0.0.1', help='CARLA服务器IP')
    argparser.add_argument('-p', '--port', default=2000, type=int, help='CARLA服务器端口')
    
    # 显示设置
    argparser.add_argument('--res', default='1280x720', help='窗口分辨率')
    argparser.add_argument('--gamma', default=2.2, type=float, help='Gamma校正')
    argparser.add_argument('--display', default=0, type=int, help='显示器编号 (0, 1, 2...)')
    argparser.add_argument('--fullscreen', action='store_true', help='全屏模式')
    argparser.add_argument('--show-lane-invasion-notification', action='store_true',
                          help='显示屏幕下方压线提示（默认关闭）')
    
    # 车辆设置
    argparser.add_argument('--filter', default='vehicle.audi.tt', help='自车蓝图')
    argparser.add_argument('--generation', default='2', help='车辆代次')
    argparser.add_argument('--rolename', default='hero', help='自车角色名')
    
    # 前车设置
    argparser.add_argument('--lead-speed', default=20.0, type=float, help='前车基准速度(m/s)')
    argparser.add_argument('--lead-random', action='store_true', help='前车使用随机速度曲线')
    argparser.add_argument('--lead-seed', default=None, type=int, help='随机种子（用于复现）')
    argparser.add_argument('--straight-drive', action='store_true', 
                          help='前车直线行驶模式（不跟随道路弯曲）')
    argparser.add_argument('--auto-reset', action='store_true',
                          help='接近弯道时自动重置到直道起点')
    argparser.add_argument('--no-straight-protect', action='store_true',
                          help='禁用直道保护警告')
    
    # 地图设置 - 默认使用CARLA当前已加载的地图（适配自编地图/Editor启动）
    argparser.add_argument('--map', default=None,
                          help='CARLA地图名 (不指定则使用当前已加载地图，例如 --map Town04)')
    argparser.add_argument('--opendrive', default=None, type=str,
                          help='加载自定义OpenDRIVE地图文件 (.xodr)（实验性）')
    argparser.add_argument('--straight-road', action='store_true', default=False,
                          help='使用自定义直道地图（实验性，可能不稳定）')
    argparser.add_argument('--spawn-point', default=None, type=int, 
                          help='生成点索引 (Town04高速直道推荐: 1, 33, 65, 97)')
    argparser.add_argument('--list-spawns', action='store_true', 
                          help='列出所有生成点后退出')
    argparser.add_argument('--lead-distance', default=75.0, type=float,
                          help='前车初始距离 (米)，默认75m')
    argparser.add_argument('--spawn-right-offset', default=2.5, type=float,
                          help='自车/前车生成点向右平移(米)，例如 3.5 表示右移一个车道宽')
    argparser.add_argument('--four-experiments', dest='four_experiments', action='store_true',
                          help='启用4组实验切换(1组跟驰不规则 + 3组超车35/50/65 km/h)')
    # 兼容旧参数名（内部统一使用 four_experiments）
    argparser.add_argument('--six-experiments', dest='four_experiments', action='store_true', help=argparse.SUPPRESS)
    argparser.add_argument('--experiment-scope', default='all', choices=['all', 'following', 'overtaking'],
                          help='实验范围：all=跟驰+超车，following=仅跟驰，overtaking=仅超车')
    argparser.add_argument('--following-experiment-duration-s', default=180.0, type=float,
                          help='跟驰实验时长（秒），默认180')
    argparser.add_argument('--overtaking-experiment-duration-s', default=120.0, type=float,
                          help='超车实验时长（秒），默认120')
    argparser.add_argument('--experiment-cooldown-s', default=10.0, type=float,
                          help='4组实验中：每次实验开始前冷却时间(秒)，冷却期前车速度保持0（默认10）')
    argparser.add_argument('--experiment-start-x', default=EXPERIMENT_START_X, type=float,
                          help='4组实验重启时的起点X坐标')
    argparser.add_argument('--experiment-start-y', default=EXPERIMENT_START_Y, type=float,
                          help='4组实验重启时的起点Y坐标')
    
    # 模式设置
    argparser.add_argument('--sync', action='store_true', default=True, help='启用同步模式')
    argparser.add_argument('--no-sync', dest='sync', action='store_false', help='禁用同步模式')
    argparser.add_argument('--keyboard', action='store_true', help='键盘控制模式')
    argparser.add_argument('--cabin', action='store_true', help='驾驶舱控制模式')
    
    # 驾驶舱通信设置
    argparser.add_argument('--cabin-ip', default=CABIN_IP, help='驾驶舱IP地址')
    argparser.add_argument('--cabin-port', default=CABIN_PORT, type=int, help='驾驶舱端口')
    argparser.add_argument(
        '--cabin-echo-interval',
        default=None,
        type=float,
        metavar='N',
        help='终端打印驾驶舱 UDP 回传间隔(秒)，须填数字，例如 0.25；0=关闭。默认 cabin=0.25、键盘=0',
    )
    
    args = argparser.parse_args()
    
    # 解析分辨率
    args.width, args.height = [int(x) for x in args.res.split('x')]
    
    # 确定输入模式
    if args.cabin:
        args.input_mode = 'cabin'
    else:
        args.input_mode = 'keyboard'

    # 直道实验默认启用；也可显式 --four-experiments 开启
    args.enable_experiment_mode = args.four_experiments or args.straight_road or bool(args.opendrive)
    
    if args.cabin_echo_interval is None:
        args.cabin_echo_interval = 0.25 if args.input_mode == 'cabin' else 0.0
    
    print(__doc__)
    
    try:
        game_loop(args)
    except KeyboardInterrupt:
        print('\n用户中断')


if __name__ == '__main__':
    main()