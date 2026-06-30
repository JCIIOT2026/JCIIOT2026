import numpy as np
import mujoco
import time


# ============================================================================
# 方式一：从零搭建场景（程序化构建）
# ============================================================================

def build_scene_from_scratch():
    """
    完全从零构建一个仿真场景：
    - 一个 Panda 机器人 + PandaGripper 夹爪，放置在桌面上
    - 一个木色方块 + 一个红色球体
    - 使用 MuJoCo 原生接口直接运行仿真

    Step:
        1. 创建空白世界 (MujocoWorldBase)
        2. 实例化机器人模型 + 夹爪 → 合并入世界
        3. 创建工作台 (TableArena)
        4. 创建物体 (BoxObject / BallObject) → 附加入世界
        5. world.get_model() 生成 MjModel → 直接驱动仿真
    """
    from robosuite.models import MujocoWorldBase
    from robosuite.models.robots import Panda # 引入panda机器人，该机器人为固定底座，默认夹爪为PandaGripper，默认底座为RethinkMount
    from robosuite.models.grippers import gripper_factory, PandaGripper
    from robosuite.models.arenas import TableArena
    from robosuite.models.objects import BoxObject, BallObject # box对象与ball对象
    from robosuite.utils.mjcf_utils import CustomMaterial, xml_path_completion, array_to_string

    world = MujocoWorldBase() # 创建空白世界
    mujoco_robot=Panda() # 创建机器人
    gripper = PandaGripper() # 创建夹爪实例
    mujoco_robot.add_gripper(gripper,arm_name="robot0_right_hand") # 右手指定夹爪
    mujoco_robot.set_base_xpos([0, 0, 0]) # 设置机器人的初始化坐标
    world.merge(mujoco_robot) # 将机器人合并进世界

    # step3:创建工作台,该实例会创建一个桌子和地板平面
    arena = TableArena(
        table_full_size=(0.8, 0.8, 0.05),   # 桌面尺寸 (x, y, z)
        table_friction=(1.0, 0.005, 0.0001),
        table_offset=(0, 0, 0.82),           # 桌面在全局坐标中的偏移
    )
    arena.set_origin([0.16, 0, 0])           # 整体原点偏移
    world.merge(arena)

    # step4:添加对象
    tex_attrib = {
        "type": "cube",
        "file": xml_path_completion("textures/wood.png"),
    }

    mat_attrib = {
        "texture": "tex-wood",
        "texrepeat": "3 3",
        "specular": "0.4",
        "shininess": "0.1",
    }
    
    wood_material = CustomMaterial(
        texture="WoodDark",
        tex_name="tex-wood",
        mat_name="wood_mat",
        tex_attrib=tex_attrib,
        mat_attrib=mat_attrib,
    )

    box = BoxObject(
        name="box",
        size=[0.04, 0.04, 0.04],        # half-size: 8cm 边长方块
        rgba=[0.8, 0.6, 0.2, 1.0],      # 颜色
        material=wood_material,           # 贴木质纹理
    )

    world.merge_assets(box)
    box_obj = box.get_obj()
    box_obj.set("pos", array_to_string([0.0, -0.15, 1.05]))
    world.worldbody.append(box_obj)

    # 4.2 红色金属球
    sphere = BallObject(
        name="sphere",
        size=[0.04],                     # 半径 4cm
        rgba=[1.0, 0.2, 0.2, 1.0],      # 红色
        density=500,                      # 较重
        friction=(0.5, 0.3, 0.1),
    )
    world.merge_assets(sphere)
    sphere_obj = sphere.get_obj()
    sphere_obj.set("pos", array_to_string([0.0, 0.15, 1.05]))
    world.worldbody.append(sphere_obj)

    from robosuite.models.objects import CylinderObject
    cylinder = CylinderObject(
        name="cylinder",
        size=[0.03, 0.06],               # (半径, half-length)
        rgba=[0.2, 0.8, 0.3, 1.0],       # 绿色
    )

    world.merge_assets(cylinder)
    cyl_obj = cylinder.get_obj()
    cyl_obj.set("pos", array_to_string([0.15, 0.0, 1.05]))
    world.worldbody.append(cyl_obj)

    model = world.get_model(mode="mujoco")
    data = mujoco.MjData(model)
    print("[INFO] Running simulation for 3 seconds...")
    viewer = None
    try:
        # 尝试启动交互式查看器
        viewer = mujoco.viewer.launch_passive(model, data)
        viewer.cam.lookat[:] = [0.3, 0.0, 1.0]
        viewer.cam.distance = 1.8
    except Exception:
        print("[WARN] Could not launch viewer. Running headless...")
    
    for step in range(int(3.0 / model.opt.timestep)):
        mujoco.mj_step(model, data)
        if viewer is not None:
            viewer.sync()
            time.sleep(0.001)
    if viewer is not None:
        viewer.close()
    mujoco.glfw.glfw.terminate()
    print("[DONE] Simulation finished.\n")

if __name__=="__main__":
    build_scene_from_scratch()

