
from configuration import get_cfg

cfg = get_cfg()



# def cmap_to_bgr(color):
#     color = map(lambda x: x * 255, color)
#     color = map(int, color)
#     color = list(color)
#     color = color[0:3]
#     r = color[0]
#     g = color[1]
#     b = color[2]
#     return b, g, r


color_map_rgb = [
    [255,192,203], # 粉红
    [199,21,133],   # 适中的紫罗兰红色
    [0,191,255], # 深天蓝
    [255,69,0],   # 橙红色
    [255,255,0],  # 黄
    [144,238,144],  # 淡绿色
    [0,255,255],   # 青色
    [255, 0, 0],   # 红
]


def rgb2bgr(colors):
    return [[color[2], color[1], color[0]] for color in colors]


color_map_bgr = rgb2bgr(color_map_rgb)


class DrawConfig:
    joint_line_thickness = 5
    # cmap = cm.viridis
    # joints_norm = matplotlib.colors.Normalize(vmin=0, vmax=len(cfg.JOINTS_DEF))

    # joint_colors_bgr = {k: cmap_to_bgr(cmap(joints_norm(v["idx"]))) for k, v in cfg.JOINTS_DEF.items()}



    joint_colors_bgr = {k: color_map_bgr[v["idx"]%len(color_map_bgr)] for k, v in cfg.JOINTS_DEF.items()}


    keypoint_circle_diameter = 10
    DRAW_KEYPOINTS_TEXT = True
    keypoint_circle_color = (255, 255, 255)
    keypoint_text_color = (125, 125, 125)
    keypoint_text_thickness = 1
    keypoint_text_scale = 0.5

