
import cv2

from draw.draw_cfg import DrawConfig


class SkeletonDrawer:
    def __init__(self, img):
        self.img = img
        self.draw_config = DrawConfig

    def _scale_flip_coord(self, coord):
        y = coord[0]
        x = coord[1]
        scaled_y = int(y * self.img.shape[0])
        scaled_x = int(x * self.img.shape[1])
        return scaled_x, scaled_y

    def joint_draw(self, start_coord, end_coord, joint_name):
        start_coord = self._scale_flip_coord(start_coord)
        end_coord = self._scale_flip_coord(end_coord)

        color = self.draw_config.joint_colors_bgr[joint_name]
        cv2.line(img=self.img, pt1=start_coord, pt2=end_coord, color=color,
                 thickness=self.draw_config.joint_line_thickness, lineType=cv2.LINE_AA)

    def kpt_draw(self, kpt_coord, kpt_name):
        kpt_coord = self._scale_flip_coord(kpt_coord)

        cv2.circle(img=self.img, center=kpt_coord, radius=self.draw_config.keypoint_circle_diameter,
                   color=self.draw_config.keypoint_circle_color)

        if self.draw_config.DRAW_KEYPOINTS_TEXT:
            cv2.putText(img=self.img, text=kpt_name, org=kpt_coord, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=self.draw_config.keypoint_text_scale,
                        thickness=self.draw_config.keypoint_text_thickness,
                        color=self.draw_config.keypoint_text_color)