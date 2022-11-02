from statistics import mean
from tqdm import tqdm
import cv2
import numpy as np
from PIL import Image, ImageDraw

from detector import pose_estimation

from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


class Visualizer:
    def __init__(self):
        self.img = None
        self.poses = None
        self.curr_frame = None
        self.x_traces = {}
        self.y_traces = {}
        self.t_traces = {}
        self.colors = plt.get_cmap("tab20")
        plt.ion()

    def update(self, img, poses, frame_idx):
        self.img = img
        self.poses = poses
        self.curr_frame = frame_idx

    def update_traces(self):
        # First, cut traces of poses that disappeared
        for id in list(self.x_traces):
            # dict id not in current pose list
            if id not in [pose.id for pose in self.poses]:
                if self.t_traces[id][-1] == self.curr_frame-1:
                    tqdm.write(f"id {id} disappeared from tracking area.")
                #self.x_traces[id] = self.x_traces[id].append(None)
                #self.y_traces[id] = self.y_traces[id].append(None)
        # Second, add new poses to existing traces or create new trace
        for pose in self.poses:
            # id not in list
            if self.x_traces.get(pose.id) is None:
                tqdm.write(f"id {pose.id} entered tracking area.")
                self.x_traces[pose.id]=[(pose.keypoints[0])[0]]
                self.y_traces[pose.id]=[(pose.keypoints[0])[1]]
                self.t_traces[pose.id] = [self.curr_frame]
            # id in list
            else:
                #print(f"id {pose.id} continues to be in tracking area.")
                self.x_traces[pose.id].append(
                    (pose.keypoints[0])[0])
                self.y_traces[pose.id].append(
                    (pose.keypoints[0])[1])
                self.t_traces[pose.id].append(self.curr_frame)

    def cut_traces(self):
        for id in list(self.x_traces):
            if len(self.x_traces[id]) > 100:
                tqdm.write(f"shortening trace of id {id}.")
                self.x_traces[id] = self.x_traces.get(id)[:100]
                self.y_traces[id] = self.y_traces.get(id)[:100]
                self.t_traces[id] = self.t_traces.get(id)[:100]

    def clean_traces(self):
        for id in list(self.x_traces):
            if all(val is None for val in self.x_traces.get(id)):
                tqdm.write(f"removing trace of id {id}, "
                      f"who disappeared 100 frames ago.")
                del self.x_traces[id]
                del self.y_traces[id]
                del self.t_traces[id]

    def create_plot(self):
        plt.ion()
        self.update_traces()
        self.cut_traces()
        self.clean_traces()
        fig = plt.figure(num='diid2')
        ax = plt.axes(projection='3d')
        ax.yaxis.set_ticklabels([])
        ax.zaxis.set_ticklabels([])
        for line in ax.yaxis.get_ticklines():
            line.set_visible(False)
        for line in ax.zaxis.get_ticklines():
            line.set_visible(False)
        # Data for a three-dimensional line

        for id in list(self.x_traces):
            ax.plot(self.t_traces.get(id),
                    self.x_traces.get(id),
                    self.y_traces.get(id))
        plt.draw()
        plt.pause(0.0001)



    """
    if self.sc is None:
            self.update_traces()
            self.cut_traces()
            self.clean_traces()
            for id in list(self.x_traces):
                self.fig = plt.figure(num='diid2')
                self.ax = plt.axes(projection='3d')
                self.sc = self.ax.scatter(self.t_traces.get(id),
                                       self.x_traces.get(id),
                                       self.y_traces.get(id),
                                       label=f'id {id}')
        else:
            self.update_traces()
            self.cut_traces()
            self.clean_traces()
            for id in list(self.x_traces):

                self.sc._offsets3d = (self.t_traces.get(id),
                                      self.x_traces.get(id),
                                      self.y_traces.get(id))
        return self.fig
    """

    def draw_bounding_boxes(self, track):
        for pose in self.poses:
            cv2.rectangle(
                self.img,
                (pose.bbox[0], pose.bbox[1]),
                (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]),
                (0, 255, 0),
            )
            if track:
                cv2.putText(
                    self.img,
                    "diid: {}".format(pose.id),
                    (pose.bbox[0], pose.bbox[1] - 16),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.5,
                    (0, 0, 255),
                )

    def skeleton_overlay(self):
        skeleton_keypoint_pairs = (
            pose_estimation.PoseEstimator.skeleton_keypoint_pairs
        )
        for pose in self.poses:
            colors = []
            if self.relevant_poses is None or pose not in self.relevant_poses:
                colors = [
                    [0, 0, 0] for _ in range(len(skeleton_keypoint_pairs))
                ]
            else:
                for key, value in self.synchrony.items():
                    color = [0,0,0]
                    colors.append(color)
            pose.draw(self.img, colors)

    def text_overlay(self):
        if len(self.poses) < 2:
            overlay_text = f"Avg synch: {np.nan}; Dist: {np.nan} px"
            dash = self.overlay_dashboard(
                overlay_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
            )
        else:
            synch_vals_avail = [
                val for val in self.synchrony.values() if val != np.nan
            ]
            synch_mean = (
                mean(synch_vals_avail) if len(synch_vals_avail) != 0 else False
            )
            dist_temp = self.distance
            text_dist = np.nan if dist_temp == -1 else f"{dist_temp:.1f}px"
            text_synch = np.nan if synch_mean is False else f"{synch_mean:.1f}"
            overlay_text = f"Avg synch: {text_synch}; Dist: {text_dist}"
            dash = self.overlay_dashboard(
                overlay_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
            )

        x_offset = y_offset = 10
        y_start = y_offset
        y_stop = y_offset + dash.shape[0]
        x_start = x_offset
        x_stop = x_offset + dash.shape[1]
        self.img[y_start:y_stop, x_start:x_stop] = dash

    def round_rectangle(self, size, radius, fill):
        """Draw a rounded rectangle"""
        width, height = size
        rectangle = Image.new("RGB", size, fill)
        corner = self.round_corner(radius, fill)
        rectangle.paste(corner, (0, 0))
        rectangle.paste(
            corner.rotate(90), (0, height - radius)
        )  # Rotate the corner and paste it
        rectangle.paste(corner.rotate(180), (width - radius, height - radius))
        rectangle.paste(corner.rotate(270), (width - radius, 0))
        return rectangle

    @staticmethod
    def round_corner(radius, fill):
        """Draw a round corner"""
        corner = Image.new("RGB", (radius, radius), (0, 0, 0, 0))
        draw = ImageDraw.Draw(corner)
        draw.pieslice((0, 0, radius * 2, radius * 2), 180, 270, fill=fill)
        return corner

    @staticmethod
    def draw_text(
        img,
        text,
        font=cv2.FONT_HERSHEY_PLAIN,
        pos=(0, 0),
        font_scale=3,
        font_thickness=2,
        text_color=(0, 255, 0),
        text_color_bg=(0, 0, 0),
    ):
        x, y = pos
        text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_w, text_h = text_size
        cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
        cv2.putText(
            img,
            text,
            (x, y + text_h + font_scale - 1),
            font,
            font_scale,
            text_color,
            font_thickness,
        )

        return text_size

    def overlay_dashboard(self, text, font, font_scale, font_thickness):
        textsize = cv2.getTextSize(text, font, 1, 2)[0]
        background = self.round_rectangle(
            (textsize[0], textsize[1] * 2), 10, "white"
        )
        dash = np.array(background)
        # Convert RGB to BGR
        dash = dash[:, :, ::-1].copy()
        # get coords based on boundary
        textX = (dash.shape[1] - textsize[0]) / 2
        textY = (dash.shape[0] + textsize[1]) / 2

        # add text centered on image
        cv2.putText(dash, text, (int(textX), int(textY)), font, 1, (0, 0, 0), 2)

        return dash
