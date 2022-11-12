from statistics import mean

import cv2
import numpy as np
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
from tqdm import tqdm
from detector import pose_estimation
from detector.synchrony_detection import SynchronyDetector
from random import randint
from PIL import ImageColor


class Visualizer:
    def __init__(self, trace_len):
        self.img = None
        self.poses = None
        self.curr_frame = None
        self.x_traces = {}
        self.y_traces = {}
        self.t_traces = {}
        #self.colors = plt.get_cmap("tab20")
        plt.ion()
        self.trace_len = trace_len
        self.last_new_appearance = None
        self.total_count = 0
        self.colors = []
        n = 100
        for i in range(n):
            self.colors.append('#%06X' % randint(0, 0xFFFFFF))

    def update(self, img, poses, frame_idx):
        self.img = img
        self.poses = poses
        self.curr_frame = frame_idx
        self.update_traces()
        self.cut_traces()
        # self.clean_traces()

    def update_traces(self):
        # First, cut traces of poses that disappeared
        for id in list(self.x_traces):
            # dict id not in current pose list
            if id not in [pose.id for pose in self.poses]:
                if self.t_traces[id][-1] == self.curr_frame - 1:
                    tqdm.write(f"id {id} disappeared from tracking area.")
                    del self.x_traces[id]
                    del self.y_traces[id]
                    del self.t_traces[id]
                # self.x_traces[id] = self.x_traces[id].append(None)
                # self.y_traces[id] = self.y_traces[id].append(None)
        # Second, add new poses to existing traces or create new trace
        for pose in self.poses:
            # id not in list
            if self.x_traces.get(pose.id) is None:
                self.total_count += 1
                self.last_new_appearance = self.curr_frame
                tqdm.write(f"id {pose.id} entered tracking area.")
                if (pose.keypoints[4])[0] == -1 or (
                        pose.keypoints[4])[1] == -1:
                    print("new appearance -1 case, len trace == 0")
                    continue
                else:
                    self.x_traces[pose.id] = [(pose.keypoints[4])[0]]
                    self.y_traces[pose.id] = [(pose.keypoints[4])[1]]
                    self.t_traces[pose.id] = [self.curr_frame]
            # id in list
            else:
                # print(f"id {pose.id} continues to be in tracking area.")
                if (pose.keypoints[4])[0] == -1 or (
                        pose.keypoints[4])[1] == -1:
                    print("-1 case, len trace > 0")
                    self.x_traces[pose.id].append(self.x_traces[pose.id][-1])
                    self.y_traces[pose.id].append(self.y_traces[pose.id][-1])
                else:
                    self.x_traces[pose.id].append((pose.keypoints[4])[0])
                    self.y_traces[pose.id].append((pose.keypoints[4])[1])
                self.t_traces[pose.id].append(self.curr_frame)

    def cut_traces(self):
        for id in list(self.x_traces):
            if len(self.x_traces[id]) > self.trace_len:
                # tqdm.write(f"shortening trace of id {id}.")
                del self.x_traces.get(id)[:1]
                del self.y_traces.get(id)[:1]
                del self.t_traces.get(id)[:1]

    def create_plot(self):
        if self.curr_frame % self.trace_len in range(10):
            self.img = self.img
        else:
            self.img = np.full_like(self.img, 0)
        """
        if self.last_new_appearance is None or self.last_new_appearance == 0:
            self.img = np.full_like(self.img, 0)
        elif self.curr_frame % self.last_new_appearance in range(10):
            self.img = self.img
        else:
            self.img = np.full_like(self.img, 0)
        """
        for id in list(self.x_traces):
            if id < len(self.colors):
                col = ImageColor.getcolor(self.colors[id],"RGB")
            else:
                col = ImageColor.getcolor(self.colors[id % len(self.colors)]
                                          ,"RGB")

            for point_idx in range(len(list(self.x_traces.get(id))) - 1):
                cv2.line(
                    self.img,
                    (int(self.x_traces.get(id)[point_idx]),
                     int(self.y_traces.get(id)[point_idx])),
                    (int(self.x_traces.get(id)[point_idx + 1]),
                     int(self.y_traces.get(id)[point_idx + 1])),
                    col, 4
                )
            # [255, 0, 0]
            x_min = min(self.x_traces.get(id))
            x_max = max(self.x_traces.get(id))
            y_min = min(self.y_traces.get(id))
            y_max = max(self.y_traces.get(id))
            cv2.rectangle(
                self.img,
                (x_min, y_max),
                (x_max, y_min),
                [255, 255, 255],
                thickness=2
            )
            cv2.putText(
                self.img,
                "diid: {}".format(id),
                (x_min, y_max - 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                thickness=2
            )
        return self.img

    def fadeIn(self, img1, img2, x):  # pass images here to fade between
        fadein = x / 10
        dst = cv2.addWeighted(img1, 1 - fadein, img2, fadein, 0)
        return dst

    def counter_overlay(self):
        overlay_text = f"Individuals tracked: {self.total_count}"
        dash = self.overlay_dashboard(
            overlay_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        x_offset = y_offset = 10
        y_start = y_offset
        y_stop = y_offset + dash.shape[0]
        x_start = x_offset
        x_stop = x_offset + dash.shape[1]
        self.img[y_start:y_stop, x_start:x_stop] = dash

    def overlay_dashboard(self, text, font, font_scale, font_thickness):
        textsize = cv2.getTextSize(text, font, 1, 2)[0]
        background = self.round_rectangle(
            (textsize[0], textsize[1] * 2), 10, "black"
        )
        dash = np.array(background)
        # Convert RGB to BGR
        dash = dash[:, :, ::-1].copy()
        # get coords based on boundary
        textX = (dash.shape[1] - textsize[0]) / 2
        textY = (dash.shape[0] + textsize[1]) / 2

        # add text centered on image
        cv2.putText(dash, text, (int(textX), int(textY)), font, 1,
                    (255, 255, 255), 2)

        return dash

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
