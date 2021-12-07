from itertools import cycle
from numpy.random import randint
from pygame import Rect, init, time, display
from pygame.event import pump
from pygame.image import load
from pygame.surfarray import array3d, pixels_alpha
from pygame.transform import rotate
import numpy as np


class Pipe(object):
    def __init__(self):
        self.pipe_images = [rotate(load('assets/sprites/pipe-green.png').convert_alpha(), 180),
                            load('assets/sprites/pipe-green.png').convert_alpha()]
        self.pipe_hitmask = [pixels_alpha(image).astype(bool)
                             for image in self.pipe_images]
        self.pipe_gap_size = 100
        self.pipe_velocity_x = -4
        self.x_upper = -1
        self.y_upper = -1
        self.x_lower = -1
        self.y_lower = -1

    def set_x_y(self, screen_width, base_y, pipe_height):
        x = screen_width + 10
        gap_y = randint(2, 10) * 10 + int(base_y / 5)
        self.x_upper = x
        self.y_upper = gap_y - pipe_height
        self.x_lower = x
        self.y_lower = gap_y + self.pipe_gap_size


    def get_width(self):
        return self.pipe_images[0].get_width()

    def get_height(self):
        return self.pipe_images[0].get_height()


class FlappyBird():

    def __init__(self):

        init()
        self.fps_clock = time.Clock()
        self.screen_width = 288
        self.screen_height = 512
        self.screen = display.set_mode((self.screen_width, self.screen_height))
        display.set_caption('Deep Q-Network Flappy Bird')
        self.base_image = load('assets/sprites/base.png').convert_alpha()
        self.background_image = load('assets/sprites/background-black.png').convert()

        self.bird_images = [load('assets/sprites/redbird-upflap.png').convert_alpha(),
                            load('assets/sprites/redbird-midflap.png').convert_alpha(),
                            load('assets/sprites/redbird-downflap.png').convert_alpha()]

        self.bird_hitmask = [pixels_alpha(image).astype(bool)
                             for image in self.bird_images]
        # pipe_hitmask = [pixels_alpha(image).astype(bool) for image in pipe_images]
        self.bird_index_generator = cycle([0, 1, 2, 1])
        self.iter = self.bird_index = self.score = 0

        self.bird_width = self.bird_images[0].get_width()
        self.bird_height = self.bird_images[0].get_height()

        self.bird_x = int(self.screen_width / 5)
        self.bird_y = int((self.screen_height - self.bird_height) / 2)

        self.base_x = 0
        self.base_y = self.screen_height * 0.79
        self.base_shift = self.base_image.get_width() - self.background_image.get_width()

        self.pipes = [Pipe(), Pipe()]
        # self.pipe_width = self.pipe_images[0].get_width()
        # self.pipe_height = self.pipe_images[0].get_height()
        self.pipe_width = self.pipes[0].get_width()
        self.pipe_height = self.pipes[0].get_height()
        # set up for the first 2 pipe
        self.pipes[0].x_upper = self.pipes[0].x_lower = self.screen_width
        self.pipes[1].x_upper = self.pipes[1].x_lower = self.screen_width * 1.5

        self.current_velocity_y = 0
        self.max_velocity_y = 10
        self.downward_speed = 1
        self.upward_speed = -9
        self.is_flapped = False

        self.fps = 30

    def is_collided(self):
        # Check if the bird touch ground
        if self.bird_height + self.bird_y + 1 >= self.base_y:
            return True
        bird_bbox = Rect(self.bird_x, self.bird_y,
                         self.bird_width, self.bird_height)
        pipe_boxes = []
        for pipe in self.pipes:
            pipe_boxes.append(
                Rect(pipe.x_upper, pipe.y_upper, self.pipe_width, self.pipe_height))
            pipe_boxes.append(
                Rect(pipe.x_lower, pipe.y_lower, self.pipe_width, self.pipe_height))
            # Check if the bird's bounding box overlaps to the bounding box of any pipe
            if bird_bbox.collidelist(pipe_boxes) == -1:
                return False
            for i in range(2):
                cropped_bbox = bird_bbox.clip(pipe_boxes[i])
                min_x1 = cropped_bbox.x - bird_bbox.x
                min_y1 = cropped_bbox.y - bird_bbox.y
                min_x2 = cropped_bbox.x - pipe_boxes[i].x
                min_y2 = cropped_bbox.y - pipe_boxes[i].y
                if np.any(self.bird_hitmask[self.bird_index][min_x1:min_x1 + cropped_bbox.width,
                          min_y1:min_y1 + cropped_bbox.height] * pipe.pipe_hitmask[i][
                                                                 min_x2:min_x2 + cropped_bbox.width,
                                                                 min_y2:min_y2 + cropped_bbox.height]):
                    return True
        return False

    def next_frame(self, action):
        pump()
        reward = 0.1
        terminal = False
        # Check input action
        if action == 1:
            self.current_velocity_y = self.upward_speed
            self.is_flapped = True

        # Update score
        bird_center_x = self.bird_x + self.bird_width / 2
        for pipe in self.pipes:
            pipe_center_x = pipe.x_upper + self.pipe_width / 2
            if pipe_center_x < bird_center_x < pipe_center_x + 5:
                self.score += 1
                reward = 1
                break

        # Update index and iteration
        if (self.iter + 1) % 3 == 0:
            self.bird_index = next(self.bird_index_generator)
            self.iter = 0
        self.base_x = -((-self.base_x + 100) % self.base_shift)

        # Update bird's position
        if self.current_velocity_y < self.max_velocity_y and not self.is_flapped:
            self.current_velocity_y += self.downward_speed
        if self.is_flapped:
            self.is_flapped = False
        self.bird_y += min(self.current_velocity_y, self.bird_y -
                           self.current_velocity_y - self.bird_height)
        if self.bird_y < 0:
            self.bird_y = 0

        # Update pipes' position
        for pipe in self.pipes:
            pipe.x_upper += pipe.pipe_velocity_x
            pipe.x_lower += pipe.pipe_velocity_x
        # Update pipes
        if 0 < self.pipes[0].x_lower < 5:
            new_pipe = Pipe()
            new_pipe.set_x_y(self.screen_width, self.base_y, self.pipe_height)
            self.pipes.append(new_pipe)
        if self.pipes[0].x_lower < -self.pipe_width:
            del self.pipes[0]
        if self.is_collided():
            terminal = True
            reward = -1
            self.__init__()

        # Draw everything
        self.screen.blit(self.background_image, (0, 0))
        self.screen.blit(self.base_image, (self.base_x, self.base_y))
        self.screen.blit(
            self.bird_images[self.bird_index], (self.bird_x, self.bird_y))
        for pipe in self.pipes:
            self.screen.blit(
                pipe.pipe_images[0], (pipe.x_upper, pipe.y_upper))
            self.screen.blit(
                pipe.pipe_images[1], (pipe.x_lower, pipe.y_lower))
        image = array3d(display.get_surface())
        display.update()
        self.fps_clock.tick(self.fps)
        return image, reward, terminal