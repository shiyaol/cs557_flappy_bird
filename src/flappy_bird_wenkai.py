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
        self.pipe_images_mask = [pixels_alpha(image).astype(bool)
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
    init()
    def __init__(self):


        self.fps_clock = time.Clock()
        self.screen_width = 288
        self.screen_height = 512
        self.screen = display.set_mode((self.screen_width, self.screen_height))
        display.set_caption('Flappy Bird Demo created by Wenkai and Shiyao')
        self.base_image = load('assets/sprites/base.png').convert_alpha()
        self.background_image = load('assets/sprites/background-black.png').convert()

        self.bird_images = [load('assets/sprites/redbird-upflap.png').convert_alpha(),
                            load('assets/sprites/redbird-midflap.png').convert_alpha(),
                            load('assets/sprites/redbird-downflap.png').convert_alpha()]

        self.bird_images_mask = [pixels_alpha(image).astype(bool)
                             for image in self.bird_images]
        # pipe_images_mask = [pixels_alpha(image).astype(bool) for image in pipe_images]
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
        self.pipes[0].set_x_y(self.screen_width, self.base_y, self.pipe_height)
        self.pipes[1].set_x_y(self.screen_width, self.base_y, self.pipe_height)
        self.pipes[0].x_upper = self.pipes[0].x_lower = self.screen_width
        self.pipes[1].x_upper = self.pipes[1].x_lower = self.screen_width * 1.5

        self.current_velocity_y = 0
        self.max_velocity_y = 10
        self.downward_speed = 1
        self.upward_speed = -9
        self.flapped = False

        self.fps = 30

    def is_collided(self):
        # Check if the bird touch ground
        if self.bird_height + self.bird_y + 1 >= self.base_y:
            return True
        bird_rect = Rect(self.bird_x, self.bird_y,
                         self.bird_width, self.bird_height)
        pipe_coll = []
        for pipe in self.pipes:
            pipe_coll.append(
                Rect(pipe.x_upper, pipe.y_upper, self.pipe_width, self.pipe_height))
            pipe_coll.append(
                Rect(pipe.x_lower, pipe.y_lower, self.pipe_width, self.pipe_height))
             #Check if the bird's bounding box overlaps to the bounding box of any pipe
            if bird_rect.collidelist(pipe_coll) == -1:
                return False
            for i in range(2):
                rect = bird_rect.clip(pipe_coll[i])
                start_x1 = rect.x - bird_rect.x
                end_x1 = start_x1 + rect.width
                
                start_y1 = rect.y - bird_rect.y
                end_y1 = start_y1 + rect.height
                
                start_x2 = rect.x - pipe_coll[i].x
                end_x2 = start_x2 + rect.width
                
                start_y2 = rect.y - pipe_coll[i].y
                end_y2 = start_y2 + rect.height

                if np.any(self.bird_images_mask[self.bird_index][start_x1:end_x1,start_y1:end_y1] * pipe.pipe_images_mask[i][
                                                                 start_x2:end_x2,
                                                                 start_y2:end_y2]):
                    return True
        return False

    def update_score(self):
        bird_center_x = self.bird_x + self.bird_width / 2
        for pipe in self.pipes:
            pipe_center_x = pipe.x_upper + self.pipe_width / 2
            if pipe_center_x < bird_center_x and bird_center_x < pipe_center_x + 5:
                self.score += 1
                reward = 1
                break
    
    def update_bird_pos(self):
        # Update bird's position
        if not self.flapped and self.current_velocity_y < self.max_velocity_y:
            self.current_velocity_y += self.downward_speed
       
        self.bird_y += min(self.current_velocity_y, self.bird_y -
                           self.current_velocity_y - self.bird_height)

        if self.flapped:
            self.flapped = False   

        if self.bird_y < 0:
            self.bird_y = 0
    
    def update_pipe(self):
        # Update pipes' position
        for pipe in self.pipes:
            pipe.x_upper += pipe.pipe_velocity_x
            pipe.x_lower += pipe.pipe_velocity_x
        # Update pipes
        if 0 < self.pipes[0].x_lower and self.pipes[0].x_lower < 5:
            new_pipe = Pipe()
            new_pipe.set_x_y(self.screen_width, self.base_y, self.pipe_height)
            self.pipes.append(new_pipe)
        if self.pipes[0].x_lower < 0 - self.pipe_width:
            del self.pipes[0]

    def draw_image(self):
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
        return image

        
    def next_frame(self, action):
        pump()
        reward = 0.1
        terminal = False
        self.update_score()
        # Check input action
        if action == 1:
            self.flapped = True
            self.current_velocity_y = self.upward_speed


        # Update index and iteration
        if (self.iter + 1) % 3 == 0:
            self.bird_index = next(self.bird_index_generator)
            self.iter = 0
        self.base_x = -((100-self.base_x) % self.base_shift)

        self.update_bird_pos()
        self.update_pipe()
        # if collided we need to restart
        if self.is_collided():
            terminal = True
            reward = -1
            self.__init__()

        # Draw everything on pygame
        image = self.draw_image()
        return image, reward, terminal