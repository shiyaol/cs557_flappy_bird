from pygame import Rect, init, time, display
from pygame.event import pump
from pygame.image import load
from pygame.surfarray import array3d, pixels_alpha
from pygame.transform import rotate
from itertools import cycle
from numpy.random import randint
import numpy as np


class Pipe(object):
    def __init__(self):
        self.pipe_images = [rotate(load('assets/sprites/pipe-green.png').convert_alpha(), 180),
                            load('assets/sprites/pipe-green.png').convert_alpha()]
        self.pipe_images_mask = [pixels_alpha(image).astype(bool)
                             for image in self.pipe_images]
        self.gap_size = 100
        self.speed_x = -4
        self.upper_x = -1
        self.upper_y = -1
        self.lower_x = -1
        self.lower_y = -1

    def set_x_y(self, screen_width, base_y, pipe_height):
        x = 10 + screen_width
        pipe_gap_y = randint(2, 10) * 10 + int(base_y / 5)
        self.upper_x = x
        self.upper_y = pipe_gap_y - pipe_height
        self.lower_x = x
        self.lower_y = pipe_gap_y + self.gap_size


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
        self.ground_image = load('assets/sprites/base.png').convert_alpha()
        self.background_image = load('assets/sprites/background-black.png').convert()

        self.bird_images = [load('assets/sprites/redbird-upflap.png').convert_alpha(),
                            load('assets/sprites/redbird-midflap.png').convert_alpha(),
                            load('assets/sprites/redbird-downflap.png').convert_alpha()]

        self.bird_images_mask = [pixels_alpha(image).astype(bool)
                             for image in self.bird_images]
        # pipe_images_mask = [pixels_alpha(image).astype(bool) for image in pipe_images]
        self.index_generator = cycle([0, 1, 2, 1])
        self.iter = self.index = self.score = 0

        self.bird_width = self.bird_images[0].get_width()
        self.bird_height = self.bird_images[0].get_height()

        self.bird_x = int(self.screen_width / 5)
        self.bird_y = int((self.screen_height - self.bird_height) / 2)

        self.ground_x = 0
        self.ground_y = self.screen_height * 0.79
        self.ground_shift = self.ground_image.get_width() - self.background_image.get_width()

        self.pipes = [Pipe(), Pipe()]
        self.pipe_width = self.pipes[0].get_width()
        self.pipe_height = self.pipes[0].get_height()
    
        self.pipes[0].set_x_y(self.screen_width, self.ground_y, self.pipe_height)
        self.pipes[1].set_x_y(self.screen_width, self.ground_y, self.pipe_height)
        self.pipes[0].upper_x = self.pipes[0].lower_x = self.screen_width
        self.pipes[1].upper_x = self.pipes[1].lower_x = 1.5 * self.screen_width 

        self.cur_speed_y = 0
        self.down_speed = 1
        self.up_speed = -9
        self.flapped = False

        self.fps = 30

    def collided(self):
        # Check if the bird touch ground
        if self.bird_height + self.bird_y + 1 >= self.ground_y:
            return True
        bird_rect = Rect(self.bird_x, self.bird_y,
                         self.bird_width, self.bird_height)
        pipe_coll = []
        for pipe in self.pipes:
            pipe_coll.append(
                Rect(pipe.upper_x, pipe.upper_y, self.pipe_width, self.pipe_height))
            pipe_coll.append(
                Rect(pipe.lower_x, pipe.lower_y, self.pipe_width, self.pipe_height))
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

                if np.any(self.bird_images_mask[self.index][start_x1:end_x1,start_y1:end_y1] * pipe.pipe_images_mask[i][
                                                                 start_x2:end_x2,
                                                                 start_y2:end_y2]):
                    return True
        return False

    def update_score(self):
        center_x = self.bird_x + self.bird_width / 2
        for pipe in self.pipes:
            pipe_center_x = self.pipe_width / 2 + pipe.upper_x 
            if pipe_center_x < center_x and center_x < pipe_center_x + 5:
                self.score += 1
                reward = 1
                return reward
        return 0.1        
    
    def update_bird_pos(self):
    
        if not self.flapped and self.cur_speed_y < 10:
            self.cur_speed_y += self.down_speed
       
        self.bird_y += min(self.cur_speed_y, self.bird_y -
                           self.cur_speed_y - self.bird_height)

        if self.flapped:
            self.flapped = False   

        if self.bird_y < 0:
            self.bird_y = 0
    
    def update_pipe(self):

        for pipe in self.pipes:
            pipe.lower_x += pipe.speed_x
            pipe.upper_x += pipe.speed_x
            
        if 0 < self.pipes[0].lower_x and self.pipes[0].lower_x < 5:
            new_pipe = Pipe()
            new_pipe.set_x_y(self.screen_width, self.ground_y, self.pipe_height)
            self.pipes.append(new_pipe)
        if self.pipes[0].lower_x < 0 - self.pipe_width:
            del self.pipes[0]

    def draw_image(self):
        self.screen.blit(self.background_image, (0, 0))
        self.screen.blit(self.ground_image, (self.ground_x, self.ground_y))
        self.screen.blit(
            self.bird_images[self.index], (self.bird_x, self.bird_y))
        for pipe in self.pipes:
            self.screen.blit(
                pipe.pipe_images[0], (pipe.upper_x, pipe.upper_y))
            self.screen.blit(
                pipe.pipe_images[1], (pipe.lower_x, pipe.lower_y))
        image = array3d(display.get_surface())
        display.update()
        self.fps_clock.tick(self.fps)
        return image

        
    def next_frame(self, action):
        pump()
        rv1 = 0.1
        rv2 = False
        rv1 = self.update_score()
      
        if action == 1:
            self.flapped = True
            self.cur_speed_y = self.up_speed


        if (self.iter + 1) % 3 == 0:
            self.index = next(self.index_generator)
            self.iter = 0
        self.ground_x = 0 - ((100-self.ground_x) % self.ground_shift)

        self.update_bird_pos()
        self.update_pipe()
      
        if self.collided():
            rv2 = True
            rv1 = -1
            self.__init__()

        rv = self.draw_image()
        return rv, rv1, rv2