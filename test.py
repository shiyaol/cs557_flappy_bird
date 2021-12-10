from src.deep_q_network import DeepQNetwork
from src.flappy_bird import FlappyBird
from src.utils import pre_processing
import argparse
import torch
import cv2

image_size = 84
saved_path = "trained_models"

def test_flap(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)

    model = torch.load("{}/flappy_bird_final_newnn".format(saved_path), map_location=lambda storage, loc: storage)
    model.eval()
    fb = FlappyBird()
    init_image, reward, stop = fb.next_frame(0)
    init_image = pre_processing(init_image[:fb.screen_width :int(fb.ground_y)], image_size, image_size)
    init_image = torch.from_numpy(init_image)
    if torch.cuda.is_available():
        model.cuda()
        init_image = init_image.cuda()
    state = torch.cat(tuple(image for i in range(4)))[None, :, :, :]

    while True:
        pred = model(state)
        action = torch.argmax(pred)

        next_image, reward, stop = fb.next_frame(action)
        next_image = pre_processing(next_image[:fb.screen_width, :int(fb.ground_y)], image_size,
                                    image_size)
        next_image = torch.from_numpy(next_image)
        if torch.cuda.is_available():
            next_image = next_image.cuda()
        next_state = torch.cat((state[0, 1:, :, :], next_image))[None, :, :, :]

        state = next_state
        if stop:
            break


if __name__ == "__main__":
    test_flap()
