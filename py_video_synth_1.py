import pygame
import numpy as np
import cv2
from PIL import Image

# Configurable parameters
WINDOW_SIZE = (800, 800)
GRID_SIZE = (128, 128)
PIXEL_SIZE = (WINDOW_SIZE[0] // GRID_SIZE[0], WINDOW_SIZE[1] // GRID_SIZE[1])
FPS = 30

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode(WINDOW_SIZE)
pygame.display.set_caption("Reaction-Diffusion with Video Input")

class ReactionDiffusion:
    def __init__(self, width, height, du=1.0, dv=0.5, feed=0.055, kill=0.062):
        self.width = width
        self.height = height

        self.U = np.ones((height, width), dtype=np.float32)
        self.V = np.zeros((height, width), dtype=np.float32)

        self.du = du
        self.dv = dv
        self.feed = feed
        self.kill = kill

        self.seed_default()

    def seed_default(self):
        self.U[self.height // 2 - 5: self.height // 2 + 5, self.width // 2 - 5: self.width // 2 + 5] = 0.50
        self.V[self.height // 2 - 5: self.height // 2 + 5, self.width // 2 - 5: self.width // 2 + 5] = 0.25

    def laplacian(self, arr):
        return (
            -arr
            + 0.2 * (np.roll(arr, 1, axis=0) + np.roll(arr, -1, axis=0)
                    + np.roll(arr, 1, axis=1) + np.roll(arr, -1, axis=1))
            + 0.05 * (np.roll(np.roll(arr, 1, axis=0), 1, axis=1) + np.roll(np.roll(arr, 1, axis=0), -1, axis=1)
                      + np.roll(np.roll(arr, -1, axis=0), 1, axis=1) + np.roll(np.roll(arr, -1, axis=0), -1, axis=1))
        )

    def update(self, time):
        self.feed = 0.055 + 0.01 * np.sin(time * 0.02)
        self.kill = 0.062 + 0.01 * np.cos(time * 0.015)

        U = self.U
        V = self.V

        Lu = self.laplacian(U)
        Lv = self.laplacian(V)

        reaction = U * V * V

        self.U += (self.du * Lu - reaction + self.feed * (1 - U))
        self.V += (self.dv * Lv + reaction - (self.kill + self.feed) * V)

    def get_pattern(self):
        img = (self.V * 255).clip(0, 255).astype(np.uint8)
        return img

def run():
    clock = pygame.time.Clock()
    frame_count = 0
    running = True

    # Start video capture
    video_capture = cv2.VideoCapture(0)

    if not video_capture.isOpened():
        print("Error: Could not open video source.")
        return

    reaction_diffusion = ReactionDiffusion(GRID_SIZE[0], GRID_SIZE[1])

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        ret, frame = video_capture.read()

        ret, frame = video_capture.read()

        if ret:
            # Rotate the frame 90 degrees clockwise
            rotated_frame = np.rot90(frame, k=1)  # k=1 => 90 degrees counter-clockwise == 90 clockwise

            # Convert to grayscale after rotation
            gray_frame = cv2.cvtColor(rotated_frame, cv2.COLOR_BGR2GRAY)

            # Resize and normalize
            resized_frame = cv2.resize(gray_frame, (GRID_SIZE[0], GRID_SIZE[1]))
            normalized_frame = resized_frame.astype(np.float32) / 255.0

            # Feed into the reaction-diffusion system
            reaction_diffusion.V = normalized_frame.copy()


        # Update RD and oscillate feed/kill rates
        reaction_diffusion.update(frame_count)

        # Get pattern and resize to window
        pattern = reaction_diffusion.get_pattern()
        pattern_resized = cv2.resize(pattern, WINDOW_SIZE, interpolation=cv2.INTER_NEAREST)

        # Convert pattern to RGB surface for Pygame
        surf = pygame.surfarray.make_surface(np.stack([pattern_resized]*3, axis=-1))

        screen.blit(surf, (0, 0))
        pygame.display.flip()

        frame_count += 1
        clock.tick(FPS)

    # Release resources
    video_capture.release()
    pygame.quit()

if __name__ == "__main__":
    run()
