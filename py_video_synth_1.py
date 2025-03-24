import pygame
import numpy as np
import cv2
import os
from datetime import datetime
from PIL import Image

# Configurable parameters
WINDOW_SIZE = (800, 800)
GRID_SIZE = (128, 128)  # Higher res, adjust if needed
PIXEL_SIZE = (WINDOW_SIZE[0] // GRID_SIZE[0], WINDOW_SIZE[1] // GRID_SIZE[1])
FPS = 30

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode(WINDOW_SIZE)
pygame.display.set_caption("Reaction-Diffusion Synth")

class ReactionDiffusion:
    def __init__(self, width, height, du=1.0, dv=0.5, feed=0.055, kill=0.062, image_path=None):
        self.width = width
        self.height = height

        # Chemical concentrations
        self.U = np.ones((height, width), dtype=np.float32)
        self.V = np.zeros((height, width), dtype=np.float32)

        # Parameters
        self.du = du
        self.dv = dv
        self.feed = feed
        self.kill = kill

        # Load image and initialize V
        if image_path:
            self.initialize_with_image(image_path)
        else:
            self.seed_default()

    def initialize_with_image(self, image_path):
        # Load and resize image to fit grid
        img = Image.open(image_path).convert('L')  # Grayscale
        img = img.resize((self.width, self.height))
        img_np = np.asarray(img).astype(np.float32) / 255.0  # Normalize to [0, 1]

        # Use image intensity to initialize V field
        self.V = img_np.copy()

        # Optionally invert or threshold
        # self.V = 1.0 - self.V

        print(f"Initialized V field from image: {image_path}")

    def seed_default(self):
        # Seed a basic spot in the middle if no image provided
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
        # Oscillate feed/kill rates over time
        self.feed = 0.045 + 0.015 * np.sin(time * 0.01)
        self.kill = 0.06 + 0.01 * np.cos(time * 0.008)


        U = self.U
        V = self.V

        # Diffusion
        Lu = self.laplacian(U)
        Lv = self.laplacian(V)

        # Reaction
        reaction = U * V * V

        # Update equations
        self.U += (self.du * Lu - reaction + self.feed * (1 - U))
        self.V += (self.dv * Lv + reaction - (self.kill + self.feed) * V)

    def get_pattern(self):
        # Return grayscale image based on V concentration
        img = (self.V * 255).clip(0, 255).astype(np.uint8)
        return img

def run():
    clock = pygame.time.Clock()
    frame_count = 0
    running = True

    # Load your image path
    image_path = "/Users/natesparks/Documents/HYLLLIC/HYllliC (1)/1.jpg"

    # Create Reaction-Diffusion object seeded with the image
    reaction_diffusion = ReactionDiffusion(GRID_SIZE[0], GRID_SIZE[1], image_path=image_path)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # Keyboard controls (optional)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        # Update the RD system with time to oscillate feed/kill rates
        reaction_diffusion.update(frame_count)

        # Get the current pattern and resize to screen
        pattern = reaction_diffusion.get_pattern()
        pattern_resized = cv2.resize(pattern, WINDOW_SIZE, interpolation=cv2.INTER_NEAREST)

        # Convert to a pygame surface
        surf = pygame.surfarray.make_surface(np.stack([pattern_resized]*3, axis=-1))

        screen.blit(surf, (0, 0))
        pygame.display.flip()

        frame_count += 1
        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    run()
