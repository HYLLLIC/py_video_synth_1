import pygame
import numpy as np
import cv2
import os
from datetime import datetime
import torch
import torch.nn as nn

# Configurable parameters
WINDOW_SIZE = (800, 800)
GRID_SIZE = (64, 64)  # The "pixel" resolution of the pattern
PIXEL_SIZE = (WINDOW_SIZE[0] // GRID_SIZE[0], WINDOW_SIZE[1] // GRID_SIZE[1])
FPS = 30

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode(WINDOW_SIZE)
pygame.display.set_caption("Python Pixel Video Synth")

# Frame saving setup
record_frames = False
output_frames = []
video_folder = "output_frames"
video_file = "output_video.mp4"

if not os.path.exists(video_folder):
    os.makedirs(video_folder)

# Generate a simple pattern (e.g., animated noise)
def generate_pattern(frame_count, density=0.5, speed=1.0):
    """
    Generates a black and white noise pattern that animates over time.
    """
    noise = np.random.rand(GRID_SIZE[1], GRID_SIZE[0])
    
    # Optional: Make it time-dependent (animate the threshold)
    threshold = 0.5 + 0.5 * np.sin(frame_count * 0.05 * speed)
    pattern = (noise > threshold * density).astype(np.uint8) * 255
    return pattern

def draw_pattern(surface, pattern):
    """
    Draws the pixel pattern to the pygame surface.
    """
    for y in range(GRID_SIZE[1]):
        for x in range(GRID_SIZE[0]):
            color = pattern[y, x]
            rect = pygame.Rect(
                x * PIXEL_SIZE[0],
                y * PIXEL_SIZE[1],
                PIXEL_SIZE[0],
                PIXEL_SIZE[1]
            )
            pygame.draw.rect(surface, (color, color, color), rect)

def save_frame(surface, frame_num):
    """
    Saves the current Pygame surface as an image frame.
    """
    pixels = pygame.surfarray.array3d(surface)
    frame = np.transpose(pixels, (1, 0, 2))  # Pygame stores pixels in (x, y), OpenCV expects (y, x)
    filename = os.path.join(video_folder, f"frame_{frame_num:04d}.png")
    cv2.imwrite(filename, frame)

def export_video():
    """
    Uses OpenCV to export the saved frames into a video.
    """
    image_files = sorted([
        os.path.join(video_folder, f)
        for f in os.listdir(video_folder)
        if f.endswith(".png")
    ])

    if not image_files:
        print("No frames found to compile into video.")
        return

    frame = cv2.imread(image_files[0])
    height, width, _ = frame.shape

    out = cv2.VideoWriter(video_file, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (width, height))

    for filename in image_files:
        img = cv2.imread(filename)
        out.write(img)

    out.release()
    print(f"Video exported as {video_file}")

class ReactionDiffusion:
    def __init__(self, width, height, du=1.0, dv=0.5, feed=0.055, kill=0.062):
        self.width = width
        self.height = height

        # Chemical concentrations
        self.U = np.ones((height, width), dtype=np.float32)
        self.V = np.zeros((height, width), dtype=np.float32)

        # Seed initial V in the middle
        self.U[height // 2 - 5: height // 2 + 5, width // 2 - 5: width // 2 + 5] = 0.50
        self.V[height // 2 - 5: height // 2 + 5, width // 2 - 5: width // 2 + 5] = 0.25

        # Parameters
        self.du = du
        self.dv = dv
        self.feed = feed
        self.kill = kill

    def laplacian(self, arr):
        return (
            -arr
            + 0.2 * (np.roll(arr, 1, axis=0) + np.roll(arr, -1, axis=0)
                    + np.roll(arr, 1, axis=1) + np.roll(arr, -1, axis=1))
            + 0.05 * (np.roll(np.roll(arr, 1, axis=0), 1, axis=1) + np.roll(np.roll(arr, 1, axis=0), -1, axis=1)
                      + np.roll(np.roll(arr, -1, axis=0), 1, axis=1) + np.roll(np.roll(arr, -1, axis=0), -1, axis=1))
        )

    def update(self):
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
        img = (self.V * 255).astype(np.uint8)
        return img

class TextureGenerator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=1):
        super(TextureGenerator, self).__init__()
        self.main = nn.Sequential(
            # Input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # State size: (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # State size: (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # State size: (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # State size: (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()  # Output between -1 and 1
            # Final size: (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate generator
nz = 100  # Size of latent vector
generator = TextureGenerator(nz=nz).to(device)
generator.eval()  # Inference mode

def generate_texture():
    # Random noise input
    noise = torch.randn(1, nz, 1, 1, device=device)

    # Generate fake image
    with torch.no_grad():
        fake = generator(noise).detach().cpu()

    # Reshape & scale to 0-255 grayscale
    img = fake.squeeze().numpy()
    img = ((img + 1) * 127.5).astype(np.uint8)  # Convert [-1,1] to [0,255]

    return img

# Main loop
def run():
    global record_frames
    clock = pygame.time.Clock()
    frame_count = 0
    running = True

    density = 0.5
    speed = 1.0

    # Create Reaction-Diffusion object
    reaction_diffusion = ReactionDiffusion(GRID_SIZE[0], GRID_SIZE[1])

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # Keyboard controls
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    record_frames = not record_frames
                    print(f"Recording {'started' if record_frames else 'stopped'}")
                elif event.key == pygame.K_UP:
                    density = min(density + 0.1, 1.0)
                    print(f"Density: {density}")
                elif event.key == pygame.K_DOWN:
                    density = max(density - 0.1, 0.1)
                    print(f"Density: {density}")
                elif event.key == pygame.K_RIGHT:
                    speed += 0.1
                    print(f"Speed: {speed}")
                elif event.key == pygame.K_LEFT:
                    speed = max(speed - 0.1, 0.1)
                    print(f"Speed: {speed}")
                    
                elif event.key == pygame.K_w:
                    reaction_diffusion.feed += 0.001
                    print(f"Feed: {reaction_diffusion.feed}, Kill: {reaction_diffusion.kill}")
                elif event.key == pygame.K_s:
                    reaction_diffusion.feed -= 0.001
                    print(f"Feed: {reaction_diffusion.feed}, Kill: {reaction_diffusion.kill}")
                elif event.key == pygame.K_a:
                    reaction_diffusion.kill -= 0.001
                    print(f"Feed: {reaction_diffusion.feed}, Kill: {reaction_diffusion.kill}")
                elif event.key == pygame.K_d:
                    reaction_diffusion.kill += 0.001
                    print(f"Feed: {reaction_diffusion.feed}, Kill: {reaction_diffusion.kill}")


        # Generate and draw pattern
        reaction_diffusion.update()
        pattern = generate_texture()

        draw_pattern(screen, pattern)

        # Save frame if recording
        if record_frames:
            save_frame(screen, frame_count)

        pygame.display.flip()
        frame_count += 1
        clock.tick(FPS)

    # Export video after quitting
    if record_frames:
        export_video()

    pygame.quit()

if __name__ == "__main__":
    run()

