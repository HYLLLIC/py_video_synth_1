import pygame
import numpy as np
import cv2
from PIL import Image

# Configurable parameters
BASE_WINDOW_HEIGHT = 800
BASE_GRID_HEIGHT = 128
FPS = 30

# Initialize Pygame
pygame.init()

# Colorize function
def colorize_pattern(pattern, motion_mask, color_strength=1.0):
    """
    Colorizes the RD pattern based on motion intensity.
    """
    base = pattern.astype(np.float32) / 255.0
    motion = motion_mask.astype(np.float32) / 255.0

    # Color mapping: R, G, B channels
    r = base + motion * color_strength
    g = base + motion * color_strength
    b = base * (1.0 - motion * color_strength)

    # Clip and convert to uint8
    rgb = np.stack([r, g, b], axis=-1)
    rgb = np.clip(rgb, 0.0, 1.0)
    rgb = (rgb * 255).astype(np.uint8)

    return rgb

# Reaction-Diffusion Class
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

# Main loop
def run():
    clock = pygame.time.Clock()
    frame_count = 0
    running = True
    rotate_output = False  # Toggle rotation of final display output

    # Start video capture
    video_capture = cv2.VideoCapture(0)

    if not video_capture.isOpened():
        print("Error: Could not open video source.")
        return

    # Get actual camera resolution (native)
    cam_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Maintain the camera's aspect ratio
    aspect_ratio = cam_width / cam_height

    # Set grid and window sizes based on camera resolution and aspect ratio
    grid_height = BASE_GRID_HEIGHT
    grid_width = int(BASE_GRID_HEIGHT * aspect_ratio)
    GRID_SIZE = (grid_width, grid_height)

    window_height = BASE_WINDOW_HEIGHT
    window_width = int(BASE_WINDOW_HEIGHT * aspect_ratio)
    WINDOW_SIZE = (window_width, window_height)

    print(f"Camera resolution: {cam_width}x{cam_height}")
    print(f"GRID_SIZE: {GRID_SIZE}")
    print(f"WINDOW_SIZE: {WINDOW_SIZE}")

    # Initialize Pygame window with updated WINDOW_SIZE
    screen = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption("Reaction-Diffusion with Video and Motion Coloring")

    # Initialize RD system
    reaction_diffusion = ReactionDiffusion(GRID_SIZE[0], GRID_SIZE[1])

    # Initialize previous frame for motion detection
    ret, prev_frame = video_capture.read()
    if not ret:
        print("Error: Could not read initial frame.")
        return

    prev_gray_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_gray_frame = cv2.resize(prev_gray_frame, (GRID_SIZE[0], GRID_SIZE[1]))

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    rotate_output = not rotate_output
                    print(f"Rotate output: {rotate_output}")

        ret, frame = video_capture.read()

        if ret:
            # No rotation on input! Keep it native
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized_frame = cv2.resize(gray_frame, (GRID_SIZE[0], GRID_SIZE[1]))
            normalized_frame = resized_frame.astype(np.float32) / 255.0

            # --- Motion detection ---
            diff_frame = cv2.absdiff(resized_frame, prev_gray_frame)
            _, motion_mask = cv2.threshold(diff_frame, 25, 255, cv2.THRESH_BINARY)

            # Normalize mask to 0-1 float
            motion_mask_float = motion_mask.astype(np.float32) / 255.0

            # --- Overlay motion into V ---
            motion_influence = 0.5
            reaction_diffusion.V = np.clip(
                reaction_diffusion.V + motion_mask_float * motion_influence,
                0.0, 1.0
            )

            # Update previous frame
            prev_gray_frame = resized_frame.copy()

        # --- Update RD system ---
        reaction_diffusion.update(frame_count)

        # --- Get RD pattern and resize to window size ---
        pattern = reaction_diffusion.get_pattern()
        pattern_resized = cv2.resize(pattern, (WINDOW_SIZE[0], WINDOW_SIZE[1]), interpolation=cv2.INTER_NEAREST)

        # --- Resize motion mask for color mapping ---
        motion_resized = cv2.resize(motion_mask, (WINDOW_SIZE[0], WINDOW_SIZE[1]), interpolation=cv2.INTER_NEAREST)

        # --- Colorize based on motion intensity ---
        colorized_pattern = colorize_pattern(pattern_resized, motion_resized, color_strength=1.5)

        # --- Rotate the final output display if toggled ---
        if rotate_output:
            final_display_pattern = np.rot90(colorized_pattern, k=1)  # Rotate counter-clockwise 90°
        else:
            final_display_pattern = colorized_pattern

        # --- Convert to Pygame surface ---
        surf = pygame.surfarray.make_surface(np.transpose(final_display_pattern, (1, 0, 2)))

        screen.blit(surf, (0, 0))
        pygame.display.flip()

        frame_count += 1
        clock.tick(FPS)

    video_capture.release()
    pygame.quit()

if __name__ == "__main__":
    run()
