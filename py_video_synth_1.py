import pygame
import numpy as np
import cv2
import os
from datetime import datetime

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

# Main loop
def run():
    global record_frames
    clock = pygame.time.Clock()
    frame_count = 0
    running = True

    density = 0.5
    speed = 1.0

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

        # Generate and draw pattern
        pattern = generate_pattern(frame_count, density=density, speed=speed)
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

