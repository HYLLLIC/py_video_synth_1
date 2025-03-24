import pygame
import numpy as np
import cv2
from PIL import Image
import dearpygui.dearpygui as dpg

# === Configurable parameters ===
BASE_WINDOW_HEIGHT = 800
BASE_GRID_HEIGHT = 128
FPS = 30

# === Global GUI Variables ===
resolution_options = {"128": 128, "64": 64, "32": 32}
current_grid_height = BASE_GRID_HEIGHT
should_reset_resolution = False

motion_color = [255, 255, 0]  # Yellow default
color_strength = 1.5

# === Colorize Function ===
def colorize_pattern(pattern, motion_mask, color_strength=1.0, motion_color=[255, 255, 0]):
    base = pattern.astype(np.float32) / 255.0
    motion = motion_mask.astype(np.float32) / 255.0

    # Normalize motion color to 0-1
    motion_color_norm = np.array(motion_color) / 255.0

    # Apply motion color weighted by motion mask and color_strength
    r = base + motion * color_strength * motion_color_norm[0]
    g = base + motion * color_strength * motion_color_norm[1]
    b = base + motion * color_strength * motion_color_norm[2]

    rgb = np.stack([r, g, b], axis=-1)
    rgb = np.clip(rgb, 0.0, 1.0)
    rgb = (rgb * 255).astype(np.uint8)

    return rgb

# === Reaction-Diffusion Class ===
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

# === DearPyGui Callbacks ===
def resolution_callback(sender, app_data, user_data):
    global current_grid_height, should_reset_resolution
    selected = app_data
    current_grid_height = resolution_options[selected]
    should_reset_resolution = True
    print(f"[GUI] Resolution set to: {current_grid_height}")

def update_motion_color(sender, app_data, user_data):
    global motion_color
    motion_color = [int(c * 255) for c in app_data[:3]]  # RGB only
    print(f"[GUI] Motion Color set to: {motion_color}")

def update_color_strength(sender, app_data, user_data):
    global color_strength
    color_strength = app_data
    print(f"[GUI] Color Strength set to: {color_strength}")

# === Initialize DearPyGui ===
def setup_gui():
    dpg.create_context()

    with dpg.window(label="Controls", width=400, height=300):
        dpg.add_radio_button(items=["128", "64", "32"], label="Grid Resolution", default_value="128", callback=resolution_callback)
        dpg.add_color_picker(label="Motion Color", default_value=[1.0, 1.0, 0.0, 1.0], callback=update_motion_color, no_alpha=True)
        dpg.add_slider_float(label="Color Strength", default_value=1.5, min_value=0.0, max_value=5.0, callback=update_color_strength)

    dpg.create_viewport(title='Video Synth Controls', width=420, height=350)
    dpg.setup_dearpygui()
    dpg.show_viewport()

# === Main Loop ===
def run():
    global should_reset_resolution, current_grid_height

    clock = pygame.time.Clock()
    frame_count = 0
    running = True
    rotate_output = False

    # Video setup
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("Error: Could not open video source.")
        return

    cam_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    aspect_ratio = cam_width / cam_height

    # === Initialize / Reset RD and Window ===
    def initialize_rd(grid_height):
        grid_width = int(grid_height * aspect_ratio)
        window_height = BASE_WINDOW_HEIGHT
        window_width = int(window_height * aspect_ratio)

        GRID_SIZE = (grid_width, grid_height)
        WINDOW_SIZE = (window_width, window_height)

        rd_system = ReactionDiffusion(GRID_SIZE[0], GRID_SIZE[1])

        print(f"[RD RESET] GRID_SIZE: {GRID_SIZE}, WINDOW_SIZE: {WINDOW_SIZE}")
        return rd_system, GRID_SIZE, WINDOW_SIZE

    # === Initialize RD System ===
    reaction_diffusion, GRID_SIZE, WINDOW_SIZE = initialize_rd(current_grid_height)
    screen = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption("Reaction-Diffusion with GUI Controls")

    # Init motion detection frame
    ret, prev_frame = video_capture.read()
    prev_gray_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_gray_frame = cv2.resize(prev_gray_frame, (GRID_SIZE[0], GRID_SIZE[1]))

    while running:
        dpg.render_dearpygui_frame()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    rotate_output = not rotate_output
                    print(f"Rotate output: {rotate_output}")

        # Check for resolution change
        if should_reset_resolution:
            reaction_diffusion, GRID_SIZE, WINDOW_SIZE = initialize_rd(current_grid_height)
            screen = pygame.display.set_mode(WINDOW_SIZE)
            should_reset_resolution = False

        ret, frame = video_capture.read()
        if ret:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized_frame = cv2.resize(gray_frame, (GRID_SIZE[0], GRID_SIZE[1]))
            normalized_frame = resized_frame.astype(np.float32) / 255.0

            diff_frame = cv2.absdiff(resized_frame, prev_gray_frame)
            _, motion_mask = cv2.threshold(diff_frame, 25, 255, cv2.THRESH_BINARY)
            motion_mask_float = motion_mask.astype(np.float32) / 255.0

            motion_influence = 0.5
            reaction_diffusion.V = np.clip(
                reaction_diffusion.V + motion_mask_float * motion_influence,
                0.0, 1.0
            )

            prev_gray_frame = resized_frame.copy()

        reaction_diffusion.update(frame_count)

        pattern = reaction_diffusion.get_pattern()
        pattern_resized = cv2.resize(pattern, (WINDOW_SIZE[0], WINDOW_SIZE[1]), interpolation=cv2.INTER_NEAREST)
        motion_resized = cv2.resize(motion_mask, (WINDOW_SIZE[0], WINDOW_SIZE[1]), interpolation=cv2.INTER_NEAREST)

        colorized_pattern = colorize_pattern(
            pattern_resized,
            motion_resized,
            color_strength=color_strength,
            motion_color=motion_color
        )

        if rotate_output:
            final_display_pattern = np.rot90(colorized_pattern, k=1)
        else:
            final_display_pattern = colorized_pattern

        surf = pygame.surfarray.make_surface(np.transpose(final_display_pattern, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        frame_count += 1
        clock.tick(FPS)

    video_capture.release()
    pygame.quit()
    dpg.destroy_context()

# === Main Entry ===
if __name__ == "__main__":
    setup_gui()
    run()
