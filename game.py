import pygame
import numpy as np
import sys


class Game:
    def __init__(self, screen_width, screen_height):
        self.screen_width = screen_width
        self.screen_height = screen_height

        self.clock = pygame.time.Clock()

        self.ai_error_magnitude = 10

        self.ball_vel = 7
        self.slider_vel = 5

        self.ball_r = 7
        self.slider_gap = 20
        self.slider_sizes = [10, 60]  # width, height

        self.colors = {"black": (0, 0, 0), "white": (255, 255, 255), "grey": (80, 80, 80)}

        self.slider1_pos = pygame.Vector2(self.slider_gap, screen_height / 2 - self.slider_sizes[1] / 2)
        self.slider2_pos = pygame.Vector2(screen_width - self.slider_gap - self.slider_sizes[0],
                screen_height / 2 - self.slider_sizes[1] / 2)

        self.ball_loc = pygame.Vector2(screen_width / 5, screen_height / 2)
        self.prev_ball_loc = None
        self.ball_dir = pygame.Vector2()
        self.ball_dir.from_polar((1, 0 + np.random.uniform(-20, 20)))

    def ball_hits_left_slider_right_side(self):
        hits_x = self.ball_loc.x - self.ball_r <= self.slider_gap + self.slider_sizes[0]
        hits_y = self.slider1_pos.y <= self.ball_loc.y <= self.slider1_pos.y + self.slider_sizes[1]
        return hits_x and hits_y

    def ball_hits_right_slider_left_side(self):
        hits_x = self.ball_loc.x + self.ball_r >= self.screen_width - self.slider_gap - self.slider_sizes[0]
        hits_y = self.slider2_pos.y <= self.ball_loc.y <= self.slider2_pos.y + self.slider_sizes[1]
        return hits_x and hits_y

    def handle_ball_collisions(self, slider1_direction, slider2_direction):
        # left wall -> player score
        if self.ball_loc.x <= 0 + self.ball_r:
            return 2
        # right wall -> AI scores
        if self.ball_loc.x >= self.screen_width - self.ball_r:
            return 1
        # top wall
        if self.ball_loc.y <= 0 + self.ball_r:
            self.ball_dir.y *= -1
        # bottom wall
        if self.ball_loc.y >= self.screen_height - self.ball_r:
            self.ball_dir.y *= -1
        # left slider
        if self.ball_hits_left_slider_right_side():
            self.ball_dir.x *= -1
            self.ball_dir.y += slider1_direction * self.slider_vel / 5
        # right slider
        if self.ball_hits_right_slider_left_side():
            self.ball_dir.x *= -1
            self.ball_dir.y += slider2_direction * self.slider_vel / 5
        return 0

    def handle_player_controls(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            self.slider2_pos.y -= self.slider_vel
            if self.slider2_pos.y <= 0:
                self.slider2_pos.y = 0
            return -1

        if keys[pygame.K_DOWN]:
            self.slider2_pos.y += self.slider_vel
            if self.slider2_pos.y + self.slider_sizes[1] >= self.screen_height:
                self.slider2_pos.y = self.screen_height - self.slider_sizes[1]
            return 1
        
        return 0

    def update_ball(self):
        self.prev_ball_loc = self.ball_loc
        self.ball_loc = self.ball_loc + self.ball_vel * self.ball_dir
        self.ball_dir = self.ball_dir.normalize()

    def ai_move(self, current_frame):
        slider_center = self.slider1_pos.y + (self.slider_sizes[1] / 2)
        # ai_velocity = np.random.normal(self.slider_vel, scale=self.ai_error_magnitude)
        ai_velocity = self.slider_vel
        if current_frame % 5 == 0:
            return 0

        if slider_center <= self.ball_loc.y:
            self.slider1_pos.y += ai_velocity
            if self.slider1_pos.y + self.slider_sizes[1] >= self.screen_height:
                self.slider1_pos.y = self.screen_height - self.slider_sizes[1]
            return -1
        else:
            self.slider1_pos.y -= ai_velocity
            if self.slider1_pos.y <= 0:
                self.slider1_pos.y = 0
            return 1

    def render(self, display, font, ai_score, player_score):
        display.fill(self.colors["black"])

        pygame.draw.circle(display, self.colors["white"], self.ball_loc, self.ball_r)
        pygame.draw.rect(display, self.colors["white"], list(self.slider1_pos) + self.slider_sizes)
        pygame.draw.rect(display, self.colors["white"], list(self.slider2_pos) + self.slider_sizes)

        ai_texture = font.render(str(ai_score), True, self.colors["grey"])
        ai_texture_x = self.screen_width / 4 - ai_texture.get_bounding_rect().width / 2
        ai_texture_y = self.screen_height / 2 - ai_texture.get_bounding_rect().height / 2
        display.blit(ai_texture, [ai_texture_x, ai_texture_y])

        player_texture = font.render(str(player_score), True, self.colors["grey"])
        player_texture_x = 3 * self.screen_width / 4 - player_texture.get_bounding_rect().width / 2
        player_texture_y = self.screen_height / 2 - player_texture.get_bounding_rect().height / 2
        display.blit(player_texture, [player_texture_x, player_texture_y])

        pygame.display.update()

    def reset(self, score):
        self.slider1_pos = pygame.Vector2(self.slider_gap, self.screen_height / 2 - self.slider_sizes[1] / 2)
        self.slider2_pos = pygame.Vector2(self.screen_width - self.slider_gap - self.slider_sizes[0],
                self.screen_height / 2 - self.slider_sizes[1] / 2)

        if score == 1:
            # AI has won last round -> start in direction to AI (left)
            self.ball_loc = pygame.Vector2(4 * self.screen_width / 5, self.screen_height / 2)
            self.prev_ball_loc = None
            self.ball_dir.from_polar((1, 180 + np.random.uniform(-20, 20)))
        else:
            # player has won last round -> start in direction to player (right)
            self.ball_loc = pygame.Vector2(self.screen_width / 5, self.screen_height / 2)
            self.prev_ball_loc = None
            self.ball_dir.from_polar((1, 0 + np.random.uniform(-20, 20)))

    def run(self, max_score=5, render=True):
        pygame.init()

        if render:
            display = pygame.display.set_mode((self.screen_width, self.screen_height))
            font = pygame.font.SysFont(None, 100)
            pygame.display.set_caption("DeepRL-Pong")
         
        ai_score = 0
        player_score = 0

        while ai_score < max_score and player_score < max_score:
            print("AI score: {}, player score: {}".format(ai_score, player_score))
            score_round = self.run_round(display, font, ai_score, player_score)
            if score_round == 1:
                ai_score += 1
            else:
                player_score += 1

        return player_score > ai_score

    def run_round(self, display, font, ai_score, player_score):
        # store the direction of the slider from the last frame to add spinning effects
        slider1_last_direction = 0  # +1: upwards, -1: downwards
        slider2_last_direction = 0
        current_frame = 0

        while True:
            slider2_last_direction = self.handle_player_controls()
            slider1_last_direction = self.ai_move(current_frame)

            score = self.handle_ball_collisions(slider1_last_direction, slider2_last_direction)
            self.update_ball()

            if display is not None:
                self.render(display, font, ai_score, player_score)

            self.clock.tick(60)  # run at 60 frames per second
            current_frame += 1
    
            if score:
                self.reset(score)
                return score  # return a 1 for the AI winning and 2 for the player winning

