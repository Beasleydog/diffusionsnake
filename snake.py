import pygame
import random
from enum import Enum
from collections import namedtuple
import math
import os
from datetime import datetime

pygame.init()

GRASS_GREEN = (34, 139, 34)
LIGHT_GREEN = (144, 238, 144)
SNAKE_GREEN = (50, 205, 50)
FRUIT_COLORS = {
    'apple': (255, 59, 59),    # Red
    'orange': (255, 165, 0),   # Orange
    'watermelon': (0, 255, 127),  # Greenish for Watermelon
    'grape': (147, 81, 182)    # Purple
}
WHITE = (255, 255, 255)
GOLD = (255, 215, 0)

BLOCK_SIZE = 20
SPEED = 15
    
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

class ParticleEffect:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = color
        self.lifetime = 20
        self.radius = random.randint(2, 5)
        self.dx = random.uniform(-2, 2)
        self.dy = random.uniform(-2, 2)
    
    def update(self):
        self.x += self.dx
        self.y += self.dy
        self.lifetime -= 1
        self.radius *= 0.9
        
    def draw(self, surface):
        alpha = int((self.lifetime / 20) * 255)
        color = (*self.color, alpha)
        surf = pygame.Surface((self.radius * 2, self.radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(surf, color, (self.radius, self.radius), self.radius)
        surface.blit(surf, (self.x - self.radius, self.y - self.radius))

class SnakeGame:
    def __init__(self, w=800, h=600):
        self.w = w
        self.h = h
        
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Garden Snake Paradise')
        self.clock = pygame.time.Clock()
        
        self.background = self._create_background()
        
        self.direction = Direction.RIGHT
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head]
        self.bodyColors = [SNAKE_GREEN] 
        self.score = 0
        self.food = None
        self.food_animation = 0
        self.particles = []
        self.fruit_type = None
        self._place_food()
        
        # Modified session handling
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = f"sessions/session_{self.session_id}"
        self.frames_dir = f"{self.session_dir}/frames"
        self.keylog_file = f"{self.session_dir}/keylog.txt"
        
        # Create directory structure
        os.makedirs(self.frames_dir, exist_ok=True)
        
        # Initialize keylog file with header
        with open(self.keylog_file, 'w') as f:
            f.write("frame,key_pressed\n")
        
        # Initialize frame counter
        self.frame_count = 0
        
    def _create_background(self):
        # Create a surface with a gradient and pattern
        surface = pygame.Surface((self.w, self.h))
        for y in range(self.h):
            # Create vertical gradient
            color_value = int(220 + (y / self.h) * 35)
            color = (color_value - 30, color_value, color_value - 30)
            pygame.draw.line(surface, color, (0, y), (self.w, y))
        
        # Add subtle pattern
        for _ in range(300):
            x = random.randint(0, self.w)
            y = random.randint(0, self.h)
            size = random.randint(2, 5)
            alpha = random.randint(20, 40)
            pattern_surf = pygame.Surface((size, size), pygame.SRCALPHA)
            pygame.draw.circle(pattern_surf, (*LIGHT_GREEN, alpha), (size//2, size//2), size//2)
            surface.blit(pattern_surf, (x, y))
            
        return surface
        
    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        self.food = Point(x, y)
        self.fruit_type = random.choice(list(FRUIT_COLORS.keys()))
        if self.food in self.snake:
            self._place_food()
        
    def _draw_fruit(self, pos):
        x, y = pos.x, pos.y
        color = FRUIT_COLORS[self.fruit_type]
        
        if self.fruit_type == 'apple':
            # Draw apple body
            pygame.draw.circle(self.display, color, 
                             (x + BLOCK_SIZE//2, y + BLOCK_SIZE//2), 
                             BLOCK_SIZE//2)
            # Draw stem
            pygame.draw.rect(self.display, (101, 67, 33),
                           (x + BLOCK_SIZE//2 - 2, y, 4, 5))
            # Draw leaf
            pygame.draw.ellipse(self.display, (34, 139, 34),
                              (x + BLOCK_SIZE//2 + 2, y, 6, 4))
            
        elif self.fruit_type == 'orange':
            # Draw orange body
            pygame.draw.circle(self.display, color,
                             (x + BLOCK_SIZE//2, y + BLOCK_SIZE//2),
                             BLOCK_SIZE//2)
            # Draw detailed texture
            for angle in range(0, 360, 30):  # More texture lines
                rad = math.radians(angle)
                line_length = BLOCK_SIZE//2
                end_x = x + BLOCK_SIZE//2 + math.cos(rad) * line_length
                end_y = y + BLOCK_SIZE//2 + math.sin(rad) * line_length
                pygame.draw.line(self.display, (255, 140, 0),
                               (x + BLOCK_SIZE//2, y + BLOCK_SIZE//2),
                               (end_x, end_y), 1)
            # Add orange highlight
            pygame.draw.circle(self.display, (255, 200, 100),
                             (x + BLOCK_SIZE//3, y + BLOCK_SIZE//3),
                             BLOCK_SIZE//6)
                
        elif self.fruit_type == 'watermelon':
            # Draw triangular watermelon slice
            center = (x + BLOCK_SIZE // 2, y + BLOCK_SIZE // 2)
            
            # Draw red flesh first (full triangle)
            flesh_points = [
                (x, y + BLOCK_SIZE),                # Bottom left
                (x + BLOCK_SIZE, y + BLOCK_SIZE),   # Bottom right
                (x + BLOCK_SIZE // 2, y)            # Top middle
            ]
            pygame.draw.polygon(self.display, (255, 75, 75), flesh_points)
            
            # Draw green rind on top (curved shape)
            pygame.draw.arc(self.display, (34, 139, 34),
                           (x - BLOCK_SIZE//5, y - BLOCK_SIZE//5,
                            BLOCK_SIZE * 1.5, BLOCK_SIZE * 1.5),
                           3.94, 5.44, 5)  
            
        elif self.fruit_type == 'grape':
            # Draw more detailed grape cluster
            grape_positions = [
                (0,0), (6,2), (-6,2),  # Top row
                (3,6), (-3,6),         # Middle row
                (0,10)                 # Bottom grape
            ]
            # Draw shadows first
            for dx, dy in grape_positions:
                pygame.draw.circle(self.display, (100, 50, 120),
                                 (x + BLOCK_SIZE//2 + dx + 1, 
                                  y + BLOCK_SIZE//2 + dy + 1), 5)
            # Draw grapes
            for dx, dy in grape_positions:
                pygame.draw.circle(self.display, color,
                                 (x + BLOCK_SIZE//2 + dx, 
                                  y + BLOCK_SIZE//2 + dy), 5)
                # Add highlight to each grape
                pygame.draw.circle(self.display, (200, 150, 220),
                                 (x + BLOCK_SIZE//2 + dx - 1,
                                  y + BLOCK_SIZE//2 + dy - 1), 2)
            # Add stem
            pygame.draw.line(self.display, (101, 67, 33),
                           (x + BLOCK_SIZE//2, y + 2),
                           (x + BLOCK_SIZE//2, y + 8), 2)
                
    
    def play_step(self):
        game_over = False
        key_pressed = "none"
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT and self.direction != Direction.RIGHT:
                    self.direction = Direction.LEFT
                    key_pressed = "LEFT"
                    break
                elif event.key == pygame.K_RIGHT and self.direction != Direction.LEFT:
                    self.direction = Direction.RIGHT
                    key_pressed = "RIGHT"
                    break
                elif event.key == pygame.K_UP and self.direction != Direction.DOWN:
                    self.direction = Direction.UP
                    key_pressed = "UP"
                    break
                elif event.key == pygame.K_DOWN and self.direction != Direction.UP:
                    self.direction = Direction.DOWN
                    key_pressed = "DOWN"
                    break
        
        self._move(self.direction)
        
        if self._is_collision():
            game_over = True
            self._handle_collision()
            self._update_ui()
            return game_over, self.score
        
        self.snake.insert(0, self.head)
        
        if self.head == self.food:
            self.score += 1
            
            for _ in range(15):
                particle = ParticleEffect(
                    self.food.x + BLOCK_SIZE/2,
                    self.food.y + BLOCK_SIZE/2,
                    FRUIT_COLORS[self.fruit_type]
                )
                self.particles.append(particle)
            self.bodyColors.append(FRUIT_COLORS[self.fruit_type])

            self._place_food()

        else:
            self.snake.pop()
        
        self._update_ui()
        
        # Save frame in jpg format
        frame = pygame.Surface((self.w, self.h))
        frame.blit(self.display, (0, 0))
        pygame.image.save(frame, f"{self.frames_dir}/frame_{self.frame_count:06d}.jpg")
        
        with open(self.keylog_file, 'a') as f:
            f.write(f"{self.frame_count},{key_pressed}\n")
        self.frame_count += 1
        
        self.clock.tick(SPEED)
        
        return game_over, self.score
    
    def _is_collision(self):
        # Hits boundary
        if (self.head.x > self.w - BLOCK_SIZE or self.head.x < 0 or 
            self.head.y > self.h - BLOCK_SIZE or self.head.y < 0):
            return True
        
        # Hits itself
        if self.head in self.snake[1:]:
            return True
        return False
        
    def _update_ui(self):
        self.display.blit(self.background, (0, 0))
        
        # Update and draw particles
        for particle in self.particles[:]:
            particle.update()
            if particle.lifetime <= 0:
                self.particles.remove(particle)
            else:
                particle.draw(self.display)
        
        # Draw snake with improved graphics
        for i, pt in enumerate(self.snake):
            color = self.bodyColors[i]
            
            # Add slight vertical movement based on horizontal position
            offset = math.sin(pt.x / 50 + pygame.time.get_ticks() * 0.004) * 3
            pos = (int(pt.x + BLOCK_SIZE/2), int(pt.y + BLOCK_SIZE/2 + offset))
            
            # Draw snake segment with glow effect
            glow_radius = BLOCK_SIZE * 0.8
            glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, (*color, 50), (glow_radius, glow_radius), glow_radius)
            self.display.blit(glow_surf, (pos[0] - glow_radius, pos[1] - glow_radius))
            
            # Draw main snake body
            pygame.draw.circle(self.display, color, pos, BLOCK_SIZE/2)
            
            # Draw eyes on head
            if i == 0:
                self._draw_snake_face(pos)
        
        self._draw_fruit(self.food)
        
        # Draw  score display
        score_surface = pygame.Surface((120, 40), pygame.SRCALPHA)
        pygame.draw.rect(score_surface, (0, 0, 0, 160), 
                        score_surface.get_rect(), 
                        border_radius=10)
        score_text = pygame.font.SysFont('arial', 24, bold=True).render(
            f'Score: {self.score}', 
            True, GOLD)
        score_surface.blit(score_text, 
                          (10, (40 - score_text.get_height())//2))
        
        self.display.blit(score_surface, 
                         (self.w//2 - 60, self.h - 50))
        
        pygame.display.flip()
    
    def _draw_snake_face(self, pos):
        # Draw eyes based on direction
        eye_color = WHITE
        eye_size = BLOCK_SIZE // 5
        pupil_size = eye_size // 2
        
        eye_offset = BLOCK_SIZE // 3
        if self.direction == Direction.RIGHT:
            left_eye = (pos[0] + eye_offset, pos[1] - eye_offset)
            right_eye = (pos[0] + eye_offset, pos[1] + eye_offset)
        elif self.direction == Direction.LEFT:
            left_eye = (pos[0] - eye_offset, pos[1] - eye_offset)
            right_eye = (pos[0] - eye_offset, pos[1] + eye_offset)
        elif self.direction == Direction.UP:
            left_eye = (pos[0] - eye_offset, pos[1] - eye_offset)
            right_eye = (pos[0] + eye_offset, pos[1] - eye_offset)
        else:
            left_eye = (pos[0] - eye_offset, pos[1] + eye_offset)
            right_eye = (pos[0] + eye_offset, pos[1] + eye_offset)
        
        # Draw white of eyes
        pygame.draw.circle(self.display, eye_color, left_eye, eye_size)
        pygame.draw.circle(self.display, eye_color, right_eye, eye_size)
        
        # Draw pupils
        pygame.draw.circle(self.display, (0, 0, 0), left_eye, pupil_size)
        pygame.draw.circle(self.display, (0, 0, 0), right_eye, pupil_size)
    
    def _handle_collision(self):
        # Add particle effects at collision point
        for _ in range(20):
            particle = ParticleEffect(
                self.head.x + BLOCK_SIZE/2,
                self.head.y + BLOCK_SIZE/2,
                SNAKE_GREEN
            )
            self.particles.append(particle)
    
    def _move(self, direction):
        x = self.head.x
        y = self.head.y
        if direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif direction == Direction.UP:
            y -= BLOCK_SIZE
        self.head = Point(x, y)
    
    def _show_game_over(self):
        overlay = pygame.Surface((self.w, self.h), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 128))
        self.display.blit(overlay, (0, 0))
        
        # Game Over text with shadow effect
        font = pygame.font.SysFont('arial', 64, bold=True)
        shadow_text = font.render('Game Over!', True, (0, 0, 0))
        text = font.render('Game Over!', True, (255, 50, 50))
        
        text_rect = text.get_rect(center=(self.w/2, self.h/2 - 50))
        self.display.blit(shadow_text, text_rect.move(2, 2))
        self.display.blit(text, text_rect)
        
        # Score display
        score_font = pygame.font.SysFont('arial', 36)
        score_text = score_font.render(f'Final Score: {self.score}', True, GOLD)
        score_rect = score_text.get_rect(center=(self.w/2, self.h/2 + 10))
        self.display.blit(score_text, score_rect)
        
        # Instructions
        inst_font = pygame.font.SysFont('arial', 24)
        inst_text = inst_font.render('Press any key to play again', True, WHITE)
        inst_rect = inst_text.get_rect(center=(self.w/2, self.h/2 + 60))
        self.display.blit(inst_text, inst_rect)
        
        pygame.display.flip()

if __name__ == '__main__':
    game = SnakeGame()
    
    while True:
        game_over, score = game.play_step()
        
        if game_over:
            game._show_game_over()
            waiting = True
            while waiting:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        quit()
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            pygame.quit()
                            quit()
                        else:
                            game = SnakeGame()
                            waiting = False
