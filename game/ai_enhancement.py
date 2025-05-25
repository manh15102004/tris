import numpy as np
import time
from collections import deque, defaultdict
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple
import pickle

@dataclass
class GestureData:
    gesture_type: str
    confidence: float
    timestamp: float
    game_state: str
    success: bool

class GesturePatternLearner:
    def __init__(self):
        self.gesture_history = deque(maxlen=1000)
        self.user_patterns = defaultdict(list)
        self.performance_metrics = {
            'accuracy': 0.0,
            'speed': 0.0,
            'consistency': 0.0,
            'preferred_gestures': {},
            'difficulty_adaptation': 1.0
        }
        self.learning_rate = 0.1
        
    def record_gesture(self, gesture_data: GestureData):
        self.gesture_history.append(gesture_data)
        context = self._get_game_context(gesture_data)
        self.user_patterns[context].append(gesture_data)
        self._update_performance_metrics(gesture_data)
        
    def _get_game_context(self, gesture_data: GestureData) -> str:
        return f"{gesture_data.game_state}_{gesture_data.gesture_type}"
    
    def _update_performance_metrics(self, gesture_data: GestureData):
        recent_gestures = list(self.gesture_history)[-50:]
        if recent_gestures:
            success_rate = sum(g.success for g in recent_gestures) / len(recent_gestures)
            self.performance_metrics['accuracy'] = (
                self.performance_metrics['accuracy'] * (1 - self.learning_rate) +
                success_rate * self.learning_rate
            )
        if len(recent_gestures) > 1:
            time_diffs = [recent_gestures[i].timestamp - recent_gestures[i-1].timestamp 
                         for i in range(1, len(recent_gestures))]
            avg_speed = 1.0 / np.mean(time_diffs) if time_diffs else 0
            self.performance_metrics['speed'] = (
                self.performance_metrics['speed'] * (1 - self.learning_rate) +
                avg_speed * self.learning_rate
            )
        gesture_type = gesture_data.gesture_type
        if gesture_type not in self.performance_metrics['preferred_gestures']:
            self.performance_metrics['preferred_gestures'][gesture_type] = 0
        self.performance_metrics['preferred_gestures'][gesture_type] += 1
    
    def predict_next_gesture(self, current_game_state: str) -> str:
        context_patterns = self.user_patterns.get(current_game_state, [])
        if not context_patterns:
            return "move_left"
        gesture_counts = defaultdict(int)
        for pattern in context_patterns[-20:]:
            gesture_counts[pattern.gesture_type] += 1
        if not gesture_counts:
            return "move_left"
        return max(gesture_counts.items(), key=lambda x: x[1])[0]
    
    def get_difficulty_recommendation(self) -> float:
        accuracy = self.performance_metrics['accuracy']
        speed = self.performance_metrics['speed']
        combined_score = (accuracy * 0.7 + min(speed / 10.0, 1.0) * 0.3)
        if combined_score > 0.8:
            return min(self.performance_metrics['difficulty_adaptation'] * 1.2, 2.0)
        elif combined_score < 0.4:
            return max(self.performance_metrics['difficulty_adaptation'] * 0.8, 0.5)
        else:
            return self.performance_metrics['difficulty_adaptation']
    
    def save_learning_data(self, filename: str):
        data = {
            'patterns': dict(self.user_patterns),
            'metrics': self.performance_metrics
        }
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
    
    def load_learning_data(self, filename: str):
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.user_patterns = defaultdict(list, data['patterns'])
                self.performance_metrics = data['metrics']
        except FileNotFoundError:
            print("Không tìm thấy file dữ liệu học tập")

class TetrisAIBot:
    def __init__(self, difficulty_level: float = 1.0):
        self.difficulty = difficulty_level
        self.thinking_delay = max(0.1, 1.0 / difficulty_level)
        self.error_rate = max(0.0, 0.2 - difficulty_level * 0.1)
        self.strategy_weights = {
            'clear_lines': 10.0 * difficulty_level,
            'avoid_holes': 8.0 * difficulty_level,
            'minimize_height': 6.0 * difficulty_level,
            'stack_evenly': 4.0 * difficulty_level
        }
        
    def evaluate_board(self, grid: List[List[int]]) -> float:
        score = 0.0
        rows, cols = len(grid), len(grid[0])
        full_rows = sum(1 for row in grid if all(cell != 0 for cell in row))
        score += full_rows * self.strategy_weights['clear_lines']
        holes = 0
        for col in range(cols):
            found_block = False
            for row in range(rows):
                if grid[row][col] != 0:
                    found_block = True
                elif found_block and grid[row][col] == 0:
                    holes += 1
        score -= holes * self.strategy_weights['avoid_holes']
        heights = []
        for col in range(cols):
            height = 0
            for row in range(rows):
                if grid[row][col] != 0:
                    height = rows - row
                    break
            heights.append(height)
        avg_height = np.mean(heights)
        score -= avg_height * self.strategy_weights['minimize_height']
        height_variance = np.var(heights)
        score -= height_variance * self.strategy_weights['stack_evenly']
        return score
    
    def find_best_position(self, piece: Dict, grid: List[List[int]]) -> Tuple[int, int, int]:
        best_score = float('-inf')
        best_position = (piece['x'], piece['y'], 0)
        current_shape = piece['shape']
        for rotation in range(4):
            for x in range(-len(current_shape[0]), len(grid[0])):
                for y in range(len(grid)):
                    test_piece = {
                        'shape': current_shape,
                        'x': x,
                        'y': y
                    }
                    if self._is_valid_position(test_piece, grid):
                        temp_grid = [row[:] for row in grid]
                        self._merge_piece_temp(test_piece, temp_grid)
                        score = self.evaluate_board(temp_grid)
                        if score > best_score:
                            best_score = score
                            best_position = (x, y, rotation)
            current_shape = self._rotate_shape(current_shape)
        return best_position
    
    def get_next_move(self, current_piece: Dict, grid: List[List[int]]) -> str:
        if random.random() < self.error_rate:
            return random.choice(['move_left', 'move_right', 'rotate', 'drop'])
        target_x, target_y, target_rotation = self.find_best_position(current_piece, grid)
        current_x = current_piece['x']
        current_rotation = current_piece.get('rotation', 0)
        if current_rotation != target_rotation:
            return 'rotate'
        if current_x < target_x:
            return 'move_right'
        elif current_x > target_x:
            return 'move_left'
        return 'drop'
    
    def _is_valid_position(self, piece: Dict, grid: List[List[int]]) -> bool:
        shape = piece['shape']
        for y in range(len(shape)):
            for x in range(len(shape[0])):
                if shape[y][x] != 0:
                    new_x = piece['x'] + x
                    new_y = piece['y'] + y
                    if (new_x < 0 or new_x >= len(grid[0]) or 
                        new_y >= len(grid) or 
                        (new_y >= 0 and grid[new_y][new_x] != 0)):
                        return False
        return True
    
    def _merge_piece_temp(self, piece: Dict, grid: List[List[int]]):
        shape = piece['shape']
        for y in range(len(shape)):
            for x in range(len(shape[0])):
                if shape[y][x] != 0:
                    grid[piece['y'] + y][piece['x'] + x] = shape[y][x]
    
    def _rotate_shape(self, shape: List[List[int]]) -> List[List[int]]:
        return [[shape[len(shape) - 1 - j][i] for j in range(len(shape))] 
                for i in range(len(shape[0]))]

class AdaptiveDifficultySystem:
    def __init__(self):
        self.pattern_learner = GesturePatternLearner()
        self.ai_bot = TetrisAIBot()
        self.current_difficulty = 1.0
        self.adaptation_rate = 0.05
        
    def update_difficulty(self, game_metrics: Dict):
        recommended_difficulty = self.pattern_learner.get_difficulty_recommendation()
        self.current_difficulty += (
            (recommended_difficulty - self.current_difficulty) * self.adaptation_rate
        )
        self.ai_bot.difficulty = self.current_difficulty
        return self.current_difficulty
    
    def get_ai_hint(self, current_piece: Dict, grid: List[List[int]]) -> str:
        return self.ai_bot.get_next_move(current_piece, grid)

class AIGameIntegration:
    def __init__(self):
        self.adaptive_system = AdaptiveDifficultySystem()
        self.ai_mode = False
        self.hint_mode = True
        self.last_hint_time = 0
        self.hint_cooldown = 2.0
        
    def process_gesture_with_ai(self, gesture: Dict, current_piece: Dict, 
                               grid: List[List[int]], game_state: str) -> Dict:
        gesture_data = GestureData(
            gesture_type=max(gesture.items(), key=lambda x: x[1])[0] if gesture else 'none',
            confidence=1.0,
            timestamp=time.time(),
            game_state=game_state,
            success=True
        )
        self.adaptive_system.pattern_learner.record_gesture(gesture_data)
        game_metrics = {'score': 0, 'level': 1}
        _ = self.adaptive_system.update_difficulty(game_metrics)
        result = gesture.copy()
        # Luôn trả về gợi ý AI (không cần điều kiện)
        ai_suggestion = self.adaptive_system.get_ai_hint(current_piece, grid)
        result['ai_hint'] = ai_suggestion
        return result
    
    def get_ai_move_for_demo(self, current_piece: Dict, grid: List[List[int]]) -> str:
        return self.adaptive_system.ai_bot.get_next_move(current_piece, grid)
    
    def save_ai_data(self):
        self.adaptive_system.pattern_learner.save_learning_data('user_patterns.pkl')
    
    def load_ai_data(self):
        self.adaptive_system.pattern_learner.load_learning_data('user_patterns.pkl')