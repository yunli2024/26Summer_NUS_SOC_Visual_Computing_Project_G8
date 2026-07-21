"""Score smoothing, combo, and total score management."""

from __future__ import annotations

from collections import Counter

from . import config


class ScoreManager:
    def __init__(self):
        self.smoothed_score = 0.0
        self.total_score = 0
        self.combo = 0
        self.best_combo = 0
        self.last_official_time = 0.0
        self.feedback = "Ready"
        self.official_samples = 0
        self.raw_samples = 0
        self.score_sum = 0.0
        self.best_score = 0.0
        self.feedback_counts = Counter()
        self.coarse_score = 0.0
        self.position_score = 0.0
        self.angle_score = 0.0
        self.vector_score = 0.0
        self.error_summary = ""
        self.user_buffer_size = 0
        self.matched_user_frame = 0
        self.has_score = False

    def reset(self):
        self.__init__()

    def update(self, raw_score: float, feedback: str, timestamp: float, details: dict | None = None):
        alpha = config.SCORE_EMA_ALPHA
        if self.has_score:
            self.smoothed_score = alpha * raw_score + (1.0 - alpha) * self.smoothed_score
        else:
            self.smoothed_score = raw_score
            self.has_score = True
        self.feedback = feedback
        self.raw_samples += 1
        self.best_score = max(self.best_score, raw_score, self.smoothed_score)
        if details:
            self.coarse_score = float(details.get("coarse", self.coarse_score))
            self.position_score = float(details.get("position", self.position_score))
            self.angle_score = float(details.get("angle", self.angle_score))
            self.vector_score = float(details.get("vector", self.vector_score))
            self.error_summary = str(details.get("error_summary", self.error_summary))
            self.user_buffer_size = int(details.get("user_buffer_size", self.user_buffer_size))
            self.matched_user_frame = int(details.get("matched_user_frame", self.matched_user_frame))
        if timestamp - self.last_official_time >= config.OFFICIAL_SCORE_INTERVAL:
            self.last_official_time = timestamp
            self.official_samples += 1
            self.score_sum += self.smoothed_score
            self.feedback_counts[feedback] += 1
            if feedback in {"Perfect", "Super", "Good"}:
                self.combo += 1
                self.best_combo = max(self.best_combo, self.combo)
                bonus = min(self.combo, 20)
                self.total_score += int(self.smoothed_score + bonus)
            elif feedback == "Miss":
                self.combo = 0
        return self.state()

    def state(self):
        average_score = self.score_sum / self.official_samples if self.official_samples else 0.0
        return {
            "smooth": self.smoothed_score,
            "feedback": self.feedback,
            "total": self.total_score,
            "combo": self.combo,
            "best_combo": self.best_combo,
            "average": average_score,
            "best_score": self.best_score,
            "samples": self.official_samples,
            "perfect": self.feedback_counts["Perfect"],
            "super": self.feedback_counts["Super"],
            "good": self.feedback_counts["Good"],
            "miss": self.feedback_counts["Miss"],
            "coarse": self.coarse_score,
            "position": self.position_score,
            "angle": self.angle_score,
            "vector": self.vector_score,
            "error_summary": self.error_summary,
            "user_buffer_size": self.user_buffer_size,
            "matched_user_frame": self.matched_user_frame,
        }

    def summary_text(self):
        state = self.state()
        if state["samples"] == 0:
            return "No scored samples yet. Start the reference video and webcam first."
        return (
            f"Final Total: {state['total']}\n"
            f"Average Score: {state['average']:.1f}\n"
            f"Best Score: {state['best_score']:.1f}\n"
            f"Best Combo: {state['best_combo']}\n"
            f"Perfect/Super/Good/Miss: {state['perfect']}/"
            f"{state['super']}/{state['good']}/{state['miss']}\n"
            f"Scored Samples: {state['samples']}"
        )
