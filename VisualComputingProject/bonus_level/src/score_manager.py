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
        self.pose_score = 0.0
        self.position_score = 0.0
        self.angle_score = 0.0
        self.motion_score = 0.0
        self.coverage = 0.0
        self.player_motion = 0.0
        self.reference_motion = 0.0
        self.lag_values = []
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
            self.pose_score = float(details.get("pose", self.pose_score))
            self.position_score = float(details.get("position", self.position_score))
            self.angle_score = float(details.get("angle", self.angle_score))
            self.motion_score = float(details.get("motion", self.motion_score))
            self.coverage = float(details.get("coverage", self.coverage))
            self.player_motion = float(details.get("player_motion", self.player_motion))
            self.reference_motion = float(details.get("reference_motion", self.reference_motion))
            self.error_summary = str(details.get("error_summary", self.error_summary))
            self.user_buffer_size = int(details.get("user_buffer_size", self.user_buffer_size))
            self.matched_user_frame = int(details.get("matched_user_frame", self.matched_user_frame))
        default_event = feedback in {"Perfect", "Super", "Good", "Miss", "Move!"}
        score_event = (
            bool(details.get("score_event", default_event))
            if details
            else default_event
        )
        if score_event and timestamp - self.last_official_time >= config.OFFICIAL_SCORE_INTERVAL:
            self.last_official_time = timestamp
            self.official_samples += 1
            self.score_sum += self.smoothed_score
            self.feedback_counts[feedback] += 1
            if details and "lag_seconds" in details:
                self.lag_values.append(float(details["lag_seconds"]))
            if feedback in {"Perfect", "Super", "Good"}:
                self.combo += 1
                self.best_combo = max(self.best_combo, self.combo)
                combo_bonus = min(self.combo, 20) * 10
                self.total_score += config.GRADE_POINTS[feedback] + combo_bonus
            elif feedback in {"Miss", "Move!"}:
                self.combo = 0
        return self.state()

    def state(self):
        average_score = self.score_sum / self.official_samples if self.official_samples else 0.0
        sorted_lags = sorted(self.lag_values)
        median_lag = percentile(sorted_lags, 0.50)
        p90_lag = percentile(sorted_lags, 0.90)
        average_lag = (
            sum(sorted_lags) / len(sorted_lags) if sorted_lags else 0.0
        )
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
            "move": self.feedback_counts["Move!"],
            "pose": self.pose_score,
            "position": self.position_score,
            "angle": self.angle_score,
            "motion": self.motion_score,
            "coverage": self.coverage,
            "player_motion": self.player_motion,
            "reference_motion": self.reference_motion,
            "average_lag": average_lag,
            "median_lag": median_lag,
            "p90_lag": p90_lag,
            "lag_samples": len(sorted_lags),
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
            f"Delay avg/median/p90: {state['average_lag']:.2f}/"
            f"{state['median_lag']:.2f}/{state['p90_lag']:.2f} s\n"
            f"Scored Samples: {state['samples']}"
        )


def percentile(sorted_values, fraction: float) -> float:
    if not sorted_values:
        return 0.0
    index = int(round((len(sorted_values) - 1) * fraction))
    return float(sorted_values[index])
