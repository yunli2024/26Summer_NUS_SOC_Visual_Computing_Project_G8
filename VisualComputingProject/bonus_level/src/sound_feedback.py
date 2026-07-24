"""Small non-blocking feedback sounds for scoring events."""

from __future__ import annotations

import threading

from . import config


class SoundFeedback:
    def __init__(self):
        self.last_feedback = ""

    def reset(self):
        self.last_feedback = ""

    def play(self, feedback: str):
        if not config.SOUND_ENABLED or feedback == self.last_feedback:
            return
        self.last_feedback = feedback
        if feedback not in {"Perfect", "Super", "Good", "Miss"}:
            return
        threading.Thread(target=self._play_sync, args=(feedback,), daemon=True).start()

    @staticmethod
    def _play_sync(feedback: str):
        try:
            import winsound

            tones = {
                "Perfect": (1046, 90),
                "Super": (880, 80),
                "Good": (660, 70),
                "Miss": (220, 110),
            }
            freq, duration = tones[feedback]
            winsound.Beep(freq, duration)
        except Exception:
            pass
