"""Generate original pixel sprites and chiptune sound effects for the demo."""

from __future__ import annotations

import math
from pathlib import Path
import struct
import wave

from PIL import Image, ImageDraw


ASSETS = Path(__file__).resolve().parent / "assets"


def sprite() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGBA", (32, 48), (0, 0, 0, 0))
    return image, ImageDraw.Draw(image)


def hero(name: str, leg_phase: int = 0, jumping: bool = False, crouching: bool = False) -> None:
    image, draw = sprite()
    y = 12 if crouching else 0
    # Original side-view red-cap plumber adventurer. The broad archetype is
    # familiar, but this is not traced from or pixel-identical to any game.
    red, dark_red = "#e6423a", "#8d211e"
    blue, dark_blue = "#2872c7", "#153d79"
    skin, skin_dark = "#f0ad7e", "#8c5437"
    brown, dark_brown = "#613727", "#2c1c1b"
    # Hair, cap and forward-facing cap bill.
    draw.rectangle((7, y + 5, 22, y + 10), fill=brown, outline=dark_brown)
    draw.rectangle((8, y + 2, 24, y + 8), fill=red, outline=dark_red)
    draw.rectangle((13, y, 23, y + 3), fill="#f15b4f", outline=dark_red)
    draw.rectangle((21, y + 6, 29, y + 9), fill=red, outline=dark_red)
    # Side-profile face, nose, eye and moustache.
    draw.rectangle((10, y + 9, 24, y + 20), fill=skin, outline=skin_dark)
    draw.rectangle((23, y + 11, 29, y + 16), fill=skin, outline=skin_dark)
    draw.rectangle((21, y + 10, 23, y + 12), fill="#172033")
    draw.rectangle((20, y + 16, 27, y + 19), fill=dark_brown)
    draw.rectangle((8, y + 18, 14, y + 21), fill=brown, outline=dark_brown)
    # Red shirt, blue overalls and gold buttons.
    draw.rectangle((6, y + 21, 25, y + 34), fill=red, outline=dark_red)
    draw.rectangle((11, y + 24, 25, y + 39), fill=blue, outline=dark_blue)
    draw.rectangle((9, y + 22, 13, y + 30), fill=blue, outline=dark_blue)
    draw.rectangle((21, y + 22, 25, y + 30), fill=blue, outline=dark_blue)
    draw.rectangle((12, y + 25, 14, y + 27), fill="#ffd166")
    draw.rectangle((22, y + 25, 24, y + 27), fill="#ffd166")
    # Back arm and forward white glove.
    draw.rectangle((4, y + 23, 9, y + 33), fill=red, outline=dark_red)
    draw.rectangle((3, y + 31, 9, y + 36), fill="#f5f5ed", outline="#777a85")
    draw.rectangle((24, y + 24, 28, y + 33), fill=red, outline=dark_red)
    draw.rectangle((26, y + 31, 31, y + 36), fill="#f5f5ed", outline="#777a85")
    if crouching:
        draw.rectangle((7, 39, 17, 44), fill=blue, outline=dark_blue)
        draw.rectangle((16, 39, 27, 44), fill=blue, outline=dark_blue)
        draw.rectangle((4, 43, 17, 47), fill=brown, outline=dark_brown)
        draw.rectangle((18, 43, 31, 47), fill=brown, outline=dark_brown)
    elif jumping:
        draw.rectangle((7, 35, 15, 41), fill=blue, outline=dark_blue)
        draw.rectangle((19, 34, 27, 41), fill=blue, outline=dark_blue)
        draw.rectangle((2, 40, 15, 46), fill=brown, outline=dark_brown)
        draw.rectangle((20, 40, 31, 46), fill=brown, outline=dark_brown)
    else:
        left = 4 if leg_phase < 0 else 9
        right = 21 if leg_phase <= 0 else 25
        draw.rectangle((left, 36, left + 7, 44), fill=blue, outline=dark_blue)
        draw.rectangle((right, 36, min(right + 6, 31), 44), fill=blue, outline=dark_blue)
        draw.rectangle((max(0, left - 2), 43, left + 9, 47), fill=brown, outline=dark_brown)
        draw.rectangle((right - 1, 43, min(right + 8, 31), 47), fill=brown, outline=dark_brown)
    image.save(ASSETS / f"{name}_right.png")
    image.transpose(Image.Transpose.FLIP_LEFT_RIGHT).save(ASSETS / f"{name}_left.png")


def make_sprites() -> None:
    hero("hero_idle")
    hero("hero_run1", -1)
    hero("hero_run2", 1)
    hero("hero_jump", jumping=True)
    hero("hero_crouch", crouching=True)

    enemy = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    draw = ImageDraw.Draw(enemy)
    draw.ellipse((3, 5, 29, 29), fill="#7a4fc6", outline="#30205e", width=2)
    draw.rectangle((7, 20, 25, 30), fill="#5535a0", outline="#30205e")
    draw.polygon(((5, 12), (0, 5), (10, 8)), fill="#f4c542", outline="#8f6712")
    draw.polygon(((27, 12), (32, 5), (22, 8)), fill="#f4c542", outline="#8f6712")
    draw.rectangle((9, 13, 12, 17), fill="white")
    draw.rectangle((20, 13, 23, 17), fill="white")
    enemy.save(ASSETS / "enemy.png")

    coin = Image.new("RGBA", (20, 28), (0, 0, 0, 0))
    draw = ImageDraw.Draw(coin)
    draw.ellipse((2, 2, 17, 25), fill="#ffd43b", outline="#a96500", width=2)
    draw.polygon(((10, 6), (12, 11), (17, 11), (13, 15), (15, 21), (10, 17), (5, 21), (7, 15), (3, 11), (8, 11)), fill="#fff4a3")
    coin.save(ASSETS / "coin.png")

    ground = Image.new("RGBA", (32, 32), "#a65d34")
    draw = ImageDraw.Draw(ground)
    draw.rectangle((0, 0, 31, 8), fill="#45b34f", outline="#1e7530")
    draw.line((4, 13, 14, 25), fill="#71391f", width=3)
    draw.line((25, 11, 16, 29), fill="#cf8650", width=2)
    ground.save(ASSETS / "ground_tile.png")

    block = Image.new("RGBA", (32, 22), "#db8a39")
    draw = ImageDraw.Draw(block)
    draw.rectangle((0, 0, 31, 21), outline="#7e421f", width=2)
    draw.line((0, 10, 31, 10), fill="#9d5429", width=2)
    draw.line((10, 0, 10, 10), fill="#9d5429", width=2)
    draw.line((22, 10, 22, 21), fill="#9d5429", width=2)
    block.save(ASSETS / "platform_tile.png")

    flag = Image.new("RGBA", (64, 128), (0, 0, 0, 0))
    draw = ImageDraw.Draw(flag)
    draw.rectangle((5, 7, 10, 126), fill="#e8edf5", outline="#667085")
    draw.ellipse((2, 1, 13, 12), fill="#ffd166", outline="#a05e00")
    draw.polygon(((10, 14), (59, 30), (10, 52)), fill="#20bfa9", outline="#075f62")
    draw.polygon(((28, 25), (34, 28), (31, 34), (24, 31)), fill="#eafff8")
    flag.save(ASSETS / "goal_flag.png")


def tone_samples(
    notes: list[tuple[float, float]],
    volume: float = 0.28,
    sample_rate: int = 22050,
    waveform: str = "square",
    sustain: bool = False,
) -> bytes:
    output: list[int] = []
    for frequency, duration in notes:
        count = int(duration * sample_rate)
        for index in range(count):
            phase = index / sample_rate
            attack = min(index / max(sample_rate * 0.008, 1), 1.0)
            if sustain:
                release_samples = max(int(sample_rate * 0.035), 1)
                release = min((count - index) / release_samples, 1.0)
                envelope = attack * release
            else:
                envelope = attack * max(0.0, 1.0 - index / count)
            if waveform == "triangle":
                cycle = frequency * phase
                signal = 2.0 * abs(2.0 * (cycle - math.floor(cycle + 0.5))) - 1.0
            else:
                signal = 1.0 if math.sin(2.0 * math.pi * frequency * phase) >= 0 else -1.0
            output.append(int(32767 * volume * envelope * signal))
    return struct.pack("<" + "h" * len(output), *output)


def save_sound(
    name: str,
    notes: list[tuple[float, float]],
    volume: float = 0.28,
    waveform: str = "square",
    sustain: bool = False,
    tail_silence: float = 0.0,
) -> None:
    with wave.open(str(ASSETS / name), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(22050)
        frames = tone_samples(notes, volume, waveform=waveform, sustain=sustain)
        frames += b"\x00\x00" * int(22050 * tail_silence)
        audio.writeframes(frames)


def make_sounds() -> None:
    # Jumping is intentionally silent; remove the legacy asset on regeneration.
    (ASSETS / "jump.wav").unlink(missing_ok=True)
    # Original 8-bit cues with deliberately different contours and registers.
    save_sound(
        "coin.wav",
        [(784, 0.12), (1175, 0.16), (1568, 0.24)],
        0.52,
        waveform="triangle",
        sustain=True,
        tail_silence=0.08,
    )
    save_sound("bump.wav", [(294, 0.045), (147, 0.13)], 0.45)
    save_sound("stomp.wav", [(440, 0.045), (220, 0.07), (110, 0.16)], 0.44)
    save_sound("hurt.wav", [(466, 0.10), (349, 0.10), (233, 0.12), (117, 0.22)], 0.40)
    save_sound("checkpoint.wav", [(523, 0.10), (659, 0.10), (784, 0.12), (1047, 0.26)], 0.34)
    save_sound("win.wav", [(392, 0.13), (523, 0.13), (659, 0.13), (784, 0.18), (1047, 0.42)], 0.38)


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    make_sprites()
    make_sounds()
    print(f"Generated original assets in {ASSETS}")


if __name__ == "__main__":
    main()
