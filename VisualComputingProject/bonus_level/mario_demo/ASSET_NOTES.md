# Original Asset Notes

The demo does not contain copied Nintendo or Mario artwork, level layouts, music, or sound effects.

## Generated background

The background at `assets/pixel_landscape.png` was created with the built-in image generation tool using this final prompt:

> Use case: stylized-concept. Asset type: original background art for a single-level 2D side-scrolling platform game. Create a polished original 16-bit pixel-art landscape background for a cheerful platform adventure, inspired by the broad genre of classic 1990s console platformers but not copying any existing game, character, level, logo, or recognizable composition. Bright cyan sky, layered rounded green hills, distant blue mountains, soft blocky clouds, subtle forest silhouettes; empty playable foreground area along the bottom. Crisp 16-bit pixel art, deliberately limited palette, hard pixel edges, seamless-looking horizontal composition. Background only, no foreground platforms, characters, enemies, items, HUD, text, logos, watermark, Mario, Nintendo, mushrooms, question blocks, pipes, or recognizable copyrighted game assets.

## Locally generated assets

`build_assets.py` deterministically creates:

- An original side-view red-cap plumber adventurer with a red shirt, blue overalls, white gloves, brown shoes, a large profile nose, and a moustache. Idle, two run frames, jump, and crouch each have deterministic left- and right-facing versions. The sprites are original and are not traced or pixel-identical copies of a commercial character.
- An original purple beetle-like enemy.
- A star token, teal goal pennant, ground tiles, and floating-platform tiles.
- Original square-wave chiptune effects for jump, token collection, enemy stomp, damage, and level completion.

These assets can be regenerated with:

```powershell
python build_assets.py
```
