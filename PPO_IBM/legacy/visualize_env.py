import pygame
import numpy as np
import sys
import argparse
import os
import csv
from datetime import datetime
import matplotlib.pyplot as plt
from recurrent_ppo import ActionSmoothnessWrapper

ENV_DIR = os.path.join(os.path.dirname(__file__), 'environments')
if ENV_DIR not in sys.path:
    sys.path.append(ENV_DIR)

GeneticPhotobioreactorEnv  = None
AlphaPhotobioreactorEnv    = None
HeavyPhotobioreactorEnv    = None
TotalPhotobioreactorEnv    = None
_IMPORT_ERRORS = {}

try:    from genetic_env import GeneticPhotobioreactorEnv
except Exception as exc:  _IMPORT_ERRORS['genetic'] = exc
try:    from alpha_env   import AlphaPhotobioreactorEnv
except Exception as exc:  _IMPORT_ERRORS['alpha']   = exc
try:    from heavy_env   import HeavyPhotobioreactorEnv
except Exception as exc:  _IMPORT_ERRORS['heavy']   = exc
try:    from total_env   import TotalPhotobioreactorEnv
except Exception as exc:  _IMPORT_ERRORS['total']   = exc

# ── Layout ──────────────────────────────────────────────────────────────────────
TANK_W    = 900
TANK_H    = 700
UI_W      = 360
WIN_W     = TANK_W + UI_W
WIN_H     = TANK_H
FPS       = 60
SPARK_LEN = 300      # rolling history for sparklines

# ── Premium dark colour palette ──────────────────────────────────────────────────
C_WATER_TOP = (10,  50,  82)
C_WATER_BOT = ( 5,  22,  45)
C_TEXT      = (200, 215, 235)
C_UI_BG     = ( 12,  18,  30)
C_BORDER    = ( 30,  50,  80)
C_SEP       = ( 40,  65,  95)
C_GREEN     = ( 50, 220, 120)
C_BLUE      = ( 80, 160, 255)
C_YELLOW    = (255, 210,  80)
C_RED       = (255,  70,  70)
C_ORANGE    = (255, 145,  40)
C_WIDGET    = ( 18,  28,  46)


# ══════════════════════════════════════════════════════════════════════
#  CO2 Bubble particle system
# ══════════════════════════════════════════════════════════════════════

class BubbleSystem:
    MAX = 150

    def __init__(self):
        # each entry: [x, y, radius, alpha, rise_speed, wobble_vx]
        self.pool: list = []

    def update(self, co2_norm: float):
        for _ in range(int(co2_norm * 6)):
            if len(self.pool) < self.MAX:
                self.pool.append([
                    float(np.random.uniform(40, TANK_W - 40)),   # x
                    float(TANK_H - np.random.uniform(5, 25)),     # y
                    float(np.random.uniform(2, 5)),               # radius
                    230.0,                                         # alpha
                    float(np.random.uniform(0.7, 2.0) * (0.5 + co2_norm * 0.5)),  # vy
                    float(np.random.uniform(-0.4, 0.4)),          # wobble vx
                ])
        alive = []
        for b in self.pool:
            b[1] -= b[4]   # rise
            b[0] += b[5]   # wobble
            b[3] -= 4      # fade
            if b[3] > 0 and b[1] > 0:
                alive.append(b)
        self.pool = alive

    def draw(self, surface: pygame.Surface, co2_norm: float):
        for b in self.pool:
            x, y, r, a = b[0], b[1], b[2], b[3]
            a2 = int(min(255, max(0, a)))
            int_r = max(1, int(r))
            s = pygame.Surface((int_r * 4, int_r * 4), pygame.SRCALPHA)
            col = (int(100 + 155 * co2_norm), 200, 255)
            pygame.draw.circle(s, (*col, a2 // 3), (int_r * 2, int_r * 2), int_r * 2)
            pygame.draw.circle(s, (*col, a2),       (int_r * 2, int_r * 2), int_r, 1)
            surface.blit(s, (int(x) - int_r * 2, int(y) - int_r * 2))


# ══════════════════════════════════════════════════════════════════════
#  Stirring vortex flow particle
# ══════════════════════════════════════════════════════════════════════

class FlowParticle:
    def __init__(self):
        self._spawn()

    def _spawn(self):
        self.x        = float(np.random.uniform(0, TANK_W))
        self.y        = float(np.random.uniform(0, TANK_H))
        self.life     = int(np.random.randint(20, 80))
        self.max_life = self.life
        self.trail: list = []

    def update(self, stir_norm: float):
        if self.life <= 0:
            self._spawn()
            return
        cx, cy = TANK_W / 2.0, TANK_H / 2.0
        dx = self.x - cx
        dy = self.y - cy
        dist = max(1.0, float(np.hypot(dx, dy)))
        spd  = stir_norm * 3.0
        vx   = (-dy / dist) * spd + (-dx / dist) * 0.15
        vy   = ( dx / dist) * spd + (-dy / dist) * 0.15
        self.trail.append((self.x, self.y))
        if len(self.trail) > 7:
            self.trail.pop(0)
        self.x = float(np.clip(self.x + vx, 0, TANK_W))
        self.y = float(np.clip(self.y + vy, 0, TANK_H))
        self.life -= 1

    def draw(self, surface: pygame.Surface, stir_norm: float):
        if len(self.trail) < 2:
            return
        frac      = self.life / max(1, self.max_life)
        intensity = int(55 * frac * stir_norm)
        col = (intensity, intensity + 15, intensity + 35)
        for i in range(1, len(self.trail)):
            pygame.draw.line(
                surface, col,
                (int(self.trail[i - 1][0]), int(self.trail[i - 1][1])),
                (int(self.trail[i    ][0]), int(self.trail[i    ][1])), 1)


# ══════════════════════════════════════════════════════════════════════
#  Inline sparkline graph widget
# ══════════════════════════════════════════════════════════════════════

class Sparkline:
    def __init__(self, rect: pygame.Rect, label: str, color: tuple,
                 min_val: float, max_val: float, ref_lines=None):
        self.rect      = rect
        self.label     = label
        self.color     = color
        self.min_val   = min_val
        self.max_val   = max_val
        self.ref_lines = ref_lines or []   # list of (value, RGBA)
        self.history: list = []

    def push(self, v: float):
        self.history.append(float(v))
        if len(self.history) > SPARK_LEN:
            self.history.pop(0)

    def draw(self, surface: pygame.Surface, font_sm: pygame.font.Font):
        r = self.rect
        pygame.draw.rect(surface, C_WIDGET, r, border_radius=4)
        pygame.draw.rect(surface, C_SEP,    r, 1, border_radius=4)

        # Optional reference lines (e.g., optimal pH zone)
        for ref_v, ref_col in self.ref_lines:
            norm = np.clip((ref_v - self.min_val) / max(1e-9, self.max_val - self.min_val), 0, 1)
            ry   = r.bottom - int(norm * r.h)
            rs   = pygame.Surface((r.w, 1), pygame.SRCALPHA)
            rs.fill(ref_col)
            surface.blit(rs, (r.left, ry))

        # Data line
        if len(self.history) >= 2:
            n   = len(self.history)
            pts = []
            for i, v in enumerate(self.history):
                norm = np.clip((v - self.min_val) / max(1e-9, self.max_val - self.min_val), 0, 1)
                pts.append((r.left + int(i / (n - 1) * r.w),
                             r.bottom - int(norm * r.h)))
            pygame.draw.lines(surface, self.color, False, pts, 2)

        # Label + live value
        surface.blit(font_sm.render(self.label, True, C_TEXT),
                     (r.left + 4, r.top + 2))
        if self.history:
            val_s = font_sm.render(f"{self.history[-1]:.4f}", True, self.color)
            surface.blit(val_s, (r.right - val_s.get_width() - 4, r.top + 2))


# ══════════════════════════════════════════════════════════════════════
#  Slider bar helper
# ══════════════════════════════════════════════════════════════════════

def draw_slider(surface, font_sm, x, y, w, value, lo, hi, label, unit, hint, col):
    norm = float(np.clip((value - lo) / max(1e-9, hi - lo), 0, 1))
    H    = 10
    surface.blit(
        font_sm.render(f"{hint}  {label}: {value:.1f} {unit}", True, C_TEXT),
        (x, y))
    y += 14
    track = pygame.Rect(x, y, w, H)
    pygame.draw.rect(surface, C_WIDGET, track, border_radius=4)
    pygame.draw.rect(surface, C_SEP,    track, 1, border_radius=4)
    fill_w = max(0, int(norm * (w - 2)))
    if fill_w:
        pygame.draw.rect(surface, col, pygame.Rect(x + 1, y + 1, fill_w, H - 2), border_radius=3)
    # Thumb knob
    pygame.draw.circle(surface, col, (x + fill_w, y + H // 2), H // 2 + 1)


# ══════════════════════════════════════════════════════════════════════
#  Main visualiser
# ══════════════════════════════════════════════════════════════════════

class PBRVisualizer:

    def __init__(self, env_type="genetic", **kwargs):
        pygame.init()
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        pygame.display.set_caption(f"Photobioreactor  ·  {env_type.upper()}")
        self.clock = pygame.time.Clock()

        self.font_sm  = pygame.font.SysFont("Consolas", 13)
        self.font     = pygame.font.SysFont("Consolas", 15)
        self.font_hd  = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_big = pygame.font.SysFont("Consolas", 22, bold=True)

        self.env_type = env_type
        difficulty    = int(kwargs.get("difficulty", 2))
        self.record_episode = bool(kwargs.get("record_episode", False))
        self.record_dir     = kwargs.get("record_dir", "recorded_episodes")
        self.record_prefix  = kwargs.get("record_prefix", "episode")
        self.saved_episode_count = 0
        self.current_episode_trace = []

        if env_type == "genetic":
            if GeneticPhotobioreactorEnv is None:
                raise ImportError(f"Failed to import genetic env: {_IMPORT_ERRORS.get('genetic')}")
            self.env = GeneticPhotobioreactorEnv(max_cells=300000, initial_cells=3000, difficulty=difficulty)
        elif env_type == "alpha":
            if AlphaPhotobioreactorEnv is None:
                raise ImportError(f"Failed to import alpha env: {_IMPORT_ERRORS.get('alpha')}")
            self.env = AlphaPhotobioreactorEnv(max_cells=300000, initial_cells=3000, difficulty=difficulty)
        elif env_type == "total":
            if TotalPhotobioreactorEnv is None:
                raise ImportError(f"Failed to import total env: {_IMPORT_ERRORS.get('total')}")
            self.env = TotalPhotobioreactorEnv(max_cells=300000, initial_cells=8000, difficulty=difficulty)
        else:
            if HeavyPhotobioreactorEnv is None:
                raise ImportError(f"Failed to import heavy env: {_IMPORT_ERRORS.get('heavy')}")
            self.env = HeavyPhotobioreactorEnv(max_cells=300000, initial_cells=8000, difficulty=difficulty)

        print(f"Visualizer: env={env_type}  difficulty=D{difficulty}")
        if self.record_episode:
            os.makedirs(self.record_dir, exist_ok=True)
            print(f"Recording enabled → {self.record_dir}")

        self.wrapped_env = ActionSmoothnessWrapper(self.env)
        self.obs, _ = self.wrapped_env.reset()
        self.action  = np.zeros(4, dtype=np.float32)

        # ── Particle systems ──────────────────────────────────────────
        self.MAX_VIS    = 12000
        self.bubbles    = BubbleSystem()
        self.flow_ptcl  = [FlowParticle() for _ in range(80)]

        # ── Division flash + crash warning state ──────────────────────
        self.prev_active  = int(getattr(self.env, "num_active", 0))
        self.flash_frames = 0
        self.warn_frames  = 0

        # ── Pre-baked water background (constant, built once) ─────────
        self._water_surf = self._make_water_surf()

        # ── Cached light-penetration overlay (invalidated on change) ──
        self._light_surf  = pygame.Surface((TANK_W, TANK_H), pygame.SRCALPHA)
        self._light_cache = None   # (rounded light_norm, rounded od)

        # ── Sparkline widgets ─────────────────────────────────────────
        sw = UI_W - 24
        self.od_spark = Sparkline(
            pygame.Rect(TANK_W + 12, 0, sw, 64),
            "OD", C_GREEN, 0.0, 0.15)
        self.ph_spark = Sparkline(
            pygame.Rect(TANK_W + 12, 0, sw, 64),
            "pH", C_BLUE, 6.0, 13.0,
            ref_lines=[(10.5, (80, 255, 80, 55))])

    # ─────────────────────────────── background helpers ────────────────────────

    @staticmethod
    def _make_water_surf() -> pygame.Surface:
        surf = pygame.Surface((TANK_W, TANK_H))
        for row in range(TANK_H):
            t = row / TANK_H
            r = int(C_WATER_TOP[0] + (C_WATER_BOT[0] - C_WATER_TOP[0]) * t)
            g = int(C_WATER_TOP[1] + (C_WATER_BOT[1] - C_WATER_TOP[1]) * t)
            b = int(C_WATER_TOP[2] + (C_WATER_BOT[2] - C_WATER_TOP[2]) * t)
            pygame.draw.line(surf, (r, g, b), (0, row), (TANK_W, row))
        return surf

    def _update_light_surf(self, light_norm: float, od: float):
        """Rebuild the alpha light-gradient surface only when inputs change."""
        key = (round(light_norm, 2), round(od, 4))
        if key == self._light_cache:
            return
        self._light_cache = key
        self._light_surf.fill((0, 0, 0, 0))
        # OD self-shading: higher OD → shorter light penetration depth
        penetration = max(20, int(TANK_H * np.exp(-od * 70)))
        N = 24   # gradient bands
        for i in range(N):
            top  = int(i * penetration / N)
            h    = max(1, int(penetration / N))
            fade = (1.0 - i / N) ** 1.4
            a    = int(light_norm * fade * 90)
            rc   = min(255, int(255 * fade * light_norm))
            gc   = min(255, int(220 * fade * light_norm))
            bc   = min(255, int( 60 * fade * light_norm))
            pygame.draw.rect(self._light_surf, (rc, gc, bc, a), (0, top, TANK_W, h))

    # ─────────────────────────────── recording ─────────────────────────────────

    def _record_step(self, reward: float):
        if not self.record_episode:
            return
        self.current_episode_trace.append({
            "step":             int(self.env.step_count),
            "reward":           float(reward),
            "pop":              int(getattr(self.env, "num_active", 0)),
            "od":               float(self.obs[0]),
            "ph":               float(self.obs[1]),
            "nutrients":        float(self.obs[2]),
            "temp":             float(self.obs[3]),
            "conductivity":     float(self.obs[4]) if len(self.obs) > 4 else 0.0,
            "rgb":              float(self.obs[5]) if len(self.obs) > 5 else 0.0,
            "action_stir":      float(self.action[0]),
            "action_light":     float(self.action[1]),
            "action_nutrients": float(self.action[2]),
            "action_co2":       float(self.action[3]),
        })

    def _save_episode_trace(self):
        if not self.record_episode or not self.current_episode_trace:
            return
        self.saved_episode_count += 1
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = (
            f"{self.record_prefix}_{self.env_type}"
            f"_D{getattr(self.env,'difficulty',0)}"
            f"_ep{self.saved_episode_count:03d}_{stamp}.csv"
        )
        path = os.path.join(self.record_dir, fname)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(self.current_episode_trace[0].keys()))
            writer.writeheader()
            writer.writerows(self.current_episode_trace)
        print(f"  Episode saved → {path}")

    # ─────────────────────────────── input ─────────────────────────────────────

    def handle_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
        step = 0.04
        keys = pygame.key.get_pressed()
        if   keys[pygame.K_UP]:    self.action[0] = min( 1.0, self.action[0] + step)
        elif keys[pygame.K_DOWN]:  self.action[0] = max(-1.0, self.action[0] - step)
        if   keys[pygame.K_RIGHT]: self.action[1] = min( 1.0, self.action[1] + step)
        elif keys[pygame.K_LEFT]:  self.action[1] = max(-1.0, self.action[1] - step)
        if   keys[pygame.K_w]:     self.action[2] = min( 1.0, self.action[2] + step)
        elif keys[pygame.K_s]:     self.action[2] = max(-1.0, self.action[2] - step)
        if   keys[pygame.K_d]:     self.action[3] = min( 1.0, self.action[3] + step)
        elif keys[pygame.K_a]:     self.action[3] = max(-1.0, self.action[3] - step)

    # ─────────────────────────────── tank drawing ──────────────────────────────

    def draw_tank(self):
        # 1. Static water gradient
        self.screen.blit(self._water_surf, (0, 0))

        # 2. Dynamic light penetration overlay
        light_norm = float(np.interp(self.action[1], [-1, 1], [0, 1]))
        od_val     = float(self.obs[0]) if len(self.obs) > 0 else 0.0
        self._update_light_surf(light_norm, od_val)
        self.screen.blit(self._light_surf, (0, 0))

        # 3. Stirring vortex flow particles (rendered behind algae)
        stir_rpm  = float(np.interp(self.action[0], [-1, 1], [50, 200]))
        stir_norm = (stir_rpm - 50) / 150.0
        for fp in self.flow_ptcl:
            fp.update(stir_norm)
            fp.draw(self.screen, stir_norm)

        # 4. Algae particles ────────────────────────────────────────────
        num_active  = self.env.num_active
        active_mask = self.env.active_mask

        if num_active > 0:
            z_pos    = self.env.cells_z[active_mask]
            screen_y = (z_pos / self.env.reactor_depth) * TANK_H

            if hasattr(self.env, "cells_x"):
                screen_x = (self.env.cells_x[active_mask] / self.env.reactor_width) * TANK_W
            else:
                screen_x = np.random.uniform(0, TANK_W, num_active)

            # Particle size from clump mass
            if hasattr(self.env, "clump_mass"):
                radii = np.sqrt(self.env.clump_mass[active_mask]) * 2
            else:
                radii = np.ones(num_active) * 2

            # ── Health & stress colour ────────────────────────────────
            pigment = float(getattr(self.env, "pigment", 1.0))
            f_ph    = float(getattr(self.env, "f_pH",    1.0))
            health  = float(np.clip(f_ph * pigment, 0.0, 1.0))

            # Three-zone gradient: green → yellow → red
            if health > 0.5:
                t = (1.0 - health) * 2.0
                base_r = int(50  + 150 * t)
                base_g = 200
                base_b = int(50  -  30 * t)
            else:
                t = health * 2.0
                base_r = int(220 -  20 * t)
                base_g = int( 40 + 160 * t)
                base_b = int( 20 +  30 * t)

            # Pigment bleach overlay (stressed → yellow-white)
            base_r = int(base_r * pigment + 220 * (1 - pigment))
            base_g = int(base_g * pigment + 200 * (1 - pigment))
            base_b = int(base_b * pigment + 100 * (1 - pigment))

            # Per-particle light attenuation with depth
            light_factor = (
                np.exp(-3.0 * (z_pos / self.env.reactor_depth))
                * (0.25 + 0.75 * light_norm)
            )

            # Division flash brightness boost
            fb = int(self.flash_frames / 10 * 90) if self.flash_frames > 0 else 0

            # Downsample for GPU-friendly draw call count
            vis_n = min(num_active, self.MAX_VIS)
            idx   = np.random.choice(num_active, vis_n, replace=False)

            for i in idx:
                lf  = float(light_factor[i])
                col = (
                    min(255, max(0, int(base_r * lf) + fb)),
                    min(255, max(0, int(base_g * lf) + fb)),
                    min(255, max(0, int(base_b * lf))),
                )
                pygame.draw.circle(
                    self.screen, col,
                    (int(screen_x[i]), int(screen_y[i])),
                    max(1, int(radii[i])))

        # 5. CO2 bubble system (front layer)
        co2_norm = float(np.interp(self.action[3], [-1, 1], [0, 1]))
        self.bubbles.update(co2_norm)
        self.bubbles.draw(self.screen, co2_norm)

        # 6. Division flash frame countdown
        if self.flash_frames > 0:
            self.flash_frames -= 1

        # 7. Crash / warning border  (pulsing red)
        if self.warn_frames > 0:
            pulse     = abs((self.warn_frames % 30) - 15) / 15.0
            intensity = int(210 * pulse)
            pygame.draw.rect(self.screen, (intensity, 0, 0),
                             pygame.Rect(0, 0, TANK_W, TANK_H), 5)
            self.warn_frames -= 1

    # ─────────────────────────────── UI panel ──────────────────────────────────

    def draw_ui(self, reward: float):
        pygame.draw.rect(self.screen, C_UI_BG, pygame.Rect(TANK_W, 0, UI_W, WIN_H))
        pygame.draw.line(self.screen, C_BORDER, (TANK_W, 0), (TANK_W, WIN_H), 2)

        x = TANK_W + 12
        y = 12

        # ── Title ──────────────────────────────────────────────────────
        title = f"  PBR  {self.env_type.upper()}"
        if hasattr(self.env, "difficulty"):
            title += f"  [D{self.env.difficulty}]"
        self.screen.blit(self.font_big.render(title, True, C_YELLOW), (x, y)); y += 30

        self.screen.blit(self.font.render(f"Step {self.env.step_count:>7}", True, C_TEXT), (x, y))
        self.screen.blit(self.font.render(f"Pop  {self.env.num_active:>7,}", True, C_TEXT), (x + 155, y)); y += 22

        rew_col = C_GREEN if reward >= 0 else C_RED
        self.screen.blit(self.font_hd.render(f"Reward:  {reward:+.4f}", True, rew_col), (x, y)); y += 28

        pygame.draw.line(self.screen, C_SEP, (x, y), (x + UI_W - 24, y)); y += 10

        # ── Sensors with colour-coded bars ─────────────────────────────
        self.screen.blit(self.font_hd.render("Sensors", True, C_BLUE), (x, y)); y += 22

        def sensor_row(label, val, fmt, unit, lo, hi, opt_lo, opt_hi):
            nonlocal y
            spread = (opt_hi - opt_lo) * 0.5
            if opt_lo <= val <= opt_hi:
                col = C_GREEN
            elif (opt_lo - spread) <= val <= (opt_hi + spread):
                col = C_YELLOW
            else:
                col = C_RED
            self.screen.blit(
                self.font.render(f"{label:<5} {val:{fmt}} {unit}", True, col),
                (x, y)); y += 15
            bar_w = UI_W - 24
            norm  = float(np.clip((val - lo) / max(1e-9, hi - lo), 0, 1))
            pygame.draw.rect(self.screen, C_WIDGET, (x, y, bar_w, 5), border_radius=2)
            fill = max(0, int(norm * bar_w))
            if fill:
                pygame.draw.rect(self.screen, col, (x, y, fill, 5), border_radius=2)
            y += 11

        od_v = float(self.obs[0]) if len(self.obs) > 0 else 0.0
        ph_v = float(self.obs[1]) if len(self.obs) > 1 else 8.0
        nu_v = float(self.obs[2]) if len(self.obs) > 2 else 0.0
        te_v = float(self.obs[3]) if len(self.obs) > 3 else 25.0

        sensor_row("OD",   od_v, ".4f", "",      0.0,  0.15,  0.005, 0.100)
        sensor_row("pH",   ph_v, ".2f", "",      6.0,  13.0,  9.5,   11.0)
        sensor_row("N",    nu_v, ".1f", "mg/L",  0.0, 500.0, 100.0, 400.0)
        sensor_row("Temp", te_v, ".1f", "°C",   10.0,  42.0, 25.0,   32.0)
        if len(self.obs) > 4:
            sensor_row("Cond", float(self.obs[4]), ".0f", "µS", 0, 5000, 1000, 3500)
        if len(self.obs) > 5:
            sensor_row("RGB",  float(self.obs[5]), ".3f", "",   0,  1.0,  0.3,   0.9)

        y += 6
        pygame.draw.line(self.screen, C_SEP, (x, y), (x + UI_W - 24, y)); y += 10

        # ── Control sliders ────────────────────────────────────────────
        self.screen.blit(self.font_hd.render("Controls", True, C_ORANGE), (x, y)); y += 22

        stir_rpm = float(np.interp(self.action[0], [-1, 1], [50,   200]))
        light_ue = float(np.interp(self.action[1], [-1, 1], [ 0,  2000]))
        nut_flow = float(np.interp(self.action[2], [-1, 1], [ 0,   100]))
        co2_max  = float(getattr(self.env, "max_co2_flow_lpm", 0.005) * 1000)
        co2_spar = float(np.interp(self.action[3], [-1, 1], [ 0, co2_max]))

        sw = UI_W - 24
        for label, val, lo, hi, unit, hint, col in [
            ("Stir",  stir_rpm,  50,   200,       "RPM",    "[↑↓]", C_BLUE),
            ("Light", light_ue,   0,  2000,       "µE",     "[←→]", C_YELLOW),
            ("Nutr",  nut_flow,   0,   100,       "mg/hr",  "[W/S]", C_GREEN),
            ("CO₂",   co2_spar,   0,   co2_max,   "mL/min", "[A/D]", (160, 220, 255)),
        ]:
            draw_slider(self.screen, self.font_sm, x, y, sw,
                        val, lo, hi, label, unit, hint, col)
            y += 34

        y += 4
        pygame.draw.line(self.screen, C_SEP, (x, y), (x + UI_W - 24, y)); y += 10

        # ── Live sparklines ────────────────────────────────────────────
        self.screen.blit(self.font_hd.render("Live Trends", True, (160, 185, 230)), (x, y)); y += 22

        self.od_spark.rect.y = y
        self.od_spark.draw(self.screen, self.font_sm)
        y += self.od_spark.rect.h + 8

        self.ph_spark.rect.y = y
        self.ph_spark.draw(self.screen, self.font_sm)

    # ─────────────────────────────── post-episode plot ─────────────────────────

    def plot_od_history(self):
        if not self.od_spark.history:
            return
        plt.figure(figsize=(9, 5))
        plt.plot(self.od_spark.history, color="#32dc78", linewidth=2)
        plt.title(f"OD Growth Curve — {self.env_type.upper()}", color="white")
        plt.xlabel("Step", color="white")
        plt.ylabel("OD",   color="white")
        plt.gca().set_facecolor("#0c1520")
        plt.gcf().patch.set_facecolor("#0c1520")
        plt.tick_params(colors="white")
        plt.grid(True, alpha=0.15, color="white")
        plt.tight_layout()
        plt.savefig("latest_od_plot.png", dpi=150, facecolor="#0c1520")
        print("  Growth curve → latest_od_plot.png")
        plt.show(block=False)
        plt.pause(0.5)

    # ─────────────────────────────── main loop ─────────────────────────────────

    def run(self):
        total_reward = 0.0
        while True:
            self.handle_input()
            self.obs, reward, done, _, info = self.wrapped_env.step(self.action)
            total_reward += reward
            self._record_step(reward)

            # Feed sparklines
            if len(self.obs) > 0: self.od_spark.push(self.obs[0])
            if len(self.obs) > 1: self.ph_spark.push(self.obs[1])

            # Division burst detection → flash
            cur_active = int(getattr(self.env, "num_active", 0))
            if cur_active > self.prev_active * 1.025 and cur_active > 200:
                self.flash_frames = min(18, self.flash_frames + 8)
            self.prev_active = cur_active

            # Crash warning border
            if 0 < cur_active < 400:
                self.warn_frames = max(self.warn_frames, 60)

            if done:
                print(f"Episode Done  total_reward={total_reward:.2f}")
                self._save_episode_trace()
                self.plot_od_history()
                self.obs, _ = self.wrapped_env.reset()
                total_reward = 0.0
                self.od_spark.history.clear()
                self.ph_spark.history.clear()
                self.current_episode_trace.clear()
                self.warn_frames  = 0
                self.flash_frames = 0

            self.draw_tank()
            self.draw_ui(reward)
            pygame.display.flip()
            self.clock.tick(FPS)


# ── Entry point ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PBR Visualizer")
    parser.add_argument("--env",            type=str, default="genetic",
                        choices=["genetic", "alpha", "heavy", "total"])
    parser.add_argument("--difficulty",     type=int, default=2, choices=[0, 1, 2])
    parser.add_argument("--record-episode", action="store_true")
    parser.add_argument("--record-dir",     type=str, default="recorded_episodes")
    parser.add_argument("--record-prefix",  type=str, default="episode")
    args = parser.parse_args()

    PBRVisualizer(
        env_type=args.env,
        difficulty=args.difficulty,
        record_episode=args.record_episode,
        record_dir=args.record_dir,
        record_prefix=args.record_prefix,
    ).run()
