import time
import cv2
import mss
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyautogui
import os
import glob
import sys
import warnings
import keyboard
import ctypes
import soundcard as sc
from tqdm import tqdm
from ctypes import wintypes
from datetime import datetime
from threading import Event, Thread
from x_transformers import Decoder
from prodigyopt import Prodigy

warnings.filterwarnings("ignore")

IMG_H, IMG_W = 128, 256
PATCH_H, PATCH_W = 16, 16
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CKPT_DIR = 'ckpts_predictive_action'
SAVE_INTERVAL = 900
MAX_CKPTS = 3
MOUSE_SPEED = 50
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.0

INPUT_DURATION = 0.3
MOUSE_DURATION = 0.04

KEY_LIST = ['w', 'a', 's', 'd', 'space', 'shift', 'ctrl', 'e', 'q']

CONTROLS_MAP = []
for k in KEY_LIST:
    CONTROLS_MAP.append({'type': 'key', 'val': k})

CONTROLS_MAP.extend([
    {'type': 'click', 'btn': 'left'},
    {'type': 'click', 'btn': 'right'},
    {'type': 'scroll', 'val': 60},
    {'type': 'scroll', 'val': -60},
    {'type': 'move', 'x': 0, 'y': -MOUSE_SPEED},
    {'type': 'move', 'x': 0, 'y': MOUSE_SPEED},
    {'type': 'move', 'x': -MOUSE_SPEED, 'y': 0},
    {'type': 'move', 'x': MOUSE_SPEED, 'y': 0},
])

NUM_CONTROLS = len(CONTROLS_MAP)

stop_event = Event()

if not os.path.exists(CKPT_DIR): os.makedirs(CKPT_DIR)

class WindowMgr:
    def __init__(self):
        self._rect = None
        self._hwnd = None

    def capture_active(self):
        self._hwnd = ctypes.windll.user32.GetForegroundWindow()
        rect = wintypes.RECT()
        ctypes.windll.user32.GetWindowRect(self._hwnd, ctypes.byref(rect))
        self._rect = (rect.left, rect.top, rect.right, rect.bottom)

        length = ctypes.windll.user32.GetWindowTextLengthW(self._hwnd)
        buff = ctypes.create_unicode_buffer(length + 1)
        ctypes.windll.user32.GetWindowTextW(self._hwnd, buff, length + 1)
        return buff.value

    def get_region(self):
        return self._rect

    def force_focus(self):
        if self._hwnd:
            current = ctypes.windll.user32.GetForegroundWindow()
            if current != self._hwnd:
                ctypes.windll.user32.SetForegroundWindow(self._hwnd)

    def lock_cursor(self):
        if not self._rect: return
        rect = wintypes.RECT(*self._rect)
        ctypes.windll.user32.ClipCursor(ctypes.byref(rect))

    def unlock_cursor(self):
        ctypes.windll.user32.ClipCursor(None)

class AudioEar:
    def __init__(self):
        self.left_vol = 0.0
        self.right_vol = 0.0
        self.running = True
        self.thread = Thread(target=self._listen, daemon=True)
        self.thread.start()

    def _listen(self):
        try:
            default_speaker = sc.default_speaker()
            mic = sc.get_microphone(id=str(default_speaker.name), include_loopback=True)
            with mic.recorder(samplerate=44100) as recorder:
                while self.running and not stop_event.is_set():
                    data = recorder.record(numframes=1024)
                    if data.shape[1] >= 2:
                        self.left_vol = np.sqrt(np.mean(data[:, 0]**2))
                        self.right_vol = np.sqrt(np.mean(data[:, 1]**2))
                    else:
                        vol = np.sqrt(np.mean(data**2))
                        self.left_vol = vol
                        self.right_vol = vol
        except Exception:
            self.running = False

    def get_levels(self):
        return np.clip(self.left_vol * 5.0, 0, 1), np.clip(self.right_vol * 5.0, 0, 1)

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=0.5)

class CheckpointManager:
    @staticmethod
    def save(model, opt, step):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(CKPT_DIR, f'predictive_action_{timestamp}.pth')
        torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'step': step}, path)
        files = sorted(glob.glob(os.path.join(CKPT_DIR, 'predictive_action_*.pth')), key=os.path.getmtime, reverse=True)
        for f in files[MAX_CKPTS:]:
            try: os.remove(f)
            except: pass

    @staticmethod
    def load(model, opt):
        files = glob.glob(os.path.join(CKPT_DIR, 'predictive_action_*.pth'))
        if not files: return 0
        latest = max(files, key=os.path.getmtime)
        ckpt = torch.load(latest, map_location=DEVICE)
        model.load_state_dict(ckpt['model'])
        opt.load_state_dict(ckpt['opt'])
        return ckpt.get('step', 0)

class Transformer(nn.Module):
    def __init__(self, image_dim, audio_dim=2, embed_dim=256, heads=8, depth=4, num_actions=NUM_CONTROLS):
        super().__init__()
        self.embed = nn.Linear(image_dim + audio_dim, embed_dim)

        self.decoder = Decoder(
            dim=embed_dim,
            depth=depth,
            heads=heads,
            rotary_pos_emb=True
        )

        self.next_frame_head = nn.Linear(embed_dim, image_dim + audio_dim)
        self.action_head = nn.Linear(embed_dim, num_actions)

    def forward(self, x):
        emb = self.embed(x)
        out = self.decoder(emb)
        last = out[:, -1, :]
        next_pred = self.next_frame_head(last)
        action_logits = self.action_head(last)
        return next_pred, action_logits

def grab_frame(sct, rect):
    mon = {"left": rect[0], "top": rect[1], "width": rect[2]-rect[0], "height": rect[3]-rect[1]}
    raw = np.array(sct.grab(mon))
    img = cv2.resize(raw, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    return img

def image_to_flat(img):
    return (img.astype(np.float32).reshape(-1) / 255.0)

def main():
    wm = WindowMgr()
    print("Open and focus your target window. Locking in 5s...")
    time.sleep(5)
    title = wm.capture_active()
    print(f"LOCKED TARGET: '{title}'")
    wm.lock_cursor()
    wm.force_focus()

    sct = mss.mss()
    audio = AudioEar()

    image_dim = IMG_H * IMG_W * 3
    audio_dim = 2

    model = Transformer(image_dim=image_dim, audio_dim=audio_dim).to(DEVICE)
    opt = Prodigy(model.parameters(), lr=0.04)

    step = CheckpointManager.load(model, opt)

    SEQ_LEN = 4096
    
    ctx_buffer = torch.zeros((SEQ_LEN + 1, image_dim + audio_dim), device=DEVICE)
    ctx_count = 0

    pbar = tqdm(initial=step)

    def on_exit():
        tqdm.write('\n!!! STOP REQUEST RECEIVED (F9) - SAVING AND EXITING !!!')
        stop_event.set()
    keyboard.add_hotkey('f9', on_exit)

    try:
        while not stop_event.is_set():
            rect = wm.get_region()
            if not rect:
                time.sleep(0.1)
                continue

            img = grab_frame(sct, rect)
            l, r = audio.get_levels() if audio.running else (0.0, 0.0)

            flat_numpy = np.concatenate([image_to_flat(img), np.array([l, r], dtype=np.float32)])
            flat_tensor = torch.from_numpy(flat_numpy).to(DEVICE)

            ctx_buffer = torch.roll(ctx_buffer, -1, 0)
            ctx_buffer[-1] = flat_tensor
            
            if ctx_count < SEQ_LEN + 1:
                ctx_count += 1

            if ctx_count < 32:
                continue

            valid_window = ctx_buffer[-ctx_count:]
            inp = valid_window[:-1].unsqueeze(0)

            model.train()
            pred_next, action_logits = model(inp)

            action_probs = torch.sigmoid(action_logits)
            dist = torch.distributions.Bernoulli(action_probs)
            action_sample = dist.sample()
            active_indices = (action_sample[0] == 1).nonzero(as_tuple=True)[0]

            try:
                for idx in active_indices:
                    ctrl = CONTROLS_MAP[idx.item()]
                    if ctrl['type'] == 'key':
                        pyautogui.keyDown(ctrl['val'])
                    elif ctrl['type'] == 'click':
                        pyautogui.mouseDown(button=ctrl['btn'])
                    elif ctrl['type'] == 'scroll':
                        pyautogui.scroll(ctrl['val'])
                    elif ctrl['type'] == 'move':
                        pyautogui.moveRel(ctrl['x'], ctrl['y'], duration=0)

                time.sleep(INPUT_DURATION)

                for idx in active_indices:
                    ctrl = CONTROLS_MAP[idx.item()]
                    if ctrl['type'] == 'key':
                        pyautogui.keyUp(ctrl['val'])
                    elif ctrl['type'] == 'click':
                        pyautogui.mouseUp(button=ctrl['btn'])
            except Exception as e:
                tqdm.write(f"Action execution error: {e}")

            img_after = grab_frame(sct, rect)
            l2, r2 = audio.get_levels() if audio.running else (0.0, 0.0)
            
            flat_after_numpy = np.concatenate([image_to_flat(img_after), np.array([l2, r2], dtype=np.float32)])
            true_after_t = torch.from_numpy(flat_after_numpy).to(DEVICE).unsqueeze(0)

            loss_vis = F.l1_loss(pred_next, true_after_t)

            probs = torch.clamp(action_probs, 1e-6, 1.0 - 1e-6)
            entropy = -(probs * torch.log(probs) + (1 - probs) * torch.log(1 - probs)).mean()
            loss = loss_vis - 1e-3 * entropy

            opt.zero_grad()
            loss.backward()
            opt.step()

            pbar.update(1)
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'vis': f'{loss_vis.item():.4f}', 'ent': f'{entropy.item():.4f}'})
            step += 1

            if time.time() % SAVE_INTERVAL < 1:
                CheckpointManager.save(model, opt, step)

    except KeyboardInterrupt:
        pass
    except Exception as e:
        tqdm.write(f"CRITICAL ERROR: {e}")
    finally:
        pbar.close()
        audio.stop()
        wm.unlock_cursor()
        CheckpointManager.save(model, opt, step)
        print("Clean exit complete.")
        sys.exit(0)

if __name__ == '__main__':
    main()