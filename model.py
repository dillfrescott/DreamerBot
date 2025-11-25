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
PATCH_H, PATCH_W = 32, 32
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CKPT_DIR = 'ckpts_predictive_action'
SAVE_INTERVAL = 900
MAX_CKPTS = 3
MOUSE_SPEED = 50
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.0

INPUT_DURATION = 0.3
MOUSE_DURATION = 0.04
DREAM_HORIZON = 16

KEY_LIST = ['w', 'a', 's', 'd', 'space', 'shift', 'ctrl', 'q', 'e']

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

class HumanListener:
    def __init__(self):
        self.last_mouse_pos = pyautogui.position()
        
    def get_active_vector(self):
        active_vec = torch.zeros(NUM_CONTROLS).to(DEVICE)
        
        curr_mouse_x, curr_mouse_y = pyautogui.position()
        delta_x = curr_mouse_x - self.last_mouse_pos[0]
        delta_y = curr_mouse_y - self.last_mouse_pos[1]
        self.last_mouse_pos = (curr_mouse_x, curr_mouse_y)
        
        vk_lbutton = 0x01
        vk_rbutton = 0x02
        left_clicked = (ctypes.windll.user32.GetAsyncKeyState(vk_lbutton) & 0x8000) != 0
        right_clicked = (ctypes.windll.user32.GetAsyncKeyState(vk_rbutton) & 0x8000) != 0

        for i, ctrl in enumerate(CONTROLS_MAP):
            is_active = False
            
            if ctrl['type'] == 'key':
                if keyboard.is_pressed(ctrl['val']):
                    is_active = True
            
            elif ctrl['type'] == 'click':
                if ctrl['btn'] == 'left' and left_clicked:
                    is_active = True
                elif ctrl['btn'] == 'right' and right_clicked:
                    is_active = True
            
            elif ctrl['type'] == 'move':
                sensitivity_thresh = 5
                if ctrl['y'] < 0 and delta_y < -sensitivity_thresh: is_active = True # Up
                if ctrl['y'] > 0 and delta_y > sensitivity_thresh: is_active = True # Down
                if ctrl['x'] < 0 and delta_x < -sensitivity_thresh: is_active = True # Left
                if ctrl['x'] > 0 and delta_x > sensitivity_thresh: is_active = True # Right
            
            if is_active:
                active_vec[i] = 1.0
                
        return active_vec

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
    def __init__(self, image_dim, audio_dim=2, embed_dim=256, heads=4, depth=4, num_actions=NUM_CONTROLS):
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

    def dream(self, x, steps=5):
        predictions = []
        current_seq = x.clone()

        for _ in range(steps):
            emb = self.embed(current_seq)
            out = self.decoder(emb)
            last = out[:, -1, :]
            next_step_pred = self.next_frame_head(last)
            next_step_token = next_step_pred.unsqueeze(1)
            current_seq = torch.cat((current_seq, next_step_token), dim=1)
            predictions.append(next_step_pred)

        return predictions

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
    
    print("\n=== SYSTEM CONTROL ===")
    print("[1] PLAY MODE (Autonomous Control)")
    print("[2] WATCH MODE (Imitation Learning)")
    mode_sel = input("Select mode (1/2): ").strip()
    
    is_watch_mode = (mode_sel == '2')
    mode_str = "WATCH" if is_watch_mode else "PLAY"
    print(f"initializing {mode_str} mode...")

    print("Open and focus your target window. Locking in 5s...")
    time.sleep(5)
    title = wm.capture_active()
    print(f"LOCKED TARGET: '{title}'")
    
    if not is_watch_mode:
        wm.lock_cursor()
    
    wm.force_focus()

    sct = mss.mss()
    audio = AudioEar()
    human = HumanListener()

    image_dim = IMG_H * IMG_W * 3
    audio_dim = 2

    model = Transformer(image_dim=image_dim, audio_dim=audio_dim).to(DEVICE)
    opt = Prodigy(model.parameters(), lr=0.04)
    criterion_action = nn.BCEWithLogitsLoss()

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
            
            human_actions = None
            if is_watch_mode:
                human_actions = human.get_active_vector()

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
            
            with torch.no_grad():
                imagined_trajectory = model.dream(inp, steps=DREAM_HORIZON)

            pred_next, action_logits = model(inp)

            if not is_watch_mode:
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
            else:
                time.sleep(INPUT_DURATION)

            img_after = grab_frame(sct, rect)
            l2, r2 = audio.get_levels() if audio.running else (0.0, 0.0)
            
            flat_after_numpy = np.concatenate([image_to_flat(img_after), np.array([l2, r2], dtype=np.float32)])
            true_after_t = torch.from_numpy(flat_after_numpy).to(DEVICE).unsqueeze(0)

            loss_vis = F.l1_loss(pred_next, true_after_t)
            
            loss = loss_vis 

            loss_act_val = 0.0
            if is_watch_mode and human_actions is not None:
                loss_act = criterion_action(action_logits, human_actions.unsqueeze(0))
                loss += loss_act
                loss_act_val = loss_act.item()
            else:
                probs = torch.sigmoid(action_logits)
                probs = torch.clamp(probs, 1e-6, 1.0 - 1e-6)
                entropy = -(probs * torch.log(probs) + (1 - probs) * torch.log(1 - probs)).mean()
                loss = loss - 1e-3 * entropy

            opt.zero_grad()
            loss.backward()
            opt.step()

            pbar.update(1)
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}', 
                'vis': f'{loss_vis.item():.4f}', 
                'act': f'{loss_act_val:.4f}'
            })
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
        if not is_watch_mode:
            wm.unlock_cursor()
        CheckpointManager.save(model, opt, step)
        print("Clean exit complete.")
        sys.exit(0)

if __name__ == '__main__':
    main()