import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import glob
import json
import sys
from tqdm import tqdm
from datetime import datetime
from x_transformers import Decoder
from prodigyopt import Prodigy

IMG_H, IMG_W = 128, 256
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CKPT_DIR = 'ckpts_predictive_action'
VIDEO_DIR = 'training_videos'
CONTROLS_JSON = 'controls_allowlist.json'
SEQ_LEN = 4096
DREAM_HORIZON = 16

def get_num_controls():
    if os.path.exists(CONTROLS_JSON):
        try:
            with open(CONTROLS_JSON, 'r') as f:
                data = json.load(f)
                return len(data.get('keys',[])) + len(data.get('mouse',[])) + 6 
        except:
            pass
    return 15

NUM_CONTROLS = get_num_controls()

class Transformer(nn.Module):
    def __init__(self, image_dim, audio_dim=2, embed_dim=256, heads=4, depth=4, num_actions=NUM_CONTROLS):
        super().__init__()
        self.embed = nn.Linear(image_dim + audio_dim, embed_dim)
        self.decoder = Decoder(dim=embed_dim, depth=depth, heads=heads, rotary_pos_emb=True)
        self.next_frame_head = nn.Linear(embed_dim, image_dim + audio_dim)
        self.action_head = nn.Linear(embed_dim, num_actions)

    def forward(self, x):
        emb = self.embed(x)
        out = self.decoder(emb)
        last = out[:, -1, :]
        return self.next_frame_head(last)

def train_on_video(video_path, model, opt, step):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Training on {os.path.basename(video_path)} ({total_frames} frames)...")
    
    image_dim = IMG_H * IMG_W * 3
    audio_dim = 2
    
    ctx_buffer = torch.zeros((SEQ_LEN + 1, image_dim + audio_dim), device=DEVICE)
    ctx_count = 0
    
    pbar = tqdm(total=total_frames)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.resize(frame, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        flat_img = (frame.astype(np.float32).reshape(-1) / 255.0)
        flat_audio = np.array([0.0, 0.0], dtype=np.float32)
        
        flat_numpy = np.concatenate([flat_img, flat_audio])
        flat_tensor = torch.from_numpy(flat_numpy).to(DEVICE)

        ctx_buffer = torch.roll(ctx_buffer, -1, 0)
        ctx_buffer[-1] = flat_tensor
        if ctx_count < SEQ_LEN + 1: ctx_count += 1

        if ctx_count < 32:
            pbar.update(1)
            continue

        valid_window = ctx_buffer[-ctx_count:]
        inp = valid_window[:-1].unsqueeze(0)
        true_next = valid_window[-1].unsqueeze(0).unsqueeze(0)

        model.train()
        pred_next = model(inp)

        loss = F.l1_loss(pred_next, true_next)

        opt.zero_grad()
        loss.backward()
        opt.step()

        pbar.set_postfix({'loss_vis': f'{loss.item():.4f}'})
        pbar.update(1)
        step += 1
        
        if step % 1000 == 0:
            save_checkpoint(model, opt, step)

    cap.release()
    return step

def save_checkpoint(model, opt, step):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(CKPT_DIR, f'predictive_action_{timestamp}.pth')
    torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'step': step}, path)
    
    files = sorted(glob.glob(os.path.join(CKPT_DIR, 'predictive_action_*.pth')), key=os.path.getmtime, reverse=True)
    for f in files[3:]:
        try: os.remove(f)
        except: pass

def load_checkpoint(model, opt):
    files = glob.glob(os.path.join(CKPT_DIR, 'predictive_action_*.pth'))
    if not files: return 0
    latest = max(files, key=os.path.getmtime)
    print(f"Loading checkpoint: {latest}")
    
    ckpt = torch.load(latest, map_location=DEVICE)
    state_dict = ckpt['model']

    saved_actions = state_dict['action_head.weight'].shape[0]
    current_actions = model.action_head.weight.shape[0]

    if saved_actions != current_actions:
        print(f"!!! ACTION SPACE MISMATCH ({saved_actions} vs {current_actions}) !!!")
        print("Adapting Model: Keeping Vision Brain, Resetting Action Muscles.")
        
        del state_dict['action_head.weight']
        del state_dict['action_head.bias']
        
        model.load_state_dict(state_dict, strict=False)
        print("Note: Optimizer reset due to architecture change.")
        return ckpt.get('step', 0)
    else:
        model.load_state_dict(state_dict)
        opt.load_state_dict(ckpt['opt'])
        return ckpt.get('step', 0)

def main():
    if not os.path.exists(VIDEO_DIR):
        os.makedirs(VIDEO_DIR)
        print(f"Created folder '{VIDEO_DIR}'. Put .mp4 files there and run again.")
        return

    videos = glob.glob(os.path.join(VIDEO_DIR, "*.mp4"))
    if not videos:
        print(f"No videos found in '{VIDEO_DIR}'.")
        return

    image_dim = IMG_H * IMG_W * 3
    audio_dim = 2
    
    model = Transformer(image_dim=image_dim, audio_dim=audio_dim).to(DEVICE)
    opt = Prodigy(model.parameters(), lr=0.04)
    
    step = load_checkpoint(model, opt)
    
    print(f"Starting Video Pre-training from step {step}")
    
    try:
        for video_path in videos:
            step = train_on_video(video_path, model, opt, step)
            save_checkpoint(model, opt, step)
    except KeyboardInterrupt:
        print("Stopped by user.")
    
    print("Video training batch complete.")

if __name__ == "__main__":
    main()