import cv2
import torch
import torch.nn.functional as F
import numpy as np
import os
import glob
from tqdm import tqdm
from datetime import datetime
from prodigyopt import Prodigy
from model import Transformer, IMG_H, IMG_W, DEVICE, CKPT_DIR, NUM_CONTROLS

VIDEO_DIR = 'training_videos'
SEQ_LEN = 2048

def train_on_video(video_path, model, opt, step):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Training on {os.path.basename(video_path)} ({total_frames} frames)...")
    
    image_dim = IMG_H * IMG_W * 3
    audio_dim = 2

    total_input_dim = image_dim + audio_dim + NUM_CONTROLS
    
    ctx_buffer = torch.zeros((SEQ_LEN + 1, total_input_dim), device=DEVICE)
    ctx_count = 0
    
    dummy_action = np.zeros(NUM_CONTROLS, dtype=np.float32)

    pbar = tqdm(total=total_frames)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.resize(frame, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        flat_img = (frame.astype(np.float32).reshape(-1) / 255.0)
        flat_audio = np.array([0.0, 0.0], dtype=np.float32)
        
        flat_numpy = np.concatenate([flat_img, flat_audio, dummy_action])
        flat_tensor = torch.from_numpy(flat_numpy).to(DEVICE)

        ctx_buffer = torch.roll(ctx_buffer, -1, 0)
        ctx_buffer[-1] = flat_tensor
        if ctx_count < SEQ_LEN + 1: ctx_count += 1

        if ctx_count < 32:
            pbar.update(1)
            continue

        valid_window = ctx_buffer[-ctx_count:]
        inp = valid_window[:-1].unsqueeze(0)
        
        target_state_size = image_dim + audio_dim
        true_next_state = valid_window[-1, :target_state_size].unsqueeze(0)

        model.train()
        pred_next_state = model(inp)

        loss = F.l1_loss(pred_next_state, true_next_state)

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

    try:
        model.load_state_dict(state_dict)
        opt.load_state_dict(ckpt['opt'])
        return ckpt.get('step', 0)
    except RuntimeError:
        print("!!! ARCHITECTURE MISMATCH !!!")
        print("This likely means the checkpoint is from the old (non-action) version.")
        print("Starting fresh/partial load...")
        return 0

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
