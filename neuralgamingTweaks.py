import asyncio
asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # Fix asyncio warning on Windows

from vpython import *
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
import time
from pathlib import Path
from math import sqrt  # For 2D distance calculation

# **Device Setup**
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# **Neural Network Model**
class EvasionNet(nn.Module):
    def __init__(self):
        super(EvasionNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(29, 512),  # Increased capacity for better feature processing
            nn.LeakyReLU(0.01),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.01),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.01),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.01),
            nn.Linear(64, 5)  # Outputs: dx, dz, shoot (logit), tx, tz
        )

    def forward(self, x):
        return self.network(x)

# **Dataset for Training**
class EvasionDataset(Dataset):
    def __init__(self, data):
        features = []
        targets = []
        for d in data:
            central_x, central_z, dist_to_center, \
            shot1_x, shot1_z, shot1_vx, shot1_vz, \
            shot2_x, shot2_z, shot2_vx, shot2_vz, \
            shot3_x, shot3_z, shot3_vx, shot3_vz, \
            shot4_x, shot4_z, shot4_vx, shot4_vz, \
            shot5_x, shot5_z, shot5_vx, shot5_vz, \
            dist_to_left, dist_to_right, dist_to_top, dist_to_bottom, \
            shots_in_radius, shot_counter, \
            target_dx, target_dz, target_shoot, target_tx, target_tz = d
            feature = [
                central_x / 25, central_z / 25, dist_to_center / 25,
                shot1_x / 25, shot1_z / 25, shot1_vx / 10, shot1_vz / 10,
                shot2_x / 25, shot2_z / 25, shot2_vx / 10, shot2_vz / 10,
                shot3_x / 25, shot3_z / 25, shot3_vx / 10, shot3_vz / 10,
                shot4_x / 25, shot4_z / 25, shot4_vx / 10, shot4_vz / 10,
                shot5_x / 25, shot5_z / 25, shot5_vx / 10, shot5_vz / 10,
                dist_to_left / 25, dist_to_right / 25, dist_to_top / 25, dist_to_bottom / 25,
                shots_in_radius / 10, shot_counter / 10
            ]
            target = [target_dx, target_dz, target_shoot, target_tx / 25, target_tz / 25]
            features.append(feature)
            targets.append(target)
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

# **Scene Setup**
scene.background = color.rgb_to_hsv(vector(0.6, 0.8, 1))  # Soft sky blue
scene.range = 25
scene.lights = []
distant_light(direction=vector(0.7, 0.7, -0.5), color=color.white)
scene.caption = "Phase 2: Enhanced Evasion with Corner and Center Bots"

# **Constants**
MOVE_SPEED = 30          # Units per second
DT = 1/60                # Time step (approximately 60 FPS)
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.999    # Slower decay for more exploration
MAX_SHOT_TIME = 10       # Maximum time a shot can exist (seconds)

# **Platform**
platform = box(pos=vector(0, -0.1, 0), size=vector(50, 0.2, 50), color=vector(0.4, 0.7, 0.3))

# **Central Spheres**
central_spheres = []
for i in range(1):
    pos = vector(random.uniform(-5, 5), 0, random.uniform(-5, 5))
    sphere_obj = sphere(pos=pos, radius=1.2, color=vector(1, 0.5, 0))
    central_spheres.append({'sphere': sphere_obj, 'shot_counter': 0, 'last_teleport_time': 0})

# **Player Spheres (Bots at Corners and Center)**
player_positions = [vector(-24, 0, -24), vector(24, 0, -24), vector(-24, 0, 24), vector(24, 0, 24), vector(0, 0, 0)]
player_spheres = [{'sphere': sphere(pos=pos, radius=0.4, color=vector(0, 0, 1)), 'last_shoot_time': 0} for pos in player_positions]
player_spheres[0]['sphere'].color = vector(0, 1, 0)  # Player-controlled bot (green)

# **Shots List**
shots = []

# **Mode**
mode = 'training'

# **Training Data**
training_data = []

# **Epsilon**
epsilon = EPSILON_START

# **Training Metrics**
training_time = 0
train_metrics = {'cycles': 0, 'avg_loss': 0.0, 'epsilon': EPSILON_START}
shot_down_count = 0
hit_count = 0
zoom_factor = 25

# **Model Path**
model_path = Path("phase2_evasion_model98888888888888.pth")

# **Load or Initialize Model**
def load_model():
    global model, optimizer, scheduler, epsilon
    model = EvasionNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)  # AdamW for better generalization
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)
    if model_path.exists():
        checkpoint = torch.load(model_path, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epsilon = checkpoint.get('epsilon', EPSILON_START)
        train_metrics['epsilon'] = epsilon
        print(f"Loaded model from {model_path} with epsilon = {epsilon:.3f}")
    else:
        print(f"No model found at {model_path}. Starting new.")
    model.eval()
    return model, optimizer, scheduler

# **Save Model**
def save_model(model, optimizer, scheduler, epsilon):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epsilon': epsilon
    }, model_path)
    train_metrics['epsilon'] = epsilon
    print(f"Saved model to {model_path}, Epsilon = {epsilon:.3f}")

# Initialize Model
model, optimizer, scheduler = load_model()
criterion_mse = nn.MSELoss()
criterion_bce = nn.BCEWithLogitsLoss()

# **Shooting Functions**
def shoot_from_player(player, target_cs=None, current_time=None):
    direction = (target_cs['sphere'].pos - player.pos).norm() if target_cs else vector(random.uniform(-1, 1), 0, random.uniform(-1, 1)).norm()
    noise = vector(random.gauss(0, 0.05), 0, random.gauss(0, 0.05))  # Reduced noise for more accurate shots
    shot_direction = (direction + noise).norm()
    shot_velocity = 10 * shot_direction
    shot_pos = player.pos + vector(0, player.radius, 0)
    make_trail = False  # No trails for player shots in any mode
    shot = sphere(pos=shot_pos, radius=0.2, color=vector(1, 1, 0), make_trail=make_trail)
    shots.append({'sphere': shot, 'velocity': shot_velocity, 'target': target_cs if target_cs else None, 'creation_time': current_time})
    print(f"Shot fired from {shot_pos} with velocity {shot_velocity}")

def shoot_from_central(central, target_shot, current_time):
    direction = (target_shot['sphere'].pos - central['sphere'].pos).norm()
    shot_velocity = 15 * direction
    shot_pos = central['sphere'].pos + vector(0, central['sphere'].radius, 0)
    shot = sphere(pos=shot_pos, radius=0.2, color=vector(1, 0, 0), make_trail=False)
    shots.append({'sphere': shot, 'velocity': shot_velocity, 'from_central': True, 'source': central, 'creation_time': current_time})

# **Key Handling**
def keydown(evt):
    global mode, zoom_factor, t
    key = evt.key.lower()
    if key == 'r' and mode == 'training':
        mode = 'real_play'
        print("Switched to real play mode")
        for central in central_spheres[1:]:
            central['sphere'].visible = False
        central_spheres[0]['shot_counter'] = 0
        central_spheres[0]['last_teleport_time'] = t
        for shot in shots:
            shot['sphere'].clear_trail()
            shot['sphere'].visible = False
        shots.clear()
    elif key == 'q' and mode == 'real_play':
        mode = 'training'
        print("Switched to training mode")
        for central in central_spheres:
            central['sphere'].visible = True
            central['shot_counter'] = 0
            central['last_teleport_time'] = t
        for shot in shots:
            shot['sphere'].clear_trail()
            shot['sphere'].visible = False
        shots.clear()
    elif key == 'y':
        zoom_factor += 5
    elif key == 'i':
        zoom_factor = max(5, zoom_factor - 5)
    if mode == 'real_play' and key in ['up', 'down', 'left', 'right', ' ']:
        player = player_spheres[0]['sphere']
        if key == 'up':
            player.pos.z -= 1
        elif key == 'down':
            player.pos.z += 1
        elif key == 'left':
            player.pos.x -= 1
        elif key == 'right':
            player.pos.x += 1
        elif key == ' ':  # Spacebar to shoot in real-play mode
            shoot_from_player(player, central_spheres[0], t)
            central_spheres[0]['shot_counter'] += 1
        player.pos.x = max(-24, min(24, player.pos.x))
        player.pos.z = max(-24, min(24, player.pos.z))

scene.bind('keydown', keydown)

# **Metrics Display**
metrics_label = label(pos=vector(0, 12, 0), text='Metrics: Initializing...', height=16, border=8)

# **Main Loop**
t = 0
last_reset = 0

while True:
    rate(60)
    t += DT

    active_central_spheres = central_spheres if mode == 'training' else [central_spheres[0]]

    # **Player Shooting**
    if mode == 'training':
        for player in player_spheres[:4]:  # Corner bots shoot every 0.5 seconds
            if t - player['last_shoot_time'] >= 0.5:
                shoot_from_player(player['sphere'], current_time=t)
                for central in active_central_spheres:
                    central['shot_counter'] += 1
                player['last_shoot_time'] = t
        if t - player_spheres[4]['last_shoot_time'] >= 2.0:  # Center bot shoots every 2 seconds
            shoot_from_player(player_spheres[4]['sphere'], current_time=t)
            for central in active_central_spheres:
                central['shot_counter'] += 1
            player_spheres[4]['last_shoot_time'] = t

    # **Update Shots**
    shots_to_remove = []
    for shot in shots[:]:
        if t - shot['creation_time'] > MAX_SHOT_TIME:
            shot['sphere'].clear_trail()
            shot['sphere'].visible = False
            shots_to_remove.append(shot)
            continue
        shot['sphere'].pos += shot['velocity'] * DT
        removed = False

        if 'from_central' in shot:  # Central sphere projectile colliding with player/bot shots
            for target in [s for s in shots if 'from_central' not in s]:
                dx = shot['sphere'].pos.x - target['sphere'].pos.x
                dz = shot['sphere'].pos.z - target['sphere'].pos.z
                distance = sqrt(dx**2 + dz**2)
                if distance < shot['sphere'].radius + target['sphere'].radius:
                    shot['sphere'].clear_trail()
                    target['sphere'].clear_trail()
                    shot['sphere'].visible = False
                    target['sphere'].visible = False
                    shots_to_remove.append(shot)
                    shots_to_remove.append(target)
                    shot_down_count += 1
                    print("Central sphere shot down a player/bot shot!")
                    removed = True
                    break
        else:  # Player/bot shots colliding with central spheres
            for central in active_central_spheres:
                dx = shot['sphere'].pos.x - central['sphere'].pos.x
                dz = shot['sphere'].pos.z - central['sphere'].pos.z
                distance = sqrt(dx**2 + dz**2)
                if distance < central['sphere'].radius + shot['sphere'].radius:
                    hit_count += 1
                    print("Central sphere hit!")
                    shot['sphere'].clear_trail()
                    shot['sphere'].visible = False
                    shots_to_remove.append(shot)
                    removed = True
                    break

        if not removed and (abs(shot['sphere'].pos.x) > 25.5 or abs(shot['sphere'].pos.z) > 25.5):
            print(f"Removing shot at {shot['sphere'].pos} for going out of bounds")
            shot['sphere'].clear_trail()
            shot['sphere'].visible = False
            shots_to_remove.append(shot)

    for shot in shots_to_remove:
        if shot in shots:
            shots.remove(shot)

    print(f"Active Shots: {len(shots)}")

    # **Central Spheres Logic**
    for central in active_central_spheres:
        targeting_shots = [s for s in shots if 'from_central' not in s and (central['sphere'].pos - s['sphere'].pos).dot(s['velocity']) > 0]
        dist_to_center = mag(central['sphere'].pos)
        closest_shots = sorted(shots, key=lambda s: mag(s['sphere'].pos - central['sphere'].pos))[:5]
        while len(closest_shots) < 5:
            closest_shots.append({'sphere': sphere(pos=vector(0, 0, 0), visible=False), 'velocity': vector(0, 0, 0)})

        dist_to_left = 25 + central['sphere'].pos.x
        dist_to_right = 25 - central['sphere'].pos.x
        dist_to_top = 25 - central['sphere'].pos.z
        dist_to_bottom = 25 + central['sphere'].pos.z
        shots_in_radius = sum(1 for s in shots if mag(s['sphere'].pos - central['sphere'].pos) < 10)

        input_data = torch.tensor([
            central['sphere'].pos.x / 25, central['sphere'].pos.z / 25, dist_to_center / 25
        ] + [s['sphere'].pos.x / 25 for s in closest_shots] +
            [s['sphere'].pos.z / 25 for s in closest_shots] +
            [s['velocity'].x / 10 for s in closest_shots] +
            [s['velocity'].z / 10 for s in closest_shots] +
            [dist_to_left / 25, dist_to_right / 25, dist_to_top / 25, dist_to_bottom / 25,
             shots_in_radius / 10, central['shot_counter'] % 10 / 10], dtype=torch.float32).to(device)

        if mode == 'training':
            if len(shots) > 0:
                avg_pos = vector(0, 0, 0)
                avg_vel = vector(0, 0, 0)
                for shot in shots:
                    avg_pos += shot['sphere'].pos
                    avg_vel += shot['velocity']
                avg_pos /= len(shots)
                avg_vel /= len(shots)
                evade_dir = (central['sphere'].pos - avg_pos).norm()
                vel_dir = (-avg_vel).norm()
                center_dir = -central['sphere'].pos.norm()
                wall_repulsion = vector(0, 0, 0)
                if dist_to_left < 10:
                    wall_repulsion.x += 1 / (dist_to_left ** 2 + 1)
                if dist_to_right < 10:
                    wall_repulsion.x -= 1 / (dist_to_right ** 2 + 1)
                if dist_to_top < 10:
                    wall_repulsion.z -= 1 / (dist_to_top ** 2 + 1)
                if dist_to_bottom < 10:
                    wall_repulsion.z += 1 / (dist_to_bottom ** 2 + 1)
                wall_repulsion = wall_repulsion.norm()
                target_dir = (0.4 * evade_dir + 0.3 * vel_dir + 0.1 * center_dir + 0.2 * wall_repulsion).norm()
                safe_dist = 10
                target_tx = max(-24, min(24, central['sphere'].pos.x + safe_dist * target_dir.x))
                target_tz = max(-24, min(24, central['sphere'].pos.z + safe_dist * target_dir.z))
            else:
                target_dir = vector(0, 0, 0)
                target_tx = central['sphere'].pos.x
                target_tz = central['sphere'].pos.z
            target_shoot = 1 if central['shot_counter'] >= 10 and len(targeting_shots) > 0 else 0

            training_data.append((
                central['sphere'].pos.x, central['sphere'].pos.z, dist_to_center,
                *sum(([s['sphere'].pos.x, s['sphere'].pos.z, s['velocity'].x, s['velocity'].z] for s in closest_shots), []),
                dist_to_left, dist_to_right, dist_to_top, dist_to_bottom,
                shots_in_radius, central['shot_counter'] % 10,
                target_dir.x, target_dir.z, target_shoot, target_tx, target_tz
            ))

            if random.random() < epsilon:
                direction = vector(random.uniform(-1, 1), 0, random.uniform(-1, 1)).norm()
                shoot_decision = random.choice([0, 1])
                tx = random.uniform(-24, 24)
                tz = random.uniform(-24, 24)
            else:
                model.eval()
                with torch.no_grad():
                    output = model(input_data.unsqueeze(0)).cpu().numpy()[0]
                direction = vector(output[0], 0, output[1]).norm()
                shoot_decision = output[2] > 0
                tx = output[3] * 25
                tz = output[4] * 25
        else:  # Real-play mode
            model.eval()
            with torch.no_grad():
                output = model(input_data.unsqueeze(0)).cpu().numpy()[0]
            direction = vector(output[0], 0, output[1]).norm()
            shoot_decision = output[2] > 0
            tx = output[3] * 25
            tz = output[4] * 25
            print(f"Shoot Decision: {shoot_decision}, Shot Counter: {central['shot_counter']}, Targeting Shots: {len(targeting_shots)}")

        if t - central['last_teleport_time'] >= 3:
            central['sphere'].pos.x = max(-24, min(24, tx))
            central['sphere'].pos.z = max(-24, min(24, tz))
            central['last_teleport_time'] = t

        wall_repulsion = vector(0, 0, 0)
        if central['sphere'].pos.x < -22:
            wall_repulsion.x += 2 / (abs(central['sphere'].pos.x + 24) + 1)
        elif central['sphere'].pos.x > 22:
            wall_repulsion.x -= 2 / (abs(central['sphere'].pos.x - 24) + 1)
        if central['sphere'].pos.z < -22:
            wall_repulsion.z += 2 / (abs(central['sphere'].pos.z + 24) + 1)
        elif central['sphere'].pos.z > 22:
            wall_repulsion.z -= 2 / (abs(central['sphere'].pos.z - 24) + 1)
        direction = (direction + 0.3 * wall_repulsion).norm()
        central['sphere'].pos += direction * MOVE_SPEED * DT
        central['sphere'].pos.x = max(-24, min(24, central['sphere'].pos.x))
        central['sphere'].pos.z = max(-24, min(24, central['sphere'].pos.z))

        if shoot_decision and central['shot_counter'] >= 10 and len(targeting_shots) > 0:
            target_shot = min(targeting_shots, key=lambda s: mag(s['sphere'].pos - central['sphere'].pos))
            shoot_from_central(central, target_shot, t)
            central['shot_counter'] = 0

    # **Training and Reset**
    if t - last_reset >= 72 and mode == 'training':
        start_time = time.time()
        if len(training_data) >= 1000:
            dataset = EvasionDataset(training_data)
            loader = DataLoader(dataset, batch_size=256, shuffle=True)
            model.train()
            total_loss = 0
            for epoch in range(20):  # Increased to 20 epochs
                epoch_loss = 0
                for features, targets in loader:
                    features, targets = features.to(device), targets.to(device)
                    optimizer.zero_grad()
                    outputs = model(features)
                    # Custom loss: MSE for continuous outputs, BCE for shoot
                    loss_mse = criterion_mse(outputs[:, :2], targets[:, :2]) + criterion_mse(outputs[:, 3:], targets[:, 3:])
                    loss_bce = criterion_bce(outputs[:, 2], targets[:, 2])
                    loss = loss_mse + loss_bce
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                total_loss += epoch_loss / len(loader)
            train_metrics['avg_loss'] = total_loss / 20  # Average over 20 epochs
            scheduler.step(train_metrics['avg_loss'])  # Step scheduler with avg_loss
            epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
            save_model(model, optimizer, scheduler, epsilon)
            training_data = training_data[-50000:]
            model.eval()
        train_metrics['cycles'] += 1
        training_time += time.time() - start_time

        for central in central_spheres:
            central['sphere'].pos = vector(random.uniform(-5, 5), 0, random.uniform(-5, 5))
            central['shot_counter'] = 0
            central['last_teleport_time'] = t
        for player in player_spheres:
            player['last_shoot_time'] = t
        for shot in shots:
            shot['sphere'].clear_trail()
            shot['sphere'].visible = False
        shots.clear()
        last_reset = t

    # **Metrics Update**
    metrics_text = (f"Mode: {mode}, Cycles: {train_metrics['cycles']}, Avg Loss: {train_metrics['avg_loss']:.6f}, "
                   f"Epsilon: {train_metrics['epsilon']:.3f}, Training Time: {training_time:.2f}s, "
                   f"Shot Down: {shot_down_count}, Hits: {hit_count}, Shots: {len(shots)}")
    metrics_label.text = metrics_text

    # **Camera**
    scene.range = zoom_factor
    scene.center = (central_spheres[0]['sphere'].pos + vector(0, 5, 0) if mode == 'training' else
                    player_spheres[0]['sphere'].pos + vector(0, 5, 0))
    scene.forward = vector(0, -0.3, -1)
