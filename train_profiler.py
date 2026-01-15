import torch
from torch import nn, optim
from torch.profiler import profile, record_function, ProfilerActivity, schedule, tensorboard_trace_handler
from src.assignment.model import ConvolutionalNetwork
from src.assignment.data import ImageFolderDataModule
from pathlib import Path

# --------------------
# Config
# --------------------
DATA_DIR = "PokemonData"
BATCH_SIZE = 64
IMG_SIZE = 224
LR = 1e-3
MAX_EPOCHS = 1
LIMIT_BATCHES = 10
SEED = 42
LOG_DIR = "profiler_logs/train"

# Device: GPU if available, else CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Make sure log directory exists
Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

# --------------------
# Data
# --------------------
dm = ImageFolderDataModule(
    data_dir=DATA_DIR,
    batch_size=BATCH_SIZE,
    num_workers=0,
    img_size=IMG_SIZE,
    seed=SEED
)
dm.setup()
train_loader = dm.train_dataloader()

print("Num classes:", dm.num_classes)
print("Train loader batches:", len(train_loader))

# --------------------
# Model, optimizer, loss
# --------------------
model = ConvolutionalNetwork(num_classes=dm.num_classes).to(DEVICE)

# Only use torch.compile if CUDA is available; CPU compile may not give speedup
if torch.cuda.is_available():
    print("Compiling model with torch.compile() for GPU...")
    model = torch.compile(model, mode="reduce-overhead")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# --------------------
# Profiler
# --------------------
profiler_activities = [ProfilerActivity.CPU]
if torch.cuda.is_available():
    profiler_activities.append(ProfilerActivity.CUDA)

with profile(
    activities=profiler_activities,
    schedule=schedule(wait=0, warmup=0, active=LIMIT_BATCHES, repeat=1),
    on_trace_ready=tensorboard_trace_handler(LOG_DIR),
    record_shapes=True,
    with_stack=True,
    with_flops=True  # optional, extra stats
) as prof:
    for epoch in range(MAX_EPOCHS):
        for step, (xb, yb) in enumerate(train_loader):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            with record_function("forward_backward"):
                optimizer.zero_grad()
                outputs = model(xb)
                loss = criterion(outputs, yb)
                loss.backward()
                optimizer.step()
            prof.step()
            if step + 1 >= LIMIT_BATCHES:
                break

print("Profiling complete. TensorBoard events written to:", LOG_DIR)
print("Run: tensorboard --logdir", LOG_DIR)
