import torch
from torch import nn, optim
from torch.profiler import profile, record_function, ProfilerActivity, schedule, tensorboard_trace_handler
from src.assignment.model import ConvolutionalNetwork
from src.assignment.data import ImageFolderDataModule

# --- Config ---
DATA_DIR = "PokemonData"
BATCH_SIZE = 32
IMG_SIZE = 224
LR = 1e-3
MAX_EPOCHS = 1        # keep small for profiling
LIMIT_BATCHES = 10    # limit batches for fast profiling
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Data ---
dm = ImageFolderDataModule(
    data_dir=DATA_DIR,
    batch_size=BATCH_SIZE,
    num_workers=0,
    img_size=IMG_SIZE,
    seed=SEED,
)
dm.setup()
train_loader = dm.train_dataloader()

# --- Model, optimizer, loss ---
model = ConvolutionalNetwork(num_classes=dm.num_classes).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# --- Torch Profiler ---
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA] if torch.cuda.is_available() else [ProfilerActivity.CPU],
    schedule=schedule(wait=1, warmup=1, active=3, repeat=1),
    record_shapes=True,
    with_stack=True,
    on_trace_ready=tensorboard_trace_handler("profiler_logs/train")
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

print("Training profiling complete.")
print("View TensorBoard logs: tensorboard --logdir profiler_logs/train")
