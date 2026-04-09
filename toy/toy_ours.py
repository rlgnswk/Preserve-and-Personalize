import torch, torch.nn as nn, matplotlib.pyplot as plt, os, math, random, numpy as np
from tqdm import tqdm

# =========================
# 🔧 Top-level hyperparameters (edit only here)
# =========================
N_PER_CLASS    = 1000   # Number of samples per class
N_BASE_CLASSES = 5      # Number of base training classes (regular polygon vertices)
N_UNITS        = 96     # Hidden size of the model
reg_lambda     = 50       # Regularization weight

NEW_CLASS_IDX   = N_BASE_CLASSES
N_CLASSES_TOTAL = N_BASE_CLASSES + 1

# ---------- Fix random seeds ----------
seed = 42
torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
np.random.seed(seed);  random.seed(seed)
torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False

# ---------- Output directory ----------
fig_dir = "toy_ours"; os.makedirs(fig_dir, exist_ok=True)

# ---------- Data generation (regular polygon vertices) ----------
def make_pentagon_mog(n_per_class=1000, std=0.5, radius=3.0, seed=42):
    """
    Classes 0..N_BASE_CLASSES-1 are placed on the vertices of
    a regular N-gon in clockwise order.
    """
    torch.manual_seed(seed)
    centers = []
    for i in range(N_BASE_CLASSES):  # ★ 5 -> N_BASE_CLASSES
        theta = 2*math.pi*i/N_BASE_CLASSES
        centers.append([radius*math.cos(theta), radius*math.sin(theta)])
    centers = torch.tensor(centers, dtype=torch.float32)
    data, labels = [], []
    for i, center in enumerate(centers):
        pts = center + std*torch.randn(n_per_class, 2)
        data.append(pts)
        labels.append(torch.full((n_per_class,), i, dtype=torch.long))
    return torch.cat(data, 0), torch.cat(labels, 0)

# ---------- Noise schedule ----------
def cosine_beta_schedule(timesteps, s=0.008):
    steps = torch.arange(timesteps+1, dtype=torch.float32)
    f = torch.cos(((steps/timesteps+s)/(1+s))*math.pi/2)**2
    ab = f/f[0]
    return 1 - ab[1:]/ab[:-1]

# ---------- Model ----------
class ConditionalDiffusionModel(nn.Module):
    def __init__(self, nfeatures=2, nclasses=6, nunits=32):
        super().__init__()
        self.embed_class = nn.Embedding(nclasses, 8)
        self.embed_time  = nn.Embedding(100, 8)
        self.in_layer    = nn.Linear(nfeatures+8+8, nunits)
        self.hidden      = nn.Sequential(nn.ReLU(),
                                         nn.Linear(nunits, nunits),
                                         nn.ReLU())
        self.out_layer   = nn.Linear(nunits, nfeatures)
    def forward(self, x, t, c):
        t_emb = self.embed_time(t.view(-1))
        c_emb = self.embed_class(c.view(-1))
        h = torch.cat([x, t_emb, c_emb], 1)
        h = self.hidden(self.in_layer(h))
        return self.out_layer(h)

# ---------- forward diffusion ----------
def q_sample(x_start, t, alpha_bars):
    noise = torch.randn_like(x_start)
    sqrt_ab  = alpha_bars[t].sqrt().unsqueeze(1)
    sqrt_mab = (1-alpha_bars[t]).sqrt().unsqueeze(1)
    return sqrt_ab*x_start + sqrt_mab*noise, noise

# ---------- Reverse diffusion sampler ----------
@torch.no_grad()
def sample(model, class_id, n_samples=100):
    x = torch.randn(n_samples, 2, device=device)
    c = torch.full((n_samples,1), class_id, device=device, dtype=torch.long)
    for t in reversed(range(T)):
        t_batch = torch.full((n_samples,1), t, device=device, dtype=torch.long)
        a, ab = alphas[t], torch.clamp(alpha_bars[t], 1e-5)
        denom = torch.sqrt(torch.clamp(1-ab, 1e-5))
        pred  = model(x, t_batch, c)
        z     = torch.randn_like(x) if t else 0
        x     = (1/a.sqrt())*(x - (1-a)/denom*pred) + betas[t].sqrt()*z
        x     = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
    return x.cpu().numpy()

# ---------- Setup ----------
device = "cuda" if torch.cuda.is_available() else "cpu"
T = 100
betas  = cosine_beta_schedule(T)
alphas = 1 - betas
alpha_bars = torch.cumprod(alphas, 0).to(device)

# Connect the total number of classes and hidden size to the top-level hyperparameters
model = ConditionalDiffusionModel(nclasses=N_CLASSES_TOTAL, nunits=N_UNITS).to(device)
optimizer, loss_fn = torch.optim.Adam(model.parameters(), 2e-3), nn.MSELoss()

# ---------- 1) Base training data ----------
x, y = make_pentagon_mog(n_per_class=N_PER_CLASS)  # Reflect N_PER_CLASS
X, Y = x.to(device), y.to(device)

# ---------- 2) New class for fine-tuning (origin) ----------
mean_new = torch.tensor([0.0, 0.0])
std_new  = 0.5
X_new    = mean_new + std_new*torch.randn(N_PER_CLASS, 2)  # Reflect N_PER_CLASS
Y_new    = torch.full((N_PER_CLASS,), NEW_CLASS_IDX, dtype=torch.long)  # Reflect NEW_CLASS_IDX

# ---------- 3) Plot range ----------
init_pts, new_pts = X.cpu().numpy(), X_new.numpy()
all_pts = np.vstack([init_pts, new_pts])
x_min, x_max = all_pts[:,0].min(), all_pts[:,0].max()
y_min, y_max = all_pts[:,1].min(), all_pts[:,1].max()
x_m, y_m = .2*(x_max-x_min), .2*(y_max-y_min)
xlim_global, ylim_global = (x_min-x_m, x_max+x_m), (y_min-y_m, y_max+y_m)

# ---------- 4) Initial training ----------
for _ in tqdm(range(1000), desc="Initial Training"):
    idx = torch.randperm(len(X))
    for i in range(0, len(X), 512):
        xb, yb = X[idx][i:i+512], Y[idx][i:i+512]
        tb = torch.randint(0, T, (len(xb),1), device=device)
        noised, noise = q_sample(xb, tb.squeeze(), alpha_bars)
        loss = loss_fn(model(noised, tb, yb), noise)
        optimizer.zero_grad(); loss.backward(); optimizer.step()

# ---------- Colors for visualization ----------
colors = [
    "#0072B2",  # Class 0 - blue
    "#009E73",  # Class 1 - green
    "#C44E52",  # Class 2 – muted red
    "#E69F00",  # Class 3 - orange
    "#56B4E9",  # Class 4 - light blue
    "#F0E442",  # Class 5 - yellow
]
assert N_CLASSES_TOTAL <= len(colors), \
    f"The current palette has {len(colors)} colors. Extend `colors` to support N_CLASSES_TOTAL={N_CLASSES_TOTAL}."

def setup_axes(title=None):
    plt.xlim(*xlim_global); plt.ylim(*ylim_global)
    ax = plt.gca()
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title is not None:
        plt.title(title)
    plt.tight_layout()

# 5) Training data distribution
plt.figure(figsize=(6,6))
for cls in range(N_BASE_CLASSES):  # ★ 5 -> N_BASE_CLASSES
    pts = init_pts[Y.cpu().numpy()==cls]
    plt.scatter(pts[:,0], pts[:,1], alpha=.8, label=f"Class {cls}", color=colors[cls])
plt.legend(); setup_axes(f"Training Data (Classes 0–{N_BASE_CLASSES-1})")
plt.savefig(f"{fig_dir}/data_distribution.png", dpi=300); plt.show()

# 6) Pretrained samples
plt.figure(figsize=(6,6))
for cls in range(N_BASE_CLASSES):  # ★ 5 -> N_BASE_CLASSES
    plt.scatter(*sample(model, cls, 1000).T, alpha=.8, color=colors[cls], label=f"Class {cls}")
plt.legend(); setup_axes("Pretrained Samples")
plt.savefig(f"{fig_dir}/pretrained_samples.png", dpi=300); plt.show()

# 7) Target data
plt.figure(figsize=(6,6))
plt.scatter(new_pts[:,0], new_pts[:,1], alpha=.8, color=colors[NEW_CLASS_IDX], label=f"Target Class {NEW_CLASS_IDX} (0,0)")
plt.legend(); setup_axes(f"Target Data (Class {NEW_CLASS_IDX} = Origin)")
plt.savefig(f"{fig_dir}/target_data.png", dpi=300); plt.show()

# ---------- 8) Fine-tuning (class NEW_CLASS_IDX, keep regularization) ----------
import copy
orig_model = copy.deepcopy(model).to(device)  # Keep the copied model on the correct device

loss_history = []
for _ in tqdm(range(5000), desc="Fine-tune"):
    idx = torch.randperm(len(X_new))
    for i in range(0, len(X_new), 512):
        xb = X_new[idx][i:i+512].to(device)
        yb = Y_new[idx][i:i+512].to(device)
        tb = torch.randint(0, T, (len(xb),1), device=device)

        noised, noise = q_sample(xb, tb.squeeze(), alpha_bars)
        pred_loss = loss_fn(model(noised, tb, yb), noise)

        # Lipschitz-style regularization (keep the original formulation)
        reg_loss = 0.0
        for p_orig, p in zip(orig_model.parameters(), model.parameters()):
            reg_loss += loss_fn(p, p_orig)

        loss = pred_loss + reg_lambda * reg_loss

        optimizer.zero_grad(); loss.backward(); optimizer.step()
        loss_history.append(loss.item())

# Loss plot disabled: do not generate or save finetune_loss.png
# plt.figure(); plt.plot(loss_history); plt.title(f"Finetune Loss (Class {NEW_CLASS_IDX})")
# plt.xlabel("Iteration"); plt.ylabel("MSE"); plt.grid(True); plt.tight_layout()
# plt.savefig(f"{fig_dir}/finetune_loss.png", dpi=300); plt.show()

# 9) Personalized samples
plt.figure(figsize=(6,6))
for cls in range(N_CLASSES_TOTAL):  # ★ 6 -> N_CLASSES_TOTAL
    plt.scatter(*sample(model, cls, 1000).T, alpha=.8, color=colors[cls])
setup_axes()
plt.savefig(f"{fig_dir}/personalized_samples.png", dpi=300); plt.show()
import torch, torch.nn as nn, matplotlib.pyplot as plt, os, math, random, numpy as np
from tqdm import tqdm
