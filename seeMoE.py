import base64
import io
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init
import mlflow

# Ensure every computation happens on the GPU when available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the text data
text_path = "./input.txt"
with open(text_path, 'r', encoding='utf-8') as f:
    text = f.read()

# Create mappings between characters and integers
chars = sorted(list(set(text)))
stoi = {ch: i for i, ch in enumerate(chars)}
stoi['<pad>'] = 65
itos = {i: ch for i, ch in enumerate(chars)}
itos[65] = '<pad>'
encode = lambda s: [stoi[c] for c in s]  # Encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l])  # Decoder: take a list of integers, output a string
vocab_size = len(stoi.keys())

# PatchEmbeddings class to extract patch embeddings from an image
class PatchEmbeddings(nn.Module):
    def __init__(self, img_size=96, patch_size=16, hidden_dim=512):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.conv = nn.Conv2d(in_channels=3, out_channels=hidden_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, X):
        X = self.conv(X)
        X = X.flatten(2)
        X = X.transpose(1, 2)
        return X

# MLP class for the transformer blocks
class MLP(nn.Module):
    def __init__(self, n_embd, dropout=0.1, is_decoder=True):
        super().__init__()
        layers = [
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU() if is_decoder else nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# Head class for the self-attention mechanism
class Head(nn.Module):
    def __init__(self, n_embd, head_size, dropout=0.1, is_decoder=False):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.is_decoder = is_decoder

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)

        if self.is_decoder:
            tril = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
            wei = wei.masked_fill(tril == 0, float('-inf'))

        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v
        return out

# MultiHeadAttention class for multi-head self-attention
class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, num_heads, dropout=0.1, is_decoder=False):
        super().__init__()
        assert n_embd % num_heads == 0, "n_embd must be divisible by num_heads"
        self.heads = nn.ModuleList([
            Head(n_embd, n_embd // num_heads, dropout, is_decoder)
            for _ in range(num_heads)
        ])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        head_outputs = [h(x) for h in self.heads]
        out = torch.cat(head_outputs, dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out

# Block class for the transformer blocks
class Block(nn.Module):
    def __init__(self, n_embd, num_heads, dropout=0.1, is_decoder=False):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, num_heads, dropout, is_decoder)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffn = MLP(n_embd)

    def forward(self, x):
        original_x = x
        x = self.ln1(x)
        attn_output = self.attn(x)
        x = original_x + attn_output
        x = self.ln2(x)
        ffn_output = self.ffn(x)
        x = x + ffn_output
        return x

# ViT class for the vision transformer encoder
class ViT(nn.Module):
    def __init__(self, img_size, patch_size, num_hiddens, num_heads, num_blks, emb_dropout, blk_dropout):
        super().__init__()
        self.patch_embedding = PatchEmbeddings(img_size, patch_size, num_hiddens)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, num_hiddens))
        num_patches = (img_size // patch_size) ** 2
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, num_hiddens))
        self.dropout = nn.Dropout(emb_dropout)
        self.blocks = nn.ModuleList([Block(num_hiddens, num_heads, blk_dropout, is_decoder=False) for _ in range(num_blks)])
        self.layer_norm = nn.LayerNorm(num_hiddens)

    def forward(self, X):
        x = self.patch_embedding(X)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.layer_norm(x[:, 0])
        return x

# MultiModalProjector class for projecting image embeddings to text embedding space
class MultiModalProjector(nn.Module):
    def __init__(self, n_embd, image_embed_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(image_embed_dim, 4 * image_embed_dim),
            nn.GELU(),
            nn.Linear(4 * image_embed_dim, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.net(x)
        return x

# Expert class for the mixture of experts
class Expert(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

# NoisyTopkRouter class for routing in the mixture of experts
class NoisyTopkRouter(nn.Module):
    def __init__(self, n_embed, num_experts, top_k):
        super(NoisyTopkRouter, self).__init__()
        self.top_k = top_k
        self.topkroute_linear = nn.Linear(n_embed, num_experts)
        self.noise_linear = nn.Linear(n_embed, num_experts)

    def forward(self, mh_output):
        logits = self.topkroute_linear(mh_output)
        noise_logits = self.noise_linear(mh_output)
        noise = torch.randn_like(logits) * F.softplus(noise_logits)
        noisy_logits = logits + noise

        top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
        zeros = torch.full_like(noisy_logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_output = F.softmax(sparse_logits, dim=-1)
        return router_output, indices

# SparseMoE class for the sparse mixture of experts module
class SparseMoE(nn.Module):
    def __init__(self, n_embed, num_experts, top_k):
        super(SparseMoE, self).__init__()
        self.router = NoisyTopkRouter(n_embed, num_experts, top_k)
        self.experts = nn.ModuleList([Expert(n_embed) for _ in range(num_experts)])
        self.top_k = top_k

    def forward(self, x):
        gating_output, indices = self.router(x)
        final_output = torch.zeros_like(x)

        flat_x = x.view(-1, x.size(-1))
        flat_gating_output = gating_output.view(-1, gating_output.size(-1))

        for i, expert in enumerate(self.experts):
            expert_mask = (indices == i).any(dim=-1)
            flat_mask = expert_mask.view(-1)

            if flat_mask.any():
                expert_input = flat_x[flat_mask]
                expert_output = expert(expert_input)
                gating_scores = flat_gating_output[flat_mask, i].unsqueeze(1)
                weighted_output = expert_output * gating_scores
                final_output[expert_mask] += weighted_output.squeeze(1)

        return final_output

# SparseMoEBlock class for the sparse mixture of experts transformer block
class SparseMoEBlock(nn.Module):
    def __init__(self, n_embd, num_heads, num_experts, top_k, dropout=0.1, is_decoder=False):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, num_heads, dropout, is_decoder)
        self.ln2 = nn.LayerNorm(n_embd)
        self.sparseMoE = SparseMoE(n_embd, num_experts, top_k)

    def forward(self, x):
        original_x = x
        x = self.ln1(x)
        attn_output = self.attn(x)
        x = original_x + attn_output
        x = self.ln2(x)
        sparseMoE_output = self.sparseMoE(x)
        x = x + sparseMoE_output
        return x

# MoEDecoderLanguageModel class for the sparse mixture of experts language model decoder
class MoEDecoderLanguageModel(nn.Module):
    def __init__(self, n_embd, image_embed_dim, vocab_size, num_heads, n_layer, num_experts, top_k, use_images=False):
        super().__init__()
        self.use_images = use_images
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(1000, n_embd)

        if use_images:
            self.image_projection = MultiModalProjector(n_embd, image_embed_dim)

        self.sparseMoEBlocks = nn.Sequential(*[SparseMoEBlock(n_embd, num_heads, num_experts, top_k, is_decoder=True) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, image_embeds=None, targets=None):
        tok_emb = self.token_embedding_table(idx)

        if self.use_images and image_embeds is not None:
            img_emb = self.image_projection(image_embeds).unsqueeze(1)
            tok_emb = torch.cat([img_emb, tok_emb], dim=1)

        pos_emb = self.position_embedding_table(torch.arange(tok_emb.size(1), device=device)).unsqueeze(0)
        x = tok_emb + pos_emb
        x = self.sparseMoEBlocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is not None:
            if self.use_images and image_embeds is not None:
                batch_size = idx.size(0)
                targets = torch.cat([torch.full((batch_size, 1), -100, dtype=torch.long, device=device), targets], dim=1)

            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
            return logits, loss

        return logits

    def generate(self, idx, image_embeds, max_new_tokens):
        B, T = idx.shape
        generated = idx

        if self.use_images and image_embeds is not None:
            img_emb = self.image_projection(image_embeds).unsqueeze(1)
            current_output = torch.cat([img_emb, self.token_embedding_table(idx)], dim=1)
        else:
            current_output = self.token_embedding_table(idx)

        for i in range(max_new_tokens):
            T_current = current_output.size(1)
            current_pos_emb = self.position_embedding_table(torch.arange(T_current, device=device)).unsqueeze(0)
            current_output += current_pos_emb

            for block in self.sparseMoEBlocks:
                current_output = block(current_output)

            logits = self.lm_head(current_output[:, -1, :])
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            generated = torch.cat((generated, idx_next), dim=1)
            idx_next_emb = self.token_embedding_table(idx_next)
            current_output = torch.cat((current_output, idx_next_emb), dim=1)

        return generated

# VisionMoELanguageModel class for the vision mixture of experts language model
class VisionMoELanguageModel(nn.Module):
    def __init__(self, n_embd, image_embed_dim, vocab_size, n_layer, img_size, patch_size, num_heads, num_blks, emb_dropout, blk_dropout, num_experts, top_k):
        super().__init__()

        # Set num_hiddens equal to image_embed_dim
        num_hiddens = image_embed_dim

        # Assert that num_hiddens is divisible by num_heads
        assert num_hiddens % num_heads == 0, "num_hiddens must be divisible by num_heads"

        # Initialize the vision encoder (ViT)
        self.vision_encoder = ViT(img_size, patch_size, num_hiddens, num_heads, num_blks, emb_dropout, blk_dropout)

        # Initialize the language model decoder (DecoderLanguageModel)
        self.decoder = MoEDecoderLanguageModel(n_embd, image_embed_dim, vocab_size, num_heads, n_layer,num_experts, top_k, use_images=True)

    def forward(self, img_array, idx, targets=None):
        # Get the image embeddings from the vision encoder
        image_embeds = self.vision_encoder(img_array)

        # Check if the image embeddings are valid
        if image_embeds.nelement() == 0 or image_embeds.shape[1] == 0:
            raise ValueError("Something is wrong with the ViT model. It's returning an empty tensor or the embedding dimension is empty.")

        if targets is not None:
            # If targets are provided, compute the logits and loss
            logits, loss = self.decoder(idx, image_embeds, targets)
            return logits, loss
        else:
            # If targets are not provided, compute only the logits
            logits = self.decoder(idx, image_embeds)
            return logits

    def generate(self, img_array, idx, max_new_tokens):
        # Get the image embeddings from the vision encoder
        image_embeds = self.vision_encoder(img_array)

        # Check if the image embeddings are valid
        if image_embeds.nelement() == 0 or image_embeds.shape[1] == 0:
            raise ValueError("Something is wrong with the ViT model. It's returning an empty tensor or the embedding dimension is empty.")

        # Generate new tokens using the language model decoder
        generated_tokens = self.decoder.generate(idx, image_embeds, max_new_tokens)
        return generated_tokens

def base64_to_tensor(base64_str, img_size=96):
   image = Image.open(io.BytesIO(base64.b64decode(base64_str)))
   if image.mode != 'RGB':
       image = image.convert('RGB')
   transform = transforms.Compose([
       transforms.Resize((img_size, img_size)),
       transforms.ToTensor(),
       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
   ])
   return transform(image).unsqueeze(0)  # Add batch dimension

def get_batch(df, batch_size, split='train', img_size=96, val_batch_size=8):
   n = int(0.9 * len(df))  # first 90% will be train, rest val
   df_train = df.iloc[:n]
   df_val = df.iloc[n:]
   data = df_train if split == 'train' else df_val
   batch_size = batch_size if split == 'train' else val_batch_size
   replace = False if split == 'train' else True
   batch = data.sample(n=batch_size, replace=replace)

   images = torch.cat([base64_to_tensor(img, img_size) for img in batch['b64string_images']], dim=0).to(device)
   text_indices = [torch.tensor(encode(desc), dtype=torch.long) for desc in batch['caption']]
   max_length = max(len(t) for t in text_indices)

   padded_text = torch.full((batch_size, max_length), fill_value=stoi['<pad>'], dtype=torch.long).to(device)
   for i, text in enumerate(text_indices):
       padded_text[i, :len(text)] = text

   targets = torch.cat([padded_text[:, 1:], torch.full((batch_size, 1), fill_value=stoi['<pad>'], dtype=torch.long, device=device)], dim=1)

   if targets.size(1) > padded_text.size(1):
       targets = targets[:, :padded_text.size(1)]
   elif targets.size(1) < padded_text.size(1):
       targets = torch.cat([targets, torch.full((batch_size, padded_text.size(1) - targets.size(1)), fill_value=stoi['<pad>'], dtype=torch.long, device=device)], dim=1)

   return images, padded_text, targets

def train_model(model, df, epochs, vocab_size, img_size=96):
   optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
   model.to(device)
   with mlflow.start_run():
       for epoch in range(epochs):
           model.train()
           for _ in range(max_iters):
               images, idx, targets = get_batch(df, batch_size, 'train', img_size)
               optimizer.zero_grad()
               logits, loss = model(images, idx, targets)
               loss.backward()
               optimizer.step()
               if _ % eval_interval == 0:
                   metrics = {"train_loss": loss.item()}
                   mlflow.log_metrics(metrics, step=_)
           val_loss = estimate_loss(model, df, 'val', img_size, val_batch_size=8)
           metrics = {"val_loss": val_loss}
           mlflow.log_metrics(metrics, step=epoch)

def estimate_loss(model, df, split, img_size=96, val_batch_size=8):
   losses = []
   model.eval()
   for _ in range(eval_iters):
       images, idx, targets = get_batch(df, batch_size, split, img_size, val_batch_size=val_batch_size)
       _, loss = model(images, idx, targets)
       losses.append(loss.item())
   return sum(losses) / len(losses)

def kaiming_init_weights(m):
   if isinstance(m, (nn.Linear)):
       init.kaiming_normal_(m.weight)

if __name__ == "__main__":
   df = pd.read_csv("./inputs.csv")
   df = pd.concat([df] * 30)[['b64string_images', 'caption']]

   batch_size = 16
   block_size = 32
   max_iters = 100
   eval_interval = 10
   learning_rate = 1e-3
   epochs = 1
   device = 'cuda' if torch.cuda.is_available() else 'cpu'
   eval_iters = 40
   num_blks = 3
   head_size = 16
   n_embd = 128
   n_head = 8
   n_layer = 8
   dropout = 0.1
   img_size = 96
   patch_size = 16
   image_embed_dim = 512
   emb_dropout = blk_dropout = 0.1
   num_experts, top_k = 8, 2

   model = VisionMoELanguageModel(n_embd, image_embed_dim, vocab_size, n_layer, img_size, patch_size, n_head, num_blks, emb_dropout, blk_dropout, num_experts, top_k)
   model.apply(kaiming_init_weights)
   model.to(device)

   dummy_img = torch.randn(1, 3, img_size, img_size).to(device)
   dummy_idx = torch.randint(0, vocab_size, (1, block_size)).to(device)
   model(dummy_img, dummy_idx)  # Forward pass to initialize all parameters

   train_model(model, df, epochs, vocab_size, img_size)
