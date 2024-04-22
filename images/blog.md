# seemore: Implement a Vision Language Model from Scratch

TL;DR: In  this blog I implement a vision language model consisting of an image encoder, a multimodal projection module and a decoder language model in pure pytorch. Think of this as a simplified version of what you see in GPT-4 or Claude 3 in terms of vision capabilities demonstrated by a language model. The name ‘seemore’ is my way of paying homage to Andrej Karpathy’s project ‘makemore’ because here I use a character level autoregressive language model much like in his nanoGPT/ makemore implementation.  My goal is for you to gain an intuitive understanding of how it all works by reading this blog and stepping through the code in the repository.


<div align="center">
 <img src="https://github.com/AviSoori1x/seemore/blob/main/images/seemorelogo.png?raw=true" width="300" height="300" alt="seemore">
</div>

The Github repo here provides the end-to-end implementation: https://github.com/AviSoori1x/seemore

#### Motivation

Vision language models have become a topic of great interest in the machine learning community due to the capabilities displayed by GPT-4, Grok 1.5, Claude 3 and Google Gemini. In addition to these proprietary multimodal (primarily vision-language) models, there have been a number of highly performant open models such as LLaVa, Kosmos from Microsoft and most recently, Idefics2 from Hugging Face.

Although the term vision language model could mean a number of things, the current wave of this class of models tend to demonstrate instruction following capabilities over both image and text inputs. In essence, you can expect a vision language model to write you a poem about how great sushi is and at the same time be able to count the number of sushi rolls on a given plate, given an image. I want to make this clear as there’s a rich collection of other types of vision language models such as CLIP and more recent variations such as SigLIP that are very important but quite different in how they are used. As a matter of fact, we will look at how components from these architectures are used in the current crop of vision language models.

For the purpose of this blog, I will focus on this type of vision language models that can be instruction tuned to perform useful tasks. More specifically, here I will specify a common architectural pattern that seems to be taking shape and proving to be highly versatile.

#### The General Architecture

<div align="center">
 <img src="https://github.com/AviSoori1x/seemore/blob/main/images/vlm.png?raw=true" width="600" height="600" alt="seemore">
</div>

In ‘seemore’, my simple implementation of a vision language model (VLM), there are 3 main components. 

1. Image Encoder to extract visual features from images. In this case I use a from scratch implementation of the original vision transformer used in CLIP. This is actually a popular choice in many modern VLMs. The one notable exception is the Fuyu series of models from Adept, that passes the patchified images directly to the projection layer.

2. Vision-Language Projector - Image embeddings are not of the same shape as text embeddings used by the decoder. So we need to ‘project’ i.e. change dimensionality of image features extracted by the image encoder to match what’s observed in the text embedding space. So image features become ‘visual tokens’ for the decoder. This could be a single layer or an MLP. I’ve used an MLP because it’s worth showing.

3. A decoder only language model. This is the component that ultimately generates text. In my implementation I’ve deviated from what you see in LLaVA a bit by incorporating the projection module to my decoder. Typically this is not observed, and you leave the architecture of the decoder (which is usually an already pretrained model) untouched.

So in summary, an image encoder extracts features from a given image, passes these image embeddings to a vision-language projector which projects these image embeddings to the text embedding space, that is then concatenated with the text embeddings from the text inputs and used to autoregressively generate text by a decoder only language model.

When you zoom out, it’s not all that complicated and honestly, quite clever. It’s also kind of amazing that this works. Just like everything else in deep learning. 

##### Let’s start with the image encoder

As mentioned earlier, here I choose to implement a vision transformer similar to the one used in CLIP. 

<div align="center">
 <img src="https://github.com/AviSoori1x/seemore/blob/main/images/clip.png?raw=true" width="600" height="400" alt="seemore">
</div>
source: https://openai.com/research/clip

There has been a trend of vision language models getting much better performance using a vision transformer from an improved version of CLIP known as SigLIP that uses a sigmoid loss instead of the cross entropy loss used in the contrastive learning task of CLIP. A great example of a tiny vision language model using the vision transformer from SigLIP punching way above its weight (literally. it’s only 1.6B parameters in total) is moondream 2 by vikhyat: https://github.com/vikhyat/moondream.

However, for the sake of simplicity, we assume that the CLIP version is used here but the implementation would be identical. In seemore, I use the embedding corresponding to the  ‘[CLS]’ token as the feature vector that represents the entire image. This is done for the sake of simplicity. However, it is possible, and likely better, to choose all the feature vectors from the last layer of the vision transformer. My assumption is that this will help with tasks such as counting and OCR, where spatial information is available in a less compressed manner for the decoder. 

<div align="center">
<img src="https://github.com/AviSoori1x/seemore/blob/main/images/vit.png?raw=true" width="600" height="300" alt="seemore">
</div>
source: https://arxiv.org/pdf/2010.11929.pdf

To implement this vision transformer from scratch we have to create a PatchEmbeddings class that can take an image and create a sequence of patches. This process is crucial for enabling the transformer architecture to process visual data effectively, specifically using the attention blocks in the subsequent steps of the architecture. 
This can be implemented quite simply as follows:

```python
class PatchEmbeddings(nn.Module):
    def __init__(self, img_size=96, patch_size=16, hidden_dim=512):
        super().__init__()
        
        # Store the input image size
        self.img_size = img_size
        
        # Store the size of each patch
        self.patch_size = patch_size
        
        # Calculate the total number of patches
        self.num_patches = (img_size // patch_size) ** 2
        
        # Create a convolutional layer to extract patch embeddings
        # in_channels=3 assumes the input image has 3 color channels (RGB)
        # out_channels=hidden_dim sets the number of output channels to match the hidden dimension
        # kernel_size=patch_size and stride=patch_size ensure each patch is separately embedded
        self.conv = nn.Conv2d(in_channels=3, out_channels=hidden_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, X):
        # Extract patch embeddings from the input image
        X = self.conv(X)
        
        # Flatten the spatial dimensions (height and width) of the patch embeddings
        # This step flattens the patch dimensions into a single dimension
        X = X.flatten(2)
        
        # Transpose the dimensions to obtain the shape [batch_size, num_patches, hidden_dim]
        # This step brings the num_patches dimension to the second position
        X = X.transpose(1, 2)
        
        return X
```

In the above code, the input image is broken down to (img_size // patch_size) ** 2 patches using the convolution layer and projected into vectors with a channel dimension (the C, in [B, T, C] shape commonly encountered in pytorch implementations for 3D tensors) of 512.

### Attention Mechanism across both the vision encoder and language decoder

Things get interesting when building the components seen in the transformer blocks. i.e. The attention head implementation, multi head attention, the multilayer perceptron seen in each transformer block and the transformer block itself. These components are mostly identical across the vision transformer we are implementing for the ‘visual token’ generation and the decoder language model for the actual text output generation. 

The only key difference is the masking applied in each attention head in the decoder language model. This is done to ensure the integrity of the autoregressive language generation process, particularly in a decoder-only model, the code implements masking. This masking technique is crucial as it obscures any information following the current token's position, thereby directing the model's attention to only the preceding parts of the sequence. Such an attention mechanism is known as causal self-attention.

<div align="center">
<img src="https://github.com/AviSoori1x/seemore/blob/main/images/mhsa.png?raw=true" width="800" height="400" alt="seemore">
</div>

In the above image, the lower triangular mask is only applied in the case of a decoder model. Consider the bright blue triangle in matrix W absent in the case of visualizing the process in each attention head in the vision encoder. 

So here I implement these components in such a manner that they can be shared for both the vision encoder and language decoder by passing in an is_decoder boolean argument to the class constructor. 

The code for causal self attention and multi-head causal self attention can be organized as follows. Multi-head self attention applies multiple attention heads in parallel, each focusing on a separate section of the channel (the embedding dimension). Multi-head self attention essentially improves the learning process and improves efficiency of model training due to the inherently parallel implementation. Notice I have used dropout throughout this implementation for regularization i.e. preventing overfitting.

The implementation of the attention head looks like this: 


```python
class Head(nn.Module):
    def __init__(self, n_embd, head_size, dropout=0.1, is_decoder=False):
        super().__init__()
        
        # Linear layer for key projection
        self.key = nn.Linear(n_embd, head_size, bias=False)
        
        # Linear layer for query projection
        self.query = nn.Linear(n_embd, head_size, bias=False)
        
        # Linear layer for value projection
        self.value = nn.Linear(n_embd, head_size, bias=False)
        
        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Flag indicating whether this head is used in the decoder
        self.is_decoder = is_decoder

    def forward(self, x):
        # Get the batch size (B), sequence length (T), and embedding dimension (C) from the input tensor
        B, T, C = x.shape
        
        # Compute key, query, and value projections
        k = self.key(x)   # Shape: [B, T, head_size]
        q = self.query(x) # Shape: [B, T, head_size]
        v = self.value(x) # Shape: [B, T, head_size]
        
        # Compute attention scores by taking the dot product of query and key
        # and scaling by the square root of the embedding dimension
        wei = q @ k.transpose(-2, -1) * (C ** -0.5) # Shape: [B, T, T]
        
        if self.is_decoder:
            # If this head is used in the decoder, apply a causal mask to the attention scores
            # to prevent attending to future positions
            tril = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
            wei = wei.masked_fill(tril == 0, float('-inf'))
        
        # Apply softmax to the attention scores to obtain attention probabilities
        wei = F.softmax(wei, dim=-1) # Shape: [B, T, T]
        
        # Apply dropout to the attention probabilities for regularization
        wei = self.dropout(wei)
        
        # Perform weighted aggregation of values using the attention probabilities
        out = wei @ v # Shape: [B, T, head_size]
        
        return out

```

The implementation of multihead attention is as follows: 

``` python
class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, num_heads, dropout=0.1, is_decoder=False):
        super().__init__()
        
        # Ensure that the embedding dimension is divisible by the number of heads
        assert n_embd % num_heads == 0, "n_embd must be divisible by num_heads"
        
        # Create a ModuleList of attention heads
        self.heads = nn.ModuleList([
            Head(n_embd, n_embd // num_heads, dropout, is_decoder)
            for _ in range(num_heads)
        ])
        
        # Linear layer for projecting the concatenated head outputs
        self.proj = nn.Linear(n_embd, n_embd)
        
        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Apply each attention head to the input tensor
        head_outputs = [h(x) for h in self.heads]
        
        # Concatenate the outputs from all heads along the last dimension
        out = torch.cat(head_outputs, dim=-1)
        
        # Apply the projection layer to the concatenated outputs
        out = self.proj(out)
        
        # Apply dropout to the projected outputs for regularization
        out = self.dropout(out)
        
        return out

```
The multilayer perceptron that follows each multihead attention module is quite straightforward. Please note that I’ve noticed GELU being used quite often in Vision Transformers and ReLU used in text transformers, so I have this conditional logic to switch between the two based on where this MLP will be inserted. However, it seems that GELU is being used for both due to its resultant model performance, regardless of the fact that it’s more computationally expensive that RELU.

```python
class MLP(nn.Module):
    def __init__(self, n_embd, dropout=0.1, is_decoder=True):
        super().__init__()
        
        # Define the layers of the MLP
        layers = [
            # First linear layer that expands the input dimension from n_embd to 4 * n_embd
            nn.Linear(n_embd, 4 * n_embd),
            
            # Activation function: ReLU if is_decoder is True, else GELU
            nn.ReLU() if is_decoder else nn.GELU(),
            
            # Second linear layer that projects the intermediate dimension back to n_embd
            nn.Linear(4 * n_embd, n_embd),
            
            # Dropout layer for regularization
            nn.Dropout(dropout)
        ]
        
        # Create a sequential container to hold the layers
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # Pass the input through the MLP layers
        return self.net(x)
```

Multihead attention and MLP can be combined into transformer blocks. As discussed before the is_decoder boolean flag will allow us to turn the mask on and off, allowing us to create encoder and decoder blocks quite easily.

``` python
class Block(nn.Module):
    def __init__(self, n_embd, num_heads, dropout=0.1, is_decoder=False):
        super().__init__()
        
        # Layer normalization for the input to the attention layer
        self.ln1 = nn.LayerNorm(n_embd)
        
        # Multi-head attention module
        self.attn = MultiHeadAttention(n_embd, num_heads, dropout, is_decoder)
        
        # Layer normalization for the input to the FFN
        self.ln2 = nn.LayerNorm(n_embd)
        
        # Feed-forward neural network (FFN)
        self.ffn = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),  # Expand the dimension
            nn.GELU(),  # Activation function
            nn.Linear(4 * n_embd, n_embd),  # Project back to the original dimension
        )

    def forward(self, x):
        original_x = x  # Save the input for the residual connection
        
        # Apply layer normalization to the input
        x = self.ln1(x)
        
        # Apply multi-head attention
        attn_output = self.attn(x)
        
        # Add the residual connection (original input) to the attention output
        x = original_x + attn_output
        
        # Apply layer normalization to the input to the FFN
        x = self.ln2(x)
        
        # Apply the FFN
        ffn_output = self.ffn(x)
        
        # Add the residual connection (input to FFN) to the FFN output
        x = x + ffn_output
        
        return x
```

### Putting the vision encoder together

Now the patchification logic and attention blocks can be combined to create the vision transformer (ViT)

```python
class ViT(nn.Module):
    def __init__(self, img_size, patch_size, num_hiddens, num_heads, num_blks, emb_dropout, blk_dropout):
        super().__init__()
        
        # Patch embedding layer to convert the input image into patches
        self.patch_embedding = PatchEmbeddings(img_size, patch_size, num_hiddens)
        
        # Learnable classification token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, num_hiddens))
        
        # Calculate the number of patches
        num_patches = (img_size // patch_size) ** 2
        
        # Learnable position embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, num_hiddens))
        
        # Dropout layer for the embeddings
        self.dropout = nn.Dropout(emb_dropout)
        
        # Stack of transformer blocks
        self.blocks = nn.ModuleList([Block(num_hiddens, num_heads, blk_dropout, is_decoder=False) for _ in range(num_blks)])
        
        # Layer normalization for the final representation
        self.layer_norm = nn.LayerNorm(num_hiddens)

    def forward(self, X):
        # Convert the input image into patch embeddings
        x = self.patch_embedding(X)
        
        # Expand the classification token to match the batch size
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        
        # Concatenate the classification token with the patch embeddings
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add the position embedding to the patch embeddings
        x += self.pos_embedding
        
        # Apply dropout to the embeddings
        x = self.dropout(x)
        
        # Pass the embeddings through the transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Apply layer normalization to the final representation
        x = self.layer_norm(x[:, 0])
        
        return x

```

Overall, the ViT class encapsulates the architecture and forward pass of a Vision Transformer model. It takes an input image, converts it into patch embeddings, adds positional information, and processes the embeddings through a series of transformer blocks to generate a meaningful representation of the image. The final representation returned is the embedding corresponding to the CLS token, which is then used to condition the text generation in the language decoder. 

### Vision-Language Projection Module

However, we can’t directly concatenate this to the text embeddings. We need to project this from the dimensionality of image embeddings from the vision transformer to the dimensionality of text embeddings. This is done by the vision-language projector. As mentioned before, this can be a single learnable layer followed by a non-linearity or an MLP. Here I implement an MLP for a couple of reasons. 

1. This is an implementation to understand how things work in a VLM. So this is more interesting than a single projection layer.
   
2. There is an interesting current trend of keeping both the pretrained vision encoder and language decoder frozen during the VLM training phase. Therefore, allocating more parameters to the connection module could enhance the overall VLM's ability to generalize and aid in the downstream instruction-tuning process.

Here’s the implementation of this projection module. It’s not too different from the MLP used in the transformer blocks. 

```python
class MultiModalProjector(nn.Module):
    def __init__(self, n_embd, image_embed_dim, dropout=0.1):
        super().__init__()
        
        # Define the projection network
        self.net = nn.Sequential(
            # Linear layer to expand the image embedding dimension
            nn.Linear(image_embed_dim, 4 * image_embed_dim),
            
            # GELU activation function
            nn.GELU(),
            
            # Linear layer to project the expanded image embeddings to the text embedding dimension
            nn.Linear(4 * image_embed_dim, n_embd),
            
            # Dropout layer for regularization
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Pass the input through the projection network
        x = self.net(x)
        return x
```

### Building the Decoder Language Model

The final component we need to look at is the decoder language model. Here I’ve remained within the confines of the modern VLM architecture but deviated a bit in the implementation. I have integrated the projection module into the decoder model class implementation. This is because I built everything from scratch and wanted to retain the causal language model architecture from Andrej Karpathy’s makemore. There’s no easy way to directly feed in reshaped embeddings in this implementation, so I’ve had to improvise a little. Please keep in mind that in using pretrained models with the Hugging Face API or any other modern library that allows you to use pretrained large language models, you can directly feed embeddings as input to the model (e.g. using inputs_embeds parameter: https://huggingface.co/docs/transformers/en/model_doc/gpt2#transformers.GPT2Model.forward.inputs_embeds).

That being said, what I’ve done here is an interesting exercise in that it allows you to see in pretty simple code:

- How the image embeddings are reshaped using the vision language projector to match that of text embeddings.

- Then concatenated with token embedding.

- Subsequently combined with position embeddings and used to eventually calculate a loss function (and finally generate text). 

Essentially the text generation is conditioned on the initial image input. This can be modified in a number of ways to work with interleaved text and images, which will be useful for multi-turn conversation i.e. chat scenarios using the finetuned VLM.


The crucial parts of this decoder implementation is given below. Note how the is_decoder flag is passed as ‘True’ to use the masked version of the self attention blocks, resulting in causal scaled dot product self attention in the language decoder. Please refer to the GitHub repo linked above for the full implementation.

```python
class DecoderLanguageModel(nn.Module):
    def __init__(self, n_embd, image_embed_dim, vocab_size, num_heads, n_layer, use_images=False):
        super().__init__()
        
        self.use_images = use_images
        
        # Token embedding table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        
        # Position embedding table
        self.position_embedding_table = nn.Embedding(1000, n_embd)
        
        if use_images:
            # Image projection layer to align image embeddings with text embeddings
            self.image_projection = MultiModalProjector(n_embd, image_embed_dim)
        
        # Stack of transformer decoder blocks
        self.blocks = nn.Sequential(*[Block(n_embd, num_heads, is_decoder=True) for _ in range(n_layer)])
        
        # Final layer normalization
        self.ln_f = nn.LayerNorm(n_embd)
        
        # Language modeling head
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, image_embeds=None, targets=None):
        # Get token embeddings from the input indices
        tok_emb = self.token_embedding_table(idx)
        
        if self.use_images and image_embeds is not None:
            # Project and concatenate image embeddings with token embeddings
            img_emb = self.image_projection(image_embeds).unsqueeze(1)
            tok_emb = torch.cat([img_emb, tok_emb], dim=1)
        
        # Get position embeddings
        pos_emb = self.position_embedding_table(torch.arange(tok_emb.size(1), device=device)).unsqueeze(0)
        
        # Add position embeddings to token embeddings
        x = tok_emb + pos_emb
        
        # Pass through the transformer decoder blocks
        x = self.blocks(x)
        
        # Apply final layer normalization
        x = self.ln_f(x)
        
        # Get the logits from the language modeling head
        logits = self.lm_head(x)
        
        if targets is not None:
            if self.use_images and image_embeds is not None:
                # Prepare targets by concatenating a dummy target for the image embedding
                batch_size = idx.size(0)
                targets = torch.cat([torch.full((batch_size, 1), -100, dtype=torch.long, device=device), targets], dim=1)
            
            # Compute the cross-entropy loss
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
            return logits, loss
        
        return logits

    def generate(self, idx, image_embeds, max_new_tokens):
          # Autoregressive character geneneration conditioned on visual token and preceding tokens. Refer to repo
```

### Bringing everything together to implement Seemore: the simple Vision Language Model


Now that we have our three key components, we can put it all together into a Vision Language Model. The full implementation is given below. If you were to remove the assert statements for error handling, this looks very simple. Coming back full circle to the outline I’ve given at the beginning of the blog, all that’s happening here is:

1. Get image features from the vision encoder (Here it’s a vision transformer, but it could be any model that could generate features from an image input such as a ResNet or a traditional convolutional neural network (needless to say performance may suffer))

2. A projection module for projecting image tokens to the same embedding space as text embeddings for the decoder (this projector is integrated with the decoder in this implementation).

3. A decoder language model for generating text conditioned on a preceding image. 

```python
class VisionLanguageModel(nn.Module):
    def __init__(self, n_embd, image_embed_dim, vocab_size, n_layer, img_size, patch_size, num_heads, num_blks, emb_dropout, blk_dropout):
        super().__init__()
        
        # Set num_hiddens equal to image_embed_dim
        num_hiddens = image_embed_dim
        
        # Assert that num_hiddens is divisible by num_heads
        assert num_hiddens % num_heads == 0, "num_hiddens must be divisible by num_heads"
        
        # Initialize the vision encoder (ViT)
        self.vision_encoder = ViT(img_size, patch_size, num_hiddens, num_heads, num_blks, emb_dropout, blk_dropout)
        
        # Initialize the language model decoder (DecoderLanguageModel)
        self.decoder = DecoderLanguageModel(n_embd, image_embed_dim, vocab_size, num_heads, n_layer, use_images=True)

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
```

And now we've implemented everything we set out to implement:
<div align="center">
 <img src="https://github.com/AviSoori1x/seemore/blob/main/images/vlm.png?raw=true" width="600" height="600" alt="seemore">
</div>

The repo (here: https://github.com/AviSoori1x/seemore) has some mock data, data loaders implemented mostly from scratch and a simple training loop with cross-entropy loss calculation. Please note that in this simple example, we are training the entire system end to end, much like Kosmos-1 from Microsoft Research. I left it at this for convenience. In practice, the commonly observed sequence is:

1. Get pretrained vision encoder from SigLIP or CLIP (both come in difference sizes). Freeze weights (i.e. don’t update during backward pass in training)
   
2. Get pretrained decoder only language model e.g. all the way from TinyLLaMA, Phi-2 etc. to Llama 3 (or even much bigger in the case of GPT-4 and Grok 1.5 etc.). Freeze weights.
   
3. Implement a projection module and train a VLM module much like what we have here, but only updating the weights of this projection module. This would effectively be the pretraining phase.
  
4. Then during the instruction finetuning keep both the projection module and the decoder language model unfrozen and update weights of both in the backward pass.

I developed this on Databricks using a single T4 GPU and MLFlow for tracking loss (during the training process). I wanted to set this up this way so that I can scale up to a GPU cluster of any size I want quite easily on Databricks, should I decide to adapt this to a more performance oriented implementation. However, you can run this anywhere, with or without a GPU. Please note that even the toy training loop with 90 samples will be painfully slow on a CPU. 


Please check out the repo (https://github.com/AviSoori1x/seemore), run the code yourself, and have fun! 
