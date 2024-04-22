# seemore

<div align="center">
    <img src="images/seemorelogo.png" width="300"/>
</div>


<a href="https://www.databricks.com/product/machine-learning">
    <img src="https://raw.githubusercontent.com/AviSoori1x/makeMoE/main/images/databricks.png" width="50px" height="auto">
</a>
<br>

<span>Developed using Databricks with ❤️</span>

#### Vision Language Model from scratch in Pytorch

The Blog that walks through this: https://avisoori1x.github.io/2024/04/22/seemore-_Implement_a_Vision_Language_Model_from_Scratch.html

In this simple implementation of a vision language model (VLM), there are 3 main components. 

1. Image Encoder to extract visual features from images. In this case I use a from scratch implementation of the original vision transformer used in CLIP. This is actually a popular choice in many modern VLMs. The one notable exception is Fuyu series of models from Adept, that passes the patchified images directly to the projection layer.
   
2. Vision-Language Projector - Image embeddings are not of the same shape as text embeddings used by the decoder. So we need to ‘project’ i.e. change dimensionality of image features extracted by the image encoder to match what’s observed in the text embedding space. So image features become ‘visual tokens’ for the decoder. This could be a single layer or an MLP. I’ve used an MLP because it’s worth showing.

3. A decoder only language model. This is the component that ultimately generates text. In my implementation I’ve deviated from what you see in LLaVA etc. a bit by incorporating the projection module to my decoder. Typically this is not observed, and you leave the architecture of the decoder (which is usually an already pretrained model) untouched.

The scaled dot product self attention implementation is borrowed from Andrej Kapathy's makemore (https://github.com/karpathy/makemore). Also the decoder is an autoregressive character-level language model, just like in makemore. Now you see where the name 'seemore' came from :)

Everything is written from the ground up using pytorch. That includes the attention mechanism (both for the vision encoder and language decoder), patch creation for the vision transformer and everything else. Hope this is useful for anyone going through the repo and/ or the associated blog.

Publications heavily referenced for this implementation: 
- Large Multimodal Models: Notes on CVPR 2023 Tutorial: https://arxiv.org/pdf/2306.14895.pdf
- Visual Instruction Tuning: https://arxiv.org/pdf/2304.08485.pdf
- Language Is Not All You Need: Aligning Perception with Language Models: https://arxiv.org/pdf/2302.14045.pdf

seemore.py is the entirety of the implementation in a single file of pytorch.

seemore_from_Scratch.ipynb walks through the intuition for the entire model architecture and how everything comes together. I recommend starting here.

seemore_Concise.ipynb is the consolidated hackable implementation that I encourage you to hack, understand, improve and make your own

The input.txt with tinyshakespear and the base64 encoded string representations + corresponding descriptions are in the inputs.csv file in the images directory.

The modules subdirectory contains each of the components in their own .py file for convenience (should you choose to hack on pieces individually/ reuse for your own projects etc.)   

**The code was entirely developed on Databricks using a single A100 for compute. If you're running this on Databricks, you can scale this on an arbitrarily large GPU cluster with no issues, on the cloud provider of your choice.**

**I chose to use MLFlow (which comes pre-installed in Databricks. It's fully open source and you can pip install easily elsewhere) as I find it helpful to track and log all the metrics necessary. This is entirely optional but encouraged.**

**Please note that the implementation emphasizes readability and hackability vs. performance, so there are many ways in which you could improve this. Please try and let me know!**

Hope you find this useful. Happy hacking!!
