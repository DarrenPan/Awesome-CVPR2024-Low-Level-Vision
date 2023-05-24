# Awesome-CVPR2023-Low-Level-Vision
A Collection of Papers and Codes in CVPR2023 related to Low-Level Vision


## Related collections for low-level vision
- [Awesome-CVPR2022-Low-Level-Vision](https://github.com/DarrenPan/Awesome-CVPR2023-Low-Level-Vision/blob/main/CVPR2022-Low-Level-Vision.md)
- [Awesome-NeurIPS2022/2021-Low-Level-Vision](https://github.com/DarrenPan/Awesome-NeurIPS2022-Low-Level-Vision)
- [Awesome-ECCV2022-Low-Level-Vision](https://github.com/DarrenPan/Awesome-ECCV2022-Low-Level-Vision)
- [Awesome-AAAI2022-Low-Level-Vision](https://github.com/DarrenPan/Awesome-AAAI2022-Low-Level-Vision)
- [Awesome-ICCV2021-Low-Level-Vision](https://github.com/Kobaayyy/Awesome-ICCV2021-Low-Level-Vision)
- [Awesome-CVPR2021/2020-Low-Level-Vision](https://github.com/Kobaayyy/Awesome-CVPR2021-CVPR2020-Low-Level-Vision)
- [Awesome-ECCV2020-Low-Level-Vision](https://github.com/Kobaayyy/Awesome-ECCV2020-Low-Level-Vision)


## Catalogue

- [Image Restoration](#ImageRetoration)
  - [Video Restoration](#VideoRestoration)

- [Super Resolution](#SuperResolution)
  - [Image Super Resolution](#ImageSuperResolution)
  - [Video Super Resolution](#VideoSuperResolution)
- [Image Rescaling](#Rescaling)

- [Denoising](#Denoising)
  - [Image Denoising](#ImageDenoising)
  - [Video Denoising](#VideoDenoising)

- [Deblurring](#Deblurring)
  - [Image Deblurring](#ImageDeblurring)
  - [Video Deblurring](#VideoDeblurring)

- [Deraining](#Deraining)

- [Dehazing](#Dehazing)

- [Demosaicing](#Demosaicing)

- [HDR Imaging / Multi-Exposure Image Fusion](#HDR)

- [Frame Interpolation](#FrameInterpolation)

- [Image Enhancement](#Enhancement)
  - [Low-Light Image Enhancement](#LowLight)

- [Image Harmonization](#Harmonization)

- [Image Completion/Inpainting](#Inpainting)

- [Image Matting](#Matting)

- [Shadow Removal](#ShadowRemoval)

- [Image Compression](#ImageCompression)

- [Image Quality Assessment](#ImageQualityAssessment)

- [Style Transfer](#StyleTransfer)

- [Image Editing](#ImageEditing)

- [Image Generation/Synthesis/ Image-to-Image Translation](#ImageGeneration)
  - [Video Generation](#VideoGeneration)

- [Others](#Others)

<a name="ImageRetoration"></a>
# Image Restoration - 图像恢复 [[back](#catalogue)]

**Efficient and Explicit Modelling of Image Hierarchies for Image Restoration**
- Paper: https://arxiv.org/abs/2303.00748
- Code: https://github.com/ofsoundof/GRL-Image-Restoration
- Tags: Transformer

**Learning Distortion Invariant Representation for Image Restoration from A Causality Perspective**
- Paper: https://arxiv.org/abs/2303.06859
- Code: https://github.com/lixinustc/Casual-IRDIL

**Generative Diffusion Prior for Unified Image Restoration and Enhancement**
- Paper: https://arxiv.org/abs/2304.01247
- Code: https://github.com/Fayeben/GenerativeDiffusionPrior

**DR2: Diffusion-based Robust Degradation Remover for Blind Face Restoration**
- Paper: https://arxiv.org/abs/2303.06885

**Bitstream-Corrupted JPEG Images are Restorable: Two-stage Compensation and Alignment Framework for Image Restoration**
- Paper: https://arxiv.org/abs/2304.06976
- Code: https://github.com/wenyang001/Two-ACIR

**Contrastive Semi-supervised Learning for Underwater Image Restoration via Reliable Bank**
- Paper: https://arxiv.org/abs/2303.09101
- Code: https://github.com/Huang-ShiRui/Semi-UIR
- Tags: Underwater Image Restoration

**Nighttime Smartphone Reflective Flare Removal Using Optical Center Symmetry Prior**
- Paper: https://arxiv.org/abs/2303.15046
- Code: https://github.com/ykdai/BracketFlare
- Tags: Reflective Flare Removal

**Generating Aligned Pseudo-Supervision from Non-Aligned Data for Image Restoration in Under-Display Camera**
- Paper: https://arxiv.org/abs/2304.06019
- Code: https://github.com/jnjaby/AlignFormer

**GamutMLP - A Lightweight MLP for Color Loss Recovery**
- Paper: https://arxiv.org/abs/2304.11743
- Code: https://github.com/hminle/gamut-mlp
- Tags: restore wide-gamut color values

**ABCD : Arbitrary Bitwise Coefficient for De-quantization**
- Paper: https://ipl.dgist.ac.kr/ABCD_cvpr23.pdf
- Code: https://github.com/WooKyoungHan/ABCD
- Tags: De-quantization/Bit depth expansion

## Image Reconstruction

**Raw Image Reconstruction with Learned Compact Metadata**
- Paper: https://arxiv.org/abs/2302.12995
- Code: https://github.com/wyf0912/R2LCM

**High-resolution image reconstruction with latent diffusion models from human brain activity**
- Paper: https://www.biorxiv.org/content/10.1101/2022.11.18.517004v2
- Code: https://github.com/yu-takagi/StableDiffusionReconstruction

**Catch Missing Details: Image Reconstruction with Frequency Augmented Variational Autoencoder**
- Paper: https://arxiv.org/abs/2305.02541

**Optimization-Inspired Cross-Attention Transformer for Compressive Sensing**
- Paper: https://arxiv.org/abs/2304.13986
- Code: https://github.com/songjiechong/OCTUF
- Tags: Compressive Sensing

<a name="BurstRestoration"></a>
## Burst Restoration

**Burstormer: Burst Image Restoration and Enhancement Transformer**
- Paper: https://arxiv.org/abs/2304.01194
- Code: https://github.com/akshaydudhane16/Burstormer

**Gated Multi-Resolution Transfer Network for Burst Restoration and Enhancement**
- Paper: https://arxiv.org/abs/2304.06703

<a name="VideoRestoration"></a>
## Video Restoration

**Blind Video Deflickering by Neural Filtering with a Flawed Atlas**
- Paper: https://arxiv.org/abs/2303.08120
- Code: https://github.com/ChenyangLEI/All-In-One-Deflicker
- Tags: Deflickering 


<a name="SuperResolution"></a>
# Super Resolution - 超分辨率 [[back](#catalogue)]
<a name="ImageSuperResolution"></a>
## Image Super Resolution

**Activating More Pixels in Image Super-Resolution Transformer**
- Paper: https://arxiv.org/abs/2205.04437
- Code: https://github.com/XPixelGroup/HAT
- Tags: Transformer

**N-Gram in Swin Transformers for Efficient Lightweight Image Super-Resolution**
- Paper: https://arxiv.org/abs/2211.11436
- Code: https://github.com/rami0205/NGramSwin

**Omni Aggregation Networks for Lightweight Image Super-Resolution**
- Paper: https://arxiv.org/abs/2304.10244
- Code: https://github.com/Francis0625/Omni-SR

**OPE-SR: Orthogonal Position Encoding for Designing a Parameter-free Upsampling Module in Arbitrary-scale Image Super-Resolution**
- Paper: https://arxiv.org/abs/2303.01091

**Local Implicit Normalizing Flow for Arbitrary-Scale Image Super-Resolution**
- Paper: https://arxiv.org/abs/2303.05156

**Super-Resolution Neural Operator**
- Paper: https://arxiv.org/abs/2303.02584
- Code: https://github.com/2y7c3/Super-Resolution-Neural-Operator

**Human Guided Ground-truth Generation for Realistic Image Super-resolution**
- Paper: https://arxiv.org/abs/2303.13069
- Code: https://github.com/ChrisDud0257/PosNegGT

**Better "CMOS" Produces Clearer Images: Learning Space-Variant Blur Estimation for Blind Image Super-Resolution**
- Paper: https://arxiv.org/abs/2304.03542
- Tags: Blind

**Implicit Diffusion Models for Continuous Super-Resolution**
- Paper: https://arxiv.org/abs/2303.16491
- Code: https://github.com/Ree1s/IDM

**Zero-Shot Dual-Lens Super-Resolution**
- Paper:
- Code: https://github.com/XrKang/ZeDuSR

**CABM: Content-Aware Bit Mapping for Single Image Super-Resolution Network with Large Input**
- Paper: https://arxiv.org/abs/2304.06454

**Learning Generative Structure Prior for Blind Text Image Super-resolution**
- Paper: https://arxiv.org/abs/2303.14726
- Code: https://github.com/csxmli2016/MARCONet
- Tags: Text SR

**Guided Depth Super-Resolution by Deep Anisotropic Diffusion**
- Paper: https://arxiv.org/abs/2211.11592
- Code: https://github.com/prs-eth/Diffusion-Super-Resolution
- Tags: Guided Depth SR

**Probability-based Global Cross-modal Upsampling for Pansharpening**
- Paper: https://arxiv.org/abs/2303.13659
- Code: https://github.com/Zeyu-Zhu/PGCU
- Tags: Pansharpening(for remote sensing image)

**Quantum Annealing for Single Image Super-Resolution**
- Paper: https://arxiv.org/abs/2304.08924
- Tags: [Workshop]

**Bicubic++: Slim, Slimmer, Slimmest -- Designing an Industry-Grade Super-Resolution Network**
- Paper: https://arxiv.org/abs/2305.02126
- Code: https://github.com/aselsan-research-imaging-team/bicubic-plusplus
- Tags: [Workshop]

**Hybrid Transformer and CNN Attention Network for Stereo Image Super-resolution**
- Paper: https://arxiv.org/abs/2305.05177
- Tags: [Workshop]

<a name="VideoSuperResolution"></a>
## Video Super Resolution

**Towards High-Quality and Efficient Video Super-Resolution via Spatial-Temporal Data Overfitting**
- Paper: https://arxiv.org/abs/2303.08331
- Code: https://github.com/coulsonlee/STDO-CVPR2023

**Structured Sparsity Learning for Efficient Video Super-Resolution**
- Paper: https://github.com/Zj-BinXia/SSL
- Code: https://arxiv.org/abs/2206.07687

<a name="Rescaling"></a>
# Image Rescaling - 图像缩放 [[back](#catalogue)]

**HyperThumbnail: Real-time 6K Image Rescaling with Rate-distortion Optimization**
- Paper: https://arxiv.org/abs/2304.01064
- Code: https://github.com/AbnerVictor/HyperThumbnail


<a name="Denoising"></a>
# Denoising - 去噪 [[back](#catalogue)]

<a name="ImageDenoising"></a>
## Image Denoising

**Masked Image Training for Generalizable Deep Image Denoising**
- Paper: https://arxiv.org/abs/2303.13132
- Code: https://github.com/haoyuc/MaskedDenoising

**Spatially Adaptive Self-Supervised Learning for Real-World Image Denoising**
- Paper: https://arxiv.org/abs/2303.14934
- Cdoe: https://github.com/nagejacob/SpatiallyAdaptiveSSID
- Tags: Self-Supervised

**LG-BPN: Local and Global Blind-Patch Network for Self-Supervised Real-World Denoising**
- Paper: https://arxiv.org/abs/2304.00534
- Code: https://github.com/Wang-XIaoDingdd/LGBPN
- Tags: Self-Supervised

**Real-time Controllable Denoising for Image and Video**
- Paper: https://arxiv.org/pdf/2303.16425.pdf


<!-- 
<a name="VideoDenoising"></a>
## Video Denoising

-->

<a name="Deblurring"></a>
# Deblurring - 去模糊 [[back](#catalogue)]
<a name="ImageDeblurring"></a>
## Image Deblurring

**Structured Kernel Estimation for Photon-Limited Deconvolution**
- Paper: https://arxiv.org/abs/2303.03472
- Code: https://github.com/sanghviyashiitb/structured-kernel-cvpr23

**Blur Interpolation Transformer for Real-World Motion from Blur**
- Paper: https://arxiv.org/abs/2211.11423
- Code: https://github.com/zzh-tech/BiT

**Neumann Network with Recursive Kernels for Single Image Defocus Deblurring**
- Paper:
- Code: https://github.com/csZcWu/NRKNet

**Efficient Frequency Domain-based Transformers for High-Quality Image Deblurring**
- Paper: https://arxiv.org/abs/2211.12250
- Code: https://github.com/kkkls/FFTformer

**Hybrid Neural Rendering for Large-Scale Scenes with Motion Blur**
- Paper: https://arxiv.org/abs/2304.12652
- Code: https://github.com/CVMI-Lab/HybridNeuralRendering

<!--
<a name="VideoDeblurring"></a>
## Video Deblurring

-->

<a name="Deraining"></a>
# Deraining - 去雨 [[back](#catalogue)]

**Learning A Sparse Transformer Network for Effective Image Deraining**
- Paper: https://arxiv.org/abs/2303.11950
- Code: https://github.com/cschenxiang/DRSformer

<a name="Dehazing"></a>
# Dehazing - 去雾 [[back](#catalogue)]

**RIDCP: Revitalizing Real Image Dehazing via High-Quality Codebook Priors**
- Paper: https://arxiv.org/abs/2304.03994
- Code: https://github.com/RQ-Wu/RIDCP

**Curricular Contrastive Regularization for Physics-aware Single Image Dehazing**
- Paper: https://arxiv.org/abs/2303.14218
- Code: https://github.com/YuZheng9/C2PNet

**Video Dehazing via a Multi-Range Temporal Alignment Network with Physical Prior**
- Paper: https://arxiv.org/abs/2303.09757
- Code: https://github.com/jiaqixuac/MAP-Net

**SCANet: Self-Paced Semi-Curricular Attention Network for Non-Homogeneous Image Dehazing**
 - Paper: https://arxiv.org/abs/2304.08444
 - Code: https://github.com/gy65896/SCANet
 - Tags: [Workshop]

**Streamlined Global and Local Features Combinator (SGLC) for High Resolution Image Dehazing**
- Paper: https://arxiv.org/abs/2304.13375
- Tags: [Workshop]

<!--

<a name="Demosaicing"></a>
# Demosaicing - 去马赛克 [[back](#catalogue)]

-->

 <a name="HDR"></a>
# HDR Imaging / Multi-Exposure Image Fusion - HDR图像生成 / 多曝光图像融合 [[back](#catalogue)]

**Learning a Practical SDR-to-HDRTV Up-conversion using New Dataset and Degradation Models**
- Paper: https://arxiv.org/abs/2303.13031
- Code: https://github.com/AndreGuo/HDRTVDM

**SMAE: Few-shot Learning for HDR Deghosting with Saturation-Aware Masked Autoencoders**
- Paper: https://arxiv.org/abs/2304.06914

**A Unified HDR Imaging Method with Pixel and Patch Level**
- Paper: https://arxiv.org/abs/2304.06943

**Inverting the Imaging Process by Learning an Implicit Camera Model**
- Paper: https://arxiv.org/abs/2304.12748
- Code: https://github.com/xhuangcv/neucam
- Tags: generating all-in-focus photos & HDR imaging

<a name="FrameInterpolation"></a>
# Frame Interpolation - 插帧 [[back](#catalogue)]

**Extracting Motion and Appearance via Inter-Frame Attention for Efficient Video Frame Interpolation**
- Paper: https://arxiv.org/abs/2303.00440
- Code: https://github.com/MCG-NJU/EMA-VFI

**A Unified Pyramid Recurrent Network for Video Frame Interpolation**
- Paper: https://arxiv.org/abs/2211.03456
- Code: https://github.com/srcn-ivl/UPR-Net

**BiFormer: Learning Bilateral Motion Estimation via Bilateral Transformer for 4K Video Frame Interpolation**
- Paper: https://arxiv.org/abs/2304.02225
- Code: https://github.com/JunHeum/BiFormer

**AMT: All-Pairs Multi-Field Transforms for Efficient Frame Interpolation**
- Paper: https://arxiv.org/abs/2304.09790
- Code: https://github.com/MCG-NKU/AMT

**Exploring Discontinuity for Video Frame Interpolation**
- Paper: https://arxiv.org/abs/2202.07291
- Code: https://github.com/pandatimo/Exploring-Discontinuity-for-VFI

**Event-based Video Frame Interpolation with Cross-Modal Asymmetric Bidirectional Motion Fields**
- Paper:
- Code: https://github.com/intelpro/CBMNet
- Tags: Event-based

**Event-based Blurry Frame Interpolation under Blind Exposure**
- Paper:
- Code: https://github.com/WarranWeng/EBFI-BE
- Tags: Event-based

**Joint Video Multi-Frame Interpolation and Deblurring under Unknown Exposure Time**
- Paper: https://arxiv.org/abs/2303.15043
- Code: https://github.com/shangwei5/VIDUE
- Tags: Frame Interpolation and Deblurring

<a name="Enhancement"></a>
# Image Enhancement - 图像增强 [[back](#catalogue)]

<a name="LowLight"></a>
## Low-Light Image Enhancement

**Learning Semantic-Aware Knowledge Guidance for Low-Light Image Enhancement**
- Paper: https://arxiv.org/abs/2304.07039
- Code: https://github.com/langmanbusi/Semantic-Aware-Low-Light-Image-Enhancement

**Visibility Constrained Wide-band Illumination Spectrum Design for Seeing-in-the-Dark**
- Paper: https://arxiv.org/abs/2303.11642
- Code: https://github.com/MyNiuuu/VCSD
- Tags: NIR2RGB


<a name="Harmonization"></a>
# Image Harmonization/Composition - 图像协调/图像合成 [[back](#catalogue)]

**LEMaRT: Label-Efficient Masked Region Transform for Image Harmonization**
- Paper: https://arxiv.org/abs/2304.13166

<!--
<a name="Inpainting"></a>
# Image Completion/Inpainting - 图像修复 [[back](#catalogue)]

-->

<a name="Matting"></a>
# Image Matting - 图像抠图 [[back](#catalogue)]

**Referring Image Matting**
- Paper: https://arxiv.org/abs/2206.05149
- Code: https://github.com/JizhiziLi/RIM

**Adaptive Human Matting for Dynamic Videos**
- Paper: https://arxiv.org/abs/2304.06018
- Code: https://github.com/microsoft/AdaM


<a name="ShadowRemoval"></a>
# Shadow Removal - 阴影消除 [[back](#catalogue)]

**ShadowDiffusion: When Degradation Prior Meets Diffusion Model for Shadow Removal**
- Paper: https://arxiv.org/abs/2212.04711
- Code: https://github.com/GuoLanqing/ShadowDiffusion

<!--
<a name="Relighting"></a>
# Relighting


<a name="Stitching"></a>
# Image Stitching - 图像拼接 

-->

<a name="ImageCompression"></a>
# Image Compression - 图像压缩 [[back](#catalogue)]

**Backdoor Attacks Against Deep Image Compression via Adaptive Frequency Trigger**
- Paper: https://arxiv.org/abs/2302.14677

**Context-based Trit-Plane Coding for Progressive Image Compression**
- Paper: https://arxiv.org/abs/2303.05715
- Code: https://github.com/seungminjeon-github/CTC

**Learned Image Compression with Mixed Transformer-CNN Architectures**
- Paper: https://arxiv.org/abs/2303.14978
- Code: https://github.com/jmliu206/LIC_TCM

## Video Compression

**Neural Video Compression with Diverse Contexts**
- Paper: https://github.com/microsoft/DCVC
- Code: https://arxiv.org/abs/2302.14402


<a name="ImageQualityAssessment"></a>
# Image Quality Assessment - 图像质量评价 [[back](#catalogue)]

**Quality-aware Pre-trained Models for Blind Image Quality Assessment**
- Paper: https://arxiv.org/abs/2303.00521

**Blind Image Quality Assessment via Vision-Language Correspondence: A Multitask Learning Perspective**
- Paper: https://arxiv.org/abs/2303.14968
- Code: https://github.com/zwx8981/LIQE

**Towards Artistic Image Aesthetics Assessment: a Large-scale Dataset and a New Method**
- Paper: https://arxiv.org/abs/2303.15166
- Code: https://github.com/Dreemurr-T/BAID

**Re-IQA: Unsupervised Learning for Image Quality Assessment in the Wild**
- Paper: https://arxiv.org/abs/2304.00451

**SB-VQA: A Stack-Based Video Quality Assessment Framework for Video Enhancement**
- Paper: https://arxiv.org/abs/2305.08408
- Tags: [Workshop]

<a name="StyleTransfer"></a>
# Style Transfer - 风格迁移 [[back](#catalogue)]

**Fix the Noise: Disentangling Source Feature for Controllable Domain Translation**
- Paper: https://arxiv.org/abs/2303.11545
- Code: https://github.com/LeeDongYeun/FixNoise

**Neural Preset for Color Style Transfer**
- Paper: https://arxiv.org/abs/2303.13511
- Code: https://github.com/ZHKKKe/NeuralPreset

**CAP-VSTNet: Content Affinity Preserved Versatile Style Transfer**
- Paper: https://arxiv.org/abs/2303.17867

**StyleGAN Salon: Multi-View Latent Optimization for Pose-Invariant Hairstyle Transfer**
- Paper: https://arxiv.org/abs/2304.02744
- Project: https://stylegan-salon.github.io/

**Modernizing Old Photos Using Multiple References via Photorealistic Style Transfer**
- Paper: https://arxiv.org/abs/2304.04461
- Project: https://kaist-viclab.github.io/old-photo-modernization/

**QuantArt: Quantizing Image Style Transfer Towards High Visual Fidelity**
- Paper: https://arxiv.org/abs/2212.10431
- Code: https://github.com/siyuhuang/QuantArt

**Master: Meta Style Transformer for Controllable Zero-Shot and Few-Shot Artistic Style Transfer**
- Paper: https://arxiv.org/abs/2304.11818

<a name="ImageEditing"></a>
# Image Editing - 图像编辑 [[back](#catalogue)]

**Imagic: Text-Based Real Image Editing with Diffusion Models**
- Paper: https://arxiv.org/abs/2210.09276

**SINE: SINgle Image Editing with Text-to-Image Diffusion Models**
- Paper: https://arxiv.org/abs/2212.04489
- Code: https://github.com/zhang-zx/SINE

**CoralStyleCLIP: Co-optimized Region and Layer Selection for Image Editing**
- Paper: https://arxiv.org/abs/2303.05031

**DeltaEdit: Exploring Text-free Training for Text-Driven Image Manipulation**
- Paper: https://arxiv.org/abs/2303.06285
- Code: https://arxiv.org/abs/2303.06285

**SIEDOB: Semantic Image Editing by Disentangling Object and Background**
- Paper: https://arxiv.org/abs/2303.13062
- Code: https://github.com/WuyangLuo/SIEDOB

**DiffusionRig: Learning Personalized Priors for Facial Appearance Editing**
- Paper: https://arxiv.org/abs/2304.06711
- Code: https://github.com/adobe-research/diffusion-rig

<a name=ImageGeneration></a>
# Image Generation/Synthesis / Image-to-Image Translation - 图像生成/合成/转换 [[back](#catalogue)]
## Text-to-Image / Text Guided / Multi-Modal

**Multi-Concept Customization of Text-to-Image Diffusion**
- Paper: https://arxiv.org/abs/2212.04488
- Code: https://github.com/adobe-research/custom-diffusion

**GALIP: Generative Adversarial CLIPs for Text-to-Image Synthesis**
- Paper: https://arxiv.org/abs/2301.12959
- Code: https://github.com/tobran/GALIP

**Scaling up GANs for Text-to-Image Synthesis**
- Paper: https://arxiv.org/abs/2303.05511
- Project: https://mingukkang.github.io/GigaGAN/

**MAGVLT: Masked Generative Vision-and-Language Transformer**
- Paper: https://arxiv.org/abs/2303.12208

**Freestyle Layout-to-Image Synthesis**
- Paper: https://arxiv.org/abs/2303.14412
- Code: https://github.com/essunny310/FreestyleNet

**Variational Distribution Learning for Unsupervised Text-to-Image Generation**
- Paper: https://arxiv.org/abs/2303.16105

**Sound to Visual Scene Generation by Audio-to-Visual Latent Alignment**
- Paper: https://arxiv.org/abs/2303.17490
- Project: https://sound2scene.github.io/

**Toward Verifiable and Reproducible Human Evaluation for Text-to-Image Generation**
- Paper: https://arxiv.org/abs/2304.01816

**Shifted Diffusion for Text-to-image Generation**
- Paper: https://arxiv.org/abs/2211.15388
- Code: https://github.com/drboog/Shifted_Diffusion

**Collaborative Diffusion for Multi-Modal Face Generation and Editing**
- Paper: https://arxiv.org/abs/2304.10530
- Code: https://github.com/ziqihuangg/Collaborative-Diffusion

## Image-to-Image / Image Guided

**LANIT: Language-Driven Image-to-Image Translation for Unlabeled Data**
- Paper: https://arxiv.org/abs/2208.14889
- Code: https://github.com/KU-CVLAB/LANIT

**Person Image Synthesis via Denoising Diffusion Model**
- Paper: https://arxiv.org/abs/2211.12500
- Code: https://github.com/ankanbhunia/PIDM

**Picture that Sketch: Photorealistic Image Generation from Abstract Sketches**
- Paper: https://arxiv.org/abs/2303.11162

**Fine-Grained Face Swapping via Regional GAN Inversion**
- Paper: https://arxiv.org/abs/2211.14068
- Code: https://github.com/e4s2022/e4s

**Masked and Adaptive Transformer for Exemplar Based Image Translation**
- Paper: https://arxiv.org/abs/2303.17123
- Code: https://github.com/AiArt-HDU/MATEBIT

**Zero-shot Generative Model Adaptation via Image-specific Prompt Learning**
- Paper: https://arxiv.org/abs/2304.03119
- Code: https://github.com/Picsart-AI-Research/IPL-Zero-Shot-Generative-Model-Adaptation

## Others for image generation

**AdaptiveMix: Robust Feature Representation via Shrinking Feature Space**
- Paper: https://arxiv.org/abs/2303.01559
- Code: https://github.com/WentianZhang-ML/AdaptiveMix

**MAGE: MAsked Generative Encoder to Unify Representation Learning and Image Synthesis**
- Paper: https://arxiv.org/abs/2211.09117
- Code: https://github.com/LTH14/mage

**Regularized Vector Quantization for Tokenized Image Synthesis**
- Paper: https://arxiv.org/abs/2303.06424  

**Towards Accurate Image Coding: Improved Autoregressive Image Generation with Dynamic Vector Quantization**
- Paper:
- Code: https://github.com/CrossmodalGroup/DynamicVectorQuantization

**Not All Image Regions Matter: Masked Vector Quantization for Autoregressive Image Generation**
- Paper:
- Code: https://github.com/CrossmodalGroup/MaskedVectorQuantization

**Exploring Incompatible Knowledge Transfer in Few-shot Image Generation**
- Paper: 
- Code: https://github.com/yunqing-me/RICK

**Post-training Quantization on Diffusion Models**
- Paper: https://arxiv.org/abs/2211.15736
- Code: https://github.com/42Shawn/PTQ4DM

**LayoutDiffusion: Controllable Diffusion Model for Layout-to-image Generation**
- Paper: https://arxiv.org/abs/2303.17189
- Code: https://github.com/ZGCTroy/LayoutDiffusion

**DiffCollage: Parallel Generation of Large Content with Diffusion Models**
- Paper: https://arxiv.org/abs/2303.17076
- Project: https://research.nvidia.com/labs/dir/diffcollage/

**Few-shot Semantic Image Synthesis with Class Affinity Transfer**
- Paper: https://arxiv.org/abs/2304.02321

**NoisyTwins: Class-Consistent and Diverse Image Generation through StyleGANs**
- Paper: https://arxiv.org/abs/2304.05866
- Code: https://github.com/val-iisc/NoisyTwins

**DCFace: Synthetic Face Generation with Dual Condition Diffusion Model**
- Paper: https://arxiv.org/abs/2304.07060
- Code: https://github.com/mk-minchul/dcface

**Exploring Incompatible Knowledge Transfer in Few-shot Image Generation**
- Paper: https://arxiv.org/abs/2304.07574
- Code: https://github.com/yunqing-me/RICK

**Class-Balancing Diffusion Models**
- Paper: https://arxiv.org/abs/2305.00562

**Spider GAN: Leveraging Friendly Neighbors to Accelerate GAN Training**
- Paper: https://arxiv.org/abs/2305.07613

**Towards Accurate Image Coding: Improved Autoregressive Image Generation with Dynamic Vector Quantization**
- Paper: https://arxiv.org/abs/2305.11718
- Code: https://github.com/CrossmodalGroup/DynamicVectorQuantization

**Not All Image Regions Matter: Masked Vector Quantization for Autoregressive Image Generation**
- Paper: https://arxiv.org/abs/2305.13607
- Code: https://github.com/CrossmodalGroup/MaskedVectorQuantization

**Efficient Scale-Invariant Generator with Column-Row Entangled Pixel Synthesis**
- Paper: https://arxiv.org/abs/2303.14157
- Code: https://github.com/VinAIResearch/CREPS

<a name="VideoGeneration"></a>
## Video Generation

**Conditional Image-to-Video Generation with Latent Flow Diffusion Models**
- Paper: https://arxiv.org/abs/2303.13744
- Code: https://github.com/nihaomiao/CVPR2023_LFDM

**Video Probabilistic Diffusion Models in Projected Latent Space**
- Paper: https://arxiv.org/abs/2302.07685
- Code: https://github.com/sihyun-yu/PVDM

**DPE: Disentanglement of Pose and Expression for General Video Portrait Editing**
- Paper: https://arxiv.org/abs/2301.06281
- Code: https://github.com/Carlyx/DPE

**Decomposed Diffusion Models for High-Quality Video Generation**
- Paper: https://arxiv.org/abs/2303.08320

**Diffusion Video Autoencoders: Toward Temporally Consistent Face Video Editing via Disentangled Video Encoding**
- Paper: https://arxiv.org/abs/2212.02802
- Code: https://github.com/man805/Diffusion-Video-Autoencoders

**MoStGAN: Video Generation with Temporal Motion Styles**
- Paper: https://arxiv.org/abs/2304.02777
- Code: https://github.com/xiaoqian-shen/MoStGAN

**Align your Latents: High-Resolution Video Synthesis with Latent Diffusion Models**
- Paper: https://arxiv.org/abs/2304.08818


<a name="Others"></a>
## Others [[back](#catalogue)]

**Perspective Fields for Single Image Camera Calibration**
- Paper: https://arxiv.org/abs/2212.03239
- Code: https://github.com/jinlinyi/PerspectiveFields

**DC2: Dual-Camera Defocus Control by Learning to Refocus**
- Paper: https://arxiv.org/abs/2304.03285
- Project: https://defocus-control.github.io/

**Images Speak in Images: A Generalist Painter for In-Context Visual Learning**
- Paper: https://arxiv.org/abs/2212.02499
- Code: https://github.com/baaivision/Painter

**Make-A-Story: Visual Memory Conditioned Consistent Story Generation**
- Paper: https://arxiv.org/abs/2211.13319
- Code: https://github.com/ubc-vision/Make-A-Story

**Cross-GAN Auditing: Unsupervised Identification of Attribute Level Similarities and Differences between Pretrained Generative Models**
- Paper: https://arxiv.org/abs/2303.10774
- Code: https://github.com/mattolson93/cross_gan_auditing

**LightPainter: Interactive Portrait Relighting with Freehand Scribble**
- Paper: https://arxiv.org/abs/2303.12950
- Tags: Portrait Relighting

**Neural Texture Synthesis with Guided Correspondence**
- Paper:
- Code: https://github.com/EliotChenKJ/Guided-Correspondence-Loss
- Tags: Texture Synthesis

**Uncurated Image-Text Datasets: Shedding Light on Demographic Bias**
- Paper: https://arxiv.org/abs/2304.02828
- Code: https://github.com/noagarcia/phase

**Large-capacity and Flexible Video Steganography via Invertible Neural Network**
- Paper: https://arxiv.org/abs/2304.12300
- Code: https://github.com/MC-E/LF-VSN
- Tags: Steganography 

**Putting People in Their Place: Affordance-Aware Human Insertion into Scenes**
- Paper: https://arxiv.org/abs/2304.14406
- Code: https://github.com/adobe-research/affordance-insertion

**Controllable Light Diffusion for Portraits**
- Paper: https://arxiv.org/abs/2305.04745
- Tags: Relighting

## Talking Head Generation

**Seeing What You Said: Talking Face Generation Guided by a Lip Reading Expert**
- Paper: https://arxiv.org/abs/2303.17480
- Code: https://github.com/Sxjdwang/TalkLip

**High-Fidelity and Freely Controllable Talking Head Video Generation**
- Paper: https://arxiv.org/abs/2304.10168
- Code: https://github.com/hologerry/PECHead

**MetaPortrait: Identity-Preserving Talking Head Generation with Fast Personalized Adaptation**
- Paper: https://arxiv.org/abs/2212.08062
- Code: https://github.com/Meta-Portrait/MetaPortrait

## Virtual Try-on

**GP-VTON: Towards General Purpose Virtual Try-on via Collaborative Local-Flow Global-Parsing Learning**
- Paper: https://arxiv.org/abs/2303.13756
- Code: https://github.com/xiezhy6/GP-VTON

## Handwriting/Font Generation

**CF-Font: Content Fusion for Few-shot Font Generation**
- Paper: https://arxiv.org/abs/2303.14017
- Code: https://github.com/wangchi95/CF-Font
- Tags: Font Generation

**DeepVecFont-v2: Exploiting Transformers to Synthesize Vector Fonts with Higher Quality**
- Paper: https://arxiv.org/abs/2303.14585
- Code: https://github.com/yizhiwang96/deepvecfont-v2

**Handwritten Text Generation from Visual Archetypes**
- Paper: https://arxiv.org/abs/2303.15269
- Tags: Handwriting Generation

**Disentangling Writer and Character Styles for Handwriting Generation**
- Paper: https://arxiv.org/abs/2303.14736
- Code: https://github.com/dailenson/SDT
- Tags: Handwriting Generation

## Layout Generation

**Unifying Layout Generation with a Decoupled Diffusion Model**
- Paper: https://arxiv.org/abs/2303.05049

**Unsupervised Domain Adaption with Pixel-level Discriminator for Image-aware Layout Generation**
- Paper: https://arxiv.org/abs/2303.14377

**PosterLayout: A New Benchmark and Approach for Content-aware Visual-Textual Presentation Layout**
- Paper: https://arxiv.org/abs/2303.15937
- Code: https://github.com/PKU-ICST-MIPL/PosterLayout-CVPR2023

**LayoutDM: Discrete Diffusion Model for Controllable Layout Generation**
- Paper: https://arxiv.org/abs/2303.08137
- Code: https://github.com/CyberAgentAILab/layout-dm

**ayoutDM: Transformer-based Diffusion Model for Layout Generation**
- Paper: https://arxiv.org/abs/2305.02567
