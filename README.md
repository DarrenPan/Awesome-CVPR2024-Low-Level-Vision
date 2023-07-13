# Awesome-CVPR2023-Low-Level-Vision
A Collection of Papers and Codes in CVPR2023 related to Low-Level Vision

**[In Construction]** If you find some missing papers or typos, feel free to pull issues or requests.

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

- [Deblurring](#Deblurring)
  - [Image Deblurring](#ImageDeblurring)
  - [Video Deblurring](#VideoDeblurring)

- [Deraining](#Deraining)

- [Dehazing](#Dehazing)

- [HDR Imaging / Multi-Exposure Image Fusion](#HDR)

- [Frame Interpolation](#FrameInterpolation)

- [Image Enhancement](#Enhancement)
  - [Low-Light Image Enhancement](#LowLight)

- [Image Harmonization](#Harmonization)

- [Image Completion/Inpainting](#Inpainting)

- [Image Matting](#Matting)

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

**Comprehensive and Delicate: An Efficient Transformer for Image Restoration**
- Paper: https://openaccess.thecvf.com/content/CVPR2023/html/Zhao_Comprehensive_and_Delicate_An_Efficient_Transformer_for_Image_Restoration_CVPR_2023_paper.html
- Tags: Transformer

**Learning Distortion Invariant Representation for Image Restoration from A Causality Perspective**
- Paper: https://arxiv.org/abs/2303.06859
- Code: https://github.com/lixinustc/Casual-IRDIL

**Generative Diffusion Prior for Unified Image Restoration and Enhancement**
- Paper: https://arxiv.org/abs/2304.01247
- Code: https://github.com/Fayeben/GenerativeDiffusionPrior

**DR2: Diffusion-based Robust Degradation Remover for Blind Face Restoration**
- Paper: https://arxiv.org/abs/2303.06885
- Tags: Diffusion, Blind Face

**Bitstream-Corrupted JPEG Images are Restorable: Two-stage Compensation and Alignment Framework for Image Restoration**
- Paper: https://arxiv.org/abs/2304.06976
- Code: https://github.com/wenyang001/Two-ACIR

**All-in-One Image Restoration for Unknown Degradations Using Adaptive Discriminative Filters for Specific Degradations**
- Paper: https://openaccess.thecvf.com/content/CVPR2023/html/Park_All-in-One_Image_Restoration_for_Unknown_Degradations_Using_Adaptive_Discriminative_Filters_CVPR_2023_paper.html

**Learning Weather-General and Weather-Specific Features for Image Restoration Under Multiple Adverse Weather Conditions**
- Paper: https://openaccess.thecvf.com/content/CVPR2023/html/Zhu_Learning_Weather-General_and_Weather-Specific_Features_for_Image_Restoration_Under_Multiple_CVPR_2023_paper.html
- Code: https://github.com/zhuyr97/WGWS-Net
- Tags: Multiple Adverse Weather

**AccelIR: Task-Aware Image Compression for Accelerating Neural Restoration**
- Paper: https://openaccess.thecvf.com/content/CVPR2023/html/Ye_AccelIR_Task-Aware_Image_Compression_for_Accelerating_Neural_Restoration_CVPR_2023_paper.html
- Tags: Image Compression for Accelerating

**Robust Unsupervised StyleGAN Image Restoration**
- Paper: https://arxiv.org/abs/2302.06733
- Tags: StyleGAN

**Ingredient-Oriented Multi-Degradation Learning for Image Restoration**
- Paper: https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_Ingredient-Oriented_Multi-Degradation_Learning_for_Image_Restoration_CVPR_2023_paper.html

**Contrastive Semi-supervised Learning for Underwater Image Restoration via Reliable Bank**
- Paper: https://arxiv.org/abs/2303.09101
- Code: https://github.com/Huang-ShiRui/Semi-UIR
- Tags: Underwater Image Restoration

**Nighttime Smartphone Reflective Flare Removal Using Optical Center Symmetry Prior**
- Paper: https://arxiv.org/abs/2303.15046
- Code: https://github.com/ykdai/BracketFlare
- Tags: Reflective Flare Removal

**Robust Single Image Reflection Removal Against Adversarial Attacks**
- Paper: https://openaccess.thecvf.com/content/CVPR2023/html/Song_Robust_Single_Image_Reflection_Removal_Against_Adversarial_Attacks_CVPR_2023_paper.html
- Tags: Reflection Removal

**ShadowDiffusion: When Degradation Prior Meets Diffusion Model for Shadow Removal**
- Paper: https://arxiv.org/abs/2212.04711
- Code: https://github.com/GuoLanqing/ShadowDiffusion
- Tags: Diffusion, Shadow Removal

**Document Image Shadow Removal Guided by Color-Aware Background**
- Paper: https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_Document_Image_Shadow_Removal_Guided_by_Color-Aware_Background_CVPR_2023_paper.html
- Code: https://github.com/hyyh1314/BGShadowNet
- Tags: Shadow Removal

**Generating Aligned Pseudo-Supervision from Non-Aligned Data for Image Restoration in Under-Display Camera**
- Paper: https://arxiv.org/abs/2304.06019
- Code: https://github.com/jnjaby/AlignFormer

**GamutMLP: A Lightweight MLP for Color Loss Recovery**
- Paper: https://arxiv.org/abs/2304.11743
- Code: https://github.com/hminle/gamut-mlp
- Tags: restore wide-gamut color values

**ABCD: Arbitrary Bitwise Coefficient for De-Quantization**
- Paper: https://openaccess.thecvf.com/content/CVPR2023/papers/Han_ABCD_Arbitrary_Bitwise_Coefficient_for_De-Quantization_CVPR_2023_paper.pdf
- Code: https://github.com/WooKyoungHan/ABCD
- Tags: De-quantization/Bit depth expansion

**Visual Recognition-Driven Image Restoration for Multiple Degradation With Intrinsic Semantics Recovery**
- Paper: https://openaccess.thecvf.com/content/CVPR2023/html/Yang_Visual_Recognition-Driven_Image_Restoration_for_Multiple_Degradation_With_Intrinsic_Semantics_CVPR_2023_paper.html
- Tags: Restoration for High-Level Tasks

**Parallel Diffusion Models of Operator and Image for Blind Inverse Problems**
- Paper: https://arxiv.org/abs/2211.10656
- Code: https://github.com/BlindDPS/blind-dps
- Tags: blind deblurring, and imaging through turbulence

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

**Seeing Beyond the Brain: Conditional Diffusion Model with Sparse Masked Modeling for Vision Decoding**
- Paper: https://arxiv.org/abs/2211.06956
- Code: https://github.com/zjc062/mind-vis

<a name="BurstRestoration"></a>
## Burst Restoration

**Burstormer: Burst Image Restoration and Enhancement Transformer**
- Paper: https://arxiv.org/abs/2304.01194
- Code: https://github.com/akshaydudhane16/Burstormer

**Gated Multi-Resolution Transfer Network for Burst Restoration and Enhancement**
- Paper: https://arxiv.org/abs/2304.06703

<a name="VideoRestoration"></a>
## Video Restoration

**A Simple Baseline for Video Restoration with Grouped Spatial-temporal Shift**
- Paper: https://arxiv.org/abs/2206.10810
- Code: https://github.com/dasongli1/Shift-Net

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
- Tags: Arbitrary-Scale SR

**Local Implicit Normalizing Flow for Arbitrary-Scale Image Super-Resolution**
- Paper: https://arxiv.org/abs/2303.05156
- Tags: Normalizing Flow, Arbitrary-Scale SR

**Cascaded Local Implicit Transformer for Arbitrary-Scale Super-Resolution**
- Paper: https://arxiv.org/abs/2303.16513
- Code: https://github.com/jaroslaw1007/CLIT
- Tags: Arbitrary-Scale SR, Transformer

**Deep Arbitrary-Scale Image Super-Resolution via Scale-Equivariance Pursuit**
- Paper: https://openaccess.thecvf.com/content/CVPR2023/html/Wang_Deep_Arbitrary-Scale_Image_Super-Resolution_via_Scale-Equivariance_Pursuit_CVPR_2023_paper.html
- Code: https://github.com/neuralchen/EQSR
- Tags: Arbitrary-Scale SR

**CiaoSR: Continuous Implicit Attention-in-Attention Network for Arbitrary-Scale Image Super-Resolution**
- Paper: https://arxiv.org/abs/2212.04362
- Tags: Arbitrary-Scale SR

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
- Tags: Diffusion

**CABM: Content-Aware Bit Mapping for Single Image Super-Resolution Network with Large Input**
- Paper: https://arxiv.org/abs/2304.06454

**Spectral Bayesian Uncertainty for Image Super-Resolution**
- Paper: https://openaccess.thecvf.com/content/CVPR2023/html/Liu_Spectral_Bayesian_Uncertainty_for_Image_Super-Resolution_CVPR_2023_paper.html

**Cross-Guided Optimization of Radiance Fields With Multi-View Image Super-Resolution for High-Resolution Novel View Synthesis**
- Paper: https://openaccess.thecvf.com/content/CVPR2023/html/Yoon_Cross-Guided_Optimization_of_Radiance_Fields_With_Multi-View_Image_Super-Resolution_for_CVPR_2023_paper.html

**Image Super-Resolution Using T-Tetromino Pixels**
- Paper: https://openaccess.thecvf.com/content/CVPR2023/html/Grosche_Image_Super-Resolution_Using_T-Tetromino_Pixels_CVPR_2023_paper.html

**Memory-Friendly Scalable Super-Resolution via Rewinding Lottery Ticket Hypothesis**
- Paper: https://openaccess.thecvf.com/content/CVPR2023/html/Lin_Memory-Friendly_Scalable_Super-Resolution_via_Rewinding_Lottery_Ticket_Hypothesis_CVPR_2023_paper.html

**Equivalent Transformation and Dual Stream Network Construction for Mobile Image Super-Resolution**
- Paper: https://openaccess.thecvf.com/content/CVPR2023/html/Chao_Equivalent_Transformation_and_Dual_Stream_Network_Construction_for_Mobile_Image_CVPR_2023_paper.html
- Code: https://github.com/ECNUSR/ETDS 

**Perception-Oriented Single Image Super-Resolution using Optimal Objective Estimation**
- Paper: https://arxiv.org/abs/2211.13676
- Code: https://github.com/seungho-snu/SROOE

**OSRT: Omnidirectional Image Super-Resolution with Distortion-aware Transformer**
- Paper: https://arxiv.org/abs/2302.03453
- Code: https://github.com/Fanghua-Yu/OSRT
- Tags: Transformer, Omnidirectional SR

**B-Spline Texture Coefficients Estimator for Screen Content Image Super-Resolution**
- Paper: https://openaccess.thecvf.com/content/CVPR2023/html/Pak_B-Spline_Texture_Coefficients_Estimator_for_Screen_Content_Image_Super-Resolution_CVPR_2023_paper.html
- Code: https://github.com/ByeongHyunPak/btc

**Spatial-Frequency Mutual Learning for Face Super-Resolution**
- Paper: https://openaccess.thecvf.com/content/CVPR2023/html/Wang_Spatial-Frequency_Mutual_Learning_for_Face_Super-Resolution_CVPR_2023_paper.html
- Code: https://github.com/wcy-cs/SFMNet
- Tags: Face SR

**Learning Generative Structure Prior for Blind Text Image Super-resolution**
- Paper: https://arxiv.org/abs/2303.14726
- Code: https://github.com/csxmli2016/MARCONet
- Tags: Text SR

**Guided Depth Super-Resolution by Deep Anisotropic Diffusion**
- Paper: https://arxiv.org/abs/2211.11592
- Code: https://github.com/prs-eth/Diffusion-Super-Resolution
- Tags: Guided Depth SR

**Toward Stable, Interpretable, and Lightweight Hyperspectral Super-Resolution**
- Paper: https://openaccess.thecvf.com/content/CVPR2023/html/Xie_Toward_Stable_Interpretable_and_Lightweight_Hyperspectral_Super-Resolution_CVPR_2023_paper.html
- Code: https://github.com/WenjinGuo/DAEM
- Tags: Hyperspectral SR

**Zero-Shot Dual-Lens Super-Resolution**
- Paper: https://openaccess.thecvf.com/content/CVPR2023/html/Xu_Zero-Shot_Dual-Lens_Super-Resolution_CVPR_2023_paper.html
- Code: https://github.com/XrKang/ZeDuSR

**Probability-based Global Cross-modal Upsampling for Pansharpening**
- Paper: https://arxiv.org/abs/2303.13659
- Code: https://github.com/Zeyu-Zhu/PGCU
- Tags: Pansharpening(for remote sensing image)

**CutMIB: Boosting Light Field Super-Resolution via Multi-View Image Blending**
- Paper: https://openaccess.thecvf.com/content/CVPR2023/html/Xiao_CutMIB_Boosting_Light_Field_Super-Resolution_via_Multi-View_Image_Blending_CVPR_2023_paper.html
- Code: https://github.com/zeyuxiao1997/CutMIB
- Tags: Light Field SR

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

**Compression-Aware Video Super-Resolution**
- Paper: https://openaccess.thecvf.com/content/CVPR2023/html/Wang_Compression-Aware_Video_Super-Resolution_CVPR_2023_paper.html

**Learning Spatial-Temporal Implicit Neural Representations for Event-Guided Video Super-Resolution**
- Paper: https://arxiv.org/abs/2303.13767
- Project: https://vlis2022.github.io/cvpr23/egvsr 
- Tags: Event

**Consistent Direct Time-of-Flight Video Depth Super-Resolution**
- Paper: https://arxiv.org/abs/2211.08658
- Code: https://github.com/facebookresearch/DVSR/
- Tags: Depth SR


<a name="Rescaling"></a>
# Image Rescaling - 图像缩放 [[back](#catalogue)]

**HyperThumbnail: Real-time 6K Image Rescaling with Rate-distortion Optimization**
- Paper: https://arxiv.org/abs/2304.01064
- Code: https://github.com/AbnerVictor/HyperThumbnail

**DINN360: Deformable Invertible Neural Network for Latitude-Aware 360deg Image Rescaling**
- Paper: https://openaccess.thecvf.com/content/CVPR2023/html/Guo_DINN360_Deformable_Invertible_Neural_Network_for_Latitude-Aware_360deg_Image_Rescaling_CVPR_2023_paper.html
- Code: https://github.com/gyc9709/DINN360

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

**Zero-Shot Noise2Noise: Efficient Image Denoising without any Data**
- Paper: https://arxiv.org/abs/2303.11253
- Code: https://colab.research.google.com/drive/1i82nyizTdszyHkaHBuKPbWnTzao8HF9b
- Tags: Zero-Shot

**Patch-Craft Self-Supervised Training for Correlated Image Denoising**
- Paper: https://arxiv.org/abs/2211.09919
- Tags: Self-Supervised

**sRGB Real Noise Synthesizing with Neighboring Correlation-Aware Noise Model**
- Paper: https://openaccess.thecvf.com/content/CVPR2023/papers/Fu_sRGB_Real_Noise_Synthesizing_With_Neighboring_Correlation-Aware_Noise_Model_CVPR_2023_paper.pdf
- Code: https://github.com/xuan611/sRGB-Real-Noise-Synthesizing
- Tags: Real Noise Synthesizing

**Spectral Enhanced Rectangle Transformer for Hyperspectral Image Denoising**
- Paper: https://arxiv.org/abs/2304.00844
- Code: https://github.com/MyuLi/SERT
- Tags: Hyperspectral

**Efficient View Synthesis and 3D-based Multi-Frame Denoising with Multiplane Feature Representations**
- Paper: https://arxiv.org/abs/2303.18139
- Tags: 3D

**Structure Aggregation for Cross-Spectral Stereo Image Guided Denoising**
- Paper: https://openaccess.thecvf.com/content/CVPR2023/html/Sheng_Structure_Aggregation_for_Cross-Spectral_Stereo_Image_Guided_Denoising_CVPR_2023_paper.html
- Code: https://github.com/lustrouselixir/SANet
- Tags: Stereo Image

**Polarized Color Image Denoising**
- Paper: https://openaccess.thecvf.com/content/CVPR2023/html/Li_Polarized_Color_Image_Denoising_CVPR_2023_paper.html
- Code: https://github.com/bandasyou/pcdenoise
- Tags: Polarized Color Image

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
- Paper: https://openaccess.thecvf.com/content/CVPR2023/html/Quan_Neumann_Network_With_Recursive_Kernels_for_Single_Image_Defocus_Deblurring_CVPR_2023_paper.html
- Code: https://github.com/csZcWu/NRKNet

**Efficient Frequency Domain-based Transformers for High-Quality Image Deblurring**
- Paper: https://arxiv.org/abs/2211.12250
- Code: https://github.com/kkkls/FFTformer

**Hybrid Neural Rendering for Large-Scale Scenes with Motion Blur**
- Paper: https://arxiv.org/abs/2304.12652
- Code: https://github.com/CVMI-Lab/HybridNeuralRendering

**Self-Supervised Non-Uniform Kernel Estimation With Flow-Based Motion Prior for Blind Image Deblurring**
- Paper: https://openaccess.thecvf.com/content/CVPR2023/html/Fang_Self-Supervised_Non-Uniform_Kernel_Estimation_With_Flow-Based_Motion_Prior_for_Blind_CVPR_2023_paper.html
- Code: https://github.com/Fangzhenxuan/UFPDeblur
- Tag: Self-Supervised

**Uncertainty-Aware Unsupervised Image Deblurring with Deep Residual Prior**
- Paper: https://arxiv.org/abs/2210.05361
- Code: https://github.com/xl-tang01/UAUDeblur
- Tags: Unsupervised

**K3DN: Disparity-Aware Kernel Estimation for Dual-Pixel Defocus Deblurring**
- Paper: https://openaccess.thecvf.com/content/CVPR2023/html/Yang_K3DN_Disparity-Aware_Kernel_Estimation_for_Dual-Pixel_Defocus_Deblurring_CVPR_2023_paper.html

**Self-Supervised Blind Motion Deblurring With Deep Expectation Maximization**
- Paper: https://openaccess.thecvf.com/content/CVPR2023/html/Li_Self-Supervised_Blind_Motion_Deblurring_With_Deep_Expectation_Maximization_CVPR_2023_paper.html
- Tags: Self-Supervised

**HyperCUT: Video Sequence from a Single Blurry Image using Unsupervised Ordering**
- Paper: https://arxiv.org/abs/2304.01686
- Code: https://github.com/VinAIResearch/HyperCUT

<a name="VideoDeblurring"></a>
## Video Deblurring

**Deep Discriminative Spatial and Temporal Network for Efficient Video Deblurring**
- Paper: https://openaccess.thecvf.com/content/CVPR2023/html/Pan_Deep_Discriminative_Spatial_and_Temporal_Network_for_Efficient_Video_Deblurring_CVPR_2023_paper.html
- Code: https://github.com/xuboming8/DSTNet


<a name="Deraining"></a>
# Deraining - 去雨 [[back](#catalogue)]

**Learning A Sparse Transformer Network for Effective Image Deraining**
- Paper: https://arxiv.org/abs/2303.11950
- Code: https://github.com/cschenxiang/DRSformer

**SmartAssign: Learning a Smart Knowledge Assignment Strategy for Deraining and Desnowing**
- Paper: https://openaccess.thecvf.com/content/CVPR2023/html/Wang_SmartAssign_Learning_a_Smart_Knowledge_Assignment_Strategy_for_Deraining_and_CVPR_2023_paper.html
- Code: https://gitee.com/mindspore/models/tree/master/research/cv/SmartAssign

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

**Joint HDR Denoising and Fusion: A Real-World Mobile HDR Image Dataset**
- Paper: https://openaccess.thecvf.com/content/CVPR2023/html/Liu_Joint_HDR_Denoising_and_Fusion_A_Real-World_Mobile_HDR_Image_CVPR_2023_paper.html
- Code: https://github.com/shuaizhengliu/Joint-HDRDN

**HDR Imaging with Spatially Varying Signal-to-Noise Ratios**
- Paper: https://arxiv.org/abs/2303.17253

**1000 FPS HDR Video with a Spike-RGB Hybrid Camera**
- Paper: https://openaccess.thecvf.com/content/CVPR2023/html/Chang_1000_FPS_HDR_Video_With_a_Spike-RGB_Hybrid_Camera_CVPR_2023_paper.html

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

**Frame Interpolation Transformer and Uncertainty Guidance**
- Paper: https://openaccess.thecvf.com/content/CVPR2023/html/Plack_Frame_Interpolation_Transformer_and_Uncertainty_Guidance_CVPR_2023_paper.html

**Exploring Motion Ambiguity and Alignment for High-Quality Video Frame Interpolation**
- Paper: https://arxiv.org/abs/2203.10291

**Range-Nullspace Video Frame Interpolation With Focalized Motion Estimation**
- Paper: https://openaccess.thecvf.com/content/CVPR2023/html/Yu_Range-Nullspace_Video_Frame_Interpolation_With_Focalized_Motion_Estimation_CVPR_2023_paper.html 

**Event-based Video Frame Interpolation with Cross-Modal Asymmetric Bidirectional Motion Fields**
- Paper: https://openaccess.thecvf.com/content/CVPR2023/html/Kim_Event-Based_Video_Frame_Interpolation_With_Cross-Modal_Asymmetric_Bidirectional_Motion_Fields_CVPR_2023_paper.html
- Code: https://github.com/intelpro/CBMNet
- Tags: Event-based

**Event-based Blurry Frame Interpolation under Blind Exposure**
- Paper: https://openaccess.thecvf.com/content/CVPR2023/html/Weng_Event-Based_Blurry_Frame_Interpolation_Under_Blind_Exposure_CVPR_2023_paper.html
- Code: https://github.com/WarranWeng/EBFI-BE
- Tags: Event-based

**Event-Based Frame Interpolation with Ad-hoc Deblurring**
- Paper: https://arxiv.org/abs/2301.05191
- Tags: Event-based

**Joint Video Multi-Frame Interpolation and Deblurring under Unknown Exposure Time**
- Paper: https://arxiv.org/abs/2303.15043
- Code: https://github.com/shangwei5/VIDUE
- Tags: Frame Interpolation and Deblurring

<a name="Enhancement"></a>
# Image Enhancement - 图像增强 [[back](#catalogue)]

**Realistic Saliency Guided Image Enhancement**
- Paper: https://openaccess.thecvf.com/content/CVPR2023/html/Miangoleh_Realistic_Saliency_Guided_Image_Enhancement_CVPR_2023_paper.html
- Code: https://github.com/compphoto/RealisticImageEnhancement

<a name="LowLight"></a>
## Low-Light Image Enhancement

**Learning Semantic-Aware Knowledge Guidance for Low-Light Image Enhancement**
- Paper: https://arxiv.org/abs/2304.07039
- Code: https://github.com/langmanbusi/Semantic-Aware-Low-Light-Image-Enhancement

**Visibility Constrained Wide-band Illumination Spectrum Design for Seeing-in-the-Dark**
- Paper: https://arxiv.org/abs/2303.11642
- Code: https://github.com/MyNiuuu/VCSD
- Tags: NIR2RGB

**DNF: Decouple and Feedback Network for Seeing in the Dark**
- Paper: https://openaccess.thecvf.com/content/CVPR2023/html/Jin_DNF_Decouple_and_Feedback_Network_for_Seeing_in_the_Dark_CVPR_2023_paper.html
- Code: https://github.com/Srameo/DNF

**You Do Not Need Additional Priors or Regularizers in Retinex-Based Low-Light Image Enhancement**
- Paper: https://openaccess.thecvf.com/content/CVPR2023/html/Fu_You_Do_Not_Need_Additional_Priors_or_Regularizers_in_Retinex-Based_CVPR_2023_paper.html

**Low-Light Image Enhancement via Structure Modeling and Guidance**
- Paper: https://arxiv.org/abs/2305.05839

**Learning a Simple Low-light Image Enhancer from Paired Low-light Instances**
- Paper: https://openaccess.thecvf.com/content/CVPR2023/html/Fu_Learning_a_Simple_Low-Light_Image_Enhancer_From_Paired_Low-Light_Instances_CVPR_2023_paper.html
- Code: https://github.com/zhenqifu/pairlie


<a name="Harmonization"></a>
# Image Harmonization/Composition - 图像协调/图像合成 [[back](#catalogue)]

**LEMaRT: Label-Efficient Masked Region Transform for Image Harmonization**
- Paper: https://arxiv.org/abs/2304.13166

**Semi-supervised Parametric Real-world Image Harmonization**
- Paper: https://arxiv.org/abs/2303.00157
- Project: https://kewang0622.github.io/sprih/

**PCT-Net: Full Resolution Image Harmonization Using Pixel-Wise Color Transformations**
- Paper: https://openaccess.thecvf.com/content/CVPR2023/html/Guerreiro_PCT-Net_Full_Resolution_Image_Harmonization_Using_Pixel-Wise_Color_Transformations_CVPR_2023_paper.html
- Code: https://github.com/rakutentech/PCT-Net-Image-Harmonization/

**ObjectStitch: Object Compositing With Diffusion Model**
- Paper: https://openaccess.thecvf.com/content/CVPR2023/html/Song_ObjectStitch_Object_Compositing_With_Diffusion_Model_CVPR_2023_paper.html


<a name="Inpainting"></a>
# Image Completion/Inpainting - 图像修复 [[back](#catalogue)]

**NUWA-LIP: Language-Guided Image Inpainting With Defect-Free VQGAN**
- Paper: https://openaccess.thecvf.com/content/CVPR2023/html/Ni_NUWA-LIP_Language-Guided_Image_Inpainting_With_Defect-Free_VQGAN_CVPR_2023_paper.html
- Code: https://github.com/kodenii/NUWA-LIP

**Imagen Editor and EditBench: Advancing and Evaluating Text-Guided Image Inpainting**
- Paper: https://arxiv.org/abs/2212.06909

**SmartBrush: Text and Shape Guided Object Inpainting with Diffusion Model**
- Paper: https://arxiv.org/abs/2212.05034

**Semi-Supervised Video Inpainting with Cycle Consistency Constraints**
- Paper: https://arxiv.org/abs/2208.06807

**Deep Stereo Video Inpainting**
- Paper: https://openaccess.thecvf.com/content/CVPR2023/html/Wu_Deep_Stereo_Video_Inpainting_CVPR_2023_paper.html

<a name="Matting"></a>
# Image Matting - 图像抠图 [[back](#catalogue)]

**Referring Image Matting**
- Paper: https://arxiv.org/abs/2206.05149
- Code: https://github.com/JizhiziLi/RIM

**Adaptive Human Matting for Dynamic Videos**
- Paper: https://arxiv.org/abs/2304.06018
- Code: https://github.com/microsoft/AdaM

**Mask-Guided Matting in the Wild**
- Paper: https://openaccess.thecvf.com/content/CVPR2023/html/Park_Mask-Guided_Matting_in_the_Wild_CVPR_2023_paper.html

**End-to-End Video Matting With Trimap Propagation**
- Paper: https://openaccess.thecvf.com/content/CVPR2023/html/Huang_End-to-End_Video_Matting_With_Trimap_Propagation_CVPR_2023_paper.html
- Code: https://github.com/csvt32745/FTP-VM

**Ultrahigh Resolution Image/Video Matting With Spatio-Temporal Sparsity**
- Paper: https://openaccess.thecvf.com/content/CVPR2023/html/Sun_Ultrahigh_Resolution_ImageVideo_Matting_With_Spatio-Temporal_Sparsity_CVPR_2023_paper.html


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

**NVTC: Nonlinear Vector Transform Coding**
- Paper: https://arxiv.org/abs/2305.16025
- Code: https://github.com/USTC-IMCL/NVTC

**Multi-Realism Image Compression with a Conditional Generator**
- Paper: https://arxiv.org/abs/2212.13824

**LVQAC: Lattice Vector Quantization Coupled with Spatially Adaptive Companding for Efficient Learned Image Compression**
- Paper: https://arxiv.org/abs/2304.12319

## Video Compression

**Neural Video Compression with Diverse Contexts**
- Paper: https://github.com/microsoft/DCVC
- Code: https://arxiv.org/abs/2302.14402

**Video Compression With Entropy-Constrained Neural Representations**
- Paper: https://openaccess.thecvf.com/content/CVPR2023/html/Gomes_Video_Compression_With_Entropy-Constrained_Neural_Representations_CVPR_2023_paper.html

**Complexity-Guided Slimmable Decoder for Efficient Deep Video Compression**
- Paper: https://openaccess.thecvf.com/content/CVPR2023/html/Hu_Complexity-Guided_Slimmable_Decoder_for_Efficient_Deep_Video_Compression_CVPR_2023_paper.html

**MMVC: Learned Multi-Mode Video Compression with Block-based Prediction Mode Selection and Density-Adaptive Entropy Coding**
- Paper: https://arxiv.org/abs/2304.02273

**Motion Information Propagation for Neural Video Compression**
- Paper: https://openaccess.thecvf.com/content/CVPR2023/html/Qi_Motion_Information_Propagation_for_Neural_Video_Compression_CVPR_2023_paper.html

**Hierarchical B-Frame Video Coding Using Two-Layer CANF Without Motion Coding**
- Paper:https://openaccess.thecvf.com/content/CVPR2023/html/Alexandre_Hierarchical_B-Frame_Video_Coding_Using_Two-Layer_CANF_Without_Motion_Coding_CVPR_2023_paper.html
- Code: https://github.com/nycu-clab/tlzmc-cvpr


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

**An Image Quality Assessment Dataset for Portraits**
- Paper: https://arxiv.org/abs/2304.05772
- Code: https://github.com/DXOMARK-Research/PIQ2023

**MD-VQA: Multi-Dimensional Quality Assessment for UGC Live Videos**
- Paper: https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_MD-VQA_Multi-Dimensional_Quality_Assessment_for_UGC_Live_Videos_CVPR_2023_paper.html
- Code: https://github.com/zzc-1998/MD-VQA

**CR-FIQA: Face Image Quality Assessment by Learning Sample Relative Classifiability**
- Paper: https://openaccess.thecvf.com/content/CVPR2023/html/Boutros_CR-FIQA_Face_Image_Quality_Assessment_by_Learning_Sample_Relative_Classifiability_CVPR_2023_paper.html
- Code: https://github.com/fdbtrs/CR-FIQA

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

**Learning Dynamic Style Kernels for Artistic Style Transfer**
- Paper: https://arxiv.org/abs/2304.00414

**Inversion-Based Style Transfer with Diffusion Models**
- Paper: https://arxiv.org/abs/2211.13203
- Code: https://github.com/zyxElsa/InST


<a name="ImageEditing"></a>
# Image Editing - 图像编辑 [[back](#catalogue)]

**Imagic: Text-Based Real Image Editing with Diffusion Models**
- Paper: https://arxiv.org/abs/2210.09276

**SINE: SINgle Image Editing with Text-to-Image Diffusion Models**
- Paper: https://arxiv.org/abs/2212.04489
- Code: https://github.com/zhang-zx/SINE

**CoralStyleCLIP: Co-optimized Region and Layer Selection for Image Editing**
- Paper: https://arxiv.org/abs/2303.05031
- Code: https://github.com/JiauZhang/CoralStyleCLIP

**SIEDOB: Semantic Image Editing by Disentangling Object and Background**
- Paper: https://arxiv.org/abs/2303.13062
- Code: https://github.com/WuyangLuo/SIEDOB

**DiffusionRig: Learning Personalized Priors for Facial Appearance Editing**
- Paper: https://arxiv.org/abs/2304.06711
- Code: https://github.com/adobe-research/diffusion-rig

**Paint by Example: Exemplar-based Image Editing with Diffusion Models**
- Paper: https://arxiv.org/abs/2211.13227
- Code: https://github.com/Fantasy-Studio/Paint-by-Example

**StyleRes: Transforming the Residuals for Real Image Editing With StyleGAN**
- Paper: https://arxiv.org/abs/2212.14359
- Code: https://github.com/hamzapehlivan/StyleRes

**Delving StyleGAN Inversion for Image Editing: A Foundation Latent Space Viewpoint**
- Paper: https://arxiv.org/abs/2211.11448
- Code: https://github.com/KumapowerLIU/CLCAE

**InstructPix2Pix: Learning to Follow Image Editing Instructions**
- Paper: https://arxiv.org/abs/2211.09800
- Code: https://github.com/timothybrooks/instruct-pix2pix

**Deep Curvilinear Editing: Commutative and Nonlinear Image Manipulation for Pretrained Deep Generative Model**
- Paper: https://arxiv.org/abs/2211.14573

**Null-text Inversion for Editing Real Images using Guided Diffusion Models**
- Paper: https://arxiv.org/abs/2211.09794
- Code: https://github.com/google/prompt-to-prompt/#null-text-inversion-for-editing-real-images

**DeltaEdit: Exploring Text-free Training for Text-Driven Image Manipulation**
- Paper: https://arxiv.org/abs/2303.06285
- Code: https://github.com/Yueming6568/DeltaEdit

**Text-Guided Unsupervised Latent Transformation for Multi-Attribute Image Manipulation**
- Paper: https://openaccess.thecvf.com/content/CVPR2023/html/Wei_Text-Guided_Unsupervised_Latent_Transformation_for_Multi-Attribute_Image_Manipulation_CVPR_2023_paper.html

**EDICT: Exact Diffusion Inversion via Coupled Transformations**
- Paper: https://arxiv.org/abs/2211.12446
- Code: https://github.com/salesforce/EDICT

## Video Editing

**DPE: Disentanglement of Pose and Expression for General Video Portrait Editing**
- Paper: https://arxiv.org/abs/2301.06281
- Code: https://github.com/Carlyx/DPE

**Diffusion Video Autoencoders: Toward Temporally Consistent Face Video Editing via Disentangled Video Encoding**
- Paper: https://arxiv.org/abs/2212.02802
- Code: https://github.com/man805/Diffusion-Video-Autoencoders

**Shape-aware Text-driven Layered Video Editing**
- Paper: https://arxiv.org/abs/2301.13173
- Project: https://text-video-edit.github.io/#


<a name=ImageGeneration></a>
# Image Generation/Synthesis / Image-to-Image Translation - 图像生成/合成/转换 [[back](#catalogue)]
## Text-to-Image / Text Guided / Multi-Modal

**GALIP: Generative Adversarial CLIPs for Text-to-Image Synthesis**
- Paper: https://arxiv.org/abs/2301.12959
- Code: https://github.com/tobran/GALIP

**Scaling up GANs for Text-to-Image Synthesis**
- Paper: https://arxiv.org/abs/2303.05511
- Project: https://mingukkang.github.io/GigaGAN/

**Variational Distribution Learning for Unsupervised Text-to-Image Generation**
- Paper: https://arxiv.org/abs/2303.16105

**Toward Verifiable and Reproducible Human Evaluation for Text-to-Image Generation**
- Paper: https://arxiv.org/abs/2304.01816

**Shifted Diffusion for Text-to-image Generation**
- Paper: https://arxiv.org/abs/2211.15388
- Code: https://github.com/drboog/Shifted_Diffusion

**ReCo: Region-Controlled Text-to-Image Generation**
- Paper: https://arxiv.org/abs/2211.15518
- Code: https://github.com/microsoft/ReCo

**RIATIG: Reliable and Imperceptible Adversarial Text-to-Image Generation With Natural Prompts**
- Paper: https://openaccess.thecvf.com/content/CVPR2023/html/Liu_RIATIG_Reliable_and_Imperceptible_Adversarial_Text-to-Image_Generation_With_Natural_Prompts_CVPR_2023_paper.html
- Code: https://github.com/WUSTL-CSPL/RIATIG

**GLIGEN: Open-Set Grounded Text-to-Image Generation**
- Paper: https://arxiv.org/abs/2301.07093
- Code: https://github.com/gligen/GLIGEN

**Multi-Concept Customization of Text-to-Image Diffusion**
- Paper: https://arxiv.org/abs/2212.04488
- Code: https://github.com/adobe-research/custom-diffusion

**ERNIE-ViLG 2.0: Improving Text-to-Image Diffusion Model With Knowledge-Enhanced Mixture-of-Denoising-Experts**
- Paper: https://openaccess.thecvf.com/content/CVPR2023/html/Feng_ERNIE-ViLG_2.0_Improving_Text-to-Image_Diffusion_Model_With_Knowledge-Enhanced_Mixture-of-Denoising-Experts_CVPR_2023_paper.html

**Uncovering the Disentanglement Capability in Text-to-Image Diffusion Models**
- Paper: https://arxiv.org/abs/2212.08698
- Code: https://github.com/UCSB-NLP-Chang/DiffusionDisentanglement

**DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation**
- Paper: https://arxiv.org/abs/2208.12242
- Code: https://github.com/google/dreambooth

**Specialist Diffusion: Plug-and-Play Sample-Efficient Fine-Tuning of Text-to-Image Diffusion Models To Learn Any Unseen Style**
- Paper: https://openaccess.thecvf.com/content/CVPR2023/html/Lu_Specialist_Diffusion_Plug-and-Play_Sample-Efficient_Fine-Tuning_of_Text-to-Image_Diffusion_Models_To_CVPR_2023_paper.html
- Code: https://github.com/Picsart-AI-Research/Specialist-Diffusion

**MAGVLT: Masked Generative Vision-and-Language Transformer**
- Paper: https://arxiv.org/abs/2303.12208

**Freestyle Layout-to-Image Synthesis**
- Paper: https://arxiv.org/abs/2303.14412
- Code: https://github.com/essunny310/FreestyleNet

**Sound to Visual Scene Generation by Audio-to-Visual Latent Alignment**
- Paper: https://arxiv.org/abs/2303.17490
- Project: https://sound2scene.github.io/

**Collaborative Diffusion for Multi-Modal Face Generation and Editing**
- Paper: https://arxiv.org/abs/2304.10530
- Code: https://github.com/ziqihuangg/Collaborative-Diffusion

**SpaText: Spatio-Textual Representation for Controllable Image Generation**
- Paper: https://arxiv.org/abs/2211.14305

**Plug-and-Play Diffusion Features for Text-Driven Image-to-Image Translation**
- Paper: https://arxiv.org/abs/2211.12572
- Code: https://github.com/MichalGeyer/plug-and-play

**LANIT: Language-Driven Image-to-Image Translation for Unlabeled Data**
- Paper: https://arxiv.org/abs/2208.14889
- Code: https://github.com/KU-CVLAB/LANIT

**High-Fidelity Guided Image Synthesis with Latent Diffusion Models**
- Paper: https://arxiv.org/abs/2211.17084
- Code: https://github.com/1jsingh/GradOP-Guided-Image-Synthesis

**Safe Latent Diffusion: Mitigating Inappropriate Degeneration in Diffusion Models**
- Paper: https://arxiv.org/abs/2211.05105
- Code: https://github.com/ml-research/safe-latent-diffusion

## Image-to-Image / Image Guided

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

**StyleGene: Crossover and Mutation of Region-Level Facial Genes for Kinship Face Synthesis**
- Paper: https://openaccess.thecvf.com/content/CVPR2023/html/Li_StyleGene_Crossover_and_Mutation_of_Region-Level_Facial_Genes_for_Kinship_CVPR_2023_paper.html
- Code: https://github.com/CVI-SZU/StyleGene

**Unpaired Image-to-Image Translation With Shortest Path Regularization**
- Paper: https://openaccess.thecvf.com/content/CVPR2023/html/Xie_Unpaired_Image-to-Image_Translation_With_Shortest_Path_Regularization_CVPR_2023_paper.html
- Code: https://github.com/Mid-Push/santa

**BBDM: Image-to-image Translation with Brownian Bridge Diffusion Models**
- Paper: https://arxiv.org/abs/2205.07680
- Code: https://github.com/xuekt98/BBDM

**MaskSketch: Unpaired Structure-guided Masked Image Generation**
- Paper: https://arxiv.org/abs/2302.05496
- Code: https://github.com/google-research/masksketch

## Others for image generation

**AdaptiveMix: Improving GAN Training via Feature Space Shrinkage**
- Paper: https://openaccess.thecvf.com/content/CVPR2023/html/Liu_AdaptiveMix_Improving_GAN_Training_via_Feature_Space_Shrinkage_CVPR_2023_paper.html
- Code: https://github.com/WentianZhang-ML/AdaptiveMix

**MAGE: MAsked Generative Encoder to Unify Representation Learning and Image Synthesis**
- Paper: https://arxiv.org/abs/2211.09117
- Code: https://github.com/LTH14/mage

**Regularized Vector Quantization for Tokenized Image Synthesis**
- Paper: https://arxiv.org/abs/2303.06424  

**Exploring Incompatible Knowledge Transfer in Few-shot Image Generation**
- Paper: https://arxiv.org/abs/2304.07574
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

**Inferring and Leveraging Parts from Object Shape for Improving Semantic Image Synthesis**
- Paper: https://arxiv.org/abs/2305.19547
- Code: https://github.com/csyxwei/iPOSE

**GLeaD: Improving GANs with A Generator-Leading Task**
- Paper: https://openaccess.thecvf.com/content/CVPR2023/html/Bai_GLeaD_Improving_GANs_With_a_Generator-Leading_Task_CVPR_2023_paper.html
- Code: https://github.com/EzioBy/glead

**Where Is My Spot? Few-Shot Image Generation via Latent Subspace Optimization**
- Paper: https://openaccess.thecvf.com/content/CVPR2023/html/Zheng_Where_Is_My_Spot_Few-Shot_Image_Generation_via_Latent_Subspace_CVPR_2023_paper.html
- Code: https://github.com/chansey0529/LSO

**KD-DLGAN: Data Limited Image Generation via Knowledge Distillation**
- Paper: https://openaccess.thecvf.com/content/CVPR2023/html/Cui_KD-DLGAN_Data_Limited_Image_Generation_via_Knowledge_Distillation_CVPR_2023_paper.html

**Private Image Generation With Dual-Purpose Auxiliary Classifier**
- Paper: https://openaccess.thecvf.com/content/CVPR2023/html/Chen_Private_Image_Generation_With_Dual-Purpose_Auxiliary_Classifier_CVPR_2023_paper.html

**SceneComposer: Any-Level Semantic Image Synthesis**
- Paper: https://arxiv.org/abs/2211.11742
- Code: https://github.com/zengxianyu/scenec

**Exploring Intra-Class Variation Factors With Learnable Cluster Prompts for Semi-Supervised Image Synthesis**
- Paper: https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_Exploring_Intra-Class_Variation_Factors_With_Learnable_Cluster_Prompts_for_Semi-Supervised_CVPR_2023_paper.html

**Re-GAN: Data-Efficient GANs Training via Architectural Reconfiguration**
- Paper: https://openaccess.thecvf.com/content/CVPR2023/html/Saxena_Re-GAN_Data-Efficient_GANs_Training_via_Architectural_Reconfiguration_CVPR_2023_paper.html
- Code: https://github.com/IntellicentAI-Lab/Re-GAN

**Discriminator-Cooperated Feature Map Distillation for GAN Compression**
- Paper: https://arxiv.org/abs/2212.14169
- Code: https://github.com/poopit/DCD-official

**Wavelet Diffusion Models are fast and scalable Image Generators**
- Paper: https://arxiv.org/abs/2211.16152
- Code: https://github.com/VinAIResearch/WaveDiff

**On Distillation of Guided Diffusion Models**
- Paper: https://arxiv.org/abs/2210.03142

**Binary Latent Diffusion**
- Paper: https://arxiv.org/abs/2304.04820
- Code: https://github.com/JiauZhang/binary-latent-diffusion

**All are Worth Words: A ViT Backbone for Diffusion Models**
- Paper: https://arxiv.org/abs/2209.12152
- Code: https://github.com/baofff/U-ViT

**Towards Practical Plug-and-Play Diffusion Models**
- Paper: https://arxiv.org/abs/2212.05973
- Code: https://github.com/riiid/PPAP

**Lookahead Diffusion Probabilistic Models for Refining Mean Estimation**
- Paper: https://arxiv.org/abs/2304.11312
- Code: https://github.com/guoqiang-zhang-x/LA-DPM

**Diffusion Probabilistic Model Made Slim**
- Paper: https://arxiv.org/abs/2211.17106

**Self-Guided Diffusion Models**
- Paper: https://arxiv.org/abs/2210.06462

<a name="VideoGeneration"></a>
## Video Generation

**Conditional Image-to-Video Generation with Latent Flow Diffusion Models**
- Paper: https://arxiv.org/abs/2303.13744
- Code: https://github.com/nihaomiao/CVPR2023_LFDM

**Video Probabilistic Diffusion Models in Projected Latent Space**
- Paper: https://arxiv.org/abs/2302.07685
- Code: https://github.com/sihyun-yu/PVDM

**Decomposed Diffusion Models for High-Quality Video Generation**
- Paper: https://arxiv.org/abs/2303.08320

**MoStGAN: Video Generation with Temporal Motion Styles**
- Paper: https://arxiv.org/abs/2304.02777
- Code: https://github.com/xiaoqian-shen/MoStGAN

**Align your Latents: High-Resolution Video Synthesis with Latent Diffusion Models**
- Paper: https://arxiv.org/abs/2304.08818

**Tell Me What Happened: Unifying Text-guided Video Completion via Multimodal Masked Video Generation**
- Paper: https://arxiv.org/abs/2211.12824
- Code: https://github.com/tsujuifu/pytorch_tvc

**MM-Diffusion: Learning Multi-Modal Diffusion Models for Joint Audio and Video Generation**
- Paper: https://openaccess.thecvf.com/content/CVPR2023/html/Ruan_MM-Diffusion_Learning_Multi-Modal_Diffusion_Models_for_Joint_Audio_and_Video_CVPR_2023_paper.html
- Code: https://github.com/researchmm/MM-Diffusion

**Dimensionality-Varying Diffusion Process**
- Paper: https://arxiv.org/abs/2211.16032

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
- Paper: https://openaccess.thecvf.com/content/CVPR2023/html/Zhou_Neural_Texture_Synthesis_With_Guided_Correspondence_CVPR_2023_paper.html
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

**Identity-Preserving Talking Face Generation with Landmark and Appearance Priors**
- Paper: https://arxiv.org/abs/2305.08293
- Code: https://github.com/Weizhi-Zhong/IP_LAP

**LipFormer: High-Fidelity and Generalizable Talking Face Generation With a Pre-Learned Facial Codebook**
- Paper: https://openaccess.thecvf.com/content/CVPR2023/html/Wang_LipFormer_High-Fidelity_and_Generalizable_Talking_Face_Generation_With_a_Pre-Learned_CVPR_2023_paper.html

**High-fidelity Generalized Emotional Talking Face Generation with Multi-modal Emotion Space Learning**
- Paper: https://arxiv.org/abs/2305.02572

**DiffTalk: Crafting Diffusion Models for Generalized Audio-Driven Portraits Animation**
- Paper: https://arxiv.org/abs/2301.03786
- Code: https://github.com/sstzal/DiffTalk

## Virtual Try-on

**GP-VTON: Towards General Purpose Virtual Try-on via Collaborative Local-Flow Global-Parsing Learning**
- Paper: https://arxiv.org/abs/2303.13756
- Code: https://github.com/xiezhy6/GP-VTON

**Linking Garment With Person via Semantically Associated Landmarks for Virtual Try-On**
- Paper: https://openaccess.thecvf.com/content/CVPR2023/html/Yan_Linking_Garment_With_Person_via_Semantically_Associated_Landmarks_for_Virtual_CVPR_2023_paper.html
- Code: https://modelscope.cn/datasets/damo/SAL-HG/summary

**TryOnDiffusion: A Tale of Two UNets**
- Paper: https://openaccess.thecvf.com/content/CVPR2023/html/Zhu_TryOnDiffusion_A_Tale_of_Two_UNets_CVPR_2023_paper.html

## Handwriting/Font Generation

**CF-Font: Content Fusion for Few-shot Font Generation**
- Paper: https://arxiv.org/abs/2303.14017
- Code: https://github.com/wangchi95/CF-Font
- Tags: Font Generation

**Neural Transformation Fields for Arbitrary-Styled Font Generation**
- Paper: https://openaccess.thecvf.com/content/CVPR2023/html/Fu_Neural_Transformation_Fields_for_Arbitrary-Styled_Font_Generation_CVPR_2023_paper.html
- Code: https://github.com/fubinfb/NTF

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

**Conditional Text Image Generation With Diffusion Models**
- Paper: https://openaccess.thecvf.com/content/CVPR2023/html/Zhu_Conditional_Text_Image_Generation_With_Diffusion_Models_CVPR_2023_paper.html

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

**LayoutDM: Transformer-based Diffusion Model for Layout Generation**
- Paper: https://arxiv.org/abs/2305.02567
