# CVPR2022-Low-Level-Vision
A Collection of Papers and Codes in CVPR2022 related to Low-Level Vision


## Related collections for low-level vision
- [Awesome-AAAI2022-Low-Level-Vision](https://github.com/DarrenPan/Awesome-AAAI2022-Low-Level-Vision)
- [Awesome-NeurIPS2021-Low-Level-Vision](https://github.com/DarrenPan/Awesome-NeurIPS2021-Low-Level-Vision)
- [Awesome-ICCV2021-Low-Level-Vision](https://github.com/Kobaayyy/Awesome-ICCV2021-Low-Level-Vision)
- [Awesome-CVPR2021/CVPR2020-Low-Level-Vision](https://github.com/Kobaayyy/Awesome-CVPR2021-CVPR2020-Low-Level-Vision)
- [Awesome-ECCV2020-Low-Level-Vision](https://github.com/Kobaayyy/Awesome-ECCV2020-Low-Level-Vision)


## 

- [Image Restoration](#ImageRetoration)
  - [Burst Restoration](#BurstRestoration)
  - [Video Restoration](#VideoRestoration)
  - [Hyperspectral Image Reconstruction](#HSIR)
- [Super Resolution](#SuperResolution)
  - [Image Super Resolution](#ImageSuperResolution)
  - [Video Super Resolution](#VideoSuperResolution)
- [Image Rescaling](#Rescaling)

- [Denoising](#Denoising)
  - [Image Denoising](#ImageDenoising)
  - [Burst Denoising](#BurstDenoising)
  - [Video Denoising](#VideoDenoising)

- [Deblurring](#Deblurring)
  - [Image Deblurring](#ImageDeblurring)
  - [Video Deblurring](#VideoDeblurring)

- [Deraining](#Deraining)

- [Dehazing](#Dehazing)

- [Demoireing](#Demoireing)

- [Frame Interpolation](#FrameInterpolation)
  - [Spatial-Temporal Video Super-Resolution](#STVSR)
- [Image Enhancement](#Enhancement)
  - [Low-Light Image Enhancement](#LowLight)

- [Image Harmonization](#Harmonization)

- [Image Completion/Inpainting](#Inpainting)
  - [Video Inpainting](#VideoInpainting)
  
- [Relighting](#Relighting)

- [Image Stitching](#Stitching)

- [Image Compression](#ImageCompression)

- [Image Quality Assessment](#ImageQualityAssessment)

- [Style Transfer](#StyleTransfer)

- [Image Editing](#ImageEditing)

- [Image Generation/Synthesis](#ImageGeneration)
  - [Video Generation/Synthesis](#VideoGeneration)

- [Others](#Others)

- [NITRE](#NTIRE)

<a name="ImageRetoration"></a>
# Image Restoration - 图像恢复

**Restormer: Efficient Transformer for High-Resolution Image Restoration**
- Paper: https://arxiv.org/abs/2111.09881
- Code: https://github.com/swz30/Restormer
- Tags: Transformer

**Uformer: A General U-Shaped Transformer for Image Restoration**
- Paper: https://arxiv.org/abs/2106.03106
- Code: https://github.com/ZhendongWang6/Uformer
- Tags: Transformer

**MAXIM: Multi-Axis MLP for Image Processing**
- Paper: https://arxiv.org/abs/2201.02973
- Code: https://github.com/google-research/maxim
- Tags: MLP, also do image enhancement

**All-In-One Image Restoration for Unknown Corruption**
- Paper: http://pengxi.me/wp-content/uploads/2022/03/All-In-One-Image-Restoration-for-Unknown-Corruption.pdf
- Code: https://github.com/XLearning-SCU/2022-CVPR-AirNet

**Fourier Document Restoration for Robust Document Dewarping and Recognition**
- Paper: https://arxiv.org/abs/2203.09910
- Tags: Document Restoration

**Exploring and Evaluating Image Restoration Potential in Dynamic Scenes**
- Paper: https://arxiv.org/abs/2203.11754

**ISNAS-DIP: Image-Specific Neural Architecture Search for Deep Image Prior**
- Paper: https://arxiv.org/abs/2111.15362v2
- Code: https://github.com/ozgurkara99/ISNAS-DIP
- Tags: DIP, NAS

**Deep Generalized Unfolding Networks for Image Restoration**
- Paper: https://arxiv.org/abs/2204.13348
- Code: https://github.com/MC-E/Deep-Generalized-Unfolding-Networks-for-Image-Restoration

**Attentive Fine-Grained Structured Sparsity for Image Restoration**
- Paper: https://arxiv.org/abs/2204.12266
- Code: https://github.com/JungHunOh/SLS_CVPR2022

**End-to-End Rubbing Restoration Using Generative Adversarial Networks**
- Paper: https://arxiv.org/abs/2205.03743
- Code: https://github.com/qingfengtommy/RubbingGAN
- Tags: [Workshop], Rubbing Restoration

**GenISP: Neural ISP for Low-Light Machine Cognition**
- Paper: https://arxiv.org/abs/2205.03688
- Tags: [Workshop], ISP

<a name="BurstRestoration"></a>
## Burst Restoration

**A Differentiable Two-stage Alignment Scheme for Burst Image Reconstruction with Large Shift**
- Paper: https://arxiv.org/abs/2203.09294
- Code: https://github.com/GuoShi28/2StageAlign
- Tags: joint denoising and demosaicking

**Burst Image Restoration and Enhancement**
- Paper: https://arxiv.org/abs/2110.03680
- Code: https://github.com/akshaydudhane16/BIPNet

<a name="VideoRestoration"></a>
## Video Restoration

**Revisiting Temporal Alignment for Video Restoration**
- Paper: https://arxiv.org/abs/2111.15288
- Code: https://github.com/redrock303/Revisiting-Temporal-Alignment-for-Video-Restoration

**Neural Compression-Based Feature Learning for Video Restoration**
- Paper:https://arxiv.org/abs/2203.09208

**Bringing Old Films Back to Life**
- Paper: https://arxiv.org/abs/2203.17276
- Code: https://github.com/raywzy/Bringing-Old-Films-Back-to-Life

**Neural Global Shutter: Learn to Restore Video from a Rolling Shutter Camera with Global Reset Feature**
- Paper: https://arxiv.org/abs/2204.00974
- Code: https://github.com/lightChaserX/neural-global-shutter
- Tags: restore clean global shutter (GS) videos

**Context-Aware Video Reconstruction for Rolling Shutter Cameras**
- Paper: https://arxiv.org/abs/2205.12912
- Code: https://github.com/GitCVfb/CVR
- Tags: Rolling Shutter Cameras

<a name="HSIR"></a>
## Hyperspectral Image Reconstruction

**Mask-guided Spectral-wise Transformer for Efficient Hyperspectral Image Reconstruction**
- Paper: https://arxiv.org/abs/2111.07910
- Code: https://github.com/caiyuanhao1998/MST

**HDNet: High-resolution Dual-domain Learning for Spectral Compressive Imaging**
- Paper: https://arxiv.org/abs/2203.02149


<a name="SuperResolution"></a>
# Super Resolution - 超分辨率
<a name="ImageSuperResolution"></a>
## Image Super Resolution

**Reflash Dropout in Image Super-Resolution**
- Paper: https://arxiv.org/abs/2112.12089
- Code: https://github.com/Xiangtaokong/Reflash-Dropout-in-Image-Super-Resolution

**Residual Local Feature Network for Efficient Super-Resolution**
- Paper: https://arxiv.org/abs/2205.07514
- Code: https://github.com/fyan111/RLFN
- Tags: won the first place in the runtime track of the NTIRE 2022 efficient super-resolution challenge

**Learning the Degradation Distribution for Blind Image Super-Resolution**
- Paper: https://arxiv.org/abs/2203.04962
- Code: https://github.com/greatlog/UnpairedSR
- Tags: Blind SR

**Deep Constrained Least Squares for Blind Image Super-Resolution**
- Paper: https://arxiv.org/abs/2202.07508
- Code: https://github.com/Algolzw/DCLS
- Tags: Blind SR

**Blind Image Super-resolution with Elaborate Degradation Modeling on Noise and Kernel**
- Paper: https://arxiv.org/abs/2107.00986
- Code: https://github.com/zsyOAOA/BSRDM
- Tags: Blind SR

**Details or Artifacts: A Locally Discriminative Learning Approach to Realistic Image Super-Resolution**
- Paper: https://arxiv.org/abs/2203.09195
- Code: https://github.com/csjliang/LDL
- Tags: Real SR

**GCFSR: a Generative and Controllable Face Super Resolution Method Without Facial and GAN Priors**
- Paper: https://arxiv.org/abs/2203.07319
- Tags: Face SR

**A Text Attention Network for Spatial Deformation Robust Scene Text Image Super-resolution**
- Paper: https://arxiv.org/abs/2203.09388
- Code: https://github.com/mjq11302010044/TATT
- Tags: Text SR

**Learning Graph Regularisation for Guided Super-Resolution**
- Paper: https://arxiv.org/abs/2203.09388
- Tags: Guided SR

**Transformer-empowered Multi-scale Contextual Matching and Aggregation for Multi-contrast MRI Super-resolution**
- Paper: https://arxiv.org/abs/2203.09388
- Tags: MRI SR

**IMDeception: Grouped Information Distilling Super-Resolution Network**
- Paper: https://arxiv.org/abs/2204.11463
- Tags: [Workshop], lightweight

**A Closer Look at Blind Super-Resolution: Degradation Models, Baselines, and Performance Upper Bounds**
- Paper: https://arxiv.org/abs/2205.04910
- Code: https://github.com/WenlongZhang0517/CloserLookBlindSR
- Tags: [Workshop], Blind SR

## Burst/Multi-frame Super Resolution

**Self-Supervised Super-Resolution for Multi-Exposure Push-Frame Satellites**
- Paper: https://arxiv.org/abs/2205.02031
- Code: https://github.com/centreborelli/HDR-DSP-SR/
- Tags: Self-Supervised, multi-exposure

<a name="VideoSuperResolution"></a>
## Video Super Resolution

**BasicVSR++: Improving Video Super-Resolution with Enhanced Propagation and Alignment**
- Paper: https://arxiv.org/abs/2104.13371
- Code: https://github.com/ckkelvinchan/BasicVSR_PlusPlus

**Learning Trajectory-Aware Transformer for Video Super-Resolution**
- Paper: https://arxiv.org/abs/2204.04216
- Code: https://github.com/researchmm/TTVSR
- Tags: Transformer

**Look Back and Forth: Video Super-Resolution with Explicit Temporal Difference Modeling**
- Paper: https://arxiv.org/abs/2204.07114

**Investigating Tradeoffs in Real-World Video Super-Resolution**
- Paper: https://arxiv.org/abs/2111.12704
- Code: https://github.com/ckkelvinchan/RealBasicVSR
- Tags: Real-world, RealBaiscVSR

**Reference-based Video Super-Resolution Using Multi-Camera Video Triplets**
- Paper: https://arxiv.org/abs/2203.14537
- Code: https://github.com/codeslake/RefVSR
- Tags: Reference-based VSR

**A New Dataset and Transformer for Stereoscopic Video Super-Resolution**
- Paper: https://arxiv.org/abs/2204.10039
- Code: https://github.com/H-deep/Trans-SVSR/
- Tags: Stereoscopic Video Super-Resolution


<a name="Rescaling"></a>
# Image Rescaling - 图像缩放

**Towards Bidirectional Arbitrary Image Rescaling: Joint Optimization and Cycle Idempotence**
- Paper: https://arxiv.org/abs/2203.00911

<a name="Denoising"></a>
# Denoising - 去噪

<a name="ImageDenoising"></a>
## Image Denoising

**Self-Supervised Image Denoising via Iterative Data Refinement**
- Paper: https://arxiv.org/abs/2111.14358
- Code: https://github.com/zhangyi-3/IDR
- Tags: Self-Supervised

**Blind2Unblind: Self-Supervised Image Denoising with Visible Blind Spots**
- Paper: https://arxiv.org/abs/2203.06967
- Code: https://github.com/demonsjin/Blind2Unblind
- Tags: Self-Supervised

**AP-BSN: Self-Supervised Denoising for Real-World Images via Asymmetric PD and Blind-Spot Network**
- Paper: https://arxiv.org/abs/2203.11799
- Code: https://github.com/wooseoklee4/AP-BSN
- Tags: Self-Supervised

**CVF-SID: Cyclic multi-Variate Function for Self-Supervised Image Denoising by Disentangling Noise from Image**
- Paper: https://arxiv.org/abs/2203.13009
- Code: https://github.com/Reyhanehne/CVF-SID_PyTorch
- Tags: Self-Supervised

**Noise2NoiseFlow: Realistic Camera Noise Modeling without Clean Images**
- Paper: https://arxiv.org/abs/2206.01103
- Tags: Noise Modeling, Normalizing Flow

**Modeling sRGB Camera Noise with Normalizing Flows**
- Paper: https://arxiv.org/abs/2206.00812
- Tags: Noise Modeling, Normalizing Flow

**Multiple Degradation and Reconstruction Network for Single Image Denoising via Knowledge Distillation**
- Paper: https://arxiv.org/abs/2204.13873
- Tags: [Workshop]

<a name="BurstDenoising"></a>
## BurstDenoising

**NAN: Noise-Aware NeRFs for Burst-Denoising**
- Paper: https://arxiv.org/abs/2204.04668
- Tags: NeRFs 

<a name="VideoDenoising"></a>
## Video Denoising 

**Dancing under the stars: video denoising in starlight**
- Paper: https://arxiv.org/abs/2204.04210
- Code: https://github.com/monakhova/starlight_denoising/
- Tags: video denoising in starlight


<a name="Deblurring"></a>
# Deblurring - 去模糊
<a name="ImageDeblurring"></a>
## Image Deblurring

**Learning to Deblur using Light Field Generated and Real Defocus Images**
- Paper: https://arxiv.org/abs/2204.00367
- Code: https://github.com/lingyanruan/DRBNet
- Tags: Defocus deblurring

**Unifying Motion Deblurring and Frame Interpolation with Events**
- Paper: https://arxiv.org/abs/2203.12178
- Tags: event-based

**E-CIR: Event-Enhanced Continuous Intensity Recovery**
- Paper: https://arxiv.org/abs/2203.01935
- Code: https://github.com/chensong1995/E-CIR
- Tags: event-based

<a name="VideoDeblurring"></a>
## Video Deblurring

**Multi-Scale Memory-Based Video Deblurring**
- Paper: https://arxiv.org/abs/2203.01935
- Code: https://github.com/jibo27/MemDeblur

<a name="Deraining"></a>
# Deraining - 去雨

**Towards Robust Rain Removal Against Adversarial Attacks: A Comprehensive Benchmark Analysis and Beyond**
- Paper: https://arxiv.org/abs/2203.16931
- Code: https://github.com/yuyi-sd/Robust_Rain_Removal

**Unpaired Deep Image Deraining Using Dual Contrastive Learning**
- Paper: https://arxiv.org/abs/2109.02973
- Tags: Contrastive Learning, Unpaired

**Unsupervised Deraining: Where Contrastive Learning Meets Self-similarity**
- Paper: https://arxiv.org/abs/2203.11509
- Tags: Contrastive Learning, Unsupervised


<a name="Dehazing"></a>
# Dehazing - 去雾

**Self-augmented Unpaired Image Dehazing via Density and Depth Decomposition**
- Paper:
- Code: https://github.com/YaN9-Y/D4
- Tags: Unpaired

<a name="Demoireing"></a>
# Demoireing - 去摩尔纹

**Video Demoireing with Relation-Based Temporal Consistency**
- Paper: https://arxiv.org/abs/2204.02957
- Code: https://github.com/CVMI-Lab/VideoDemoireing


<a name="FrameInterpolation"></a>
# Frame Interpolation - 插帧

**ST-MFNet: A Spatio-Temporal Multi-Flow Network for Frame Interpolation**
- Paper: https://arxiv.org/abs/2111.15483
- Code: https://github.com/danielism97/ST-MFNet

**Long-term Video Frame Interpolation via Feature Propagation**
- Paper: https://arxiv.org/abs/2203.15427

**Many-to-many Splatting for Efficient Video Frame Interpolation**
- Paper: https://arxiv.org/abs/2204.03513
- Code: https://github.com/feinanshan/M2M_VFI

**Video Frame Interpolation with Transformer**
- Paper: https://arxiv.org/abs/2205.07230
- Code: https://github.com/dvlab-research/VFIformer
- Tags: Transformer

**IFRNet: Intermediate Feature Refine Network for Efficient Frame Interpolation**
- Paper: https://arxiv.org/abs/2205.14620
- Code: https://github.com/ltkong218/IFRNet

**TimeReplayer: Unlocking the Potential of Event Cameras for Video Interpolation**
- Paper: https://arxiv.org/abs/2203.13859
- Tags: Event Camera

**Time Lens++: Event-based Frame Interpolation with Parametric Non-linear Flow and Multi-scale Fusion**
- Paper: https://arxiv.org/abs/2203.17191
- Tags: Event-based 

**Unifying Motion Deblurring and Frame Interpolation with Events**
- Paper: https://arxiv.org/abs/2203.12178
- Tags: event-based

**Multi-encoder Network for Parameter Reduction of a Kernel-based Interpolation Architecture**
- Paper: https://arxiv.org/abs/2205.06723
- Tags: [Workshop]

<a name="STVSR"></a>
## Spatial-Temporal Video Super-Resolution

**RSTT: Real-time Spatial Temporal Transformer for Space-Time Video Super-Resolution**
- Paper: https://arxiv.org/abs/2203.14186
- Code: https://github.com/llmpass/RSTT

**Spatial-Temporal Space Hand-in-Hand: Spatial-Temporal Video Super-Resolution via Cycle-Projected Mutual Learning**
- Paper: https://arxiv.org/abs/2205.05264

**VideoINR: Learning Video Implicit Neural Representation for Continuous Space-Time Super-Resolution**
- Paper: https://arxiv.org/abs/2206.04647
- Code: https://github.com/Picsart-AI-Research/VideoINR-Continuous-Space-Time-Super-Resolution

<a name="Enhancement"></a>
# Image Enhancement - 图像增强

**AdaInt: Learning Adaptive Intervals for 3D Lookup Tables on Real-time Image Enhancement**
- Paper: https://arxiv.org/abs/2204.13983
- Code: https://github.com/ImCharlesY/AdaInt

**Exposure Correction Model to Enhance Image Quality**
- Paper: https://arxiv.org/abs/2204.10648
- Code: https://github.com/yamand16/ExposureCorrection
- Tags: [Workshop]

<a name="LowLight"></a>
## Low-Light Image Enhancement

**Abandoning the Bayer-Filter to See in the Dark**
- Paper: https://arxiv.org/abs/2203.04042
- Code: https://github.com/TCL-AILab/Abandon_Bayer-Filter_See_in_the_Dark

**Toward Fast, Flexible, and Robust Low-Light Image Enhancement**
- Paper: https://arxiv.org/abs/2204.10137
- Code: https://github.com/vis-opt-group/SCI


<a name="Harmonization"></a>
# Image Harmonization/Composition - 图像协调/图像合成

**High-Resolution Image Harmonization via Collaborative Dual Transformationsg**
- Paper: https://arxiv.org/abs/2109.06671
- Code: https://github.com/bcmi/CDTNet-High-Resolution-Image-Harmonization

**SCS-Co: Self-Consistent Style Contrastive Learning for Image Harmonization**
- Paper: https://arxiv.org/abs/2204.13962
- Code: https://github.com/YCHang686/SCS-Co-CVPR2022

<a name="Inpainting"></a>
# Image Completion/Inpainting - 图像修复

**Bridging Global Context Interactions for High-Fidelity Image Completion**
- Paper: https://arxiv.org/abs/2104.00845
- Code: https://github.com/lyndonzheng/TFill

**Incremental Transformer Structure Enhanced Image Inpainting with Masking Positional Encoding**
- Paper: https://arxiv.org/abs/2203.00867
- Code: https://github.com/DQiaole/ZITS_inpainting

**MISF: Multi-level Interactive Siamese Filtering for High-Fidelity Image Inpainting**
- Paper: https://arxiv.org/abs/2203.06304
- Code: https://github.com/tsingqguo/misf

**MAT: Mask-Aware Transformer for Large Hole Image Inpainting**
- Paper: https://arxiv.org/abs/2203.15270
- Code: https://github.com/fenglinglwb/MAT

**Reduce Information Loss in Transformers for Pluralistic Image Inpainting**
- Paper: https://arxiv.org/abs/2205.05076

**SaiNet: Stereo aware inpainting behind objects with generative networks**
- Paper: https://arxiv.org/abs/2205.07014
- Tags: [Workshop]

<a name="VideoInpainting"></a>
## Video Inpainting

**Towards An End-to-End Framework for Flow-Guided Video Inpainting**
- Paper: https://arxiv.org/abs/2204.02663
- Code: https://github.com/MCG-NKU/E2FGVI


<a name="Relighting"></a>
# Relighting

**Face Relighting with Geometrically Consistent Shadows**
- Paper: https://arxiv.org/abs/2203.16681
- Code: https://github.com/andrewhou1/GeomConsistentFR
- Tags: Face Relighting

<a name="Stitching"></a>
# Image Stitching - 图像拼接

**Deep Rectangling for Image Stitching: A Learning Baseline**
- Paper: https://arxiv.org/abs/2203.03831
- Code: https://github.com/nie-lang/DeepRectangling

<a name="ImageCompression"></a>
# Image Compression - 图像压缩

**Neural Data-Dependent Transform for Learned Image Compression**
- Paper: https://arxiv.org/abs/2203.04963v1

**The Devil Is in the Details: Window-based Attention for Image Compression**
- Paper: https://arxiv.org/abs/2203.08450
- Code: https://github.com/Googolxx/STF

**ELIC: Efficient Learned Image Compression with Unevenly Grouped Space-Channel Contextual Adaptive Coding**
- Paper: https://arxiv.org/abs/2203.10886

**Unified Multivariate Gaussian Mixture for Efficient Neural Image Compression**
- Paper: https://arxiv.org/abs/2203.10897
- Code: https://github.com/xiaosu-zhu/McQuic

**Practical Learned Lossless JPEG Recompression with Multi-Level Cross-Channel Entropy Model in the DCT Domain**
- Paper: https://arxiv.org/abs/2203.16357
- Tags: Compress JPEG 

**PO-ELIC: Perception-Oriented Efficient Learned Image Coding**
- Paper: https://arxiv.org/abs/2205.14501
- Tags: [Workshop]

## Video Compression

**Enhancing VVC with Deep Learning based Multi-Frame Post-Processing**
- Paper: https://arxiv.org/abs/2205.09458
- Tags: [Workshop]

<a name="ImageQualityAssessment"></a>
# Image Quality Assessment - 图像质量评价

**Personalized Image Aesthetics Assessment with Rich Attributes**
- Paper: https://arxiv.org/abs/2203.16754
- Tags: Aesthetics Assessment

**Incorporating Semi-Supervised and Positive-Unlabeled Learning for Boosting Full Reference Image Quality Assessment**
- Paper: https://arxiv.org/abs/2204.08763
- Code: https://github.com/happycaoyue/JSPL
- Tags: FR-IQA

**SwinIQA: Learned Swin Distance for Compressed Image Quality Assessment**
- Paper: https://arxiv.org/abs/2205.04264
- Tags: [Workshop], compressed IQA


<a name="StyleTransfer"></a>
# Style Transfer - 风格迁移

**CLIPstyler: Image Style Transfer with a Single Text Condition**
- Paper: https://arxiv.org/abs/2112.00374
- Code: https://github.com/cyclomon/CLIPstyler
- Tags: CLIP

**Style-ERD: Responsive and Coherent Online Motion Style Transfer**
- Paper: https://arxiv.org/abs/2203.02574
- Code: https://github.com/tianxintao/Online-Motion-Style-Transfer

**Exact Feature Distribution Matching for Arbitrary Style Transfer and Domain Generalization**
- Paper: https://arxiv.org/abs/2203.07740
- Code: https://github.com/YBZh/EFDM

**Pastiche Master: Exemplar-Based High-Resolution Portrait Style Transfer**
- Paper: https://arxiv.org/abs/2203.13248
- Code: https://github.com/williamyang1991/DualStyleGAN

**Industrial Style Transfer with Large-scale Geometric Warping and Content Preservation**
- Paper: https://arxiv.org/abs/2203.12835
- Code: https://github.com/jcyang98/InST


<a name="ImageEditing"></a>
# Image Editing - 图像编辑

**High-Fidelity GAN Inversion for Image Attribute Editing**
- Paper: https://arxiv.org/abs/2109.06590
- Code: https://github.com/Tengfei-Wang/HFGI

**Style Transformer for Image Inversion and Editing**
- Paper: https://arxiv.org/abs/2203.07932
- Code: https://github.com/sapphire497/style-transformer

**HairCLIP: Design Your Hair by Text and Reference Image**
- Paper: https://arxiv.org/abs/2112.05142
- Code: https://github.com/wty-ustc/HairCLIP
- Tags: CLIP

**HyperStyle: StyleGAN Inversion with HyperNetworks for Real Image Editing**
- Paper: https://arxiv.org/abs/2111.15666
- Code: https://github.com/yuval-alaluf/hyperstyle

**Blended Diffusion for Text-driven Editing of Natural Images**
- Paper: https://arxiv.org/abs/2111.14818
- Code: https://github.com/omriav/blended-diffusion
- Tags: CLIP

**FlexIT: Towards Flexible Semantic Image Translation**
- Paper: https://arxiv.org/abs/2203.04705 

**SemanticStyleGAN: Learning Compositonal Generative Priors for Controllable Image Synthesis and Editing**
- Paper: https://arxiv.org/abs/2112.02236

**SketchEdit: Mask-Free Local Image Manipulation with Partial Sketches**
- Paper: https://arxiv.org/abs/2111.15078
- Code: https://github.com/zengxianyu/sketchedit

**TransEditor: Transformer-Based Dual-Space GAN for Highly Controllable Facial Editing**
- Paper: https://arxiv.org/abs/2203.17266
- Code: https://github.com/BillyXYB/TransEditor

**HyperInverter: Improving StyleGAN Inversion via Hypernetwork**
- Paper: https://arxiv.org/abs/2112.00719
- Code: https://github.com/VinAIResearch/HyperInverter


<a name=ImageGeneration></a>
# Image Generation/Synthesis / Image-to-Image Translation - 图像生成/合成/转换

## Text-to-Image / Text Guided / Multi-Modal

**Text to Image Generation with Semantic-Spatial Aware GAN**
- Paper: https://arxiv.org/abs/2104.00567
- Code: https://github.com/wtliao/text2image

**LAFITE: Towards Language-Free Training for Text-to-Image Generation**
- Paper: https://arxiv.org/abs/2111.13792
- Code: https://github.com/drboog/Lafite

**DF-GAN: Deep Fusion Generative Adversarial Networks for Text-to-Image Synthesis**
- Paper: https://arxiv.org/abs/2008.05865
- Code: https://github.com/tobran/DF-GAN

**StyleT2I: Toward Compositional and High-Fidelity Text-to-Image Synthesis**
- Paper: https://arxiv.org/abs/2203.15799
- Code: https://github.com/zhihengli-UR/StyleT2I

**DiffusionCLIP: Text-Guided Diffusion Models for Robust Image Manipulation**
- Paper: https://arxiv.org/abs/2110.02711
- Code: https://github.com/gwang-kim/DiffusionCLIP

**Predict, Prevent, and Evaluate: Disentangled Text-Driven Image Manipulation Empowered by Pre-Trained Vision-Language Model**
- Paper: https://arxiv.org/abs/2111.13333
- Code: https://github.com/zipengxuc/PPE-Pytorch

**Sound-Guided Semantic Image Manipulation**
- Paper: https://arxiv.org/abs/2112.00007
- Code: https://github.com/kuai-lab/sound-guided-semantic-image-manipulation

**ManiTrans: Entity-Level Text-Guided Image Manipulation via Token-wise Semantic Alignment and Generation**
- Paper: https://arxiv.org/abs/2204.04428

## Image-to-Image / Image Guided

**Maximum Spatial Perturbation Consistency for Unpaired Image-to-Image Translation**
- Paper: https://arxiv.org/abs/2203.12707

**A Style-aware Discriminator for Controllable Image Translation**
- Paper: https://arxiv.org/abs/2203.15375
- Code: https://github.com/kunheek/style-aware-discriminator

**QS-Attn: Query-Selected Attention for Contrastive Learning in I2I Translation**
- Paper: https://arxiv.org/abs/2203.08483
- Code: https://github.com/sapphire497/query-selected-attention

**InstaFormer: Instance-Aware Image-to-Image Translation with Transformer**
- Paper: https://arxiv.org/abs/2203.16248

**Marginal Contrastive Correspondence for Guided Image Generation**
- Paper: https://arxiv.org/abs/2204.00442
- Code: https://github.com/fnzhan/UNITE

**Unsupervised Image-to-Image Translation with Generative Prior**
- Paper: https://arxiv.org/abs/2204.03641
- Code: https://github.com/williamyang1991/GP-UNIT

**Exploring Patch-wise Semantic Relation for Contrastive Learning in Image-to-Image Translation Tasks**
- Paper: https://arxiv.org/abs/2203.01532
- Code: https://github.com/jcy132/Hneg_SRC

## Others for image generation

**Attribute Group Editing for Reliable Few-shot Image Generation**
- Paper: https://arxiv.org/abs/2203.08422
- Code: https://github.com/UniBester/AGE

**Modulated Contrast for Versatile Image Synthesis**
- Paper: https://arxiv.org/abs/2203.09333
- Code: https://github.com/fnzhan/MoNCE

**Interactive Image Synthesis with Panoptic Layout Generation**
- Paper: https://arxiv.org/abs/2203.02104

**Autoregressive Image Generation using Residual Quantization**
- Paper: https://arxiv.org/abs/2203.01941
- Code: https://github.com/lucidrains/RQ-Transformer

**Dynamic Dual-Output Diffusion Models**
- Paper: https://arxiv.org/abs/2203.04304

**Exploring Dual-task Correlation for Pose Guided Person Image Generation**
- Paper: https://arxiv.org/abs/2203.02910
- Code: https://github.com/PangzeCheung/Dual-task-Pose-Transformer-Network

**StyleSwin: Transformer-based GAN for High-resolution Image Generation**
- Paper: https://arxiv.org/abs/2112.10762
- Code: https://github.com/microsoft/StyleSwin

**Semantic-shape Adaptive Feature Modulation for Semantic Image Synthesis**
- Paper: https://arxiv.org/abs/2203.16898
- Code: https://github.com/cszy98/SAFM

**Arbitrary-Scale Image Synthesis**
- Paper: https://arxiv.org/abs/2204.02273
- Code: https://github.com/vglsd/ScaleParty

**InsetGAN for Full-Body Image Generation**
- Paper: https://arxiv.org/abs/2203.07293

**HairMapper: Removing Hair from Portraits Using GANs**
- Paper: http://www.cad.zju.edu.cn/home/jin/cvpr2022/HairMapper.pdf
- Code: https://github.com/oneThousand1000/non-hair-FFHQ

**OSSGAN: Open-Set Semi-Supervised Image Generation**
- Paper: https://arxiv.org/abs/2204.14249
- Code: https://github.com/raven38/OSSGAN

**Retrieval-based Spatially Adaptive Normalization for Semantic Image Synthesis**
- Paper: https://arxiv.org/abs/2204.02854
- Code: https://github.com/Shi-Yupeng/RESAIL-For-SIS

**Retrieval-based Spatially Adaptive Normalization for Semantic Image Synthesis**
- Paper: https://arxiv.org/abs/2204.02854
- Code: https://github.com/Shi-Yupeng/RESAIL-For-SIS

**A Closer Look at Few-shot Image Generation**
- Paper: https://arxiv.org/abs/2205.03805
- Tags: Few-shot

**Ensembling Off-the-shelf Models for GAN Training**
- Paper: https://arxiv.org/abs/2112.09130
- Code: https://github.com/nupurkmr9/vision-aided-gan

**Few-Shot Font Generation by Learning Fine-Grained Local Styles**
- Paper: https://arxiv.org/abs/2205.09965
- Tags: Few-shot

**Modeling Image Composition for Complex Scene Generation**
- Paper: https://arxiv.org/abs/2206.00923
- Code: https://github.com/JohnDreamer/TwFA

**On Conditioning the Input Noise for Controlled Image Generation with Diffusion Models**
- Paper: https://arxiv.org/abs/2205.03859
- Tags: [Workshop]

**Generate and Edit Your Own Character in a Canonical View**
- Paper: https://arxiv.org/abs/2205.02974
- Tags: [Workshop]

**StyLandGAN: A StyleGAN based Landscape Image Synthesis using Depth-map**
- Paper: https://arxiv.org/abs/2205.06611
- Tags: [Workshop]

**Overparameterization Improves StyleGAN Inversion**
- Paper: https://arxiv.org/abs/2205.06304
- Tags: [Workshop]

<a name="VideoGeneration"></a>
## Video Generation/Synthesis

**Show Me What and Tell Me How: Video Synthesis via Multimodal Conditioning**
- Paper: https://arxiv.org/abs/2203.02573
- Code: https://github.com/snap-research/MMVID

**Playable Environments: Video Manipulation in Space and Time**
- Paper: https://arxiv.org/abs/2203.01914
- Code: https://github.com/willi-menapace/PlayableEnvironments

**StyleGAN-V: A Continuous Video Generator with the Price, Image Quality and Perks of StyleGAN2**
- Paper: https://kaust-cair.s3.amazonaws.com/stylegan-v/stylegan-v-paper.pdf
- Code: https://github.com/universome/stylegan-v

**Thin-Plate Spline Motion Model for Image Animation**
- Paper: https://arxiv.org/abs/2203.14367
- Code: https://github.com/yoyo-nb/Thin-Plate-Spline-Motion-Model

**Diverse Video Generation from a Single Video**
- Paper: https://arxiv.org/abs/2205.05725
- Tags: [Workshop]

<a name="Others"></a>
# Others

**GAN-Supervised Dense Visual Alignment**
- Paper: https://arxiv.org/abs/2112.05143
- Code: https://github.com/wpeebles/gangealing

**ClothFormer:Taming Video Virtual Try-on in All Module**
- Paper: https://arxiv.org/abs/2204.12151
- Tags: Video Virtual Try-on

**Iterative Deep Homography Estimation**
- Paper: https://arxiv.org/abs/2203.15982
- Code: https://github.com/imdumpl78/IHN

**Patch-wise Contrastive Style Learning for Instagram Filter Removal**
- Paper: https://arxiv.org/abs/2204.07486
- Code: https://github.com/birdortyedi/cifr-pytorch
- Tags: [Workshop]


<a name="NTIRE"></a>
# NTIRE2022
New Trends in Image Restoration and Enhancement workshop and challenges on image and video processing.

##  Spectral Reconstruction from RGB

**MST++: Multi-stage Spectral-wise Transformer for Efficient Spectral Reconstruction**
- Paper: https://arxiv.org/abs/2204.07908
- Code: https://github.com/caiyuanhao1998/MST-plus-plus
- Tags: 1st place

## Perceptual Image Quality Assessment: Track 1 Full-Reference / Track 2 No-Reference

**MANIQA: Multi-dimension Attention Network for No-Reference Image Quality Assessment**
- Paper: https://arxiv.org/abs/2204.08958
- Code: https://github.com/IIGROUP/MANIQA
- Tags: 1st place for track2

**Attentions Help CNNs See Better: Attention-based Hybrid Image Quality Assessment Network**
- Paper: https://arxiv.org/abs/2204.10485
- Code: https://github.com/IIGROUP/AHIQ
- Tags: 1st place for track1

**MSTRIQ: No Reference Image Quality Assessment Based on Swin Transformer with Multi-Stage Fusion**
- Paper: https://arxiv.org/abs/2205.10101
- Tags: 2nd place in track2

**Conformer and Blind Noisy Students for Improved Image Quality Assessment**
- Paper: https://arxiv.org/abs/2204.12819

## Inpainting: Track 1 Unsupervised / Track 2 Semantic

**GLaMa: Joint Spatial and Frequency Loss for General Image Inpainting**
- Paper: https://arxiv.org/abs/2205.07162
- Tags:  ranked first in terms of PSNR, LPIPS and SSIM in the track1

## Efficient Super-Resolution

- **Report**: https://arxiv.org/abs/2205.05675

**ShuffleMixer: An Efficient ConvNet for Image Super-Resolution**
- Paper: https://arxiv.org/abs/2205.15175
- Code: https://github.com/sunny2109/MobileSR-NTIRE2022
- Tags: Winner of the model complexity track 

**Edge-enhanced Feature Distillation Network for Efficient Super-Resolution**
- Paper: https://arxiv.org/abs/2204.08759
- Code: https://github.com/icandle/EFDN

**Fast and Memory-Efficient Network Towards Efficient Image Super-Resolution**
- Paper: https://arxiv.org/abs/2204.08759
- Code: https://github.com/NJU-Jet/FMEN
- Tags: Lowest memory consumption and second shortest runtime

**Blueprint Separable Residual Network for Efficient Image Super-Resolution**
- Paper: https://arxiv.org/abs/2205.05996
- Code: https://github.com/xiaom233/BSRN
- Tags: 1st place in model complexity track



## Night Photography Rendering

**Rendering Nighttime Image Via Cascaded Color and Brightness Compensation**
- Paper: https://arxiv.org/abs/2204.08970
- Code: https://github.com/NJUVISION/CBUnet
- Tags: 2nd place

## Super-Resolution and Quality Enhancement of Compressed Video: Track1 (Quality enhancement) / Track2 (Quality enhancement and x2 SR) / Track3 (Quality enhancement and x4 SR)

- **Report**: https://arxiv.org/abs/2204.09314
- **Homepage**: https://github.com/RenYang-home/NTIRE22_VEnh_SR

**Progressive Training of A Two-Stage Framework for Video Restoration**
- Paper: https://arxiv.org/abs/2204.09924
- Code: https://github.com/ryanxingql/winner-ntire22-vqe
- Tags: 1st place in track1 and track2, 2nd place in track3


## High Dynamic Range (HDR): Track 1 Low-complexity (fidelity constrain) / Track 2 Fidelity (low-complexity constrain)

- **Report**: https://arxiv.org/abs/2205.12633

**Efficient Progressive High Dynamic Range Image Restoration via Attention and Alignment Network**
- Paper: https://arxiv.org/abs/2204.09213
- Tags: 2nd palce of both two tracks

## Stereo Super-Resolution

- **Report**: https://arxiv.org/abs/2204.09197

**Parallel Interactive Transformer**
- Code: https://github.com/chaineypung/CVPR-NTIRE2022-Parallel-Interactive-Transformer-PAIT
- Tags: 7st place 

## Burst Super-Resolution: Track 2 Real

**BSRT: Improving Burst Super-Resolution with Swin Transformer and Flow-Guided Deformable Alignment**
- Code: https://github.com/Algolzw/BSRT
- Tags: 1st place 

