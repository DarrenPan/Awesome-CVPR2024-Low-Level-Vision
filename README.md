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
- [Super Resolution](#SuperResolution)
  - [Image Super Resolution](#ImageSuperResolution)
  - [Video Super Resolution](#VideoSuperResolution)
- [Image Rescaling](#Rescaling)

- [Denoising](#Denoising)

- [Deblurring](#Deblurring)

- [Frame Interpolation](#FrameInterpolation)

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

<a name="BurstRestoration"></a>
## Burst Restoration

**A Differentiable Two-stage Alignment Scheme for Burst Image Reconstruction with Large Shift**
- Paper: https://arxiv.org/abs/2203.09294
- Code: https://github.com/GuoShi28/2StageAlign
- Tags: joint denoising and demosaicking

<a name="VideoRestoration"></a>
## Video Restoration

**Neural Compression-Based Feature Learning for Video Restoration**
- Paper:https://arxiv.org/abs/2203.09208

**Bringing Old Films Back to Life**
- Paper: https://arxiv.org/abs/2203.17276
- Code: https://github.com/raywzy/Bringing-Old-Films-Back-to-Life

**Neural Global Shutter: Learn to Restore Video from a Rolling Shutter Camera with Global Reset Feature**
- Paper: https://arxiv.org/abs/2204.00974
- Code: https://github.com/lightChaserX/neural-global-shutter
- Tags: restore clean global shutter (GS) videos


<a name="SuperResolution"></a>
# Super Resolution - 超分辨率
<a name="ImageSuperResolution"></a>
## Image Super Resolution

**Reflash Dropout in Image Super-Resolution**
- Paper: https://arxiv.org/abs/2112.12089
- Code: https://github.com/Xiangtaokong/Reflash-Dropout-in-Image-Super-Resolution

**Learning the Degradation Distribution for Blind Image Super-Resolution**
- Paper: https://arxiv.org/abs/2203.04962
- Code: https://github.com/greatlog/UnpairedSR
- Tags: Blind SR

**Deep Constrained Least Squares for Blind Image Super-Resolution**
- Paper: https://arxiv.org/abs/2202.07508
- Code: https://github.com/Algolzw/DCLS
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

<a name="VideoSuperResolution"></a>
## Video Super Resolution

**BasicVSR++: Improving Video Super-Resolution with Enhanced Propagation and Alignment**
- Paper: https://arxiv.org/abs/2104.13371
- Code: https://github.com/ckkelvinchan/BasicVSR_PlusPlus

**Reference-based Video Super-Resolution Using Multi-Camera Video Triplets**
- Paper: https://arxiv.org/abs/2203.14537
- Code: https://github.com/codeslake/RefVSR
- Tags: Reference-based VSR


<a name="Rescaling"></a>
# Image Rescaling - 图像缩放

**Towards Bidirectional Arbitrary Image Rescaling: Joint Optimization and Cycle Idempotence**
- Paper: https://arxiv.org/abs/2203.00911

<a name="Denoising"></a>
# Denoising - 去噪

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


<a name="Deblurring"></a>
# Deblurring - 去模糊

**Learning to Deblur using Light Field Generated and Real Defocus Images**
- Paper: https://arxiv.org/abs/2204.00367
- Code: https://github.com/lingyanruan/DRBNet
- Tags: Defocus deblurring


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

**TimeReplayer: Unlocking the Potential of Event Cameras for Video Interpolation**
- Paper: https://arxiv.org/abs/2203.13859
- Tags: Event Camera

**Time Lens++: Event-based Frame Interpolation with Parametric Non-linear Flow and Multi-scale Fusion**
- Paper: https://arxiv.org/abs/2203.17191
- Tags: Event-based 


<a name="Enhancement"></a>
# Image Enhancement - 图像增强

<a name="LowLight"></a>
## Low-Light Image Enhancement

**Abandoning the Bayer-Filter to See in the Dark**
- Paper: https://arxiv.org/abs/2203.04042
- Code: https://github.com/TCL-AILab/Abandon_Bayer-Filter_See_in_the_Dark

<a name="Harmonization"></a>
# Image Harmonization/Composition - 图像协调/图像合成

**High-Resolution Image Harmonization via Collaborative Dual Transformationsg**
- Paper: https://arxiv.org/abs/2109.06671
- Code: https://github.com/bcmi/CDTNet-High-Resolution-Image-Harmonization


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

**Practical Learned Lossless JPEG Recompression with Multi-Level Cross-Channel Entropy Model in the DCT Domain**
- Paper: https://arxiv.org/abs/2203.16357
- Tags: Compress JPEG 


<a name="ImageQualityAssessment"></a>
# Image Quality Assessment - 图像质量评价

**Personalized Image Aesthetics Assessment with Rich Attributes**
- Paper: https://arxiv.org/abs/2203.16754
- Tags: Aesthetics Assessment


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

**Unsupervised Image-to-Image Translation with Generative Prior**
- Paper: https://arxiv.org/abs/2204.03641
- Code: https://github.com/williamyang1991/GP-UNIT


## Others for image generation

**Attribute Group Editing for Reliable Few-shot Image Generation**
- Paper: https://arxiv.org/abs/2203.08422
- Code: https://github.com/UniBester/AGE

**Modulated Contrast for Versatile Image Synthesis**
- Paper: https://arxiv.org/abs/2203.09333
- Code: https://github.com/UniBester/AGE

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


<a name=VideoGeneration></a>
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


# Others

**GAN-Supervised Dense Visual Alignment**
- Paper: https://arxiv.org/abs/2112.05143
- Code: https://github.com/wpeebles/gangealing
