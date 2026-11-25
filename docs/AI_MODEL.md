# Deployed AI model

## v1.1 model contract

| Property | Value |
| --- | --- |
| Encoder | MobileNetV3-Large |
| Representation | 960 features |
| Pretraining | BYOL + Masked InfoNCE, epoch 3 encoder |
| Classifier | Logistic-regression DR head exported as ONNX |
| Input | `fundus_image`, float32 NCHW `[1, 3, 512, 512]` |
| Output | `dr_probability`, float32 `[1, 1]`, sigmoid probability |
| Decision cutoff | `0.48`, inclusive for DR detected |
| Runtime | ONNX Runtime 1.15.1 Android CPU libraries |
| ONNX SHA-256 | `b8fc10191f69faef16f1dc92d08bf9542ee7e26f885767942e28292c5201dd90` |

The model is checked into `deployment/offline_dr_screening/assets/offirst_dr_model.onnx` so the documented Flutter build works from a fresh checkout. The verified v1.1 release also provides the standalone [model asset](https://github.com/Mayank-gitspace/OFFIRST/releases/download/offirst-android-prototype-v1.1/offirst_dr_model.onnx). Verify the SHA-256 above. The app decodes to RGB, resizes to 512 × 512 with the audited Pillow-compatible bilinear sampling path, scales to `[0,1]`, applies ImageNet channel mean and standard deviation, then arranges NCHW float32 input. EXIF orientation is not applied to inference pixels, matching the reference inference contract.

BYOL is non-contrastive self-supervised learning. Masked InfoNCE provides the explicit contrastive component. This description applies to the deployed encoder recipe; KD Run 2 is a separate teacher/student feature-distillation experiment and is not described as InfoNCE.

## Model provenance and limits

The v1.1 model and 0.48 cutoff are unchanged. Experimental v2 and v2.1 models are not packaged. The released classifier had 54.54% accuracy, 21.47% sensitivity, 98.27% specificity and AUROC 0.764 on an exploratory external DeepDRiD benchmark (1,742 images, 488 patients). Training exposure and label-definition questions remain unresolved. This model is not clinically validated; low sensitivity means false negatives are a serious limitation.

For implementation details see the [model contract](../deployment/offline_dr_screening/README.md) and [preprocessing audit](../deployment/offline_dr_screening/PREPROCESSING_AUDIT.md).
