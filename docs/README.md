# OFFIRST documentation index

Reviewers should start at the [repository front page](../README.md). This index points at the current application and at the research history without moving those folders.

## Current application, version 1.0.5+2005

| Document | Use |
| --- | --- |
| [Front page](../README.md) | Download, verified APK identity, architecture, and limits |
| [Installation](INSTALLATION.md) | ARM64 install steps for the published 1.0.5 APK |
| [Screenshots](SCREENSHOTS.md) | Why no current UI gallery is checked in, and what a device pass should capture |
| [Application source](../deployment/offline_dr_screening) | Flutter, Kotlin, tests, ONNX, and probes |
| [Probe sources](../heads) | JSON used to prepare the packaged heads |

The published reviewer file is [OFFIRST-1.0.5-arm64-v8a.apk](https://github.com/Mayank-gitspace/OFFIRST/releases/download/offirst-1.0.5/OFFIRST-1.0.5-arm64-v8a.apk).

## Read with a status banner

These pages still contain useful history. Their opening banner states that the packaged 1.0.5 contract has moved on. Do not treat their old threshold, 16-unit head, or universal APK as the current release.

| Document | Why it is still here |
| --- | --- |
| [AI model](AI_MODEL.md) | Earlier ONNX contract and preprocessing notes |
| [Architecture](ARCHITECTURE.md) | Earlier on-device flow |
| [App user guide](APP_USER_GUIDE.md) | Earlier demonstration steps |
| [Demo guide](DEMO_GUIDE.md) | Earlier live-demo script |
| [Known limitations](KNOWN_LIMITATIONS.md) | Limits recorded for the earlier classifier |
| [Research results](RESEARCH_RESULTS.md) | Traceable historical metrics and representation probes |
| [Development setup](DEVELOPMENT_SETUP.md) | Local toolchain notes |
| [App README](../deployment/offline_dr_screening/README.md) | Application directory overview from the earlier contract |
| [Release handoff](../deployment/offline_dr_screening/FINAL_RELEASE_HANDOFF.md) | Submission handoff for the earlier packaged classifier |

## Research map

| Area | Path |
| --- | --- |
| Android deployment | [deployment/offline_dr_screening](../deployment/offline_dr_screening) |
| Model assets in the app | [assets](../deployment/offline_dr_screening/assets) |
| Self-supervised ConvNeXt BYOL | [encoder_byol](../encoder_byol) |
| MobileNetV3 self-supervision | [encoder_mnv3_ssl](../encoder_mnv3_ssl) |
| Knowledge distillation | [kd_run2](../kd_run2) |
| Supervised encoder baselines | [encoder_baseline](../encoder_baseline) |
| Imaging studies | [kd_imaging](../kd_imaging) |
| Lesion segmentation, not in the APK | [lesion_segmenter](../lesion_segmenter) |
| Evaluation utilities | [tools](../tools) |
| Root audit notes | `OFFIRST_*.md` in the repository root |

Root audit notes preserve evidence trails. Several of them still quote the earlier ONNX hash `43a87f05…` and threshold `0.3675977844516245`. Read them as history next to the front page, which is the current contract.
