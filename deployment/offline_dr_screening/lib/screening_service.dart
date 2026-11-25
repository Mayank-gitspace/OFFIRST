import 'dart:io';
import 'dart:isolate';
import 'dart:typed_data';

import 'package:flutter/foundation.dart' show debugPrint;
import 'package:flutter/services.dart';

import 'package:onnxruntime/onnxruntime.dart';

import 'model_contract.dart';
import 'model_preprocessing.dart';

const screeningThreshold = 0.48;
const screeningModelVersion = 'offirst-dr-onnx-b8fc1019';
// v2.1 is the selected application model. Set the define to false only when
// deliberately building the v1.1 rollback candidate.
const useExperimentalV21 = bool.fromEnvironment(
  'OFFIRST_EXPERIMENTAL_V21',
  defaultValue: true,
);
const v21Threshold = 0.3675977844516245;
const v21ModelVersion = 'offirst-v2.1-experimental-43a87f05';
const activeScreeningThreshold =
    useExperimentalV21 ? v21Threshold : screeningThreshold;
const activeScreeningModelVersion =
    useExperimentalV21 ? v21ModelVersion : screeningModelVersion;
const activeModelAsset = useExperimentalV21
    ? 'assets/offirst_v2_1_experimental.onnx'
    : 'assets/offirst_dr_model.onnx';
const _v21Preprocessing = MethodChannel('offirst/v2_1_preprocessing');

class ScreeningResult {
  const ScreeningResult(this.probability);

  final double probability;

  bool get detected => probability >= activeScreeningThreshold;
}

class ScreeningService {
  OrtSession? _session;

  Future<void> _ensureSession() async {
    if (_session != null) return;
    final model = await rootBundle.load(activeModelAsset);
    final options = OrtSessionOptions();
    try {
      options.setIntraOpNumThreads(2);
      options.setInterOpNumThreads(1);
      options.setSessionGraphOptimizationLevel(
        GraphOptimizationLevel.ortEnableAll,
      );
      final session = OrtSession.fromBuffer(
        model.buffer.asUint8List(model.offsetInBytes, model.lengthInBytes),
        options,
      );
      try {
        validateModelContract(session);
      } catch (_) {
        session.release();
        rethrow;
      }
      _session = session;
    } finally {
      options.release();
      rootBundle.evict(activeModelAsset);
    }
  }

  Future<ScreeningResult> predict(File file) async {
    const measure = bool.fromEnvironment('OFFIRST_PROFILE_INFERENCE');
    final watch = measure ? (Stopwatch()..start()) : null;
    final cold = _session == null;
    await _ensureSession();
    final result = useExperimentalV21
        ? await _predictV21(file, _session!.address)
        : await _predictOffThread(file.path, _session!.address);
    if (watch != null) {
      // Timing only. Keep image paths and patient information out of logs.
      debugPrint(
          'OFFIRST_INFERENCE cold=$cold total_us=${watch.elapsedMicroseconds}');
    }
    return result;
  }

  void dispose() {
    _session?.release();
    _session = null;
  }
}

Future<ScreeningResult> _predictV21(File file, int sessionAddress) async {
  if (await file.length() > maxEncodedImageBytes) {
    throw const FormatException('Image file exceeds 32 MiB.');
  }
  final encoded = await file.readAsBytes();
  final tensorBytes =
      await _v21Preprocessing.invokeMethod<Uint8List>('prepareTensor', encoded);
  if (tensorBytes == null || tensorBytes.length != 3 * 512 * 512 * 4) {
    throw StateError('v2.1 preprocessing returned an invalid tensor.');
  }
  final input = Float32List.view(tensorBytes.buffer, tensorBytes.offsetInBytes,
      tensorBytes.lengthInBytes ~/ Float32List.bytesPerElement);
  return Isolate.run(() => _predictInput(input, sessionAddress),
      debugName: 'OFFIRST v2.1 inference');
}

Future<ScreeningResult> _predictOffThread(String path, int sessionAddress) =>
    Isolate.run(() => _predict(path, sessionAddress),
        debugName: 'OFFIRST inference');

ScreeningResult _predict(String path, int sessionAddress) {
  final input = prepareFundusFile(File(path));
  return _predictInput(input, sessionAddress);
}

ScreeningResult _predictInput(Float32List input, int sessionAddress) {
  final tensor =
      OrtValueTensor.createTensorWithDataList(input, modelInputShape);
  try {
    final runOptions = OrtRunOptions();
    try {
      // This isolate borrows the session. The screening screen waits for the
      // operation before disposing the owning session.
      final outputs = OrtSession.fromAddress(sessionAddress).run(
        runOptions,
        {modelInput: tensor},
        [modelOutput],
      );
      try {
        final output = outputs.single;
        if (output is! OrtValueTensor) {
          throw StateError('Model returned a non-tensor output.');
        }
        return ScreeningResult(readModelProbability(output));
      } finally {
        for (final output in outputs) {
          output?.release();
        }
      }
    } finally {
      runOptions.release();
    }
  } finally {
    tensor.release();
  }
}
