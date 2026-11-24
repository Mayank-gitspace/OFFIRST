import 'dart:io';
import 'dart:isolate';
import 'dart:typed_data';

import 'package:flutter/foundation.dart' show debugPrint;
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:onnxruntime/onnxruntime.dart';

const screeningThreshold = 0.48;
const screeningModelVersion = 'offirst-dr-onnx-b8fc1019';

class ScreeningResult {
  const ScreeningResult(this.probability);

  final double probability;

  bool get detected => probability >= screeningThreshold;
}

class ScreeningService {
  OrtSession? _session;

  Future<void> _ensureSession() async {
    if (_session != null) return;
    final model = await rootBundle.load('assets/offirst_dr_model.onnx');
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
      if (session.inputNames.length != 1 ||
          session.inputNames.single != 'fundus_image' ||
          session.outputNames.length != 1 ||
          session.outputNames.single != 'dr_probability') {
        session.release();
        throw StateError('Unexpected ONNX model inputs or outputs.');
      }
      _session = session;
    } finally {
      options.release();
      rootBundle.evict('assets/offirst_dr_model.onnx');
    }
  }

  Future<ScreeningResult> predict(File file) async {
    const measure = bool.fromEnvironment('OFFIRST_PROFILE_INFERENCE');
    final watch = measure ? (Stopwatch()..start()) : null;
    final cold = _session == null;
    await _ensureSession();
    final result = await _predictOffThread(file.path, _session!.address);
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

Future<ScreeningResult> _predictOffThread(String path, int sessionAddress) =>
    Isolate.run(() => _predict(path, sessionAddress),
        debugName: 'OFFIRST inference');

ScreeningResult _predict(String path, int sessionAddress) {
  final bytes = File(path).readAsBytesSync();
  final decoded = img.decodeImage(bytes)?.convert(numChannels: 3);
  if (decoded == null) {
    throw const FormatException('Unsupported or invalid image.');
  }

  // image's bilinear resampler operates on 8-bit RGB pixels. Keep that step
  // before float conversion, matching PIL's RGB -> bilinear -> float pipeline.
  final rgb = img.copyResize(
    decoded,
    width: 512,
    height: 512,
    interpolation: img.Interpolation.linear,
  );
  final input = Float32List(3 * 512 * 512);
  const means = [0.485, 0.456, 0.406];
  const stds = [0.229, 0.224, 0.225];
  for (var y = 0; y < 512; y++) {
    for (var x = 0; x < 512; x++) {
      final pixel = rgb.getPixel(x, y);
      final offset = y * 512 + x;
      input[offset] = ((pixel.r / 255.0) - means[0]) / stds[0];
      input[512 * 512 + offset] = ((pixel.g / 255.0) - means[1]) / stds[1];
      input[2 * 512 * 512 + offset] = ((pixel.b / 255.0) - means[2]) / stds[2];
    }
  }

  final tensor = OrtValueTensor.createTensorWithDataList(input, [
    1,
    3,
    512,
    512,
  ]);
  try {
    final runOptions = OrtRunOptions();
    try {
      // This isolate borrows the session. The screening screen waits for the
      // operation before disposing the owning session.
      final outputs = OrtSession.fromAddress(sessionAddress).run(
        runOptions,
        {'fundus_image': tensor},
        ['dr_probability'],
      );
      try {
        final output = outputs.single;
        if (output is! OrtValueTensor) {
          throw StateError('Model returned a non-tensor output.');
        }
        final data = output.value;
        final probability = _firstNumber(data);
        if (!probability.isFinite || probability < 0 || probability > 1) {
          throw StateError('Model returned an invalid probability.');
        }
        return ScreeningResult(probability);
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

double _firstNumber(Object? value) {
  if (value is num) return value.toDouble();
  if (value is List && value.isNotEmpty) return _firstNumber(value.first);
  throw StateError('Model returned an unsupported probability tensor.');
}
