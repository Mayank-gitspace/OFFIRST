import 'dart:io';
import 'dart:isolate';

import 'package:flutter/foundation.dart' show debugPrint;
import 'package:flutter/services.dart';

import 'package:onnxruntime/onnxruntime.dart';

import 'model_contract.dart';
import 'model_preprocessing.dart';

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
      try {
        validateModelContract(session);
      } catch (_) {
        session.release();
        rethrow;
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
  final input = prepareFundusTensor(File(path).readAsBytesSync());
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
