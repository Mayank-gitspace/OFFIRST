import 'dart:io';
import 'dart:isolate';

import 'package:flutter/foundation.dart';
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

enum ModelReadiness { initializing, ready, unavailable }

enum ScreeningStage {
  imageAccess,
  imageDecode,
  preprocessing,
  modelLoad,
  modelContract,
  inference,
}

class ScreeningFailure implements Exception {
  const ScreeningFailure(this.stage, this.message);
  final ScreeningStage stage;
  final String message;

  @override
  String toString() => message;
}

// Log code locations and failure categories only. Exceptions may contain image
// paths or patient information, so their raw messages are deliberately excluded.
void logScreeningFailure(ScreeningStage stage, Object error, StackTrace stack) {
  debugPrint('OFFIRST_ERROR stage=${stage.name} type=${error.runtimeType}');
  final frames = RegExp(r'^#\d+\s+([^\r\n(]+)', multiLine: true)
      .allMatches(stack.toString())
      .take(6);
  for (final frame in frames) {
    debugPrint('OFFIRST_FRAME ${frame.group(1)?.trim()}');
  }
}

class ScreeningService extends ChangeNotifier {
  ScreeningService();

  @visibleForTesting
  ScreeningService.forTesting({
    required Future<void> Function() initializeModel,
    required Future<ScreeningResult> Function(File) runPrediction,
  })  : _testInitialize = initializeModel,
        _testPredict = runPrediction;

  Future<void> Function()? _testInitialize;
  Future<ScreeningResult> Function(File)? _testPredict;
  OrtSession? _session;
  Future<void>? _initialization;
  ModelReadiness _readiness = ModelReadiness.initializing;
  ScreeningFailure? _initializationError;
  bool _disposed = false;
  int _activePredictions = 0;

  ModelReadiness get readiness => _readiness;
  ScreeningFailure? get initializationError => _initializationError;

  Future<void> initialize() {
    if (_disposed) {
      return Future.error(const ScreeningFailure(
          ScreeningStage.modelLoad, 'The screening service has closed.'));
    }
    if (_readiness == ModelReadiness.ready) return Future.value();
    return _initialization ??= _initialize().whenComplete(() {
      _initialization = null;
    });
  }

  Future<void> _initialize() async {
    _readiness = ModelReadiness.initializing;
    _initializationError = null;
    notifyListeners();
    try {
      if (_testInitialize != null) {
        await _testInitialize!();
      } else {
        await _loadSession();
      }
      if (!_disposed) {
        _readiness = ModelReadiness.ready;
        notifyListeners();
      }
    } catch (error, stack) {
      final failure = error is ScreeningFailure
          ? error
          : const ScreeningFailure(ScreeningStage.modelLoad,
              'ONNX model could not be loaded. Restart OFFIRST or reinstall the verified APK without clearing app data.');
      logScreeningFailure(failure.stage, error, stack);
      if (!_disposed) {
        _readiness = ModelReadiness.unavailable;
        _initializationError = failure;
        notifyListeners();
      }
      throw failure;
    }
  }

  Future<void> _loadSession() async {
    final model = await rootBundle.load(activeModelAsset);
    final options = OrtSessionOptions();
    try {
      options.setIntraOpNumThreads(2);
      options.setInterOpNumThreads(1);
      options.setSessionGraphOptimizationLevel(
          GraphOptimizationLevel.ortEnableAll);
      final session = OrtSession.fromBuffer(
          model.buffer.asUint8List(model.offsetInBytes, model.lengthInBytes),
          options);
      try {
        validateModelContract(session);
      } catch (error, stack) {
        session.release();
        logScreeningFailure(ScreeningStage.modelContract, error, stack);
        throw const ScreeningFailure(ScreeningStage.modelContract,
            'The packaged model has an unexpected input/output contract. Install the verified OFFIRST APK.');
      }
      if (_disposed) {
        session.release();
      } else {
        _session = session;
      }
    } finally {
      options.release();
      rootBundle.evict(activeModelAsset);
    }
  }

  Future<ScreeningResult> predict(File file) async {
    if (_disposed) {
      throw const ScreeningFailure(
          ScreeningStage.modelLoad, 'The screening service has closed.');
    }
    _activePredictions++;
    try {
      await initialize();
      if (_disposed) {
        throw const ScreeningFailure(
            ScreeningStage.modelLoad, 'The screening service has closed.');
      }
      final result = _testPredict != null
          ? await _testPredict!(file)
          : useExperimentalV21
              ? await _predictV21(file, _session!.address)
              : await _predictOffThread(file.path, _session!.address);
      if (!result.probability.isFinite ||
          result.probability < 0 ||
          result.probability > 1) {
        throw const ScreeningFailure(ScreeningStage.inference,
            'Inference returned an invalid probability. Please retry.');
      }
      return result;
    } catch (error, stack) {
      final failure = error is ScreeningFailure
          ? error
          : const ScreeningFailure(ScreeningStage.inference,
              'Inference failed. Retry this image or restart OFFIRST.');
      logScreeningFailure(failure.stage, error, stack);
      throw failure;
    } finally {
      _activePredictions--;
      if (_disposed && _activePredictions == 0) _releaseSession();
    }
  }

  void _releaseSession() {
    _session?.release();
    _session = null;
  }

  @override
  void dispose() {
    if (_disposed) return;
    _disposed = true;
    if (_activePredictions == 0) _releaseSession();
    super.dispose();
  }
}

/// StandardMethodCodec returns Uint8List views with no float-alignment guarantee.
/// Read the explicitly little-endian wire format into an aligned Float32List.
Float32List decodeV21Tensor(Uint8List bytes) {
  const count = 3 * 512 * 512;
  if (bytes.lengthInBytes != count * Float32List.bytesPerElement) {
    throw const ScreeningFailure(ScreeningStage.preprocessing,
        'Image preprocessing returned an invalid tensor size.');
  }
  final wire = ByteData.sublistView(bytes);
  final tensor = Float32List(count);
  for (var i = 0; i < count; i++) {
    final value = wire.getFloat32(i * 4, Endian.little);
    if (!value.isFinite) {
      throw const ScreeningFailure(ScreeningStage.preprocessing,
          'Image preprocessing returned invalid tensor values.');
    }
    tensor[i] = value;
  }
  return tensor;
}

Future<ScreeningResult> _predictV21(File file, int sessionAddress) async {
  final Uint8List encoded;
  try {
    final size = await file.length();
    if (size <= 0 || size > maxEncodedImageBytes) {
      throw const ScreeningFailure(ScreeningStage.imageAccess,
          'The image is empty or exceeds the 32 MiB file limit. Select another image.');
    }
    encoded = await file.readAsBytes();
  } on ScreeningFailure {
    rethrow;
  } catch (error, stack) {
    logScreeningFailure(ScreeningStage.imageAccess, error, stack);
    throw const ScreeningFailure(ScreeningStage.imageAccess,
        'Image file could not be accessed. Select it again from the gallery or camera.');
  }
  final Uint8List? tensorBytes;
  try {
    tensorBytes = await _v21Preprocessing.invokeMethod<Uint8List>(
        'prepareTensor', encoded);
  } on PlatformException catch (error, stack) {
    final stage = switch (error.code) {
      'image_access' || 'image' => ScreeningStage.imageAccess,
      'image_decode' || 'image_format' => ScreeningStage.imageDecode,
      _ => ScreeningStage.preprocessing,
    };
    logScreeningFailure(stage, error, stack);
    final message = switch (error.code) {
      'image_access' ||
      'image' =>
        'Image bytes could not be accessed. Select the image again.',
      'image_decode' =>
        'Image decoding failed. Select a valid JPEG or PNG image.',
      'image_format' => 'Unsupported image format. Select a JPEG or PNG image.',
      'image_size' =>
        'Image exceeds available memory or the 24 megapixel limit. Select a smaller image.',
      'runtime' =>
        'The OpenCV image runtime is unavailable. Restart OFFIRST or reinstall the verified APK.',
      _ =>
        'Image preprocessing failed. Retry this image or select another JPEG or PNG.',
    };
    throw ScreeningFailure(stage, message);
  } catch (error, stack) {
    logScreeningFailure(ScreeningStage.preprocessing, error, stack);
    throw const ScreeningFailure(ScreeningStage.preprocessing,
        'The Android image preprocessing service is unavailable. Restart OFFIRST.');
  }
  if (tensorBytes == null) {
    throw const ScreeningFailure(ScreeningStage.preprocessing,
        'Image preprocessing returned no tensor.');
  }
  final input = decodeV21Tensor(tensorBytes);
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
