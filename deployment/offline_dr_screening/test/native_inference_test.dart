import 'dart:io';
import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;
import 'package:offirst_dr_screening/model_contract.dart';
import 'package:offirst_dr_screening/screening_service.dart';
import 'package:onnxruntime/onnxruntime.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  final noNative =
      !Platform.isWindows || Platform.environment['OFFIRST_NATIVE_TEST'] != '1';
  final noServiceNative = noNative || useExperimentalV21;

  test('rejects wrong output type, dimensions and probability values', () {
    final wrongType =
        OrtValueTensor.createTensorWithDataList(Int32List(1), [1, 1]);
    final wrongShape =
        OrtValueTensor.createTensorWithDataList(Float32List(1), [1]);
    final outside = OrtValueTensor.createTensorWithDataList(
        Float32List.fromList([1.5]), [1, 1]);
    final good = OrtValueTensor.createTensorWithDataList(
        Float32List.fromList([.5]), [1, 1]);
    for (final tensor in [wrongType, wrongShape, outside, good]) {
      addTearDown(tensor.release);
    }
    expect(() => readModelProbability(wrongType), throwsStateError);
    expect(() => readModelProbability(wrongShape), throwsStateError);
    expect(() => readModelProbability(outside), throwsStateError);
    expect(readModelProbability(good), .5);
  }, skip: noNative);

  test('background ONNX inference matches direct inference and reuses session',
      () async {
    final model = await rootBundle.load(activeModelAsset);
    final options = OrtSessionOptions();
    final session = OrtSession.fromBuffer(model.buffer.asUint8List(), options);
    final runOptions = OrtRunOptions();
    const plane = 512 * 512;
    final input = Float32List(3 * plane)
      ..fillRange(0, plane, -0.485 / 0.229)
      ..fillRange(plane, 2 * plane, -0.456 / 0.224)
      ..fillRange(2 * plane, 3 * plane, -0.406 / 0.225);
    final tensor =
        OrtValueTensor.createTensorWithDataList(input, [1, 3, 512, 512]);
    final outputs =
        session.run(runOptions, {'fundus_image': tensor}, ['dr_probability']);
    addTearDown(() {
      for (final output in outputs) {
        output?.release();
      }
      tensor.release();
      runOptions.release();
      session.release();
      options.release();
    });
    Object? expected = (outputs.single as OrtValueTensor).value;
    while (expected is List) {
      expected = expected.single;
    }
    final probability = (expected as num).toDouble();

    final folder = Directory.systemTemp.createTempSync('offirst-native-test-');
    addTearDown(() => folder.deleteSync(recursive: true));
    final file = File('${folder.path}/black.png')
      ..writeAsBytesSync(img.encodePng(img.Image(width: 512, height: 512)));
    final service = ScreeningService.withValidatedTestInput();
    addTearDown(service.dispose);

    for (var run = 0; run < 2; run++) {
      final result = await service.predict(file);
      expect(result.probability, closeTo(probability, 1e-5));
      expect(result.detected, probability >= screeningThreshold);
    }
  }, skip: noServiceNative);

  test(
      'production service fails closed for decodable images without a detector',
      () async {
    _mockV21DecoderChannel();
    final folder = Directory.systemTemp.createTempSync('offirst-input-gate-');
    addTearDown(() => folder.deleteSync(recursive: true));
    final valid = File('${folder.path}/valid.png')
      ..writeAsBytesSync(img.encodePng(img.Image(width: 32, height: 32)));
    final service = ScreeningService();
    addTearDown(service.dispose);
    await expectLater(
      service.predict(valid),
      throwsA(isA<ScreeningFailure>().having(
          (failure) => failure.stage, 'stage', ScreeningStage.inputValidation)),
    );
    expect(service.readiness, ModelReadiness.initializing,
        reason: 'Unvalidated images must be rejected before model loading.');
  });

  test('production service rejects corrupted image bytes before classification',
      () async {
    _mockV21DecoderChannel();
    final folder = Directory.systemTemp.createTempSync('offirst-corrupt-');
    addTearDown(() => folder.deleteSync(recursive: true));
    final file = File('${folder.path}/corrupt.png')
      ..writeAsBytesSync([0x89, 0x50, 0x4e, 0x47, 0, 0, 0]);
    final service = ScreeningService();
    addTearDown(service.dispose);
    await expectLater(
      service.predict(file),
      throwsA(isA<ScreeningFailure>().having(
          (failure) => failure.stage, 'stage', ScreeningStage.imageDecode)),
    );
  });

  test('selected ONNX asset runs and returns a probability', () async {
    final model = await rootBundle.load(activeModelAsset);
    final options = OrtSessionOptions();
    final session = OrtSession.fromBuffer(
        model.buffer.asUint8List(model.offsetInBytes, model.lengthInBytes),
        options);
    addTearDown(() {
      session.release();
      options.release();
    });
    validateModelContract(session);
    final tensor = OrtValueTensor.createTensorWithDataList(
        Float32List(3 * 512 * 512), modelInputShape);
    final runOptions = OrtRunOptions();
    final outputs =
        session.run(runOptions, {modelInput: tensor}, [modelOutput]);
    addTearDown(() {
      for (final output in outputs) {
        output?.release();
      }
      tensor.release();
      runOptions.release();
    });
    final probability = readModelProbability(outputs.single as OrtValueTensor);
    expect(probability, inInclusiveRange(0.0, 1.0));
  }, skip: noNative);

  test('uses the configured threshold at, below and above its boundary', () {
    expect(const ScreeningResult(activeScreeningThreshold - 1e-9).detected,
        isFalse);
    expect(const ScreeningResult(activeScreeningThreshold).detected, isTrue);
    expect(const ScreeningResult(activeScreeningThreshold + 1e-9).detected,
        isTrue);
  });

  final v21Fixtures = Platform.environment['OFFIRST_V21_FIXTURES'];
  test(
      'v2.1 service replays Python tensors through real ONNX and handles native errors',
      () async {
    final rows =
        (jsonDecode(File('$v21Fixtures/manifest.json').readAsStringSync())
            as Map)['fixtures'] as List;
    final service = ScreeningService();
    addTearDown(service.dispose);
    const channel = MethodChannel('offirst/v2_1_preprocessing');
    final messenger =
        TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger;
    addTearDown(() => messenger.setMockMethodCallHandler(channel, null));
    for (final row in rows) {
      final file = File('$v21Fixtures/${row['name']}');
      final tensor = Uint8List.fromList(
          gzip.decode(File('${file.path}.f32.gz').readAsBytesSync()));
      messenger.setMockMethodCallHandler(channel, (call) async {
        expect(call.method, 'prepareTensor');
        expect(call.arguments, file.readAsBytesSync());
        return tensor; // Native preprocessing is mocked; actual ORT and service execute.
      });
      final result = await service.predict(file);
      expect(result.probability, closeTo(row['probability'] as num, 1e-5));
      expect(result.detected, row['at_or_above']);
      expect(service.readiness, ModelReadiness.ready);
    }
    await expectLater(
        service.predict(File('$v21Fixtures/missing.jpg')),
        throwsA(isA<ScreeningFailure>()
            .having((e) => e.stage, 'stage', ScreeningStage.imageAccess)));
    messenger.setMockMethodCallHandler(channel, (_) async {
      throw PlatformException(
          code: 'image_decode', message: 'PRIVATE PATH MUST NOT ESCAPE');
    });
    await expectLater(
        service.predict(File('$v21Fixtures/gallery.jpg')),
        throwsA(isA<ScreeningFailure>()
            .having((e) => e.stage, 'stage', ScreeningStage.imageDecode)
            .having((e) => e.message, 'message', isNot(contains('PRIVATE')))));
  }, skip: noNative || !useExperimentalV21 || v21Fixtures == null);
}

void _mockV21DecoderChannel() {
  const channel = MethodChannel('offirst/v2_1_preprocessing');
  final messenger =
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger;
  messenger.setMockMethodCallHandler(channel, (call) async {
    final encoded = call.arguments as Uint8List;
    if (img.decodeImage(encoded) == null) {
      throw PlatformException(code: 'image_decode');
    }
    return Uint8List(3 * 512 * 512 * Float32List.bytesPerElement);
  });
  addTearDown(() => messenger.setMockMethodCallHandler(channel, null));
}
