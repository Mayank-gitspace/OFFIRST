import 'dart:io';
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
    final service = ScreeningService();
    addTearDown(service.dispose);

    for (var run = 0; run < 2; run++) {
      final result = await service.predict(file);
      expect(result.probability, closeTo(probability, 1e-5));
      expect(result.detected, probability >= screeningThreshold);
    }
  }, skip: noServiceNative);

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
}
