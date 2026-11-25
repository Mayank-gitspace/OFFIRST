import 'dart:async';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:image_picker/image_picker.dart';
import 'package:offirst_dr_screening/patient_store.dart';
import 'package:offirst_dr_screening/screening_page.dart';
import 'package:offirst_dr_screening/screening_service.dart';

class CaptureStore extends PatientStore {
  VisitRecord? saved;
  String? patientId;
  @override
  Future<void> addVisit(String patient, VisitRecord visit) async {
    patientId = patient;
    saved = visit;
  }
}

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  test('real method codec payload is unaligned; decoder preserves every float',
      () {
    const count = 3 * 512 * 512;
    final wire = ByteData(count * 4);
    for (var i = 0; i < count; i++) {
      wire.setFloat32(i * 4, (i % 256 - 128) / 64, Endian.little);
    }
    const codec = StandardMethodCodec();
    final payload = codec.decodeEnvelope(
        codec.encodeSuccessEnvelope(wire.buffer.asUint8List())) as Uint8List;
    expect(payload.offsetInBytes % 4, isNot(0));
    expect(() => Float32List.view(payload.buffer, payload.offsetInBytes),
        throwsArgumentError); // Original v2.1 implementation fails here.
    final tensor = decodeV21Tensor(payload);
    for (var i = 0; i < count; i++) {
      if (tensor[i] != wire.getFloat32(i * 4, Endian.little)) {
        fail('Tensor value differs at $i');
      }
    }
    expect(
        () => decodeV21Tensor(Uint8List(2)), throwsA(isA<ScreeningFailure>()));
    wire.setFloat32(0, double.nan, Endian.little);
    expect(() => decodeV21Tensor(wire.buffer.asUint8List()),
        throwsA(isA<ScreeningFailure>()));
  });

  test('readiness waits for validation; initialization failures support retry',
      () async {
    final completion = Completer<void>();
    var attempts = 0;
    final service = ScreeningService.forTesting(
      initializeModel: () {
        attempts++;
        return completion.future;
      },
      runPrediction: (_) async => const ScreeningResult(.25),
    );
    addTearDown(service.dispose);
    final first = service.initialize();
    final second = service.initialize();
    expect(service.readiness, ModelReadiness.initializing);
    expect(attempts, 1);
    completion.complete();
    await Future.wait([first, second]);
    expect(service.readiness, ModelReadiness.ready);

    var failLoading = true;
    final retry = ScreeningService.forTesting(
      initializeModel: () async {
        if (failLoading) throw StateError('private/path');
      },
      runPrediction: (_) async => const ScreeningResult(.25),
    );
    addTearDown(retry.dispose);
    await expectLater(retry.initialize(), throwsA(isA<ScreeningFailure>()));
    expect(retry.readiness, ModelReadiness.unavailable);
    expect(retry.initializationError!.message, isNot(contains('private/path')));
    failLoading = false;
    await retry.initialize();
    expect(retry.readiness, ModelReadiness.ready);
  });

  testWidgets(
      'patient required; mixed failures retain eye on retry and save actual scores',
      (tester) async {
    tester.view.physicalSize = const Size(900, 2000);
    tester.view.devicePixelRatio = 1;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);
    final store = CaptureStore();
    var retrySucceeded = false;
    final service = ScreeningService.forTesting(
      initializeModel: () async {},
      runPrediction: (file) async {
        if (file.path == 'retry.png' && !retrySucceeded) {
          throw const ScreeningFailure(
              ScreeningStage.imageDecode, 'Image decoding failed.');
        }
        return ScreeningResult(file.path == 'left.png' ? .7 : .2);
      },
    );
    addTearDown(service.dispose);
    var picks = 0;
    await tester.pumpWidget(MaterialApp(
        home: Scaffold(
            body: ScreeningPage(
      store: store,
      service: service,
      onSaved: () {},
      patients: Future.value([
        PatientRecord(
            id: 'OFF-SYNTHETIC',
            name: 'Synthetic',
            createdAt: DateTime.utc(2026))
      ]),
      pickImages: (_) async => ++picks == 1
          ? [XFile('right.png'), XFile('retry.png')]
          : [XFile('left.png')],
    ))));
    await tester.pumpAndSettle();
    final add = find.widgetWithText(FilledButton, 'Add fundus images');
    expect(tester.widget<FilledButton>(add).onPressed, isNull);
    await tester.tap(find.byType(DropdownButtonFormField<String>));
    await tester.pumpAndSettle();
    await tester.tap(find.textContaining('OFF-SYNTHETIC').last);
    await tester.pumpAndSettle();
    await tester.tap(add);
    await tester.pumpAndSettle();
    expect(find.textContaining('Image decoding failed.'), findsOneWidget);
    final save = find.widgetWithText(FilledButton, 'Finalize and save visit');
    expect(tester.widget<FilledButton>(save).onPressed, isNull);
    retrySucceeded = true;
    await tester.ensureVisible(find.text('Retry failed images'));
    await tester.tap(find.text('Retry failed images'));
    await tester.pumpAndSettle();
    expect(find.text('2 screened image(s)'), findsOneWidget);
    await tester.ensureVisible(find.byType(SegmentedButton<Eye>));
    await tester.tap(find.text('Left eye (OS)').first);
    await tester.pumpAndSettle();
    await tester.ensureVisible(add);
    await tester.tap(add);
    await tester.pumpAndSettle();
    await tester.ensureVisible(save);
    await tester.tap(save);
    await tester.pumpAndSettle();
    expect(store.patientId, 'OFF-SYNTHETIC');
    expect(store.saved!.images.map((i) => i.eye),
        [Eye.right, Eye.right, Eye.left]);
    expect(store.saved!.images.map((i) => i.probability), [.2, .2, .7]);
    expect(
        store.saved!.images.every((i) =>
            i.modelVersion == activeScreeningModelVersion &&
            i.threshold == activeScreeningThreshold),
        isTrue);
  });
}
